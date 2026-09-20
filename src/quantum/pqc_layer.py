""" Affine-Equivariant Quantum Convolutional Neural Network (QCNN). """
import torch
import torch.nn as nn
import pennylane as qml

class PQCLayer(nn.Module):
    def __init__(self, n_qubits=8, n_layers=1):
        super(PQCLayer, self).__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Dual-Head Classical-to-Quantum Bottleneck
        self.structure_head = nn.Linear(n_qubits, n_qubits)
        self.scale_head = nn.Sequential(
            nn.Linear(n_qubits, 1),
            nn.Softplus() # Ensures scale scalar lambda is strictly positive
        )
        
        # QCNN Parameterization (Convolution and Pooling)
        self.weights_conv1 = nn.Parameter(torch.randn(4, 3) * 0.05)
        self.weights_pool1 = nn.Parameter(torch.randn(4, 3) * 0.05)
        self.weights_conv2 = nn.Parameter(torch.randn(2, 3) * 0.05)
        self.weights_pool2 = nn.Parameter(torch.randn(2, 3) * 0.05)
        
        # Projects the 2 surviving QCNN measurements back to 8 dimensions
        self.out_proj = nn.Linear(2, n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(structure, scale, w_c1, w_p1, w_c2, w_p2):
            state = torch.zeros(2**n_qubits, dtype=torch.complex128, device=structure.device)
            state[0] = 1.0 + 0.0j
            qml.StatePrep(state, wires=range(n_qubits))
            
            # Affine-Equivariant Encoding
            for i in range(n_qubits):
                qml.RX(structure[i] * scale, wires=i)
                qml.RY(structure[i] * scale, wires=i)
                qml.RZ(structure[i] * scale, wires=i)
                
            # Level 1 Convolution & Pooling
            for i, wire in enumerate([0, 2, 4, 6]):
                qml.Rot(*w_c1[i], wires=wire)
                qml.Rot(*w_c1[i], wires=wire+1)
                qml.CNOT(wires=[wire, wire+1])
            for i, wire in enumerate([0, 2, 4, 6]):
                qml.CRot(*w_p1[i], wires=[wire, wire+1])
                
            # Level 2 Convolution & Pooling
            for i, pairs in enumerate([(1,3), (5,7)]):
                qml.Rot(*w_c2[i], wires=pairs[0])
                qml.Rot(*w_c2[i], wires=pairs[1])
                qml.CNOT(wires=[pairs[0], pairs[1]])
            qml.CRot(*w_p2[0], wires=[1, 3])
            qml.CRot(*w_p2[1], wires=[5, 7])
            
            return [qml.expval(qml.PauliZ(3)), qml.expval(qml.PauliZ(7))]
        self.circuit = circuit

    def forward(self, x):
        structure = self.structure_head(x)
        scale = self.scale_head(x).squeeze(-1)
        
        batch_size = x.shape[0]
        q_out = [torch.stack(self.circuit(structure[b], scale[b],
                                           self.weights_conv1, self.weights_pool1,
                                          self.weights_conv2, self.weights_pool2))
                  for b in range(batch_size)]
        q_out = torch.stack(q_out)
        
        # Cast the 64-bit quantum tensor down to 16-bit to match AMP format
        q_out = q_out.type_as(x)
        return self.out_proj(q_out)