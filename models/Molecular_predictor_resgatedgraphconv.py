import torch
import torch.nn.functional as F
from torch.nn import GRUCell, Linear, BatchNorm1d
from torch_geometric.nn import GATConv, global_add_pool, ResGatedGraphConv

class MolPredictor(torch.nn.Module):
    
    def __init__(
        self,
        in_channels=9,
        hidden_channels=97,
        out_channels=1,
        edge_dim=3,
        num_layers=7,
        num_timesteps=7,
        dropout: float = 0.49060123078514695,
        
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        
        self.lin1 = Linear(in_channels, hidden_channels)
        self.gate_conv = ResGatedGraphConv(hidden_channels, hidden_channels)
        self.gru = GRUCell(hidden_channels, hidden_channels)
        
        self.atom_convs = torch.nn.ModuleList()
        self.atom_batchnorms = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()

        self.mol_gru_grads = None,
        self.final_mol_conv_grads = None
        self.firstlayer_atom_gru_grads = None
        self.atom_gru_grads = None

        
        for _ in range(num_layers - 1):
            conv = ResGatedGraphConv(hidden_channels, hidden_channels)
            self.atom_convs.append(conv)
            self.atom_batchnorms.append(BatchNorm1d(hidden_channels))
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))
        
        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)
        
        self.lin2 = Linear(hidden_channels, out_channels)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    
    def firstlayer_atomgru_activations_hook(self, grad):
        self.firstlayer_atom_gru_grads = grad

    def atomgru_activations_hook(self, grad):
        # pass
        # print("inside atomgru_activations_hook")
        # print(grad.shape)
        self.atom_gru_grads = grad
    
    def gru_activations_hook(self, output_grad):
        # pass
        # print("inside atomgru_activations_hook")
        # print(grad.shape)
        self.mol_gru_grads = output_grad 
    
    def conv_activations_hook(self,  output_grad):
        # pass
        # print("inside conv_activations_hook")
        self.final_mol_conv_grads = output_grad  
    
    def firstlayer_atomgru_activations_hook(self, grad):
        # pass
        # print("inside atomgru_activations_hook")
        # print(grad.shape)
        self.firstlayer_atom_gru_grads = grad

    def forward(self, x, edge_index, batch):
        # x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.gate_conv(x, edge_index))
        x = self.gru(h, x).relu_()


        with torch.enable_grad():
            self.firstlayer_atom_gru_activation = self.gru(h, x).relu_()
        self.firstlayer_atom_gru_activation.register_hook(self.firstlayer_atomgru_activations_hook)
        x = self.firstlayer_atom_gru_activation
        
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            # x = gru(h, x).relu_()
            with torch.enable_grad():
                self.atom_gru_activation = gru(h, x).relu_()
            self.atom_gru_activation.register_hook(self.atomgru_activations_hook)
            x= self.atom_gru_activation
            
        
        for conv, batchnorm, gru in zip(self.atom_convs, self.atom_batchnorms, self.atom_grus):
            h = F.elu_(conv(x, edge_index))
            h = batchnorm(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            with torch.enable_grad():
                self.atom_gru_activation = gru(h, x).relu_()
            self.atom_gru_activation.register_hook(self.atomgru_activations_hook)
            x= self.atom_gru_activation
            
            # x = gru(h, x).relu_()

        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()
        
        return self.lin2(out)