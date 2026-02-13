import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from typing import Callable, Tuple, Union
from torch import Tensor
from torch.nn import Parameter
from configs import ParticleConfig

class CustomNNConv(MessagePassing):
    def __init__(self, config : ParticleConfig):
        super().__init__()

        self.config = config


        state_channels = config.state_dim


        out_channels = config.message_out_channels


        mlp = []
        mlp.append(Linear(config.message_channels, 32))
        mlp.append(torch.nn.ReLU())
        mlp.append(Linear(32, out_channels))
        
        self.nn = torch.nn.Sequential(*mlp)

        self.lin = Linear(state_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        reset(self.lin)


    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:

        if not isinstance(x, Tensor):
            raise ValueError("I dont understand Pytorch-error!!!")
        
        x_for_propagate = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out_msg = self.propagate(edge_index, x=x_for_propagate, edge_attr=edge_attr, size=size)

        
        out = out_msg + self.lin(x)

        return out


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.nn(edge_attr)  # I think this is just what i want
        weight = self.nn(edge_attr)
        weight = weight.view(-1, self.in_channels_l, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)
