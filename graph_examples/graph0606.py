import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import pdb

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add') # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        pdb.set_trace()

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [num_edges, out_channels]

        # Step 3: Normalize node features.
        pdb.set_trace()
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        pdb.set_trace()

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [num_nodes, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

# x = torch.tensor([6,8,7,5])
# input = torch.randn(4, 20)
# edge =  torch.empty(2, 8, dtype=torch.long)
# edge[0] = torch.Tensor([1,2,3,3,0,1,0,2], type = torch.long)
# edge[1] = torch.Tensor([0,0,1,0,2,2,3,3], type = torch.long)

# edge_index = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 2],
#                            [2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
x = torch.rand([4,20]).to('cuda')
edge_index =  torch.tensor([[0, 1, 2, 3],[2, 1, 0, 3]], dtype=torch.long)


relation = torch.empty(2, 5, dtype=torch.long).to('cuda')
relation[0] = torch.Tensor([0, 1, 2, 3, 0])
relation[1] = torch.Tensor([2, 1, 0, 3, 2])

# print(edge_index.shape, type(edge_index))

print(relation.dtype)
# print(isinstance(edge_index, Tensor))
# relation = torch.empty(2, 32 * 2, dtype=torch.long).to('cuda')

# pdb.set_trace()

# relation[0] = torch.Tensor(list(range(2)) * 32)  # , type=torch.long)
# relation[1] = torch.rand(64)

# pdb.set_trace()
a = GCNConv(20,30).to('cuda')
b = a(x, relation)
print('b', b.shape)