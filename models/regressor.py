import torch
from sde import VPSDE, VESDE
from models.layers import DenseGCNConv
from utils.graph_utils import mask_x, mask_adjs


class Regressor(torch.nn.Module):
    def __init__(self, max_node_num, max_feat_num, depth, nhid, dropout):
        super().__init__()

        self.linears = torch.nn.ModuleList([torch.nn.Linear(max_feat_num, nhid)])
        for _ in range(depth - 1):
            self.linears.append(torch.nn.Linear(nhid, nhid))
        
        self.convs = torch.nn.ModuleList([DenseGCNConv(nhid, nhid) for _ in range(depth)])

        dim = max_feat_num + depth * nhid
        dim_out = nhid

        self.sigmoid_linear = torch.nn.Sequential(torch.nn.Linear(dim, dim_out), torch.nn.Sigmoid())
        self.tanh_linear = torch.nn.Sequential(torch.nn.Linear(dim, dim_out), torch.nn.Tanh())

        self.final_linear = [torch.nn.Linear(dim_out, nhid),
                             torch.nn.ReLU(),
                             torch.nn.Dropout(p=dropout),
                             torch.nn.Linear(nhid, 1),
                             torch.nn.Sigmoid()]
        self.final_linear = torch.nn.Sequential(*self.final_linear)

    def forward(self, x, adj, flags):
        xs = [x]
        out = x
        for lin, conv in zip(self.linears, self.convs):
            out = conv(lin(out), adj)
            out = torch.tanh(out)
            out = mask_x(out, flags)
            xs.append(out)
        out = torch.cat(xs, dim=-1)     # bs, max_feat_num, dim

        sigmoid_out = self.sigmoid_linear(out)
        tanh_out = self.tanh_linear(out)
        out = torch.mul(sigmoid_out, tanh_out).sum(dim=1)
        out = torch.tanh(out)

        out = self.final_linear(out)

        return out


def get_regressor_fn(sde, model):
    model_fn = model

    if isinstance(sde, VPSDE):
        def regressor_fn(x, adj, flags, t):
            pred = model_fn(x, adj, flags)
            return pred
    elif isinstance(sde, VESDE):
        def regressor_fn(x, adj, flags, t):
            pred = model_fn(x, adj, flags)
            return pred
    else:
        raise NotImplementedError(f"SDE class: {sde.__class__.__name__} not supported.")

    return regressor_fn


class RegressorScoreX(torch.nn.Module):
    def __init__(self, sde, Regressor):
        super().__init__()
        self.sde = sde
        self.regressor = get_regressor_fn(sde, Regressor)

    def forward(self, x, adj, flags, t):
        with torch.enable_grad():
            x_para = torch.nn.Parameter(x)
            F = self.regressor(x_para, adj, flags, t).sum()
            F.backward()
            score = x_para.grad
            score = mask_x(score, flags)
        return score


class RegressorScoreAdj(torch.nn.Module):
    def __init__(self, sde, Regressor):
        super().__init__()
        self.sde = sde
        self.regressor = get_regressor_fn(sde, Regressor)

    def forward(self, x, adj, flags, t):
        with torch.enable_grad():
            adj_para = torch.nn.Parameter(adj)
            F = self.regressor(x, adj_para, flags, t).sum()
            F.backward()
            score = adj_para.grad
            score = mask_adjs(score, flags)
        return score
