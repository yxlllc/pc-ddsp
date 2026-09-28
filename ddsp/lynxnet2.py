import torch
import torch.nn as nn
import torch.nn.functional as F


class ATanGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, gate):
        atan_gate = torch.atan(gate)
        decay_out = out / gate.square().add(1.0)
        ctx.save_for_backward(decay_out, atan_gate)
        return out * atan_gate

    @staticmethod
    def backward(ctx, grad_output):
        decay_out, atan_gate = ctx.saved_tensors
        grad_out_part = grad_output * atan_gate
        grad_gate_part = grad_output * decay_out
        return grad_out_part, grad_gate_part   

       
class ATanGLU(nn.Module):
    # ArcTan-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.        
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return ATanGLUFunction.apply(out, gate)

       
class SoftSignGLUFunction(torch.autograd.Function):
    """ATanGLUFunction-style memory trick for SoftSignGLU.

    softsign'(x) = 1/(1+|x|)^2 = (1-|softsign(x)|)^2, so both partial
    derivatives of y = out * softsign(gate) are precomputable in forward:
      dy/dout = softsign(gate)
      dy/dgate = out * (1-|softsign(gate)|)^2
    Saves 2 tensors (vs 3 for naive autograd) and backward is two pure
    multiplies with no softsign recompute.
    """
    @staticmethod
    def forward(ctx, out, gate):
        ss_gate = torch.nn.functional.softsign(gate)
        decay_out = out * (1.0 - ss_gate.abs()).square()
        ctx.save_for_backward(ss_gate, decay_out)
        return out * ss_gate

    @staticmethod
    def backward(ctx, grad_output):
        ss_gate, decay_out = ctx.saved_tensors
        return grad_output * ss_gate, grad_output * decay_out


class SoftSignGLU(nn.Module):
    """Gated Linear Unit with SoftSign gate: out * softsign(gate).

    More numerically stable than ATanGLU (no approximation needed in
    Triton kernels) while providing similar gating behavior.
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        if self.training:
            return SoftSignGLUFunction.apply(out, gate)
        else:
            return out * F.softsign(gate)


class DoubleSoftSignGLUFunction(torch.autograd.Function):
    """Memory-optimized backward for DoubleSoftSignGLU (same trick).

    Operates on the WHOLE Linear output x = [out | gate] (softsign is
    elementwise, so softsign-then-split == split-then-softsign). With
    a = softsign(out), b = softsign(gate), y = a * b:
      dy/dout = b * (1-|a|)^2
      dy/dgate = a * (1-|b|)^2
    Both partials are precomputed into ONE [.., 2N] tensor in forward, so
    backward is a single multiply against the (broadcast) upstream grad —
    no softsign recompute, no cat.
    """
    @staticmethod
    def forward(ctx, x, dim):
        ss = torch.nn.functional.softsign(x)
        a, b = torch.split(ss, ss.size(dim) // 2, dim=dim)
        decay_sq = (1.0 - ss.abs()).square()
        da, db = torch.split(decay_sq, decay_sq.size(dim) // 2, dim=dim)
        # decay = [b*(1-|a|)^2 | a*(1-|b|)^2], written into decay_sq's halves
        da.mul_(b)
        db.mul_(a)
        ctx.save_for_backward(decay_sq)
        ctx.dim = dim
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        decay, = ctx.saved_tensors
        grad_x = decay * torch.cat([grad_output, grad_output], dim=ctx.dim)
        return grad_x, None


class DoubleSoftSignGLU(nn.Module):
    """FastWaveD-style double-gated unit: softsign applied to the whole
    Linear output, then split and multiplied:
      y = softsign(out) * softsign(gate)
    Output is bounded in (-1, 1) since both branches saturate.
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.training:
            return DoubleSoftSignGLUFunction.apply(x, self.dim)
        # softsign whole tensor first (one elementwise op), then split
        ss = torch.nn.functional.softsign(x)
        out, gate = torch.split(ss, ss.size(self.dim) // 2, dim=self.dim)
        return out * gate


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


def _make_glu(glu_type):
    if glu_type == 'atanglu':
        return ATanGLU()
    elif glu_type == 'softsign_glu':
        return SoftSignGLU()
    elif glu_type == 'double_softsign_glu':
        return DoubleSoftSignGLU()
    else:
        raise ValueError(f'{glu_type} is not a valid activation')


class LYNXNet2Block(nn.Module):
    def __init__(self, dim, expansion_factor, kernel_size=31, dropout=0., glu_type='softsign_glu'):
        super().__init__()
        self.dim = dim
        self.glu_type = glu_type
        inner_dim = int(dim * expansion_factor)
        if float(dropout) > 0.:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()
        self.net = nn.Sequential(
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            _make_glu(glu_type),
            nn.Linear(inner_dim, inner_dim * 2),
            _make_glu(glu_type),
            nn.Linear(inner_dim, dim),
            _dropout
        )

    def forward(self, x):
        y = F.rms_norm(x, (x.size(-1), ))
        return x + self.net(y)