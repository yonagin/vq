import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable
from torch import Tensor


class LSWLoss(nn.Module):
    def __init__(self, lambda_lsw: float = 0.1, L: int = 64, p: float = 2.0, 
                 projection_type: str = "base", q: float = 2.0):
        super().__init__()
        self.lambda_lsw = lambda_lsw
        self.L = L
        self.p = p
        self.projection_type = projection_type
        self.q = q
        
    def forward(self, ze_flat: Tensor, codebook: Tensor) -> Tensor:
        """
        Compute LSW loss between encoded features and codebook.
        
        Args:
            ze_flat: Encoded features [N, D]
            codebook: Codebook embeddings [M, D]
            
        Returns:
            LSW loss scalar tensor
        """
        if self.lambda_lsw <= 0.0:
            return torch.tensor(0.0, device=ze_flat.device, dtype=ze_flat.dtype)
            
        return self.lambda_lsw * lsw_dist(
            ze_flat,
            codebook,
            L=self.L,
            p=self.p,
            projection_type=self.projection_type,
            q=self.q
        )


def _quantile_linear(sorted_vals: Tensor, u: Tensor) -> Tensor:
    n, L = sorted_vals.shape
    u = u.view(-1, 1).clamp(0.0, 1.0)
    pos = u * (n - 1)
    lo = torch.floor(pos).to(torch.long)
    hi = (lo + 1).clamp(max=n - 1)

    lo_idx = lo.expand(-1, L).clamp(0, n-1)
    hi_idx = hi.expand(-1, L).clamp(0, n-1)

    w  = (pos - lo).to(sorted_vals.dtype)
    w = w.expand(-1, L)

    v_lo = sorted_vals.gather(dim=0, index=lo_idx)
    v_hi = sorted_vals.gather(dim=0, index=hi_idx)

    return v_lo * (1.0 - w) + v_hi * w



def _wasserstein_1d(P1: Tensor, P2: Tensor, p: float = 2.0, grid_size: Optional[int] = None) -> Tensor:
    """
    P1: [N, L]  为 T1 在 L 个方向上的投影
    P2: [M, L]  为 T2 在 L 个方向上的投影
    """
    N, _ = P1.shape
    M, _ = P2.shape

    # 按列排序（每个方向单独排序）
    s1 = P1.sort(dim=0).values  # [N, L]
    s2 = P2.sort(dim=0).values  # [M, L]

    if N == M:
        d = (s1 - s2).abs().pow(p)  # [N, L]
    else:
        # 用分位数插值统一到同一个 K 点的网格上，近似积分 ∫|Q1(u)-Q2(u)|^p du
        K = grid_size if grid_size is not None else max(N, M)
        u = torch.linspace(0.0, 1.0, K, device=P1.device, dtype=P1.dtype)  # [K]

        q1 = _quantile_linear(s1, u)  # [K, L]
        q2 = _quantile_linear(s2, u)  # [K, L]
        d = (q1.sub(q2).abs().pow(p))

    return d.mean(dim=0)  # [L]


def lsw_dist(
    T1: Tensor,
    T2: Tensor,
    L: int = 256,
    p: float = 2.0,                    # 距离阶数（Wasserstein-p）
    q: float = 2.0,                    # Lehmer 平均的幂参数
    grid_size: Optional[int] = None,
    eps: float = 1e-12,
    generator: Optional[torch.Generator] = None,
    directions: Optional[Tensor] = None,
    projection_type: str = 'base',                          # 'base' | 'ortho' | 'nonlinear'
    nonlinear_activation: Optional[Callable] = None,   # 仅在 'nonlinear' 下使用
) -> Tensor:
    """
    计算基于 Lehmer 平均聚合的一种SW距离度量

    参数:
    - T1: [N, D]
    - T2: [M, D]
    - L:  投影次数（方向数）
    - p:  Wasserstein 距离阶数（通常 2）
    - grid_size: N≠M 时，用于分位数近似积分的网格点数（默认 max(N, M)）
    - eps: 数值稳定用
    - generator: 随机数生成器（可选，控制可复现）
    - directions: [L, D] 自定义投影方向（可选；若给定将跳过采样）
    - projection_type: 切片投影类型
        * 'base'      -> 单位向量随机方向
        * 'ortho'     -> 正交方向（QR）（要求 L ≤ D）
        * 'nonlinear' -> 单位向量随机方向 + 非线性激活
    - nonlinear_activation: 非线性激活函数，仅在 projection_type='nonlinear' 时生效
    - q: Lehmer 平均的幂参数。
       
    返回:
    - 标量 Tensor: L(x,q) = sum(x^q) / sum(x^{q-1})，q=1 退化为算术平均。
    """
    assert T1.dim() == 2 and T2.dim() == 2, "T1, T2 必须是 [*, D] 的2维张量"
    N, D1 = T1.shape
    M, D2 = T2.shape
    assert D1 == D2, "T1 和 T2 的特征维 D 必须一致"
    device = T1.device
    dtype = T1.dtype

    proj_type = projection_type.lower()
    if proj_type not in ('base', 'ortho', 'nonlinear'):
        raise ValueError(f"未知的投影类型: {projection_type}. 可选: 'base' | 'ortho' | 'nonlinear'")

    # 采样/处理 L 个方向（或使用给定的 directions）
    if directions is None:
        if proj_type == 'ortho':
            # 正交方向：要求 L <= D
            if L > D1:
                raise ValueError(f"PRW/正交投影要求 L <= D，但收到 L={L}, D={D1}")
            A = torch.randn((D1, L), device=device, dtype=dtype, generator=generator)  # [D, L]
            dirs, _ = torch.linalg.qr(A, mode='reduced')  # Q: [D, L] (列两两正交、单位范数)
        else:
            # base / nonlinear: 单位向量
            dirs = torch.randn((D1, L), device=device, dtype=dtype, generator=generator)
            dirs = F.normalize(dirs, dim=0, eps=eps)  # [D, L], 单位向量
    else:
        assert directions.shape == (D1, L), f"directions 形状应为 {(D1, L)}, 实际为 {directions.shape}"
        if proj_type == 'ortho':
            if L > D1:
                raise ValueError(f"PRW/正交投影要求 L <= D，但收到 L={L}, D={D1}")
            # 对给定方向做 QR 正交化
            dirs, _ = torch.linalg.qr(directions.t(), mode='reduced')  # [D, L]
        else:
            # 单位化给定方向
            dirs = F.normalize(directions, dim=0, eps=eps)  # [D, L], 单位向量

    # 线性投影到 1D：得到 [N, L] 和 [M, L]
    P1 = T1 @ dirs  # [N, L]
    P2 = T2 @ dirs  # [M, L]

    # 非线性切片
    if proj_type == 'nonlinear':
        act_fn = nonlinear_activation if nonlinear_activation is not None else torch.sigmoid
        if not callable(act_fn):
            raise TypeError("nonlinear_activation 必须是可调用对象（例如 torch.sigmoid / torch.tanh / nn.Sigmoid() 等）")
        P1 = act_fn(P1)
        P2 = act_fn(P2)

    # 每个方向上的 1D Wasserstein-p 距离的 p 次幂: w_p_values ∈ [L]
    w_p_values = _wasserstein_1d(P1, P2, p=p, grid_size=grid_size)  # [L], >= 0

    # 使用 Lehmer 平均做方向聚合
    if q == 1.0:
        # q=1 时，Lehmer 平均退化为普通算术平均
        lehmer_mean = w_p_values.mean()
    else:
        # 使用 LSE 稳定技巧
        x = w_p_values.clamp_min(eps) # 确保 x > 0

        if q > 0:
            # 按 max 缩放
            c = torch.max(x)
            x_norm = x / c
        else: # q <= 0 (q=0时,是调和平均的倒数)
            # 按 min 缩放
            c = torch.min(x)
            x_norm = x / c
        numerator = torch.sum(x_norm.pow(q))
        denominator = torch.sum(x_norm.pow(q - 1.0))
        lehmer_mean = c * (numerator / (denominator + eps))

    # 最后按 L^p 范数取 1/p 次方
    distance = torch.clamp(lehmer_mean, min=0.0).pow(1.0 / p)
    return distance