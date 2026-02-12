import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable
from torch import Tensor

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

class SWDirLoss(nn.Module):
    """
    Sliced‑Wasserstein loss that matches softmax‑induced probability vectors
    to a symmetric Dirichlet prior  Dir(α, …, α)  with  α = α₀ / K.

    Parameters
    ----------
    num_embeddings : int
        Codebook size K.
    num_projections : int
        Number of random Boolean masks M to draw per forward call.
    temperature : float
        Initial temperature for  softmax(−d / τ).
    learnable_temperature : bool
        If True, τ is a learnable parameter (softplus‑parameterised so τ > 0).
    mask_sampling : str
        How mask sizes are sampled.  One of
        • "bernoulli"    – each entry iid Bernoulli(p_mask); size concentrates
                           around K·p_mask.
        • "uniform"      – mask size ~ Uniform{1, …, K−1}, then entries
                           sampled uniformly for that size.
        • "log_uniform"  – log₂(size) ~ Uniform[0, log₂(K−1)]; biases toward
                           small masks → better sparse‑structure sensitivity.
        • "mixture"      – equal‑weight mixture of the three above; gives
                           broadest coverage.
    p_mask : float
        Bernoulli probability (only used when mask_sampling="bernoulli").
    """

    VALID_SAMPLINGS = {"mixture", "uniform", "log_uniform", "bernoulli"}

    def __init__(
        self,
        num_embeddings: int = 8192,
        alpha: float = 0.1,
        num_projections: int = 64,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        mask_sampling: str = "bernoulli",
        p_mask: float = 0.5,
    ):
        super().__init__()
        assert mask_sampling in self.VALID_SAMPLINGS, (
            f"mask_sampling must be one of {self.VALID_SAMPLINGS}, "
            f"got '{mask_sampling}'"
        )

        self.K = num_embeddings
        self.alpha = alpha  # per-component concentration
        self.M = num_projections
        self.mask_sampling = mask_sampling
        self.p_mask = p_mask

        self.learnable_temperature = learnable_temperature
        if learnable_temperature:
            raw_init = math.log(math.expm1(temperature))  # inverse softplus
            self._temperature_raw = nn.Parameter(torch.tensor(raw_init))
        else:
            self.register_buffer(
                "_temperature_raw", torch.tensor(temperature)
            )

    @property
    def temperature(self) -> torch.Tensor:
        """始终 > 0 的温度值."""
        if self.learnable_temperature:
            return F.softplus(self._temperature_raw)
        return self._temperature_raw

    # ─────────────────────── mask 采样 ───────────────────────

    @torch.no_grad()
    def _sizes_bernoulli(self, device) -> torch.Tensor:
        """每个 entry 独立 Bernoulli(p_mask)，返回 mask sizes [M]."""
        masks = torch.bernoulli(
            torch.full((self.M, self.K), self.p_mask, device=device)
        )
        return masks  # 直接返回完整 mask 矩阵

    @torch.no_grad()
    def _sizes_uniform(self, M: int, device) -> torch.Tensor:
        """size ~ Uniform{1, …, K−1}."""
        return torch.randint(1, self.K, (M,), device=device)

    @torch.no_grad()
    def _sizes_log_uniform(self, M: int, device) -> torch.Tensor:
        """log₂(size) ~ Uniform[0, log₂(K−1)] → 偏向小 mask."""
        log_max = math.log2(max(self.K - 1, 1))
        log_sizes = torch.empty(M, device=device).uniform_(0.0, log_max)
        sizes = (2.0 ** log_sizes).round().clamp(1, self.K - 1).long()
        return sizes

    @torch.no_grad()
    def _masks_from_sizes(self, sizes: torch.Tensor, device) -> torch.Tensor:
        """
        给定每个投影的 mask size，生成对应的 0/1 mask 矩阵 [M, K].
        对每一行，随机选 size[i] 个位置置 1.
        """
        M = sizes.shape[0]
        # 对每行生成随机排列，取前 size[i] 个
        rand = torch.rand(M, self.K, device=device)
        # 将 rand 按行排序，取 topk 等价：rank < size
        ranks = rand.argsort(dim=1).argsort(dim=1)  # rank matrix [M, K]
        masks = (ranks < sizes.unsqueeze(1)).float()
        return masks

    @torch.no_grad()
    def _sample_masks(self, device) -> torch.Tensor:
        """根据 self.mask_sampling 策略生成有效的 [M_valid, K] mask 矩阵."""
        if self.mask_sampling == "bernoulli":
            masks = self._sizes_bernoulli(device)

        elif self.mask_sampling == "uniform":
            sizes = self._sizes_uniform(self.M, device)
            masks = self._masks_from_sizes(sizes, device)

        elif self.mask_sampling == "log_uniform":
            sizes = self._sizes_log_uniform(self.M, device)
            masks = self._masks_from_sizes(sizes, device)

        elif self.mask_sampling == "mixture":
            # 将 M 个投影平均分给三种策略
            m1 = self.M // 3
            m2 = self.M // 3
            m3 = self.M - m1 - m2

            # bernoulli 部分
            masks_bern = torch.bernoulli(
                torch.full((m1, self.K), self.p_mask, device=device)
            )

            # uniform 部分
            sizes_uni = self._sizes_uniform(m2, device)
            masks_uni = self._masks_from_sizes(sizes_uni, device)

            # log_uniform 部分
            sizes_log = self._sizes_log_uniform(m3, device)
            masks_log = self._masks_from_sizes(sizes_log, device)

            masks = torch.cat([masks_bern, masks_uni, masks_log], dim=0)

        else:
            raise ValueError(f"Unknown mask_sampling: {self.mask_sampling}")

        # 过滤全 0 或全 1 mask（使 Beta 分布参数合法）
        mask_sums = masks.sum(dim=1)
        valid = (mask_sums > 0) & (mask_sums < self.K)
        return masks[valid]

    # ─────────────────── target 采样 ─────────────────────────

    @torch.no_grad()
    def _sample_target(
        self, mask_sizes: torch.Tensor, N: int, device
    ) -> torch.Tensor:
        """
        从 Beta(m·α, (K−m)·α) 采样并排序.

        当 α = α₀/K 时，参数变为
            a = m · α₀ / K,   b = (K − m) · α₀ / K
        总浓度 a + b = α₀（与 K 无关），只有比例 a/(a+b) = m/K 随 mask 变。
        """
        M_valid = mask_sizes.shape[0]

        a = (mask_sizes * self.alpha).unsqueeze(1).expand(-1, N)  # [M, N]
        b = ((self.K - mask_sizes) * self.alpha).unsqueeze(1).expand(-1, N)

        # 防止极端小值导致数值问题
        eps = 1e-6
        a = a.clamp(min=eps)
        b = b.clamp(min=eps)

        beta_dist = Beta(a, b)
        samples = beta_dist.sample()  # [M_valid, N]
        sorted_target, _ = torch.sort(samples, dim=1)
        return sorted_target

    # ─────────────────── forward ─────────────────────────────

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        distances : Tensor [N, K]
            每个样本到 K 个 embedding 的距离.

        Returns
        -------
        loss : scalar Tensor
            所有有效投影方向上的平均 W₁ 距离.
        """
        device = distances.device
        N, K = distances.shape
        assert K == self.K, f"Expected K={self.K}, got {K}"

        # ── 概率向量 ─────────────────────────────────────────
        tau = self.temperature  # scalar tensor (可能带梯度)
        P = F.softmax(-distances / tau, dim=-1)  # [N, K]

        # ── 掩码 ─────────────────────────────────────────────
        masks = self._sample_masks(device)  # [M_valid, K]
        M_valid = masks.shape[0]
        if M_valid == 0:
            return distances.new_tensor(0.0, requires_grad=True)

        # ── 掩码投影 + 排序 ──────────────────────────────────
        proj_emp = masks @ P.t()  # [M_valid, N]
        sorted_emp, _ = torch.sort(proj_emp, dim=1)  # [M_valid, N]

        mask_sizes = masks.sum(dim=1)  # [M_valid]
        sorted_tgt = self._sample_target(mask_sizes, N, device)  # [M_valid, N]

        # ── W₁ ───────────────────────────────────────────────
        w1 = (sorted_emp - sorted_tgt).abs().mean(dim=1)  # [M_valid]
        return w1.mean()

    def extra_repr(self) -> str:
        tau = self.temperature.item() if self.temperature.numel() == 1 else "?"
        return (
            f"K={self.K}, α={self.alpha:.6g}, "
            f"M={self.M}, τ={tau:.4f} "
            f"({'learnable' if self.learnable_temperature else 'fixed'}), "
            f"mask_sampling='{self.mask_sampling}'"
        )
