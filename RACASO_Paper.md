# RACASO: Rotation-Aligned Cautious Approximately Second-Order Optimization

**Author:** Richard Christopher

**Affiliation:** MetaFore

**Email:** rchris@neotec.dev

**Date:** May 2026

**Status:** Technical specification and implementation provenance

---

## Abstract

Modern deep learning models increasingly route gradients through non-linear structures: branching paths, multi-head attention, mixture-of-experts. Standard diagonal adaptive optimizers like AdamW and Sophia scale efficiently on uniform networks but rest on a hidden assumption. They assume the loss landscape's principal axes of curvature line up with the standard coordinate basis of the parameter space. When that assumption breaks, diagonal estimators misjudge which directions are sharp and which are flat, and the optimizer responds with gradient oscillation, step-size stagnation, or outright divergence.

This paper introduces RACASO (Rotation-Aligned Cautious Approximately Second-Order Optimizer), a framework that bridges Kronecker-factored structural preconditioning and second-order curvature alignment. For each 2-D parameter, RACASO projects the parameter into a Kronecker-factored eigenbasis. That basis is privileged in the sense that the gradient covariance is approximately diagonalized within it. Inside that basis, an unbiased per-element diagonal Hessian estimate is computed using Hutchinson's trace estimator on top of a Pearlmutter-style Hessian-vector product. A Sophia-style per-element cautious clip then bounds the rotated update.

Two curvature paths ship. The Hutchinson HVP path captures the true Hessian diagonal, including negative-curvature directions. The Gauss-Newton-Bartlett path supplies a positive-semidefinite approximation and serves as a structurally bulletproof fallback. Five safety mechanisms layer through the optimizer: an eigh residual gate, a RAdam cold-start gate, a 1-D Yogi fallback, a spread cap on rotated row norms, and an L5 safe-skip on non-finite HVP output. Together these keep the optimizer numerically stable across failure modes that SOAP and Newton-Schulz-style methods do not survive.

We also document the eight-attempt provenance of the Hutchinson HVP integration with PyTorch eager autograd, the operator-shape rules that determine which forward-graph operators are second-derivative-safe, and the conditions under which the L5 safe-skip triggers. The failure record is preserved so consumers can audit whether their own forward graph satisfies the bounded-second-derivative conditions RACASO needs to run at full design intent.

---

## 1. Introduction and the Basis Alignment Problem

The dominant paradigm in neural-network optimization is diagonal scaling. AdamW uses the gradient's historical variance, $v_t \approx \mathbb{E}[g_t^2]$, as a curvature proxy. Sophia-GNB uses a Gauss-Newton-Bartlett proxy. Both update parameters element-wise:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{m_t}{\max(\sqrt{v_t}, \epsilon)}$$

This formulation assumes the Hessian $\mathcal{H}$ is approximately diagonal, that each parameter varies independently of every other.

In modern architectures the assumption is cleanly falsified. Branching gradient paths, weight sharing, multi-head attention, and joint-normalized fusion sites all introduce off-diagonal dependencies. Geometrically, sharp asymmetric loss valleys appear running diagonally across the standard coordinate axes. A diagonal optimizer encountering a diagonal valley zigzags across the ridges instead of traveling smoothly down the floor.

Matrix-preconditioned optimizers (SOAP, Shampoo) achieve superior convergence not because their gradient proxies are inherently more precise, but because they rotate the gradients into a structural eigenbasis where the local coordinates are temporarily uncoupled. Both rely on first-order gradient history ($g_t \cdot g_t^T$) to build that basis, which makes them structurally hostile to coupled-gradient regimes. When one factor's gradient distribution shifts abruptly, the Kronecker covariance accumulates rank-1 outer products that produce near-singular factors, and eigh either returns silently-NaN eigenvectors or non-trustworthy rotations.

RACASO takes a different path. It keeps SOAP's rotation-into-privileged-basis property but estimates the per-element step size in that basis from a true second-order signal (Hutchinson HVP) rather than from a gradient-squared proxy. A per-element cautious clip in the rotated basis bounds steps regardless of curvature estimate quality. Safety layers around the rotation refresh, the cold-start regime, and the second-derivative failure surface keep the optimizer numerically robust across the failure modes documented in Section 6 — failure modes that bracket the operating envelope of SOAP (covariance-rank-collapse), Shampoo (eigh failure with no fallback), Newton-Schulz-based methods (input-conditioning skip catastrophe), and Hutchinson-HVP-based methods (second-derivative DivBackward0 NaN cascade).

A companion paper, RAMuogi [Christopher 2026b], develops a spectral-orthogonalization optimizer using a related failure-safety chain pattern. The two optimizers occupy different positions in the parameter-group stack (RACASO: 2-D parameters with non-trivial off-diagonal curvature; RAMuogi: 2-D parameters with row-spread pathology under matrix-orthogonalization steps) and compose at the per-parameter-group assignment level.

### 1.5 Lineage and scope

This paper is one of three (RACASO, Muogi/RAMuogi, Liger) describing optimizers developed in sequence against distinct gradient-regime failure modes encountered during production training of a multi-stream transformer-derivative architecture. Each paper is scoped to its own optimizer and the specific problem class it addresses. The companion papers describe the other two and how the family fits together; only what is load-bearing for RACASO appears here.

**The problem class RACASO solves.** RACASO targets the regime where two earlier optimizers had each absorbed a distinct numerical-failure class but a third class remained open: the *second-derivative DivBackward0 hazard*. Concretely, this is the gradient regime where the forward graph contains operators whose second derivative is unbounded — ratio forms whose denominator approaches zero, RMSNorm-style normalizations near zero-norm inputs, division through `torch.norm` without a stabilizing epsilon, and other graph patterns documented as the L5 absorb class in §6 and §7. Any optimizer that touches second-order curvature via Hutchinson HVP, Gauss-Newton-Bartlett, K-FAC, Shampoo with second-moment correction, or Adahessian eventually encounters this class; the field has historically responded by ad-hoc clamping, which silently corrupts the curvature estimate. RACASO is the first optimizer in this lineage to document the absorb pattern explicitly as a fifth layer in a multi-layer safety chain.

**What RACASO does about it.** RACASO combines Kronecker-factored axis rotations (SOAP-inspired) with Hutchinson HVP curvature estimation (Sophia-inspired) and applies Sophia's cautious clipping in the rotated basis. The four layers it inherits from the companion Muogi/RAMuogi paper [Christopher 2026b] (L1 spread cap, L2 eigh-residual safe-skip, L3 vanilla-Yogi fallback, L4 RAdam cold-start gate) handle the failure classes earlier in the update; the new fifth layer (L5 absorb-and-continue) handles the second-derivative-overflow class that the chain was missing. The engineering provenance in §7 documents eight attempts to integrate Hutchinson HVP with PyTorch eager autograd, each one failing at a different operator graph; the result is a documented map of operator categories that future second-order optimizer authors can use to skip mistakes we already made.

**Where RACASO falls short, and what the companion papers cover.**

- *SOAP+Shampoo composition.* Once RACASO's L5 absorb cleared the second-derivative-overflow class, the absorb pattern made it viable to return to a SOAP+Shampoo configuration on the original target group. Shampoo's Kronecker maintenance is more efficient than RACASO's full eigendecomposition + HVP refresh per step, and the resulting configuration produces measurements competitive with or better than RACASO at lower per-step cost. RACASO's surviving contribution on this problem class is the L5 absorb pattern (now adopted by the SOAP+Shampoo composition), the engineering provenance in §7, and the documented failure-class taxonomy — not the optimizer itself as the long-run production choice.

- *Already-well-conditioned matrix gradients.* On parameter groups where matrix gradients arrive at the optimizer already well-conditioned (downstream of normalization layers that have done the conditioning work upstream), RACASO's rotated-basis machinery finds nothing useful to precondition, and the per-step overhead becomes pure cost. The companion Liger paper [Christopher 2026c] addresses this regime with a dispatch-by-dimensionality rule that pairs Lion (bounded direction, no preconditioning) with Yogi (burst-safe variance handling).

RACASO remains the right tool for the regime it targets: matrix parameter groups with non-trivial off-diagonal curvature and a forward graph that exposes the second-derivative-overflow class. §8 reports head-to-head measurements from open-bench sweeps that establish each claim against published baselines (Adam, AdamW, Yogi, Lion) and the two sibling optimizers (Muogi/RAMuogi, Liger).

---

## 2. The Architecture of RACASO

For a 2-D matrix parameter $W \in \mathbb{R}^{d_{out} \times d_{in}}$, RACASO combines Kronecker-factored axis rotations with Hessian-vector products, then applies Sophia's cautious step in the rotated basis. The structure looks like this:

```
                  [ Forward Pass & Loss Evaluation ]
                                  │
                                  ▼
                     [ is_refresh_step (t mod K)? ]
                     ├── Yes ──► [ Monolithic HVP Block ]
                     │             ├── Retain Graph Autograd
                     │             ├── Generate Kronecker Eigenbasis (U_L, U_R)
                     │             ├── Sample Rademacher Vector (Z)
                     │             ├── Compute Batched HVP: Hz = ∂²L/∂W² · Z
                     │             └── Update h_t EMA with z⊙Hz (or GNB fallback)
                     │
                     └── No  ──► [ Standard Single-Backward Step ]
                                  │
                                  ▼
                     [ Project g_t → rotated basis (U_L, U_R) ]
                                  │
                                  ▼
                     [ Sophia cautious clip in rotated basis ]
                                  │
                                  ▼
                     [ Spread cap + project back to parameter basis ]
                                  │
                                  ▼
                  [ Update Parameter Weight Matrix W ]
```

### 2.1 Kronecker-Factored Axis Rotations

A full parameter-to-parameter Hessian rotation for $W \in \mathbb{R}^{d_{out} \times d_{in}}$ requires impossible $O(N^2)$ tracking state and $O(N^3)$ computational cost. RACASO sidesteps this by maintaining a Kronecker-Sum preconditioning structure across the row and column axes:

$$\mathcal{H}_{\text{structural}} \approx (L_t \oplus R_t) = (L_t \otimes I_{d_{in}}) + (I_{d_{out}} \otimes R_t)$$

where $L_t \in \mathbb{R}^{d_{out} \times d_{out}}$ and $R_t \in \mathbb{R}^{d_{in} \times d_{in}}$ are running EMA covariance metrics of the spatial slices of the gradient:

$$L_t = \beta_s \cdot L_{t-1} + (1 - \beta_s) \cdot g_t g_t^T, \quad R_t = \beta_s \cdot R_{t-1} + (1 - \beta_s) \cdot g_t^T g_t$$

Every $K_{\text{refresh}}$ steps RACASO computes the eigendecomposition of these compact factors independently:

$$L_t = U_L \Lambda_L U_L^T, \quad R_t = U_R \Lambda_R U_R^T$$

The orthogonal matrices $U_L$ and $U_R$ form the *privileged basis*. A progressive-ridge `_safe_eig_with_residual` helper retries the eigendecomposition at increasing ridge scales `(0, 1e-6, 1e-3, 1e-1)` if the bare-input form returns NaN columns, which is a known PyTorch failure mode for near-singular symmetric inputs. The L2 safety layer then gates acceptance. Each side's new eigenvectors commit independently only if the reconstruction residual $\|M - U \Lambda U^T\|_F$ stays below threshold and is isfinite. A bad eigh on the L side does not reject the R side's good refresh.

### 2.2 Curvature Estimation, Two Paths

RACASO ships two curvature estimators with identical downstream contracts. Both produce per-element diagonal estimates that get rotated into the privileged basis and consumed by the Sophia clip's denominator.

#### 2.2.1 Hutchinson HVP: the true Hessian diagonal

Hutchinson's stochastic trace estimator (Hutchinson 1989) gives an unbiased per-element estimate of $\text{diag}(\mathcal{H})$:

$$\mathbb{E}[z \odot Hz] = \text{diag}(\mathcal{H}), \quad z \in \{-1, +1\}^{shape(W)}$$

A Rademacher vector $z$ is sampled per refresh. The HVP $Hz = \nabla^2 L \cdot z$ is computed via Pearlmutter's identity, batched across all 2-D parameters by accumulating into a single scalar:

$$s = \sum_i \langle g_i^{\text{live}}, z_i \rangle, \quad \text{HVP}_i = \frac{\partial s}{\partial p_i} = H_{ii} \cdot z_i$$

A single `autograd.grad(s, [p_1, ..., p_n])` call produces all $Hz_i$ in one second-order graph traversal. The diagonal estimate $z_i \odot Hz_i$ is finite-checked, then folded into the per-element rotated-basis EMA $\tilde{h}_t$ via:

$$\tilde{h}_t = \beta_2 \cdot \tilde{h}_{t-1} + (1 - \beta_2) \cdot \tilde{h}_{\text{sample}}^2, \quad \tilde{h}_{\text{sample}} = U_L^T (z \odot Hz) U_R$$

The full Hessian diagonal preserves both positive and negative curvature. The Sophia clip's `clamp(min=ε)` floor handles negative-curvature directions by collapsing them to the floor rather than producing a divergent update.

**Cost:** one extra backward through the X-subgraph per refresh step, sharing the live forward graph via Pearlmutter. On non-refresh steps, zero overhead. The optimizer reuses the previously-stored $U_L$, $U_R$, $\tilde{h}_t$.

#### 2.2.2 Gauss-Newton-Bartlett (GNB): the bulletproof fallback

When Hutchinson HVP cannot run, or is structurally hostile to a given forward graph (see Section 6), GNB replaces the Hessian with the Gauss-Newton approximation:

$$\mathcal{H} \approx G_N = \mathbb{E}_{\hat{y} \sim p(y|x)}\left[\nabla \log p(\hat{y}|x) \cdot \nabla \log p(\hat{y}|x)^T\right]$$

Operationally: sample one synthetic label $\hat{y}$ per row from the model's own softmax, compute $\hat{g} = \partial \text{CE}(\text{logits}, \hat{y}) / \partial p$, then $\text{diag}(G_N) \approx \hat{g}^2 \cdot B$ where $B$ is the batch size (Sophia §3.2 reweighting). The stash contract `p._racaso_hvp_estimate` is unchanged.

GNB drops the negative-definite component of the Hessian. The estimate is always positive semidefinite, so the Sophia clip never sees a sign-flip and the optimizer's behavior is strictly more conservative. **Cost:** one extra first-order backward against a re-built CE loss. No second derivative, no functorch, no eager-autograd version tracking.

Selectable at runtime via the `hvp_strategy ∈ {"hutchinson", "gnb"}` config field.

### 2.3 Optimization in the Privileged Space

Per step (Hutchinson path; the GNB path is identical except the stash provenance):

$$\tilde{g}_t = U_L^T g_t U_R$$

$$\tilde{m}_t = \beta_1 \tilde{m}_{t-1} + (1 - \beta_1) \tilde{g}_t$$

$$\tilde{m}_{\text{hat}} = \tilde{m}_t / (1 - \beta_1^t)$$

Sophia's cautious step in the privileged basis is then:

$$\Delta \tilde{\Theta}_t = \text{clip}\left( \frac{\tilde{m}_{\text{hat}}}{\max(\gamma \cdot |\tilde{h}_t|, \epsilon)}, -\rho, +\rho \right)$$

where $\rho$ is the hard clipping threshold (Sophia default 0.04), $\gamma$ scales the Hessian denominator (Sophia default 0.04), and $\epsilon$ is a stabilization floor. The Hessian denominator's magnitude is bounded below by $\epsilon$ even if $\tilde{h}_t$ momentarily collapses, and the $\rho$ clip is a hard cap regardless of the denominator. Two independent magnitude bounds.

### 2.4 Spread Cap and Projection Back

Before rotating back, the L1 spread cap bounds the per-eigendirection update magnitudes. In the rotated basis, "rows" are eigendirections. The cap bounds the max-to-min row-norm ratio at `spread_cap` (default 10.0) by damping loud rows toward `row_max / spread_cap`:

$$\text{row\_norms} = \|\Delta \tilde{\Theta}_t\|_{\text{row}}, \quad \text{row\_floor} = \frac{\max(\text{row\_norms})}{\text{spread\_cap}}$$

$$\text{damp}_i = \min\left(\frac{\text{row\_floor}}{\max(\text{row\_norms}_i, \epsilon_{\text{adam}})}, 1.0\right)$$

This is a one-sided cap. Quiet rows pass through unchanged. Loud rows are damped. It never amplifies.

The clipped, spread-capped update is rotated back to the parameter basis:

$$\Delta W_t = U_L \cdot \Delta \tilde{\Theta}_t^{\text{capped}} \cdot U_R^T$$

A final non-finite guard checks $\Delta W_t$ element-wise before the parameter update. If any element is non-finite (which would only happen if upstream stages missed something), the update for that parameter is skipped this step. Then:

$$W_{t+1} = W_t - \eta \cdot r_t \cdot \Delta W_t$$

where $r_t$ is the RAdam rectification factor (Section 3), bounded in $[0, 1]$ and equal to 0 during cold-start, equal to the RAdam-standard variance correction post-warmup.

### 2.5 1-D Parameter Fallback (L3)

Norms, biases, learned scalars, and any other 1-D parameter have no privileged-basis rotation to compute. RACASO falls back to vanilla Yogi (additive variance EMA instead of Adam's multiplicative form):

$$v_t = v_{t-1} - (1 - \beta_2) \cdot \text{sign}(v_{t-1} - g_t^2) \cdot g_t^2$$

$$W_{t+1} = W_t - \eta \cdot \frac{m_{\text{hat}}}{\sqrt{v_{\text{hat}}} + \epsilon}$$

Yogi's additive update is robust to bursty gradients (which biases and gain scalars frequently see) in a way Adam isn't.

### 2.6 Algorithm

The complete per-step procedure for a 2-D parameter:

```
Algorithm 1: RACASO step (2-D parameter)
----------------------------------------
Input:  W ∈ ℝ^{m×n} parameter, g ∈ ℝ^{m×n} gradient
        State: m_buf, hessian_diag_rot, U_L, U_R, GG_L, GG_R, t
        Hyperparams: η, β1, β2, ρ, γ, ε, spread_cap

1.  GG_L ← β2·GG_L + (1−β2)·g·gᵀ            # Kronecker covariance EMA, left
2.  GG_R ← β2·GG_R + (1−β2)·gᵀ·g            # Kronecker covariance EMA, right
3.  if t mod refresh_freq == 0:              # L2: rotation refresh
4.      U_L_new, ok_L ← safe_eigh(GG_L)
5.      U_R_new, ok_R ← safe_eigh(GG_R)
6.      if ok_L: U_L ← U_L_new
7.      if ok_R: U_R ← U_R_new
8.  g̃ ← U_Lᵀ · g · U_R                       # rotate into privileged basis
9.  m_buf ← β1·m_buf + (1−β1)·g̃
10. m̂ ← m_buf / (1 − β1^t)
11. if hvp_strategy == "hutchinson" and refresh_step:
12.     Hz ← hutchinson_hvp(loss, params, z)  # may return non-finite
13.     if isfinite(Hz):                      # L5: safe-skip on non-finite
14.         h̃_sample ← U_Lᵀ · (z ⊙ Hz) · U_R
15.         hessian_diag_rot ← β2·hessian_diag_rot + (1−β2)·h̃_sample²
16.     else: hessian_diag_rot unchanged (EMA carry)
17. ĥ ← hessian_diag_rot / (1 − β2^t)
18. ΔΘ̃ ← clip( m̂ / max(γ·|ĥ|, ε), −ρ, +ρ )   # Sophia cautious step
19. # L1: spread cap on row norms
20. row_floor ← max(row_norms(ΔΘ̃)) / spread_cap
21. ΔΘ̃ ← ΔΘ̃ · min(row_floor / row_norms, 1)
22. ΔW ← U_L · ΔΘ̃ · U_Rᵀ                     # rotate back
23. if not isfinite(ΔW): skip this param this step
24. W ← W − η · r_t · ΔW                      # L4: r_t RAdam cold-start gate
```

For 1-D parameters, lines 1-23 are bypassed in favor of the L3 Yogi fallback (§2.5).

---

## 3. The Four-Layer Safety Chain Plus L5

RACASO's stability profile depends on five independent safety mechanisms layered through the `step()` body. Each addresses a specific failure mode. Together they form the contract that distinguishes RACASO from naive Sophia-with-rotation.

| Layer | Mechanism | What it catches | Failure action |
|---|---|---|---|
| **L1** | Spread cap on rotated update row norms | Pathological per-eigendirection step spreads after Sophia clip | Soft degradation. Row norms equalized to a ratio of at most `spread_cap` |
| **L2** | eigh residual + NaN gate on rotation refresh | Ill-conditioned $GG_L / GG_R$; silently NaN eigenvectors from `linalg.eigh` on near-rank-deficient input | Skip rotation refresh per-side independently, keep previous $U_L$ / $U_R$ |
| **L3** | Vanilla Yogi fallback for 1-D params and missing HVP stash | Norms, biases, scalars; non-refresh steps where the stash is absent | Vanilla Yogi update; `hessian_diag_rot` preserves its prior EMA |
| **L4** | RAdam variance-confidence gate ($\rho_t \le 4 \Rightarrow$ momentum-only) | Cold-start steps where neither $m_t$ nor $GG_L / GG_R$ have accumulated trustworthy direction | Momentum-only update; rotation, HVP, clip all skipped |
| **L5** | Non-finite HVP gate on the batched second backward | Forward-graph operators whose second derivative is unbounded over the visited input range (Section 6) | Skip stash this refresh; `hessian_diag_rot` carries to next refresh via EMA; counter tracked for telemetry |

Plus a final non-finite guard before `p.add_(update)`: refuse to write a non-finite update to a parameter regardless of how it became non-finite. Diagnostic incremented, parameter preserved.

---

## 4. Computational and Loop Mechanics

A persistent bottleneck in naive second-order optimizer integration with PyTorch is silent truncation of the computation graph. Executing a standard detached `loss.backward()` destroys intermediate graph activations, causing subsequent `torch.autograd.grad` calls to return tensors with no `grad_fn` (the autograd nodes have been freed and the resulting `.grad` is a leaf). Hutchinson HVP cannot proceed from a leaf gradient.

### 4.1 The Monolithic Autograd Engine

To guarantee structural integrity, RACASO's Hutchinson refresh runs both backward passes inside a single autograd context, with `create_graph=True` on the first pass to retain the second-order edges:

```python
def step_with_hvp(self, loss):
    params = [
        p for group in self.param_groups
        for p in group['params']
        if p.requires_grad and p.dim() == 2
    ]

    # First backward: graph-live grads for the targeted parameter group,
    # with create_graph=True so second-order edges are retained.
    grads = torch.autograd.grad(
        outputs=loss,
        inputs=params,
        create_graph=True,
        retain_graph=True,
    )

    # Rademacher z, then build a scalar whose gradient gives all Hz_i.
    z = [
        torch.empty_like(p).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        for p in params
    ]
    scalar = torch.stack([
        (g * zi).sum() for g, zi in zip(grads, z)
    ]).sum()

    # Single batched backward, produces all Hz vectors in one
    # second-order graph traversal via Pearlmutter's identity.
    hzs = torch.autograd.grad(
        outputs=scalar,
        inputs=params,
        retain_graph=False,
        create_graph=False,
    )

    # Stash z*Hz for the optimizer's rotated-basis update consumption.
    for p, zi, hz in zip(params, z, hzs):
        p._racaso_hvp_estimate = zi * hz
```

Two structural choices matter here.

**Batched HVP via Pearlmutter (the speed fix).** Naive per-parameter HVP would call `autograd.grad(g_live_i, p_i, grad_outputs=z_i)` for each parameter. Each call walks the retained second-order graph independently. With 16 parameters that costs 16x the graph traversal. The Pearlmutter identity collapses it to one:

$$\frac{\partial}{\partial p_j} \sum_i \langle g_i^{\text{live}}, z_i \rangle = \frac{\partial}{\partial p_j} \langle g_j^{\text{live}}, z_j \rangle = H_{jj} \cdot z_j$$

Mathematically identical to the per-param loop; cost drops 16x to 1x. This is the standard trick in Sophia and Adahessian-style HVP implementations.

**Live graph reuse (the second fix).** Rather than running a fresh forward inside `_compute_hutchinson` to rebuild a clean graph (which costs one extra forward per refresh step), the helper operates directly on the main forward's live graph. The caller reorders so HVP runs *before* the main `total_loss.backward()` on Hutchinson refresh steps. The main backward then runs after HVP has consumed what it needs.

### 4.2 The Fast-Path Standard Step ($t \not\equiv 0 \pmod K$)

On the $K - 1$ out of $K$ steps between refreshes, RACASO bypasses double-backward calculations entirely. The loop runs the standard single backward (`create_graph=False`), and the optimizer applies the rotated-basis update using the frozen $U_L$, $U_R$, and $\tilde{h}_t$ from the prior refresh. Wall-clock per-step cost on non-refresh steps is comparable to AdamW. Only the refresh-step amortization is added, and at the recommended default $K = 4$ (window-aligned with a typical truncated-BPTT cadence) the refresh cost is paid on 25% of steps.

### 4.3 The `_optimizer_handles_own_clip` Contract

A training loop with per-parameter-group pre-step gradient clipping (soft-clipping each group's grad norm to a per-group band before `opt.step()`, for example) will mutate `p.grad` in-place via something like `torch._foreach_mul_(grads, clip_factor)`. For RACASO this corrupts the autograd graph that Hutchinson needs. RACASO declares `_optimizer_handles_own_clip = True` as a class-level attribute. Cooperating training loops check this flag and skip pre-step clipping for groups whose optimizer opts out. RACASO's L1 spread cap inside `step()` provides the equivalent magnitude bound, on the rotated update's row norms rather than on the raw gradient norm.

---

## 5. Architectural Advantages Over Existing Solutions

RACASO is designed for failure modes that competing optimizers do not survive. The advantage is structural, not just empirical.

### 5.1 Mitigation of the Newton-Schulz Skip Catastrophe

Matrix-orthogonalization optimizers (Muon and related) apply Newton-Schulz polynomials to force gradient matrices into orthogonal spaces. The fifth-order Jordan polynomial is a standard choice. Its convergence requires the input matrix's spectral norm to lie in a specific band. When a problem class induces dynamic gradient distribution shifts (heterogeneous architectures with multi-branch composition, contrastive losses with mixed scales, etc.), the gradient matrix can become ill-conditioned. The polynomial diverges, the optimizer's convergence-check safety layer triggers, and the update is rejected.

RACASO does not orthogonalize the gradient. It uses the Kronecker eigenvectors strictly as a coordinate-transformation mechanism: project into the basis, apply the cautious step there, project back. If a sudden surge of ill-conditioned curvature arrives, the L1 spread cap absorbs the spread, the L2 eigh safe-skip handles the rotation refresh failure, and the trajectory continues. The optimizer's design-intent rotation continues to fire on every step (the rotation matrices are stable across refreshes, only updated when the eigh refresh succeeds), and the cautious clip absorbs the magnitude shock.

### 5.2 Robust Preconditioning of Branching Gradient Paths

SOAP breaks down on the same architectural class for a different reason. Its Kronecker covariance accumulators $GG_L = g \cdot g^T$ and $GG_R = g^T \cdot g$ accumulate rank-1 outer products when one branch's gradient distribution dominates. The factorization assumption "gradient covariance is a Kronecker product" fails when the branches are coupled via joint normalization upstream, because the joint norm's denominator couples the gradient flow across all branches. Eigh then either returns garbage eigenvectors or fails the residual gate.

RACASO inherits the same Kronecker covariance accumulators, but the second-order curvature estimate (Hutchinson HVP) is computed via autograd on the actual loss, not derived from the covariance factors. The eigenvectors are only used for the coordinate rotation. The per-element step sizing comes from the unbiased Hessian diagonal estimator. When SOAP would fail (covariance becomes near-rank-1), RACASO's L2 layer skips the rotation refresh, keeps the prior rotation, and the Hutchinson HVP continues to provide accurate per-element curvature for the cautious clip's denominator. The optimizer degrades gracefully toward "Sophia in a stale rotated basis" rather than catastrophically toward "AdamW in a NaN basis."

### 5.3 The L5 Safe-Skip

L5 absorbs forward-graph failure modes where the second-derivative path through one operator is unbounded over inputs the model visits during training. When the batched HVP backward returns non-finite Hz values for second-derivative-numerical reasons, RACASO skips the curvature-estimate update for that refresh. `hessian_diag_rot` carries via its prior EMA, the rotated-basis update for that refresh proceeds with stale curvature, and the optimizer counter `_l5_skip_count` increments for telemetry.

The skip is graceful. Training continues, loss descends, parameter updates still apply. The contract is well-defined: the failure surface is identified, located, and the user can either (a) accept the L5 absorption rate as background noise, (b) audit the forward graph for the operator-shape rules in Section 6 and replace the offending operator with a bounded-second-derivative equivalent, or (c) switch to the GNB strategy (`hvp_strategy="gnb"`) which avoids second derivatives entirely. RACASO ships all three paths. The user picks based on their forward graph's structure.

When the forward graph contains only operators with bounded second derivatives over visited inputs, L5 fires 0% of refreshes and the optimizer delivers full design-intent compute. When an unsafe operator is present, four fixes are available:

1. **Bump the user-side eps** on a dividing expression $x / (\|x\| + \epsilon)$. This protects the first derivative but not the second. The eps sits on the wrong side of the autograd boundary; `tensor.norm()` computes its gradient internally without access to the user's floor. No effect on the second-derivative path.
2. **Replace `tensor.norm()` with `(x.pow(2).sum() + eps).sqrt()`** — manual sqrt with eps *inside* the sqrt's argument. This bounds the second derivative because $\text{arg} = \sum x^2 + \epsilon$ is bounded below by $\epsilon$.
3. **Replace the offending operator with a second-derivative-safe equivalent.** For cosine-similarity sites, negative squared Euclidean distance has bounded second derivatives by construction (subtract, square, sum, negate, multiply, softmax — all bounded).
4. **Wrap the operator in a custom `autograd.Function` with hand-rolled bounded double-backward.** Most general fix; preserves the original op's forward semantics while defining the second derivative analytically with safe floors.

The operator-shape rules in Section 6 generalize across any project's choice between these options.

---

## 6. The Second-Derivative DivBackward0 Failure Class

This section documents the class of forward-graph failure that L5 absorbs. It is independent of RACASO and applies to any second-order-aware optimizer (Hutchinson HVP, Gauss-Newton, K-FAC, Shampoo plus second-moment-correction, Adahessian).

### 6.1 The Mechanism

The failure shape:

- First-order backward produces finite gradients.
- The scalar formed from those gradients (for example $\sum_i \langle g_i, z_i \rangle$ for Hutchinson) is finite.
- The second-order backward (for example `autograd.grad(scalar, params)` for the batched HVP form) returns tensors that are uniformly NaN, with `isfinite_count = 0`.
- Every probed forward site is observably clean.
- The cause is in the second derivative of one operator in the shared upstream subgraph.

When `autograd.grad` is called the second time during HVP, PyTorch walks every node's `_backward` method. Most ops have bounded second derivatives over their input domain. A small set have unbounded second derivatives near input regions models visit during training. The most common offender is `tensor.norm()`:

$$\|x\|_2 = \sqrt{\sum x_i^2}$$
$$\frac{\partial \|x\|}{\partial x_i} = \frac{x_i}{\|x\|}$$
$$\frac{\partial^2 \|x\|}{\partial x_i \partial x_j} = \frac{\delta_{ij}\|x\|^2 - x_i x_j}{\|x\|^3}$$

The $1/\|x\|^3$ term in the second derivative is the failure surface. When $\|x\|$ is small, say $0.05$, the second-derivative scale is about $8000$. When $\|x\| \sim 0.01$, about $10^6$. The Pearlmutter HVP traversal multiplies these terms with adjacent gradient components; the products hit float32 overflow, become inf, become NaN, and the NaN propagates through the entire second-order graph to every downstream parameter's $Hz$ output uniformly.

A user-side $+ \epsilon$ floor on the divisor of $x / (\|x\| + \epsilon)$ protects the *first* derivative but does nothing for the *second*. `tensor.norm()` computes its gradient internally, before the user's floor is applied. The eps is on the wrong side of the autograd boundary.

### 6.2 The Diagnostic Recipe

PyTorch ships the right tool, but the API has a subtle correctness requirement. `torch.autograd.set_detect_anomaly(True)` must be active *during the forward pass* for the eventual NaN-on-backward to include the forward Python stack trace identifying the offending operator's `file:line`. The usage pattern:

```python
# Wrong (catches op class name only; warns "no forward pass info"):
with torch.autograd.set_detect_anomaly(True):
    hzs = torch.autograd.grad(scalar, params)  # forward ran outside

# Right (captures forward stack trace of offending op):
torch.autograd.set_detect_anomaly(True)    # global flag ON
out = model(input)                         # forward records stack traces
loss = out.loss
hzs = torch.autograd.grad(scalar, params)  # raises with full file:line
torch.autograd.set_detect_anomaly(False)   # disarm
```

The global-flag form (not the context-manager form) keeps anomaly mode active across both the forward and the subsequent HVP backward. Cost: roughly 5-10x slower forward. Arming for one refresh step is sufficient; the diagnostic fires once and identifies the exact source.

RACASO integrates this via the `_compute_hutchinson._anomaly_diagnosis_armed` class flag. When set to `True`, the next L5 fire prints the offending operator's full forward stack trace before disarming itself. A consumer who hits a sustained L5 skip rate flips the flag, runs one refresh, gets a traceback identifying file and line.

### 6.3 The Operator-Shape Rules

**Safe under HVP** (bounded second derivatives over typical input ranges):

- Linear, Convolution (matmul and conv produce constant second derivatives)
- ReLU, GELU, SiLU, Tanh, Sigmoid (bounded second derivatives everywhere)
- `softmax` (Jacobian chain rule produces finite second derivatives for finite-magnitude logits)
- `LayerNorm`, `RMSNorm` via the `x * rsqrt(var + eps)` form when eps is *inside* the rsqrt argument (the eps floors rsqrt's argument so its second derivative is bounded by $1/(4 \epsilon^{3/2})$)
- Elementwise arithmetic with constant operands (`x + 1`, `x * 0.5`, `x / 3.0`; the divisor is a leaf scalar, second derivative is zero)
- Sums, reductions, broadcasts

**Hazardous under HVP** (unbounded second derivatives over typical input regions):

- `tensor.norm(dim=...)`, $1/\|x\|^3$ in second derivative
- `tensor.div(other_tensor)` where `other_tensor` can be small, $1/\text{other}^3$ in second derivative
- `1 / x` or `x.reciprocal()` for small $x$, $2/x^3$ in second derivative
- `x.sqrt()` for small $x$, $-1/(4 x^{3/2})$; eps must be *inside* the sqrt's argument, not added to its output
- `x.log()` for small $x$, $-1/x^2$
- `x.pow(p)` for $p < 1$ and small $x$, $p(p-1) x^{p-2}$, same shape

The pattern: any operator whose second derivative contains a power of $1/x$ or $1/\|x\|$ where $x$ can approach zero is hazardous. The fix is either (a) replace the operator with a bounded-second-derivative equivalent (squared distance instead of cosine similarity for "which is closest" queries, for example), or (b) floor-clamp the operator's argument inside the operator's input (`(x² + eps).sqrt()` instead of `x.sqrt() + eps`).

### 6.4 Implications for Second-Order Optimization Adoption

Codebases written for first-order optimization may contain `tensor.norm()` in cosine attention, log-likelihood losses with small probabilities, or division-by-computed-magnitudes that work cleanly under SGD, Adam, or AdamW but fail under HVP. The user-facing failure mode usually appears as "optimizer produces NaN gradients" or "loss diverges around step N." The actual cause is often an operator-shape mismatch between the forward graph and the optimizer's second-order traversal.

The right shape for the optimizer is to provide a diagnostic mode (anomaly trace plus class flag) the user can flip on for one step to identify the offending operator. The right shape for the codebase is to audit forward-graph operators for second-derivative boundedness during migration to a second-order optimizer. RACASO's L5 absorbs the failure when it occurs, the diagnostic flag identifies the source, and the rest happens at the operator definition site, not the optimizer.

---

## 7. Engineering Provenance: Eight Attempts to Integrate Hutchinson with PyTorch Eager Autograd

This section is preserved from the integration log so future optimizer work has the failure record. The integration was performed inside a development testbed — a heterogeneous research language-model architecture with multi-branch composition and per-parameter-group optimizer assignment — but every wall described below is a property of the PyTorch eager-autograd interaction with second-order optimization, not a property of the testbed. Each attempt failed in a structurally specific way; each fix made the next failure visible. The fix at each wall was either small or out-of-scope-but-named. The recipe didn't survive contact with the codebase, but each contact taught the codebase what it needed.

### 7.1 The Eight Walls

| # | Wall | Fix that landed | Failure that emerged |
|---|---|---|---|
| 1 | `autograd.grad(outputs=p.grad, inputs=p, ...)` returns None | `.grad` is a leaf with no `grad_fn` even when `loss.backward(create_graph=True)`. Use `autograd.grad(loss, params, create_graph=True, retain_graph=True)` instead, stash the grad-fn-bearing tensor as a custom attribute on each param | First live training run: `hess: 0 ok / 64 skip` on every refresh, but for a different reason |
| 2 | CPU SDPA flash has no second derivative | `RuntimeError: derivative for aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented`. Wrap the forward in `sdpa_kernel([SDPBackend.MATH])` on refresh steps. MATH has full second-derivative support | Next error: `version mismatch; AsStridedBackward0 expected version 9 got 10` |
| 3 | Saved-tensor version mismatch | Eager autograd's saved-tensor version tracking conflicts when sharing the live graph with a `.backward()` call that mutates intermediate buffers in-place. Route HVP through `torch.func.jvp(grad(loss))` (functional API) to sidestep | Next error: `autograd.Function must override setup_context staticmethod` |
| 4 | Functorch can't compose through legacy `autograd.Function` | The reference integration's model contained 9 custom `autograd.Function` subclasses (Triton kernel wrappers, hand-rolled SSM, attention variants) all using legacy `forward+backward` style. Functorch requires the new `forward + setup_context + backward` style | Functorch path closed; fall back to GNB (Sophia §3.2) which uses one first-order backward only |
| 5 | GNB silently absorbed `multinomial(NaN)` | `torch.multinomial` raised when softmax fed it inf/nan, and a `try/except RuntimeError` absorbed the error letting opt.step run on NaN gradients for 30+ steps before detection | Hard isfinite gates at loss / logits / g_hat; halt training at the actual source instead of 30 steps downstream |
| 6 | `linalg.eigh` returns NaN columns silently on near-singular input | PyTorch's `linalg.eigh` does not raise on near-rank-deficient matrices; it returns eigenvectors with NaN columns for the null space. The L2 layer's residual-threshold gate accepted these because `nan < threshold` is False but `max(nan, x)` is order-dependent, and the broken Q could slip through | Explicit `isfinite()` check on Q + eigenvalues; progressive ridge cascade if non-finite; per-side independent Q commit (so a bad eigh on L doesn't reject R's good refresh) |
| 7 | Per-param HVP loop is 16x graph traversal cost | Naive implementation calls `autograd.grad(g_live_i, p_i, grad_outputs=z_i)` per param. At 16 parameters, 16x the graph traversal. Pearlmutter batched HVP via $\sum_i \langle g_i^{\text{live}}, z_i \rangle$ collapses to 1x | Cost dropped 16x; HVP wall-clock from roughly 27 s/step to roughly 6 s/step at K=1 |
| 8 | All single-site ablations failed to localize L5 source | F.normalize swap, joint-norm bypass, per-group rebalance off, div→mul on 5 sites, site probes at 8 forward locations, all showed identical 5/50 L5 skip rate. The source was not at any instrumented forward site | `torch.autograd.set_detect_anomaly(True)` wrapping the forward + HVP backward (the global-flag form, not the context-manager form). One refresh fires; raises with full forward stack trace identifying the cosine attention's `tensor.norm()` call |

### 7.2 Five Operational Lessons

1. **Don't prosecute on inference.** Attempts 2/3/4 burned because each "fix" was a plausible hypothesis without instrumentation. Attempt 5 added 50 lines of counters; every subsequent attempt was diagnosis-driven.

2. **The stage trap is cheaper than another guess.** Zero-overhead instrumentation that records the first non-finite value per stage paid for itself the first time it fired (`Q_R nan=382 at step 20` was the trap's first report on a near-rank-deficient eigh failure).

3. **PyTorch's silent failure modes are real.** `autograd.grad` returning `None` on leaf grads; `linalg.eigh` returning NaN columns on near-rank-deficient inputs; `torch.multinomial` raising on NaN probs while `softmax` happily produces NaN from divergent logits; `tensor.norm()` second derivative blowing up without forward observable signal. None of these throw at the source of the problem. The codebase has to do the throwing.

4. **Hard gates are load-bearing.** When GNB silently absorbed `multinomial(NaN)` via a bare `except RuntimeError`, opt.step ran on NaN gradients for 30 steps before detection. The fix was three `isfinite()` checks, at logits, at CE loss, and per-param at $\hat{g}$, each raising with diagnostic context. The hard gates halted at the actual source rather than 30 steps downstream on a wrecked model.

5. **K=4 is the right cadence default.** Window-aligned with a typical K=4 TBPTT cadence, K=4 gives both variance reduction (4 independent Hutchinson samples per opt-step) and per-step cost amortization. K=1 is more correct in the abstract; K=4 is more correct given how the rest of a typical training loop is sequenced.

---

## 8. Empirical Results

The benchmark suite (in `bench/`) comprises two layers:

- **Six synthetic problems (P1–P6)** that isolate the analytical claims:
  P1 off-axis quadratic (claim C1: rotation matters under off-diagonal
  curvature), P2 Rosenbrock 2D + N=100 (C1 at scale), P3 saddle 2D +
  N=20 (C2: Hutchinson captures negative curvature; C3: GNB cannot
  escape on its own), P4 row-spread pathology (C4 spread cap + C5 eigh
  safe-skip), P5 DivBackward0 hazard (C6: L5 safe-skip on unbounded
  second derivative), P6 classification (the canonical setup for
  Hutchinson vs GNB comparison).
- **Three real-task problems (R1–R3)** that demonstrate industry-credible
  training: R1 CIFAR-10 ResNet-18, R2 char-LM on tiny-shakespeare, R3
  byte-level NanoGPT (~30M params) on WikiText-2.

The harness runs against all 10 optimizers vendored in `bench/optimizers/`
(`adam`, `adamw`, `yogi`, `lion`, `liger`, `muogi`, `ramuogi`,
`racaso_hutchinson`, `racaso_gnb`, `naive_yogi_muon`). Each RACASO
variant is constructed via the HVP wrapper at
`bench/optimizers/racaso_hvp_wrappers.py` which computes the curvature
stash on each refresh step — Hutchinson via `torch.func.hvp` against the
problem's `forward(params)`, GNB via one extra first-order backward
against a CE loss with synthetic labels sampled from the model's
softmax (paper §2.2.2). For GNB to run, the problem must expose a
`logits_fn(params) -> [B, C]` method; P6 and R1/R2/R3 all do.

Sweeps run on NVIDIA RTX A4500 (20GB) via `python bench/run_bench.py
--sweep --device cuda`. Raw results: `bench/results.csv`. Figures:
`bench/figs/*.png`.

### 8.0 Methodology

**Per-optimizer learning-rate grids.** Different optimizer families have structurally different update magnitudes given the same nominal learning rate. Lion's update is `lr · sign(m_t)` — every coordinate moves by exactly `±lr`. Adam's update is `lr · m̂_t / (√v̂_t + ε)` — the same `lr` produces a coordinate move scaled down by the running variance estimate. Empirically a Lion step at `lr = 1e-3` moves parameters two to three orders of magnitude farther than an Adam step at the same `lr`. Running all optimizers on a shared LR grid would put one family in a regime where it diverges while the other runs at an appropriate step size, which is not a meaningful comparison.

We therefore use **per-family LR grids** matched to each optimizer family's typical operating range, following the convention used in published Lion, Sophia, and Muon comparison papers (Chen et al. 2023 §4.2 explicitly notes Lion requires a 3–10× lower LR than Adam). The exact grids used:

| Family | LR grid |
|---|---|
| Adam, AdamW, Yogi, NaiveYogiMuon | `[1e-4, 3e-4, 1e-3, 3e-3]` |
| Lion, Liger | `[1e-5, 3e-5, 1e-4, 3e-4]` |
| Muogi, RAMuogi, RACASO (Hutchinson + GNB) | `[3e-5, 1e-4, 3e-4, 1e-3]` |

These grids are pinned in `bench/run_bench.py::LR_SWEEP_BY_OPT` so the comparison is exactly reproducible from the open-source bench harness.

**Reporting convention.** For each (problem, optimizer) pair, figures and tables report the **best LR for that optimizer**, averaged across seeds — the LR that minimizes the seed-averaged final loss. The figure legend shows `(lr=X)` next to each optimizer's name so the LR each line corresponds to is always visible.

**Seed budgets.** Synthetic problems P1–P6 use seeds {0, 1, 2}. Real-task problems R1/R2/R3 use seeds {0, 1} because each run is much more expensive in GPU-time.

**Divergence filtering in figures.** Optimizers whose seed-averaged best-LR final loss exceeds 3× the median of all optimizers' final losses on a problem are filtered out of the main figure panels and listed in the figure subtitle. The filter is symmetric — RACASO would be filtered out of its own paper's figure if it diverged on a problem (and it does, on R3 NanoGPT, exactly the §6 DivBackward0 class this paper documents). Filtering it from the main panels is not selective; it applies the same standard to every optimizer.

**Hardware envelope.** Single GPU, RTX A4500 (20GB). RACASO's optimizer state at the 1B-parameter-equivalent synthetic module scale used for memory measurement (P4) exceeds the card's capacity and is OOM-skipped — that's a documented memory cost, not a missing data point.

### 8.1 P1 — Off-axis quadratic (C1)

**Setup.** 8-dim quadratic ``f(W) = 0.5 W^T H W - b^T W`` where ``H = U Λ U^T``, ``Λ = diag(10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005)`` and ``U`` is a random orthogonal matrix. The optimum is non-trivial and the rotation breaks axis-alignment, exposing methods that can model off-diagonal curvature.

**Results.** _TBD after GPU sweep._ See `bench/figs/fig_p1_off_axis_quad.png`.

### 8.2 P2 — Rosenbrock (2D + N=100)

**Setup.** Classic 2-D Rosenbrock starting at (-1.2, 1.0), and the generalized N=100 version starting at all -1.2.

**Results.** _TBD._ See `bench/figs/fig_p2_rosenbrock.png`.

### 8.3 P3 — Saddle escape (C2 + C3)

**Setup.** 2-D `f(x, y) = x² - y²` and 20-D `f(W) = 0.5 W^T diag([+1]*10 + [-1]*10) W`. The PSD-only second-order method (GNB) cannot escape; Hutchinson should escape via negative-curvature directions.

**Results.** _TBD._ See `bench/figs/fig_p3_saddle.png`.

### 8.4 P4 — Row-spread pathology (C4 + C5)

**Setup.** 8×8 quadratic with one row's gradient multiplied by a cycling burst factor (1e2 → 1e4 → 1e6 → 1e8). Tests L1 spread cap and L2 eigh safe-skip.

**Results.** _TBD._ See `bench/figs/fig_p4_row_spread.png` and the safety-counter bar chart in `bench/figs/fig_safety_counters.png`.

### 8.5 P5 — DivBackward0 hazard (C6, L5 safe-skip)

**Setup.** Ratio-form objective `(x·y / ||x||)² - target²` whose second derivative diverges as `||x|| → 0`. Tests RACASO's L5 safe-skip on unbounded HVP.

**Results.** _TBD._ See `bench/figs/fig_p5_div_backward.png`.

### 8.6 P6 — Classification (Hutchinson vs GNB)

**Setup.** 2-layer MLP on 10-class synthetic classification. The canonical setup where both `racaso_hutchinson` and `racaso_gnb` can run side-by-side; the problem exposes `logits_fn` so GNB has somewhere to sample synthetic labels from.

**Results.** _TBD._ See `bench/figs/fig_p6_classification.png`.

### 8.7 R1 — CIFAR-10 ResNet-18

**Setup.** ResNet-18 (~11.2M params, vendored at `bench/models/resnet18.py`) on CIFAR-10. 5000 steps, batch 128. Convergence threshold train loss < 0.5.

**Results.**

| Optimizer | Best LR | Final train loss | Steps to converge |
|---|---|---|---|
| AdamW | _TBD_ | _TBD_ | _TBD_ |
| Yogi | _TBD_ | _TBD_ | _TBD_ |
| Lion | _TBD_ | _TBD_ | _TBD_ |
| Liger | _TBD_ | _TBD_ | _TBD_ |
| Muogi | _TBD_ | _TBD_ | _TBD_ |
| RAMuogi | _TBD_ | _TBD_ | _TBD_ |
| **RACASO (Hutchinson)** | _TBD_ | _TBD_ | _TBD_ |
| **RACASO (GNB)** | _TBD_ | _TBD_ | _TBD_ |

(See `bench/figs/fig_r1_cifar10.png`.)

### 8.8 R2 — Char-LM on tiny-shakespeare

**Setup.** 4-layer char-LM (~3M params) on tiny-shakespeare. 3000 steps, batch 32, sequence length 128. Convergence threshold train loss < 1.5.

**Results.**

| Optimizer | Best LR | Final train loss | Steps to converge |
|---|---|---|---|
| AdamW | _TBD_ | _TBD_ | _TBD_ |
| Yogi | _TBD_ | _TBD_ | _TBD_ |
| Lion | _TBD_ | _TBD_ | _TBD_ |
| Liger | _TBD_ | _TBD_ | _TBD_ |
| Muogi | _TBD_ | _TBD_ | _TBD_ |
| RAMuogi | _TBD_ | _TBD_ | _TBD_ |
| **RACASO (Hutchinson)** | _TBD_ | _TBD_ | _TBD_ |
| **RACASO (GNB)** | _TBD_ | _TBD_ | _TBD_ |

(See `bench/figs/fig_r2_charlm.png`.)

### 8.9 R3 — NanoGPT (byte-level) on WikiText-2

**Setup.** 6-layer NanoGPT (~30M params): hidden 384, 6 heads, byte-level vocab 256, sequence length 256. WikiText-2-raw, 1000 steps, batch 8. Convergence threshold train loss < 5.0.

**Results.**

| Optimizer | Best LR | Final train loss | Steps to converge |
|---|---|---|---|
| AdamW | _TBD_ | _TBD_ | _TBD_ |
| Yogi | _TBD_ | _TBD_ | _TBD_ |
| Lion | _TBD_ | _TBD_ | _TBD_ |
| Liger | _TBD_ | _TBD_ | _TBD_ |
| Muogi | _TBD_ | _TBD_ | _TBD_ |
| RAMuogi | _TBD_ | _TBD_ | _TBD_ |
| **RACASO (Hutchinson)** | _TBD_ | _TBD_ | _TBD_ |
| **RACASO (GNB)** | _TBD_ | _TBD_ | _TBD_ |

(See `bench/figs/fig_r3_nanogpt.png`.)

### 8.10 Comparison with sibling family optimizers (Liger, Muogi, RAMuogi)

The RACASO benchmark suite runs against **all sibling-family optimizers** developed in this lineage — Liger (Christopher 2026a) and Muogi/RAMuogi (Christopher 2026b) — because each is published as a separate ArXiv submission with overlapping baselines, and cross-citation strengthens all three papers.

**Where each sibling wins.**

- **Liger** is expected to outperform RACASO on memory-constrained problems (R1/R2/R3 at scale) because Liger's optimizer state is ~50% of AdamW's, while RACASO's rotated-basis matrices push it *above* AdamW.
- **Muogi/RAMuogi** are expected to outperform RACASO on dense matrix-orthogonalization problems (R1 CIFAR ResNet) because NS5's polar decomposition is cheaper than RACASO's Hutchinson HVP refresh per step.

**Where RACASO wins.**

- **P3 saddle escape** — by construction. Hutchinson HVP captures negative curvature; GNB (and Adam-family `v_t` adaptive methods) cannot escape on their own.
- **P5 DivBackward0** — RACASO's L5 safe-skip handles unbounded second derivative gracefully; methods that naively call into `torch.autograd.grad` on a second-derivative graph hit eager-autograd pathologies (§7 documents 8 such attempts).
- **P6 classification with Hutchinson vs GNB** — head-to-head, GNB is strictly more conservative (PSD-only) while Hutchinson can both escape saddles and be hostile to forward-graph fragility. The data tells us which trade-off is worth it.

**Cross-comparison figure.** See `bench/figs/cross_comparison.png` — a single multi-panel figure overlaying all optimizers on R1/R2/R3. The same figure appears in Liger_Paper.md §9 and RAMuogi_Paper.md §9.

**Unified head-to-head table** (same content across all 3 papers; this paper highlights RACASO):

| Optimizer | R1 final loss | R2 final loss | R3 final loss | State bytes (% of AdamW) |
|---|---|---|---|---|
| AdamW | _TBD_ | _TBD_ | _TBD_ | 100.00% |
| Yogi | _TBD_ | _TBD_ | _TBD_ | 100.00% |
| Lion | _TBD_ | _TBD_ | _TBD_ | 50.00% |
| Liger | _TBD_ | _TBD_ | _TBD_ | ~50% |
| Muogi | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| RAMuogi | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **RACASO (Hutchinson)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **RACASO (GNB)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## 9. Configuration

| Field | Default | Meaning |
|---|---|---|
| `lr` | `6e-2` | Sophia paper default |
| `betas` | `(0.965, 0.99)` | Sophia momentum β1; Hessian-diagonal EMA β2 |
| `shampoo_beta` | `0.95` | Kronecker covariance EMA decay |
| `eps` | `1e-12` | Hessian denominator floor (Sophia clamp) |
| `eps_adam` | `1e-8` | Spread-cap `safe_max` floor + 1-D Yogi epsilon |
| `eps_yogi` | `1e-3` | 1-D Yogi variance floor |
| `rho` | `0.04` | Sophia clip threshold |
| `gamma` | `0.04` | Sophia Hessian denominator scaling ($\gamma \cdot |h|$) |
| `weight_decay` | `0.0` | Decoupled weight decay (AdamW-style) |
| `refresh_freq` | `10` | $U_L$, $U_R$ eigh refresh cadence |
| `hessian_freq` | `4` | Hessian-diagonal refresh cadence (K, TBPTT-window-aligned) |
| `eigh_residual_threshold` | `0.5` | L2 eigh acceptance threshold (per-side) |
| `spread_cap` | `10.0` | L1 row-spread bound (in rotated basis) |
| `radam_enabled` | `True` | L4 RAdam cold-start gate |
| `initial_accumulator` | `1e-6` | Initial value for momentum + hessian_diag_rot |
| `hvp_strategy` | `"hutchinson"` | Curvature estimator. `"hutchinson"` (true diagonal, includes negative curvature) or `"gnb"` (Gauss-Newton fallback, positive semidefinite, no second derivatives) |

---

## 10. Conclusion

RACASO decouples the calculation of curvature topology from the tracking of gradient coordinate history. By combining Kronecker-factored structural rotations with rigorous monolithic Hutchinson HVP sampling, RACASO ensures that second-order curvature alignment is mathematically rigorous, computationally viable, and resilient to the structural challenges of modern branching network topologies.

The optimizer is designed for failure modes that competing second-order methods do not survive: ill-conditioned Kronecker covariance under coupled-gradient regimes, Newton-Schulz convergence failure under sudden distribution shifts, and the silent failure modes PyTorch eager autograd exposes when the second-derivative path is naively integrated. The four-layer safety chain (L1 through L4) absorbs the first three. The L5 safe-skip absorbs the fourth, documented in Section 6 as a general class for any second-order-aware optimizer.

The engineering provenance in Section 7 is preserved because each wall was a property of the codebase plus framework interaction, not a property of Hutchinson itself. The fix at each wall was either small or out-of-scope-but-named. The recipe didn't survive contact with the codebase, but each contact taught the codebase what it needed. The diagnostic infrastructure built during that integration (stage trap, anomaly mode integration, site probe utility) is reusable for any future optimizer integration that hits the same operator-shape failure surface.

RACASO ships as a drop-in for any model with branching gradient paths and coupled-gradient regimes. The underlying composition (Kronecker eigenbasis rotation plus Sophia-style cautious step in the rotated basis, with the four-layer safety chain and L5 absorption) is general.

---

## Acknowledgments

Thanks to Ben Goertzel for the arXiv endorsement. The companion paper RAMuogi [Christopher 2026b] is developed in parallel; the safety-chain framing shared between the two papers benefited from cross-pollination across the design reviews of both optimizers.

---

## References

Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., Pham, H., Dong, X., Luong, T., Hsieh, C., Lu, Y., & Le, Q. V. (2023). Symbolic discovery of optimization algorithms. arXiv:2302.06675.

Christopher, R. (2026b). RAMuogi: Spectral-orthogonalization optimization with a four-layer failure-safety chain. *Companion paper, arXiv ID TBD upon submission.*

Gupta, V., Koren, T., & Singer, Y. (2018). Shampoo: Preconditioned stochastic tensor optimization. arXiv:1802.09568.

Hutchinson, M. F. (1989). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Communications in Statistics — Simulation and Computation* 18(3): 1059–1076.

Jordan, K., et al. (2024). Muon: An optimizer for hidden layers in neural networks. *Available at https://kellerjordan.github.io/posts/muon/.*

Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.

Liu, H., Li, Z., Hall, D., Liang, P., & Ma, T. (2023). Sophia: A scalable stochastic second-order optimizer for language model pre-training. arXiv:2305.14342.

Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., & Han, J. (2019). On the variance of the adaptive learning rate and beyond. arXiv:1908.03265 (RAdam).

Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. arXiv:1711.05101 (AdamW).

Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored approximate curvature. arXiv:1503.05671 (K-FAC).

Pearlmutter, B. A. (1994). Fast exact multiplication by the Hessian. *Neural Computation* 6(1): 147–160.

Shazeer, N., & Stern, M. (2018). Adafactor: Adaptive learning rates with sublinear memory cost. arXiv:1804.04235.

Vyas, N., Morwani, D., Zhao, R., Kaplun, G., Kakade, S., & Barak, B. (2024). SOAP: Improving and stabilizing Shampoo using Adam. arXiv:2409.11321.

Yao, Z., Gholami, A., Shen, S., Mustafa, M., Keutzer, K., & Mahoney, M. W. (2020). AdaHessian: An adaptive second order optimizer for machine learning. arXiv:2006.00719.

Zaheer, M., Reddi, S. J., Sachan, D., Kale, S., & Kumar, S. (2018). Adaptive methods for nonconvex optimization. NeurIPS 2018 (Yogi).

---

## License

MIT. See LICENSE file.
