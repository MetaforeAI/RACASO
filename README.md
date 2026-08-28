# RACASO

**Rotation-Aligned Cautious Approximately Second-Order Optimizer.**

A Sophia-style cautious clip applied in a Kronecker-factored eigenbasis. The per-eigendirection step size comes from one of two curvature estimators, both computed *in the rotated basis* and positive by construction:

- **Hutchinson** (`curvature_mode="hutchinson"`, default) — the true rotated-basis Hessian diagonal, probed with a Rademacher vector in the eigenbasis. Captures negative curvature (saddle escape).
- **SOAP** (`curvature_mode="soap"`) — the second moment of the rotated gradient, `EMA((Qᵀ g Q)²)`. No second derivatives; Adam-in-eigenbasis.

Designed for architectures where SOAP's covariance accumulators go rank-1 and Muon-style Newton-Schulz polynomials diverge (branching paths, joint norms, coupled gradients).

Five safety layers: spread cap, eigh-residual gate, Yogi fallback for 1-D params, RAdam cold-start, and an L5 safe-skip for when the forward graph hits the second-derivative `DivBackward0` failure class. A Gauss-Newton-Bartlett variant (`RACASOGNB`) is available as a no-second-derivative alternative when your forward graph isn't HVP-safe.

## Install

Single file. Vendor it.

```bash
curl -O https://raw.githubusercontent.com/MetaforeAI/RACASO/main/racaso.py
```

Or clone and import:

```bash
git clone https://github.com/MetaforeAI/RACASO.git
```

## Use

The Hutchinson curvature estimate needs a re-evaluable forward function (it is computed via a Hessian-vector product). Pass `forward_fn` — a callable `params -> scalar loss` — and the optimizer computes and consumes the rotated-basis HVP itself on refresh steps:

```python
from racaso import RACASO

params = list(model.parameters())

def forward_fn(params):
    return model(batch).loss          # re-evaluable scalar loss over `params`

opt = RACASO(params, lr=3e-4, hessian_freq=4, curvature_mode="hutchinson",
             forward_fn=forward_fn)

for batch in loader:
    loss = model(batch).loss
    loss.backward()
    opt.step()                        # HVP refresh is handled internally on refresh steps
    opt.zero_grad()
```

The SOAP variant needs no forward function (it uses the rotated gradient's second moment):

```python
opt = RACASO(params, lr=3e-4, curvature_mode="soap")
# standard loss.backward(); opt.step(); opt.zero_grad() loop
```

If you don't want to audit your forward graph for second-derivative-unsafe operators (`tensor.norm()` and friends, see paper §6), use the Gauss-Newton-Bartlett wrapper, which needs only a `logits_fn(params) -> [B, C]`:

```python
from bench.optimizers.racaso_hvp_wrappers import RACASOGNB

opt = RACASOGNB(params, lr=3e-4)
opt.set_hvp_context(logits_fn, params)   # GNB samples synthetic labels from the model's softmax
```

## Telemetry

```python
t = opt.get_telemetry()
print(f"rot: {t['rotation_success_count']} ok / {t['rotation_skip_count']} skip  "
      f"hess: {t['hessian_success_count']} ok / {t['hessian_skip_count']} skip  "
      f"clip%={t['last_clip_fraction']*100:.1f}  r_t={t['last_r_t']:.3f}  "
      f"mode={t['curvature_mode']}")

counts = opt.get_safety_counts()   # {'l1':.., 'l2':.., 'l3':.., 'l4':.., 'l5':..}
```

If `hess: ... skip` climbs, your forward graph has an unbounded second derivative somewhere — the L5 safe-skip is absorbing non-finite HVP output. The paper (§6) lists which operators to look at first, and the GNB variant above sidesteps second derivatives entirely.

## Architecture & Visuals

- **Figure 1 (Architecture Pipeline & Safety Chain)**: `bench/figs/fig_architecture_pipeline.png`
- **Figure 2 (Second-Derivative Singularity & Operator Boundedness)**: `bench/figs/fig_divbackward0_mechanism.png`
- **Figure 3 (Morpheus Production Training Dynamics)**: `bench/figs/fig_morpheus_production_dynamics.png`
- **Benchmark Suite & Comparisons**: `bench/figs/cross_comparison.png`

## Paper

- [RACASO_Paper.md](RACASO_Paper.md) — Comprehensive paper detailing the rotated-basis trace estimation math, the 5-layer safety chain, production telemetry in Morpheus, and the 8-attempt PyTorch autograd provenance.
- [RACASO_Paper.pdf](RACASO_Paper.pdf) — Rendered PDF for distribution and arXiv preprint.

## License & Open-Source Attributions

Distributed under the **MIT License**. See [LICENSE](LICENSE) for terms.

RACASO gratefully builds upon and synthesizes core ideas from:
- **CASPR** (Cautious Adaptive Second-order Preconditioned Regularizer) for the foundational architectural intuition bridging second-order cautious step bounds with structured preconditioning.
- **SOAP** (Vyas et al. 2024, Harvard/MIT/OpenAI) [MIT License]
- **Sophia** (Liu et al. 2023, Stanford) [MIT License]
- **Shampoo** (Gupta et al. 2018, Google Research) [Apache 2.0]
- **AdaHessian** (Yao et al. 2020, UC Berkeley) [Apache 2.0]
- **RAdam** (Liu et al. 2019) [Apache 2.0 / MIT]
- **Yogi** (Zaheer et al. 2018, Google Research) [Apache 2.0 / MIT]

