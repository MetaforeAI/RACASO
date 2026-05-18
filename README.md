# RACASO

**Rotation-Aligned Cautious Approximately Second-Order Optimizer.**

Sophia-style cautious clip applied in a Kronecker-factored eigenbasis, with Hutchinson HVP for the per-element Hessian diagonal. Designed for architectures where SOAP's covariance accumulators go rank-1 and Muon-style Newton-Schulz polynomials diverge (branching paths, joint norms, coupled gradients).

Five safety layers: spread cap, eigh-residual gate, Yogi fallback for 1-D params, RAdam cold-start, and an L5 safe-skip for when the forward graph hits the second-derivative DivBackward0 failure class. GNB (Gauss-Newton-Bartlett) is available as a no-second-derivative alternative when your forward graph isn't HVP-safe.

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

```python
from racaso import RACASO

opt = RACASO(model.parameters(), lr=6e-2, hessian_freq=4)

# Hutchinson refresh steps need create_graph=True on the first backward.
# Non-refresh steps work like any optimizer.
for step, batch in enumerate(loader):
    loss = model(batch).loss
    if opt.is_refresh_step():
        opt.step_with_hvp(loss)   # batched HVP, monolithic autograd
    else:
        loss.backward()
        opt.step()
    opt.zero_grad()
```

Switch to GNB if you don't want to audit your forward graph for `tensor.norm()` and friends:

```python
opt = RACASO(model.parameters(), lr=6e-2, hvp_strategy="gnb")
```

## Telemetry

```python
t = opt.get_telemetry()
print(f"rot: {t['rotation_success_count']} ok / {t['rotation_skip_count']} skip  "
      f"hess: {t['hessian_success_count']} ok / {t['hessian_skip_count']} skip  "
      f"clip%={t['last_clip_fraction']*100:.1f}  r_t={t['last_r_t']:.3f}")
```

If `hess: ... skip` climbs, your forward graph has an unbounded second derivative somewhere. The paper (§6) lists which operators to look at first; flip `_compute_hutchinson._anomaly_diagnosis_armed = True` for one refresh and the next L5 fire prints the offending `file:line`.

## Paper

[RACASO_Paper.pdf](RACASO_Paper.pdf) — architecture, math, eight-attempt integration provenance, and the second-derivative operator-shape rules.

## License

MIT.
