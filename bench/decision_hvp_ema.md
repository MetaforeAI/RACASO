# HVP-EMA decision: squared vs linear

## Question

`racaso.py:462-465` (pre-fix) ran
```python
hessian_diag_rot.mul_(beta2).addcmul_(h_rot, h_rot, value=1.0 - beta2)
```
which is `EMA = beta2 * EMA + (1 - beta2) * h * h` — i.e. it tracks
`E[h^2]`, a Hessian magnitude/variance estimator. The sign of `h` is
destroyed before it ever reaches the denominator.

The paper Abstract and §2.2.1 explicitly claim RACASO "preserves both
positive and negative curvature." That claim is contradicted by the
squared form: positive and negative `h` collapse to the same `h^2`.

Linear-signed EMA (`add_(h, alpha=1-beta2)`) tracks `E[h]`. The sign is
preserved inside the EMA. The denominator still applies `.abs()` at the
end (so the denom is always positive — Sophia's formulation), but the
EMA dynamics, and crucially the *magnitude* of the surviving EMA when
signed `h` partially cancels, is fundamentally different.

## Experiment

`bench/decision_hvp_ema_run.py` constructs a matrix saddle
`f(W) = 0.5 * sum D_ij W_ij^2` with `W` of shape `(5, 4)`, `D` a mix of
positive and negative entries with magnitudes geometrically distributed
(so the magnitudes vary and `|D|` is non-trivially different from `D^2`).

Two variants, both with the SOAP-style rotation fix already applied
(no more `Q_L^T h_rot Q_R` similarity transform — see #2 in the
fix sheet):

  - **A (squared):** `EMA <- beta2 * EMA + (1 - beta2) * h * h`
  - **B (linear):** `EMA <- beta2 * EMA + (1 - beta2) * h`

LR sweep `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`, seeds `{0, 1, 2}`, 500 steps,
`rho=10.0` (loose so the denominator, not the clip, drives behavior),
exact Hessian-diagonal stash (no Hutchinson sampling noise — isolates
the EMA decision from autograd variance).

## Numbers

| lr      | squared loss | squared ‖W‖² | linear loss | linear ‖W‖² |
|---------|-------------:|-------------:|------------:|------------:|
| 1e-04   | -1.647e+01   | 2.938e-01    | -2.254e+01  | 3.292e-01   |
| 3e-04   | -3.793e+01   | 3.617e-01    | -7.025e+01  | 5.617e-01   |
| 1e-03   | -3.098e+02   | 1.226e+00    | -6.517e+02  | 3.006e+00   |
| 3e-03   | -1.134e+04   | 3.315e+01    | -1.508e+04  | 4.672e+01   |
| 1e-02   | -3.593e+06   | 1.001e+04    | -3.647e+06  | 1.022e+04   |

(Mean over 3 seeds. "loss" is `0.5 * Σ D_ij W_ij²`, so more negative =
deeper saddle escape. ‖W‖² is the squared parameter norm, so larger =
farther escape from origin.)

## Decision

**Ship variant B (linear-signed EMA).**

At every LR tested, the linear-EMA produces a deeper saddle escape than
the squared variant (lower / more-negative final loss, larger ‖W‖²). The
gap is largest at the moderate LRs (1.4×–2.1× larger negative loss at
LR ∈ {1e-4, 3e-4, 1e-3}) and shrinks at the extreme high LR where both
variants are running away into the negative-curvature direction
asymptotically.

The denominator still applies `.abs()` (so it's always positive — the
optimizer remains stable), but the underlying EMA tracks the
sign-preserving Hessian-diagonal estimate. The mathematical claim in
§2.2.1 about "preserving negative curvature" is now defensible.

## What this does NOT mean

The denominator itself takes `.abs()`, so the sign of `h` does not
propagate to a sign-flip on the update direction. Negative-curvature
*direction* information flows through the **momentum** path (which sees
the raw signed gradient `g`), not through the Hessian-EMA denominator
per se. What the linear-EMA gets right is the *magnitude* of the
denominator: under a mixed-sign Hessian, positive and negative
contributions in the same parameter element partially cancel in
`E[h]` (so |denom| is smaller in saddle-mixed regions, allowing larger
steps), whereas they amplify in `E[h^2]` (so |denom| is larger in
saddle-mixed regions, throttling steps exactly where saddle escape is
wanted).

§2.2.1 should be updated to say: "The Hessian-diagonal EMA preserves
sign internally (linear EMA, not squared); the denominator applies
`.abs()` for stability. Sign-preservation manifests as smaller
denominators in mixed-curvature regions, allowing the momentum-driven
update to take larger steps in saddle escape directions."

## File pointers

- Experiment script: `bench/decision_hvp_ema_run.py`
- Optimizer code: `racaso.py` and `bench/optimizers/racaso.py` (vendored
  copy — keep them in sync).
- Both updated to variant B (linear EMA) post-decision.
