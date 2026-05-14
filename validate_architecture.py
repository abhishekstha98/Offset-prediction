"""
validate_architecture.py - Structural validation of baseline and multi-channel models.

Checks (no training runs):
  1. Dummy graph matches real dataset dimensions (N=23, 17 node features, 4 edge features).
  2. Forward pass: BaselineModel (OffsetMPT) - output (N, 2), finite, no errors.
  3. Forward pass: MultiChannelOffsetModel - output (N, 2), finite, no errors.
  4. Both models produce identical output shapes.
  5. OffsetLoss (masked MAE) accepts output from both models -> finite scalar.
  6. Factory dispatch: build_model(cfg) returns correct type for each model_type.
  7. Gradient flow: .backward() runs without errors on both models.

Run from project root:
    python validate_architecture.py
"""

import copy
import os
import sys
import traceback

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"
SEP = "-" * 60


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"  {status}  {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


# ---------------------------------------------------------------------------
# 1. Build dummy graph matching real dataset dimensions
# ---------------------------------------------------------------------------
section("1. Dummy Graph Construction")

N = 23          # typical Netherlands station count
E = N * 3       # k=3 directed edges per node ~= 69
IN_FEATURES = 17  # fog-upgrade feature vector
EDGE_DIM = 4    # [distance_km, delta_lat, delta_lon, delta_height]
HIDDEN_DIM = 64
OUT_DIM = 2     # [dTmax, dTmin]

torch.manual_seed(0)

x = torch.randn(N, IN_FEATURES)
edge_index = torch.randint(0, N, (2, E), dtype=torch.long)
edge_attr = torch.randn(E, EDGE_DIM)
y = torch.randn(N, OUT_DIM)

# valid_mask: randomly mark some nodes as having NaN targets (like real data)
valid_mask = torch.rand(N, OUT_DIM) > 0.2   # ~80% valid

check("x      shape == (N=23, 17)", x.shape == (N, IN_FEATURES), str(x.shape))
check("edge_index shape == (2, E)", edge_index.shape == (2, E), str(edge_index.shape))
check("edge_attr shape == (E, 4)", edge_attr.shape == (E, EDGE_DIM), str(edge_attr.shape))
check("valid_mask shape == (N, 2)", valid_mask.shape == (N, OUT_DIM), str(valid_mask.shape))


# ---------------------------------------------------------------------------
# 2 & 3. Instantiate both models and run forward passes
# ---------------------------------------------------------------------------
section("2. Model Instantiation & Forward Pass")

all_passed = True

from src.models.mpt import OffsetMPT
from src.models.multi_channel_model import MultiChannelOffsetModel
from src.utils.loss import BackboneMultiTaskLoss, OffsetLoss

models = {
    "BaselineModel (OffsetMPT)": OffsetMPT(
        in_features=IN_FEATURES,
        hidden_dim=HIDDEN_DIM,
        heads=4,
        num_gnn_layers=2,
        edge_dim=EDGE_DIM,
        out_dim=OUT_DIM,
        dropout=0.0,
    ),
    "MultiChannelOffsetModel": MultiChannelOffsetModel(
        in_features=IN_FEATURES,
        hidden_dim=HIDDEN_DIM,
        heads=4,
        num_gnn_layers=2,
        edge_dim=EDGE_DIM,
        out_dim=OUT_DIM,
        dropout=0.0,
        num_channels=4,
    ),
}

outputs = {}
for name, model in models.items():
    print(f"\n  Model: {name}")
    model.eval()
    try:
        with torch.no_grad():
            pred = model(x, edge_index, edge_attr)
        outputs[name] = pred

        ok_shape = check("  output shape == (N, 2)", pred.shape == (N, OUT_DIM), str(pred.shape))
        ok_finite = check("  all outputs finite", torch.isfinite(pred).all().item())
        ok_dtype = check("  output dtype == float32", pred.dtype == torch.float32, str(pred.dtype))

        if not (ok_shape and ok_finite and ok_dtype):
            all_passed = False

    except Exception:
        print(f"  {FAIL}  Exception during forward pass:\n  {traceback.format_exc()}")
        all_passed = False


# ---------------------------------------------------------------------------
# 4. Shape parity between both models
# ---------------------------------------------------------------------------
section("3. Output Shape Parity")

if len(outputs) == 2:
    shapes = [v.shape for v in outputs.values()]
    check(
        "Both models return identical output shapes",
        shapes[0] == shapes[1],
        f"{shapes[0]} vs {shapes[1]}",
    )
else:
    print(f"  {FAIL}  Not enough model outputs to compare (forward pass likely failed above)")
    all_passed = False


# ---------------------------------------------------------------------------
# 5. Loss computation: OffsetLoss accepts both model outputs
# ---------------------------------------------------------------------------
section("4. Loss Computation (OffsetLoss)")

loss_fn = OffsetLoss(lambda_tmax=1.0, lambda_tmin=1.0)

# Re-run with gradients for loss backward check
for name, model in models.items():
    model.train()
    try:
        pred = model(x, edge_index, edge_attr)   # (N, 2) with grad
        loss, l_tmax, l_tmin = loss_fn(pred, y, valid_mask)

        ok_loss = check(
            f"  [{name}] loss is finite scalar",
            torch.isfinite(loss).item() and loss.ndim == 0,
            f"loss={loss.item():.4f}",
        )
        ok_tmax = check(
            f"  [{name}] Tmax component finite",
            torch.isfinite(l_tmax).item(),
            f"{l_tmax.item():.4f}",
        )
        ok_tmin = check(
            f"  [{name}] Tmin component finite",
            torch.isfinite(l_tmin).item(),
            f"{l_tmin.item():.4f}",
        )
        if not (ok_loss and ok_tmax and ok_tmin):
            all_passed = False
    except Exception:
        print(f"  {FAIL}  Exception in loss for {name}:\n  {traceback.format_exc()}")
        all_passed = False


# ---------------------------------------------------------------------------
# 6. Gradient flow: .backward() should work for both models
# ---------------------------------------------------------------------------
section("5. Gradient Flow (.backward())")

for name, model in models.items():
    model.train()
    try:
        pred = model(x, edge_index, edge_attr)
        loss, _, _ = loss_fn(pred, y, valid_mask)
        loss.backward()

        # Check at least one gradient is non-None and non-zero
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        has_grads = len(grads) > 0 and any(g.abs().sum() > 0 for g in grads)

        ok = check(f"  [{name}] gradients flow (non-zero grads: {len(grads)})", has_grads)
        if not ok:
            all_passed = False
    except Exception:
        print(f"  {FAIL}  Exception in backward for {name}:\n  {traceback.format_exc()}")
        all_passed = False


# ---------------------------------------------------------------------------
# 7. Factory dispatch
# ---------------------------------------------------------------------------
section("6. Factory Dispatch (build_model)")

try:
    from src.config import Config
    from src.models.factory import build_model

    baseline_cfg = Config()
    baseline_cfg.model.model_type = "baseline"
    m = build_model(baseline_cfg)
    check("  'baseline' -> returns OffsetMPT", isinstance(m, OffsetMPT), type(m).__name__)

    mc_cfg = Config()
    mc_cfg.model.model_type = "multi_channel"
    mc_cfg.model.num_channels = 4
    m = build_model(mc_cfg)
    check(
        "  'multi_channel' -> returns MultiChannelOffsetModel",
        isinstance(m, MultiChannelOffsetModel),
        type(m).__name__,
    )

    default_cfg = Config()   # no overrides -> should load baseline
    m = build_model(default_cfg)
    check(
        "  default config -> returns OffsetMPT (baseline)",
        isinstance(m, OffsetMPT),
        type(m).__name__,
    )

    bad_cfg = Config()
    bad_cfg.model.model_type = "unknown"
    try:
        build_model(bad_cfg)
        check("  unknown type -> raises ValueError", False)
        all_passed = False
    except ValueError:
        check("  unknown type -> raises ValueError", True)

except Exception:
    print(f"  {FAIL}  Exception in factory check:\n  {traceback.format_exc()}")
    all_passed = False


# ---------------------------------------------------------------------------
# 8. MultiChannelGraphAttention attention weight return
# ---------------------------------------------------------------------------
section("7. Optional Attention Weights (MultiChannelOffsetModel)")

try:
    mc_model = MultiChannelOffsetModel(
        in_features=IN_FEATURES,
        hidden_dim=HIDDEN_DIM,
        heads=4,
        num_gnn_layers=1,
        edge_dim=EDGE_DIM,
        out_dim=OUT_DIM,
        dropout=0.0,
        num_channels=4,
    )
    mc_model.eval()
    with torch.no_grad():
        pred = mc_model(x, edge_index, edge_attr, return_attn=False)

    check("  return_attn=False -> output still (N,2)", pred.shape == (N, 2), str(pred.shape))

except Exception:
    print(f"  {FAIL}  Exception:\n  {traceback.format_exc()}")
    all_passed = False


# ---------------------------------------------------------------------------
# 9. Shared-backbone multitask path
# ---------------------------------------------------------------------------
section("8. Shared Backbone Multitask Path")

try:
    multitask_model = OffsetMPT(
        in_features=IN_FEATURES,
        hidden_dim=HIDDEN_DIM,
        heads=4,
        num_gnn_layers=2,
        edge_dim=EDGE_DIM,
        out_dim=OUT_DIM,
        enable_fog_head=True,
        fog_out_dim=1,
        dropout=0.0,
    )
    multitask_model.eval()
    with torch.no_grad():
        outputs = multitask_model.forward_multitask(x, edge_index, edge_attr)

    check(
        "  multitask offset output shape == (N,2)",
        outputs["offset"].shape == (N, 2),
        str(outputs["offset"].shape),
    )
    check(
        "  fog head returns (N,1) logits",
        outputs["fog_logits"] is not None and outputs["fog_logits"].shape == (N, 1),
        str(outputs["fog_logits"].shape if outputs["fog_logits"] is not None else None),
    )

    fog_target = torch.randint(0, 2, (N,), dtype=torch.float32)
    fog_valid_mask = torch.rand(N) > 0.2
    multitask_loss = BackboneMultiTaskLoss()
    losses = multitask_loss(
        outputs["offset"],
        y,
        valid_mask,
        fog_logits=outputs["fog_logits"],
        fog_target=fog_target,
        fog_valid_mask=fog_valid_mask,
    )
    check(
        "  multitask total loss is finite",
        torch.isfinite(losses["total"]).item(),
        f"{losses['total'].item():.4f}",
    )
    check(
        "  fog loss is finite",
        torch.isfinite(losses["loss_fog"]).item(),
        f"{losses['loss_fog'].item():.4f}",
    )
except Exception:
    print(f"  {FAIL}  Exception:\n  {traceback.format_exc()}")
    all_passed = False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("VALIDATION SUMMARY")
if all_passed:
    print("  ALL CHECKS PASSED\n")
else:
    print("  SOME CHECKS FAILED - review output above\n")

sys.exit(0 if all_passed else 1)
