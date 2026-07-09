"""
export_anomalib_onnx.py — Export anomalib PatchCore / EfficientAD to a *pure*
ONNX graph that matches the SuperSimpleNet / SK-RD4AD contract.

Root cause of the "meaningless output" bug
-------------------------------------------
The old ``export_model_to_onnx`` used ``engine.export(ExportType.ONNX)``. In
anomalib 2.x that wraps THREE things into one graph:

    PreProcessor (Resize + ImageNet Normalize)
        -> torch model (features + scoring)
            -> PostProcessor (min-max normalization + thresholding)

The GPU runtime (``inference_simulation``) *also* resizes + ImageNet-normalizes on
the host and does its own folder-global min-max, so every image was normalized
twice and the scores rescaled twice against calibrated stats. The result was a
numerically meaningless map/score even though the trained model was fine.

Fix / uniform contract (identical to the other 3 architectures)
---------------------------------------------------------------
We export the *inner* torch module (``lightning_model.model``) directly, NOT via
``engine.export``. That module is a pure forward pass: it expects an
already-normalized tensor and returns the raw anomaly map + score. We only strip
the Gaussian blur (host-side post-processing).

    Input   "input_tensor" : float32 [B, 3, H, W]  host-resized + ImageNet-
            normalized (the anomalib PreProcessor is intentionally NOT in-graph).
    Output  "anomaly_map"   : float32 [B, 1, H, W]  raw map, no blur, no min-max,
            no threshold.
    Output  "anomaly_score" : float32 [B]           raw model score.

Only the batch axis is dynamic (``dynamic_axes``). Host post-processing (blur +
folder-global min-max) is unchanged in inference_simulation.

IMPORTANT — validation
----------------------
anomalib is a fast-moving library and its inner-module attribute names differ
between versions. This module targets anomalib **2.x** and probes attributes
defensively. It MUST be re-run with the parity check (``--verify``) in the
training environment (where anomalib + a trained memory-bank/quantiles exist);
random weights are not enough for PatchCore (empty memory bank) or EfficientAD
(uncalibrated quantiles).
"""

import os

import numpy as np
import torch
import torch.nn as nn

INPUT_NAMES = ["input_tensor"]
OUTPUT_NAMES = ["anomaly_map", "anomaly_score"]
OPSET = 17


# ---------------------------------------------------------------------------
# Export-only wrapper around the inner anomalib torch module
# ---------------------------------------------------------------------------
class AnomalibPureWrapper(nn.Module):
    """
    Wraps ``lightning_model.model`` (PatchcoreModel / EfficientAdModel) so the
    exported graph returns a plain ``(anomaly_map, anomaly_score)`` tuple with the
    uniform names/shapes. The inner module is left in eval mode and its
    ImageNet-normalized-tensor-in / raw-out semantics are preserved.

    The only post-processing removed is the Gaussian blur inside the model's
    ``AnomalyMapGenerator`` (replicated on the host). anomalib's external
    PostProcessor (min-max + threshold) is *not* part of this inner module, so it
    is already excluded by construction.
    """

    def __init__(self, inner_model: nn.Module):
        super().__init__()
        self.model = inner_model
        self.model.eval()
        self._disable_blur(self.model)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _disable_blur(module: nn.Module) -> None:
        """Replace any GaussianBlur used by an AnomalyMapGenerator with Identity.

        anomalib places the blur either at ``anomaly_map_generator.blur`` or as a
        standalone ``GaussianBlur2d`` submodule; we neutralize both defensively so
        no smoothing is baked into the graph regardless of the exact version.
        """
        amg = getattr(module, "anomaly_map_generator", None)
        if amg is not None and hasattr(amg, "blur"):
            amg.blur = nn.Identity()
        for name, child in module.named_modules():
            if child.__class__.__name__ in ("GaussianBlur2d", "GaussianBlur"):
                parent, _, attr = name.rpartition(".")
                owner = module.get_submodule(parent) if parent else module
                setattr(owner, attr, nn.Identity())

    def forward(self, input_tensor: torch.Tensor):
        out = self.model(input_tensor)

        # In eval, anomalib inner models return an InferenceBatch dataclass with
        # .anomaly_map and .pred_score; older/ONNX paths may return a tuple/dict.
        anomaly_map = _get_field(out, "anomaly_map", 0)
        anomaly_score = _get_field(out, "pred_score", 1)

        # Normalize shapes to the uniform contract: map [B,1,H,W], score [B].
        if anomaly_map.dim() == 3:                    # [B,H,W] -> [B,1,H,W]
            anomaly_map = anomaly_map.unsqueeze(1)
        anomaly_score = anomaly_score.reshape(anomaly_score.shape[0])
        return anomaly_map, anomaly_score


def _get_field(out, name, tuple_index):
    """Pull a field from an InferenceBatch / dict / tuple output uniformly."""
    if hasattr(out, name):
        return getattr(out, name)
    if isinstance(out, dict):
        return out[name]
    return out[tuple_index]


# Embedded in the .onnx file so the inference runtime (inference_simulation) can
# auto-configure blur/scoring per architecture instead of relying on CLI flags an
# operator has to remember to set correctly (that exact class of bug produced
# wrong scores for SK-RD4AD when run with SuperSimpleNet's defaults).
#
# "verified": "false" for BOTH entries below: unlike SuperSimpleNet and SK-RD4AD,
# these values were NOT checked against a live anomalib install / trained
# checkpoint (anomalib isn't available in this dev environment). score_source
# ="graph" is correct by anomalib's architecture (both PatchcoreModel.forward and
# EfficientAdModel.forward compute pred_score internally - PatchCore's from the
# reweighted nearest-neighbor distance, EfficientAd's from the quantile-normalized
# map - neither is literally map.max(), so score_from_map/map_max_blurred would be
# wrong). The blur kernel/sigma guesses (matching anomalib's typical
# GaussianBlur2d(sigma=4) default) are NOT confirmed for this repo's config and
# MUST be validated with verify_parity(...) plus a real accuracy check (AUROC on
# a labeled set) before trusting production verdicts.
_METADATA_BY_ARCH = {
    "patchcore": {
        "architecture": "patchcore",
        "score_source": "graph",
        "blur_kernel_size": "25",
        "blur_sigma": "4.0",
        "verified": "false",
    },
    "efficientad": {
        "architecture": "efficientad",
        "score_source": "graph",
        "blur_kernel_size": "0",   # best guess: EfficientAd's PDN path is not known to blur
        "blur_sigma": "0.0",
        "verified": "false",
    },
}


def _write_metadata(onnx_path: str, model_name: str) -> None:
    import onnx
    key = "efficientad" if "efficientad" in model_name.lower() else "patchcore"
    # weights_source: this exporter only ever runs on the trained LightningModule
    # coming out of engine.fit, so there is no random-weights path here; the key
    # exists for uniformity with the other exporters (the runtime logs it, and
    # refuses "random_self_test" models).
    meta = {"anomaly_export_contract": "1.0",
            "weights_source": f"trained:{model_name}",
            **_METADATA_BY_ARCH[key]}
    m = onnx.load(onnx_path)
    for k, v in meta.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(m, onnx_path)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_pure_onnx(lightning_model, input_size, onnx_path: str, device: str = "cpu") -> str:
    """
    Export a trained anomalib LightningModule (Patchcore / EfficientAd) to a pure
    ONNX graph with dynamic batch and uniform I/O names.

    Args:
        lightning_model: the trained anomalib LightningModule (must expose ``.model``).
        input_size: (H, W) the model was trained at.
        onnx_path: destination .onnx path.
        device: "cuda" or "cpu".
    """
    inner = getattr(lightning_model, "model", lightning_model)
    wrapper = AnomalibPureWrapper(inner).to(device).eval()

    h, w = input_size
    dummy = torch.randn(2, 3, h, w, dtype=torch.float32, device=device)
    dynamic_axes = {
        "input_tensor": {0: "batch"},
        "anomaly_map": {0: "batch"},
        "anomaly_score": {0: "batch"},
    }

    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy,), onnx_path,
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes, opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,           # classic exporter honours the dynamic_axes dict
            external_data=False,
        )

    import onnx
    onnx.checker.check_model(onnx_path)
    _write_metadata(onnx_path, lightning_model.__class__.__name__)
    print(f"[OK] Pure ONNX export: {onnx_path}")
    print(f"     input 'input_tensor' : float32 [B,3,{h},{w}] (host-normalized)")
    print( "     output 'anomaly_map'  : float32 [B,1,H,W] (raw, no blur)")
    print( "     output 'anomaly_score': float32 [B] (raw)")
    return onnx_path, wrapper


def verify_parity(wrapper: nn.Module, onnx_path: str, input_size, device: str = "cpu",
                  atol: float = 1e-5) -> None:
    """Strict PyTorch<->ONNX parity on a normalized dummy tensor (atol=1e-5),
    checking batch=1 and batch=4 to prove the dynamic axis."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    h, w = input_size
    print("\n--- PyTorch vs ONNX parity (atol=1e-5) ---")
    for batch in (1, 4):
        x = rng.standard_normal((batch, 3, h, w)).astype(np.float32)
        with torch.no_grad():
            tm, ts = wrapper(torch.from_numpy(x).to(device))
        om, os_ = sess.run(OUTPUT_NAMES, {INPUT_NAMES[0]: x})
        np.testing.assert_allclose(om, tm.cpu().numpy(), atol=atol, rtol=0,
                                   err_msg=f"anomaly_map mismatch (batch={batch})")
        np.testing.assert_allclose(os_, ts.cpu().numpy(), atol=atol, rtol=0,
                                   err_msg=f"anomaly_score mismatch (batch={batch})")
        print(f"  batch={batch}: map |Δ|max={np.abs(om-tm.cpu().numpy()).max():.2e}  "
              f"score |Δ|max={np.abs(os_-ts.cpu().numpy()).max():.2e}  OK")
    print("[PASS] Exact mathematical parity confirmed.")
