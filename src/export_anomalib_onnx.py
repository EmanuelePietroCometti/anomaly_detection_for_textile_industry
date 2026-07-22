"""
export_anomalib_onnx.py — Export anomalib PatchCore / EfficientAD to ONNX under
the shared export contract 3.0 (see export_common.py).

Contract 3.0 in one paragraph
-----------------------------
    Input   "image"         : float32 [B, 3, H, W]  RGB in [0,1], model
            resolution, NOT normalized — ImageNet normalization is IN-GRAPH.
    Output  "anomaly_map"   : float32 [B, 1, H, W]  FINAL map: the model's own
            upsample + Gaussian blur are kept inside the graph. No min-max,
            no threshold.
    Output  "anomaly_score" : float32 [B]           FINAL number, directly
            comparable with metadata["calibrated_threshold"].

Only the batch axis is dynamic. The C++ runtime does no decisional
post-processing: it validates the metadata contract at startup and compares
anomaly_score against the calibrated threshold.

Differences from the previous (contract 1.0) version of this file
-----------------------------------------------------------------
1. The Gaussian blur is NO LONGER stripped from the graph. Under contract 3.0
   the map output is final ("map_semantics=final_blurred"), so the blur the
   model was trained/evaluated with stays where it belongs. This also removes
   the need to *guess* blur_kernel_size/blur_sigma in the metadata: the values
   are probed from the live model (``_probe_blur``) and reported as facts.
2. ImageNet normalization moved in-graph (``InGraphNormalize`` prepended by
   ``ExportWrapper``): the graph input is plain RGB in [0,1].
3. Metadata follow the mandatory contract-3.0 schema (``build_metadata``) and
   are no longer hardcoded per architecture.
4. Trained-state guards: exporting a PatchCore with an empty memory bank or an
   EfficientAD with unset quantiles produces a syntactically valid but
   numerically meaningless graph with no error anywhere downstream — so the
   export now refuses to run in those states.

anomalib is a fast-moving library and its inner-module attribute names differ
between versions. This module targets anomalib **2.x** and probes attributes
defensively. It MUST be re-run with the parity check in the training
environment (where anomalib + a trained memory-bank/quantiles exist).
"""

from pathlib import Path

import torch
import torch.nn as nn

try:
    from .export_common import (
        ExportWrapper,
        OUTPUT_NAMES,
        build_metadata,
        export,
        verify,
    )
except ImportError:  # standalone execution outside the src package
    from export_common import (
        ExportWrapper,
        OUTPUT_NAMES,
        build_metadata,
        export,
        verify,
    )

# Protobuf (single-file .onnx, external_data=False) hard limit is 2 GB; stay
# below it with margin for the graph structure itself. PatchCore's memory bank
# is the only tensor in this repo that can realistically get there.
_PROTOBUF_SAFE_BYTES = int(1.8 * 2**30)


# ---------------------------------------------------------------------------
# Export wrapper (contract 3.0)
# ---------------------------------------------------------------------------
class AnomalibExportWrapper(ExportWrapper):
    """Wraps ``lightning_model.model`` (PatchcoreModel / EfficientAdModel).

    The inner module keeps its trained semantics untouched — blur included:
    under contract 3.0 the graph emits the FINAL map and FINAL score, so there
    is nothing left for the host to compute. ``ExportWrapper.forward`` prepends
    the in-graph ImageNet normalization and canonicalizes shapes.

    Both PatchcoreModel.forward and EfficientAdModel.forward compute pred_score
    internally (PatchCore from the reweighted nearest-neighbor distance,
    EfficientAD from the quantile-normalized map), so the score is already
    final by construction ("score_semantics=final"). anomalib's external
    PostProcessor (min-max + threshold) is not part of the inner module, so it
    is excluded by construction.
    """

    def __init__(self, inner_model: nn.Module):
        super().__init__()
        self.model = inner_model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def core(self, x: torch.Tensor):
        out = self.model(x)
        # In eval, anomalib inner models return an InferenceBatch dataclass with
        # .anomaly_map and .pred_score; older/ONNX paths may return a tuple/dict.
        return _get_field(out, "anomaly_map", 0), _get_field(out, "pred_score", 1)


def _get_field(out, name, tuple_index):
    """Pull a field from an InferenceBatch / dict / tuple output uniformly."""
    if hasattr(out, name):
        return getattr(out, name)
    if isinstance(out, dict):
        return out[name]
    return out[tuple_index]


# ---------------------------------------------------------------------------
# Probes and trained-state guards
# ---------------------------------------------------------------------------
def _probe_blur(module: nn.Module) -> tuple[int, float]:
    """Legge kernel_size/sigma effettivi dal GaussianBlur2d di anomalib,
    invece di riportare valori ipotizzati nei metadati. Se non c'e' blur
    (caso EfficientAD), restituisce 0/0 — che e' un fatto, non una stima."""
    for child in module.modules():
        if child.__class__.__name__ in ("GaussianBlur2d", "GaussianBlur"):
            ks = getattr(child, "kernel_size", 0)
            ks = ks[0] if isinstance(ks, (tuple, list, torch.Size)) else ks
            sigma = getattr(child, "sigma", 0.0)
            sigma = sigma[0] if isinstance(sigma, (tuple, list)) else sigma
            return int(ks), float(sigma)
    return 0, 0.0


def _detect_architecture(lightning_model) -> str:
    name = lightning_model.__class__.__name__.lower()
    if "efficientad" in name or "efficient_ad" in name:
        return "efficientad"
    if "patchcore" in name:
        return "patchcore"
    raise ValueError(
        f"Unsupported anomalib model {lightning_model.__class__.__name__!r}: "
        "this exporter handles Patchcore and EfficientAd only."
    )


def _assert_trained_state(inner: nn.Module, architecture: str) -> None:
    """Refuse to export a model whose trained buffers are still at init values.

    A PatchCore with an empty memory bank or an EfficientAD with all-zero
    quantiles exports to a *valid* ONNX graph that silently produces garbage;
    the parity check would even pass (both sides compute the same garbage).
    Failing here is the only place the mistake is detectable.
    """
    if architecture == "patchcore":
        bank = getattr(inner, "memory_bank", None)
        if bank is None or not torch.is_tensor(bank) or bank.numel() == 0:
            raise RuntimeError(
                "PatchCore memory bank is empty: the model has not been fitted "
                "(engine.fit populates it at the end of training). Refusing to "
                "export a graph that would compute nearest-neighbors against "
                "nothing."
            )
    elif architecture == "efficientad":
        quantiles = getattr(inner, "quantiles", None)
        if quantiles is None:
            raise RuntimeError(
                "EfficientAdModel has no 'quantiles' attribute: unsupported "
                "anomalib version, cannot verify the model is calibrated."
            )
        # anomalib initializes qa_st/qb_st/qa_ae/qb_ae to 0.0 and its own
        # forward uses `is_set()` (any value != 0) to decide whether to apply
        # the map normalization — the same criterion applies here: all-zero
        # quantiles mean the graph would bake the *uncalibrated* branch.
        if all(float(v.sum()) == 0.0 for v in quantiles.values()):
            raise RuntimeError(
                "EfficientAD quantiles (qa_st/qb_st/qa_ae/qb_ae) are all zero: "
                "the map-normalization statistics were never computed (they are "
                "filled at the end of training). Exporting now would bake the "
                "uncalibrated branch into the graph. Refusing."
            )


def _check_protobuf_size(inner: nn.Module) -> None:
    """PatchCore's memory bank can push a single-file .onnx over protobuf's
    2 GB hard limit, which surfaces as a cryptic serialization error inside
    torch.onnx.export. Check upfront and explain the actual fix."""
    state_bytes = sum(
        t.numel() * t.element_size()
        for t in list(inner.parameters()) + list(inner.buffers())
    )
    if state_bytes > _PROTOBUF_SAFE_BYTES:
        raise RuntimeError(
            f"Model state is {state_bytes / 2**30:.2f} GiB, above the ~2 GB "
            "protobuf limit for a single-file .onnx (external_data=False). "
            "For PatchCore this means the memory bank is too large: lower "
            "coreset_sampling_ratio, or switch the export to external data "
            "(which breaks the single-file deployment contract — coordinate "
            "with the C++ runtime first)."
        )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_pure_onnx(lightning_model, input_size, onnx_path: str,
                     device: str = "cpu"):
    """
    Export a trained anomalib LightningModule (Patchcore / EfficientAd) to a
    contract-3.0 ONNX graph with dynamic batch and uniform I/O names.

    Args:
        lightning_model: the trained anomalib LightningModule (must expose ``.model``).
        input_size: (H, W) the model was trained at.
        onnx_path: destination .onnx path.
        device: "cuda" or "cpu".

    Returns:
        (onnx_path, wrapper) — the wrapper is reused by ``verify_parity``.
    """
    architecture = _detect_architecture(lightning_model)
    inner = getattr(lightning_model, "model", lightning_model)

    _assert_trained_state(inner, architecture)
    _check_protobuf_size(inner)

    blur_kernel_size, blur_sigma = _probe_blur(inner)
    wrapper = AnomalibExportWrapper(inner).to(device).eval()

    h, w = input_size
    metadata = build_metadata(
        architecture=architecture,
        image_size=(h, w),
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
        dynamic_crop=False,
        weights_path=None,
        extra={
            # No checkpoint file exists at this point: the export runs on the
            # in-memory LightningModule straight out of engine.fit. Override
            # build_metadata's "random_self_test" default so the runtime's
            # random-weights guard does not reject a genuinely trained model
            # (the trained-state asserts above are the actual guarantee here).
            "weights_source": f"trained:{lightning_model.__class__.__name__}",
        },
    )

    path = export(wrapper, (h, w), Path(onnx_path), device, metadata)

    print(f"[OK] Contract-3.0 ONNX export: {path}")
    print(f"     input  'image'         : float32 [B,3,{h},{w}] RGB in [0,1] "
          "(normalization is in-graph)")
    print( "     output 'anomaly_map'   : float32 [B,1,H,W] (final, blur in-graph)")
    print( "     output 'anomaly_score' : float32 [B] (final)")
    print(f"     blur probed from model : kernel={blur_kernel_size}, sigma={blur_sigma}")
    return str(path), wrapper


def verify_parity(wrapper: nn.Module, onnx_path: str, input_size,
                  device: str = "cpu", atol: float = 1e-3) -> None:
    """PyTorch<->ONNX parity on [0,1] inputs (batch 1 and 4).

    Thin adapter over export_common.verify, kept for the existing call sites.
    Note this proves the graph matches the wrapper — NOT that the C++
    preprocessing is correct; that is the job of the end-to-end parity harness.
    """
    verify(wrapper, Path(onnx_path), tuple(input_size), device, atol=atol, rtol=atol)
