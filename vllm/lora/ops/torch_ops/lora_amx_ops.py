# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMX-accelerated dispatch for CPU LoRA shrink/expand matmuls.

Mirrors the base model's own AMX dispatch (`check_cpu_sgl_kernel`,
`dispatch_cpu_unquantized_gemm` in `vllm.model_executor.layers.utils`),
scoped specifically to `max_loras == 1`: with exactly one adapter slot,
every valid (non -1) row uses the same weight matrix, which maps directly
onto `weight_packed_linear`'s single-fixed-weight design. With more than
one adapter, different rows can need different weight matrices, which
this path doesn't handle -- callers should fall back to a per-row
gather + einsum in that case.
"""

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.utils import check_cpu_sgl_kernel

logger = init_logger(__name__)

_SUPPORTS_AMX_WEIGHT_PACKED_LINEAR = hasattr(
    torch.ops._C, "weight_packed_linear"
) and hasattr(torch.ops._C, "convert_weight_packed")


def _pack_single_adapter_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.dtype] | None:
    """Pack `weight` (the full lora_a_stacked/lora_b_stacked tensor for a
    single-adapter LoRA layer) for the AMX weight_packed_linear kernel, or
    return None if not eligible.

    Repacks on every call rather than caching: lora_a_stacked/
    lora_b_stacked are mutated in place by set_lora/reset_lora (no
    reassignment), and vLLM's model forward pass runs under
    torch.inference_mode(), where tensors don't carry a usable version
    counter (`weight._version` raises "Inference tensors do not track
    version counter") -- so there's no reliable, cheap signal here to
    detect that the weight changed since a cached pack. The weight being
    packed is small (LoRA rank x hidden_size), so repacking each call is
    expected to be cheap relative to the matmul it enables.

    Returns (packed_weight, compute_dtype) so the caller casts its input
    to the dtype the weight was actually packed in, not the packed
    tensor's own (internal, layout-specific) dtype.
    """
    if not (_SUPPORTS_AMX_WEIGHT_PACKED_LINEAR and envs.VLLM_CPU_SGL_KERNEL):
        return None
    if not torch.cpu._is_amx_tile_supported():
        return None
    if weight.shape[0] != 1:
        return None
    w2d = weight.reshape(weight.shape[-2], weight.shape[-1])
    if not check_cpu_sgl_kernel(w2d.shape[0], w2d.shape[1], w2d.dtype):
        return None

    packed = torch.ops._C.convert_weight_packed(w2d.contiguous())
    return packed, w2d.dtype


def try_amx_linear(
    weight: torch.Tensor,
    valid_inputs: torch.Tensor,
    output_dtype: torch.dtype,
    op_name: str,
) -> torch.Tensor | None:
    """Attempt the AMX-accelerated path for a single-adapter LoRA matmul.

    Returns the computed output tensor, or None if not eligible -- callers
    should fall back to the plain gather + einsum path in that case.
    """
    amx = _pack_single_adapter_weight(weight)
    if amx is None:
        return None
    packed_weight, compute_dtype = amx
    logger.debug_once(
        "CPU LoRA %s dispatch: using sgl-kernel weight_packed_linear", op_name
    )
    amx_inputs = valid_inputs.to(dtype=compute_dtype)
    return torch.ops._C.weight_packed_linear(amx_inputs, packed_weight, None, True).to(
        dtype=output_dtype
    )
