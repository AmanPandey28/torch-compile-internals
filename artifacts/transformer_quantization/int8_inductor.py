# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
from torch._C._dynamo.guards import copy_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream



# kernel path: <inductor-cache>/sh/csh4y26f66qfupveutlg6cwson7uxcevnbjxsudndtdmhknohhs7.py
# Topologically Sorted Source Nodes: [rms_norm, linear], Original ATen: [aten._fused_rms_norm, aten.amin, aten.zeros_like, aten.minimum, aten.neg, aten.amax, aten.maximum, aten.div, aten.clamp, aten.reciprocal, aten.mul, aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   linear => amax, amin, clamp_max, clamp_min, clamp_min_1, convert_element_type_2, convert_element_type_3, convert_element_type_5, convert_element_type_6, convert_element_type_7, div, full_default, full_default_1, maximum, maximum_1, minimum, mul_2, mul_3, neg, reciprocal, round_1
#   rms_norm => add, convert_element_type, convert_element_type_1, mean, mul, mul_1, pow_1, rsqrt
# Graph fragment:
#   %arg1_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %buf0 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=buf0]
#   %arg0_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %amin : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amin]
#   %amax : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amax]
#   %round_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=round_1]
#   %convert_element_type : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg0_1), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %amin : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amin.default](args = (%convert_element_type_1, [2], True), kwargs = {})
#   %full_default : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%amin, %full_default), kwargs = {})
#   %neg : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%minimum,), kwargs = {})
#   %amax : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_1, [2], True), kwargs = {})
#   %full_default_1 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%amax, %full_default_1), kwargs = {})
#   %maximum_1 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%neg, %maximum), kwargs = {})
#   %div : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%maximum_1, 127.5), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float32), kwargs = {})
#   %clamp_min : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_2, 1.1920928955078125e-07), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min, torch.bfloat16), kwargs = {})
#   %reciprocal : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convert_element_type_3,), kwargs = {})
#   %mul_2 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1.0), kwargs = {})
#   %mul_3 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %mul_2), kwargs = {})
#   %round_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.round.default](args = (%mul_3,), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%round_1, torch.float32), kwargs = {})
#   %clamp_min_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_5, -128), kwargs = {})
#   %clamp_max : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 127), kwargs = {})
#   %convert_element_type_6 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.bfloat16), kwargs = {})
#   %convert_element_type_7 : Tensor "i8[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_6, torch.int8), kwargs = {})
#   return %buf0,%amin,%amax,%round_1,%convert_element_type_7
triton_per_fused__fused_rms_norm__to_copy_add_amax_amin_clamp_div_maximum_minimum_mul_neg_reciprocal_zeros_like_0 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy_add_amax_amin_clamp_div_maximum_minimum_mul_neg_reciprocal_zeros_like_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr4': '*i8', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy_add_amax_amin_clamp_div_maximum_minimum_mul_neg_reciprocal_zeros_like_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 3, 'num_reduction': 3, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 4096, 'r0_': 1574400}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy_add_amax_amin_clamp_div_maximum_minimum_mul_neg_reciprocal_zeros_like_0(in_ptr0, in_ptr1, out_ptr1, out_ptr2, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tl.full([1, 1], 768.0, tl.float32)
    tmp8 = (tmp6 / tmp7)
    tmp9 = tl.full([1, 1], 1e-06, tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp1 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, float("inf"))
    tmp20 = triton_helpers.min2(tmp19, 1)[:, None].to(tl.float32)
    tmp22 = tl.where(r0_mask & xmask, tmp17, float("-inf"))
    tmp23 = triton_helpers.max2(tmp22, 1)[:, None].to(tl.float32)
    tmp24 = tl.full([1, 1], 0.0, tl.float32)
    tmp25 = triton_helpers.minimum(tmp20, tmp24)
    tmp26 = -tmp25
    tmp27 = triton_helpers.maximum(tmp23, tmp24)
    tmp28 = triton_helpers.maximum(tmp26, tmp27)
    tmp29 = tl.full([1, 1], 0.00784313725490196, tl.float32)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.full([1, 1], 1.1920928955078125e-07, tl.float32)
    tmp33 = triton_helpers.maximum(tmp31, tmp32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tl.full([1, 1], 1.0, tl.float32)
    tmp36 = (tmp35 / tmp34)
    tmp37 = tmp36 * tmp35
    tmp38 = tmp16 * tmp37
    tmp39 = libdevice.nearbyint(tmp38)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tl.full([1, 1], -128.0, tl.float32)
    tmp42 = triton_helpers.maximum(tmp40, tmp41)
    tmp43 = tl.full([1, 1], 127.0, tl.float32)
    tmp44 = triton_helpers.minimum(tmp42, tmp43)
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp45.to(tl.int8)
    tl.store(out_ptr4 + (r0_1 + 768*x0), tmp46, r0_mask & xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
    tl.store(out_ptr2 + (x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/4v/c4vpihvss63l2bbd5wl3meskvsxbpxpgbj2zdlfqw6r5tw3eqsv7.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   linear => sum_1
# Graph fragment:
#   %arg2_1 : Tensor "i8[2304, 768][768, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %sum_1 : Tensor "i64[2304][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%arg2_1, [-1]), kwargs = {})
#   return %sum_1
triton_per_fused_sum_1 = async_compile.triton('triton_per_fused_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 36864, 'r0_': 1769472}}
)
@triton.jit
def triton_per_fused_sum_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2304
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.int64)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/2z/c2zejs4oplpcafrk65ilxixnhfo3pnowha6vf2hvmmiv25344lwn.py
# Topologically Sorted Source Nodes: [linear, chunk, view, transpose, view_1, transpose_1, view_2, transpose_2, attended], Original ATen: [aten.zeros_like, aten.minimum, aten.neg, aten.maximum, aten.div, aten.clamp, aten.view, aten.expand, aten.mul, aten._to_copy, aten.sub, aten.split, aten.transpose, aten._scaled_dot_product_flash_attention]
# Source node to ATen node mapping:
#   attended => _scaled_dot_product_flash_attention
#   chunk => split
#   linear => clamp_min, convert_element_type_2, convert_element_type_3, convert_element_type_9, div, expand, full_default, full_default_1, full_default_3, maximum, maximum_1, minimum, mul_4, mul_5, mul_6, mul_7, neg, sub, view_10, view_11, view_12, view_8
#   transpose => permute_1
#   transpose_1 => permute_2
#   transpose_2 => permute_3
#   view => view_13
#   view_1 => view_14
#   view_2 => view_15
# Graph fragment:
#   %_int_mm : Tensor "i32[512, 2304][2304, 1]cuda:0" = PlaceHolder[target=_int_mm]
#   %amin : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amin]
#   %amax : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amax]
#   %sum_1 : Tensor "i64[2304][1]cuda:0" = PlaceHolder[target=sum_1]
#   %arg3_1 : Tensor "bf16[2304, 1][1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %full_default : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%amin, %full_default), kwargs = {})
#   %neg : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%minimum,), kwargs = {})
#   %full_default_1 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%amax, %full_default_1), kwargs = {})
#   %maximum_1 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%neg, %maximum), kwargs = {})
#   %div : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%maximum_1, 127.5), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float32), kwargs = {})
#   %clamp_min : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_2, 1.1920928955078125e-07), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min, torch.bfloat16), kwargs = {})
#   %view_8 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_3, [-1, 1]), kwargs = {})
#   %expand : Tensor "bf16[512, 2304][1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_8, [512, 2304]), kwargs = {})
#   %mul_4 : Tensor "bf16[512, 2304][2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_int_mm, %expand), kwargs = {})
#   %full_default_3 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([512, 1], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_10 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_3, [-1, 1]), kwargs = {})
#   %mul_5 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_3, %view_10), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[2304][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_1, torch.bfloat16), kwargs = {})
#   %mul_6 : Tensor "bf16[512, 2304][2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %convert_element_type_9), kwargs = {})
#   %sub : Tensor "bf16[512, 2304][2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_4, %mul_6), kwargs = {})
#   %view_11 : Tensor "bf16[2304][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg3_1, [2304]), kwargs = {})
#   %mul_7 : Tensor "bf16[512, 2304][2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %view_11), kwargs = {})
#   %view_12 : Tensor "bf16[4, 128, 2304][294912, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_7, [4, 128, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_12, 768, -1), kwargs = {})
#   %view_13 : Tensor "bf16[4, 128, 12, 64][294912, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [4, 128, 12, 64]), kwargs = {})
#   %permute_1 : Tensor "bf16[4, 12, 128, 64][294912, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_13, [0, 2, 1, 3]), kwargs = {})
#   %view_14 : Tensor "bf16[4, 128, 12, 64][294912, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [4, 128, 12, 64]), kwargs = {})
#   %permute_2 : Tensor "bf16[4, 12, 128, 64][294912, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_14, [0, 2, 1, 3]), kwargs = {})
#   %view_15 : Tensor "bf16[4, 128, 12, 64][294912, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [4, 128, 12, 64]), kwargs = {})
#   %permute_3 : Tensor "bf16[4, 12, 128, 64][294912, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_15, [0, 2, 1, 3]), kwargs = {})
#   %_scaled_dot_product_flash_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention.default](args = (%permute_1, %permute_2, %permute_3, 0.0, True), kwargs = {scale: 0.125})
#   return %buf7,%buf8,%buf9
triton_poi_fused__scaled_dot_product_flash_attention__to_copy_clamp_div_expand_maximum_minimum_mul_neg_split_sub_transpose_view_zeros_like_2 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention__to_copy_clamp_div_expand_maximum_minimum_mul_neg_split_sub_transpose_view_zeros_like_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*i64', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention__to_copy_clamp_div_expand_maximum_minimum_mul_neg_split_sub_transpose_view_zeros_like_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 11, 'num_store': 3, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 9460224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention__to_copy_clamp_div_expand_maximum_minimum_mul_neg_split_sub_transpose_view_zeros_like_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 768)
    x1 = xindex // 768
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2304*x1), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp23 = tl.load(in_ptr0 + (768 + x0 + 2304*x1), None)
    tmp26 = tl.load(in_ptr3 + (768 + x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (768 + x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp32 = tl.load(in_ptr0 + (1536 + x0 + 2304*x1), None)
    tmp35 = tl.load(in_ptr3 + (1536 + x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (1536 + x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1], 0.0, tl.float32)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = -tmp4
    tmp7 = triton_helpers.maximum(tmp6, tmp3)
    tmp8 = triton_helpers.maximum(tmp5, tmp7)
    tmp9 = tl.full([1], 0.00784313725490196, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.full([1], 1.1920928955078125e-07, tl.float32)
    tmp13 = triton_helpers.maximum(tmp11, tmp12)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp1 * tmp14
    tmp16 = tmp3 * tmp14
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 * tmp18
    tmp20 = tmp15 - tmp19
    tmp22 = tmp20 * tmp21
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24 * tmp14
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp16 * tmp27
    tmp29 = tmp25 - tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33 * tmp14
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp16 * tmp36
    tmp38 = tmp34 - tmp37
    tmp40 = tmp38 * tmp39
    tl.store(out_ptr0 + (x2), tmp22, None)
    tl.store(out_ptr1 + (x2), tmp31, None)
    tl.store(out_ptr2 + (x2), tmp40, None)
''', device_str='cuda')


# kernel path: <inductor-cache>/wb/cwbtogyw4dlrffgwxvhju7n4lbo4ufumijrkrugyf33ukeezrg5j.py
# Topologically Sorted Source Nodes: [transpose_3, merged, linear_1], Original ATen: [aten.transpose, aten.view, aten.amin, aten.zeros_like, aten.minimum, aten.neg, aten.amax, aten.maximum, aten.div, aten.clamp, aten.reciprocal, aten.mul, aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   linear_1 => amax_1, amin_1, clamp_max_1, clamp_min_2, clamp_min_3, convert_element_type_10, convert_element_type_11, convert_element_type_13, convert_element_type_14, convert_element_type_15, div_1, full_default_4, full_default_5, maximum_2, maximum_3, minimum_1, mul_8, mul_9, neg_1, reciprocal_1, round_2
#   merged => view_16
#   transpose_3 => permute_4
# Graph fragment:
#   %getitem_3 : Tensor "bf16[4, 12, 128, 64][98304, 64, 768, 1]cuda:0" = PlaceHolder[target=getitem_3]
#   %amin_1 : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amin_1]
#   %amax_1 : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amax_1]
#   %permute_4 : Tensor "bf16[4, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_3, [0, 2, 1, 3]), kwargs = {})
#   %view_16 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_4, [4, 128, 768]), kwargs = {})
#   %amin_1 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amin.default](args = (%view_16, [2], True), kwargs = {})
#   %full_default_4 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum_1 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%amin_1, %full_default_4), kwargs = {})
#   %neg_1 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%minimum_1,), kwargs = {})
#   %amax_1 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_16, [2], True), kwargs = {})
#   %full_default_5 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum_2 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%amax_1, %full_default_5), kwargs = {})
#   %maximum_3 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%neg_1, %maximum_2), kwargs = {})
#   %div_1 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%maximum_3, 127.5), kwargs = {})
#   %convert_element_type_10 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.float32), kwargs = {})
#   %clamp_min_2 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_10, 1.1920928955078125e-07), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_2, torch.bfloat16), kwargs = {})
#   %reciprocal_1 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convert_element_type_11,), kwargs = {})
#   %mul_8 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1.0), kwargs = {})
#   %mul_9 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_16, %mul_8), kwargs = {})
#   %round_2 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.round.default](args = (%mul_9,), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%round_2, torch.float32), kwargs = {})
#   %clamp_min_3 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_13, -128), kwargs = {})
#   %clamp_max_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 127), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_15 : Tensor "i8[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_14, torch.int8), kwargs = {})
#   return %amin_1,%amax_1,%convert_element_type_15
triton_per_fused__to_copy_add_amax_amin_clamp_div_maximum_minimum_mul_neg_reciprocal_transpose_view_zeros_like_3 = async_compile.triton('triton_per_fused__to_copy_add_amax_amin_clamp_div_maximum_minimum_mul_neg_reciprocal_transpose_view_zeros_like_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*i8', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_amax_amin_clamp_div_maximum_minimum_mul_neg_reciprocal_transpose_view_zeros_like_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 3, 'num_reduction': 2, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 4096, 'r0_': 1572864}}
)
@triton.jit
def triton_per_fused__to_copy_add_amax_amin_clamp_div_maximum_minimum_mul_neg_reciprocal_transpose_view_zeros_like_3(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, float("inf"))
    tmp4 = triton_helpers.min2(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = tl.where(r0_mask & xmask, tmp1, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 0.0, tl.float32)
    tmp9 = triton_helpers.minimum(tmp4, tmp8)
    tmp10 = -tmp9
    tmp11 = triton_helpers.maximum(tmp7, tmp8)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = tl.full([1, 1], 0.00784313725490196, tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1, 1], 1.1920928955078125e-07, tl.float32)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.full([1, 1], 1.0, tl.float32)
    tmp20 = (tmp19 / tmp18)
    tmp21 = tmp20 * tmp19
    tmp22 = tmp0 * tmp21
    tmp23 = libdevice.nearbyint(tmp22)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tl.full([1, 1], -128.0, tl.float32)
    tmp26 = triton_helpers.maximum(tmp24, tmp25)
    tmp27 = tl.full([1, 1], 127.0, tl.float32)
    tmp28 = triton_helpers.minimum(tmp26, tmp27)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29.to(tl.int8)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp30, r0_mask & xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/l2/cl2bdpjkn6z2ihiyuavvuo6dahlljljrxflxkkkrz7qvz3ympxe7.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   linear_1 => sum_2
# Graph fragment:
#   %arg5_1 : Tensor "i8[768, 768][768, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %sum_2 : Tensor "i64[768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%arg5_1, [-1]), kwargs = {})
#   return %sum_2
triton_per_fused_sum_4 = async_compile.triton('triton_per_fused_sum_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 12288, 'r0_': 589824}}
)
@triton.jit
def triton_per_fused_sum_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 768
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.int64)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/j5/cj5zpzpbhexus53a7ly5fiwqob7vajvbxafqqf2my3pe3utcitjb.py
# Topologically Sorted Source Nodes: [linear_1, hidden, rms_norm_1, linear_2], Original ATen: [aten.zeros_like, aten.minimum, aten.neg, aten.maximum, aten.div, aten.clamp, aten.view, aten.expand, aten.mul, aten._to_copy, aten.sub, aten.add, aten._fused_rms_norm, aten.amin, aten.amax, aten.reciprocal]
# Source node to ATen node mapping:
#   hidden => add_3
#   linear_1 => clamp_min_2, convert_element_type_10, convert_element_type_11, convert_element_type_17, div_1, expand_1, full_default_4, full_default_5, full_default_7, maximum_2, maximum_3, minimum_1, mul_10, mul_11, mul_12, mul_13, neg_1, sub_1, view_25, view_27, view_28, view_29
#   linear_2 => amax_2, amin_2, clamp_max_2, clamp_min_4, clamp_min_5, convert_element_type_20, convert_element_type_21, convert_element_type_23, convert_element_type_24, convert_element_type_25, div_2, full_default_8, full_default_9, maximum_4, maximum_5, minimum_2, mul_16, mul_17, neg_2, reciprocal_2, round_3
#   rms_norm_1 => add_4, convert_element_type_18, convert_element_type_19, mean_1, mul_14, mul_15, pow_2, rsqrt_1
# Graph fragment:
#   %arg1_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %_int_mm_1 : Tensor "i32[512, 768][768, 1]cuda:0" = PlaceHolder[target=_int_mm_1]
#   %amin_1 : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amin_1]
#   %amax_1 : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amax_1]
#   %sum_2 : Tensor "i64[768][1]cuda:0" = PlaceHolder[target=sum_2]
#   %arg6_1 : Tensor "bf16[768, 1][1, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %add_3 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_3]
#   %buf22 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=buf22]
#   %arg8_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %amin_2 : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amin_2]
#   %amax_2 : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amax_2]
#   %round_3 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=round_3]
#   %full_default_4 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum_1 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%amin_1, %full_default_4), kwargs = {})
#   %neg_1 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%minimum_1,), kwargs = {})
#   %full_default_5 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum_2 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%amax_1, %full_default_5), kwargs = {})
#   %maximum_3 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%neg_1, %maximum_2), kwargs = {})
#   %div_1 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%maximum_3, 127.5), kwargs = {})
#   %convert_element_type_10 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.float32), kwargs = {})
#   %clamp_min_2 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_10, 1.1920928955078125e-07), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_2, torch.bfloat16), kwargs = {})
#   %view_25 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_11, [-1, 1]), kwargs = {})
#   %expand_1 : Tensor "bf16[512, 768][1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_25, [512, 768]), kwargs = {})
#   %mul_10 : Tensor "bf16[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_int_mm_1, %expand_1), kwargs = {})
#   %full_default_7 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([512, 1], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_27 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_11, [-1, 1]), kwargs = {})
#   %mul_11 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_7, %view_27), kwargs = {})
#   %convert_element_type_17 : Tensor "bf16[768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.bfloat16), kwargs = {})
#   %mul_12 : Tensor "bf16[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %convert_element_type_17), kwargs = {})
#   %sub_1 : Tensor "bf16[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_10, %mul_12), kwargs = {})
#   %view_28 : Tensor "bf16[768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg6_1, [768]), kwargs = {})
#   %mul_13 : Tensor "bf16[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %view_28), kwargs = {})
#   %view_29 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_13, [4, 128, 768]), kwargs = {})
#   %add_3 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %view_29), kwargs = {})
#   %convert_element_type_18 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_18, 2), kwargs = {})
#   %mean_1 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_4 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_14 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_18, %rsqrt_1), kwargs = {})
#   %mul_15 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %arg8_1), kwargs = {})
#   %convert_element_type_19 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
#   %amin_2 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amin.default](args = (%convert_element_type_19, [2], True), kwargs = {})
#   %full_default_8 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum_2 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%amin_2, %full_default_8), kwargs = {})
#   %neg_2 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%minimum_2,), kwargs = {})
#   %amax_2 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_19, [2], True), kwargs = {})
#   %full_default_9 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum_4 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%amax_2, %full_default_9), kwargs = {})
#   %maximum_5 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%neg_2, %maximum_4), kwargs = {})
#   %div_2 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%maximum_5, 127.5), kwargs = {})
#   %convert_element_type_20 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.float32), kwargs = {})
#   %clamp_min_4 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_20, 1.1920928955078125e-07), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_4, torch.bfloat16), kwargs = {})
#   %reciprocal_2 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convert_element_type_21,), kwargs = {})
#   %mul_16 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1.0), kwargs = {})
#   %mul_17 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %mul_16), kwargs = {})
#   %round_3 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.round.default](args = (%mul_17,), kwargs = {})
#   %convert_element_type_23 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%round_3, torch.float32), kwargs = {})
#   %clamp_min_5 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_23, -128), kwargs = {})
#   %clamp_max_2 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_5, 127), kwargs = {})
#   %convert_element_type_24 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_2, torch.bfloat16), kwargs = {})
#   %convert_element_type_25 : Tensor "i8[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_24, torch.int8), kwargs = {})
#   return %add_3,%buf22,%amin_2,%amax_2,%round_3,%convert_element_type_25
triton_per_fused__fused_rms_norm__to_copy_add_amax_amin_clamp_div_expand_maximum_minimum_mul_neg_reciprocal_sub_view_zeros_like_5 = async_compile.triton('triton_per_fused__fused_rms_norm__to_copy_add_amax_amin_clamp_div_expand_maximum_minimum_mul_neg_reciprocal_sub_view_zeros_like_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*i64', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'out_ptr0': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'out_ptr5': '*i8', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__to_copy_add_amax_amin_clamp_div_expand_maximum_minimum_mul_neg_reciprocal_sub_view_zeros_like_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 4, 'num_reduction': 3, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 6144, 'r0_': 4727808}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__to_copy_add_amax_amin_clamp_div_expand_maximum_minimum_mul_neg_reciprocal_sub_view_zeros_like_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr2, out_ptr3, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tl.full([1, 1], 0.0, tl.float32)
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = -tmp5
    tmp8 = triton_helpers.maximum(tmp7, tmp4)
    tmp9 = triton_helpers.maximum(tmp6, tmp8)
    tmp10 = tl.full([1, 1], 0.00784313725490196, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.full([1, 1], 1.1920928955078125e-07, tl.float32)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp2 * tmp15
    tmp17 = tmp4 * tmp15
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 * tmp19
    tmp21 = tmp16 - tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tmp0 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
    tmp29 = tl.where(r0_mask & xmask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None].to(tl.float32)
    tmp31 = tl.full([1, 1], 768.0, tl.float32)
    tmp32 = (tmp30 / tmp31)
    tmp33 = tl.full([1, 1], 1e-06, tl.float32)
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp25 * tmp35
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp36 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
    tmp43 = tl.where(r0_mask & xmask, tmp41, float("inf"))
    tmp44 = triton_helpers.min2(tmp43, 1)[:, None].to(tl.float32)
    tmp46 = tl.where(r0_mask & xmask, tmp41, float("-inf"))
    tmp47 = triton_helpers.max2(tmp46, 1)[:, None].to(tl.float32)
    tmp48 = triton_helpers.minimum(tmp44, tmp4)
    tmp49 = -tmp48
    tmp50 = triton_helpers.maximum(tmp47, tmp4)
    tmp51 = triton_helpers.maximum(tmp49, tmp50)
    tmp52 = tmp51 * tmp10
    tmp53 = tmp52.to(tl.float32)
    tmp54 = triton_helpers.maximum(tmp53, tmp13)
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tl.full([1, 1], 1.0, tl.float32)
    tmp57 = (tmp56 / tmp55)
    tmp58 = tmp57 * tmp56
    tmp59 = tmp40 * tmp58
    tmp60 = libdevice.nearbyint(tmp59)
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tl.full([1, 1], -128.0, tl.float32)
    tmp63 = triton_helpers.maximum(tmp61, tmp62)
    tmp64 = tl.full([1, 1], 127.0, tl.float32)
    tmp65 = triton_helpers.minimum(tmp63, tmp64)
    tmp66 = tmp65.to(tl.float32)
    tmp67 = tmp66.to(tl.int8)
    tl.store(out_ptr0 + (r0_1 + 768*x0), tmp24, r0_mask & xmask)
    tl.store(out_ptr5 + (r0_1 + 768*x0), tmp67, r0_mask & xmask)
    tl.store(out_ptr2 + (x0), tmp44, xmask)
    tl.store(out_ptr3 + (x0), tmp47, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/m2/cm2c5b4ve5svjwqjif74na3nwuomxraryyskn4lmku7s4tyzeiua.py
# Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   linear_2 => sum_3
# Graph fragment:
#   %arg9_1 : Tensor "i8[4096, 768][768, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %sum_3 : Tensor "i64[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%arg9_1, [-1]), kwargs = {})
#   return %sum_3
triton_per_fused_sum_6 = async_compile.triton('triton_per_fused_sum_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 65536, 'r0_': 3145728}}
)
@triton.jit
def triton_per_fused_sum_6(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None].to(tl.int64)
    tl.store(out_ptr0 + (x0), tmp5, None)
''', device_str='cuda')


# kernel path: <inductor-cache>/sf/csfh4zy2fiezkzz2dyeqkuyduwcllxzxjrkaazkgsqa7x2mhr3wm.py
# Topologically Sorted Source Nodes: [linear_2, chunk_1, silu, mul, linear_3], Original ATen: [aten.zeros_like, aten.minimum, aten.neg, aten.maximum, aten.div, aten.clamp, aten.view, aten.expand, aten.mul, aten._to_copy, aten.sub, aten.split, aten.silu, aten.amin, aten.amax, aten.reciprocal, aten.add]
# Source node to ATen node mapping:
#   chunk_1 => split_1
#   linear_2 => clamp_min_4, convert_element_type_20, convert_element_type_21, convert_element_type_27, div_2, expand_2, full_default_11, full_default_8, full_default_9, maximum_4, maximum_5, minimum_2, mul_18, mul_19, mul_20, mul_21, neg_2, sub_2, view_38, view_40, view_41, view_42
#   linear_3 => amax_3, amin_3, clamp_max_3, clamp_min_6, clamp_min_7, convert_element_type_30, convert_element_type_31, convert_element_type_33, convert_element_type_34, convert_element_type_35, div_4, full_default_12, full_default_13, maximum_6, maximum_7, minimum_3, mul_23, mul_24, neg_4, reciprocal_3, round_4
#   mul => mul_22
#   silu => add_6, convert_element_type_28, convert_element_type_29, div_3, exp, neg_3
# Graph fragment:
#   %_int_mm_2 : Tensor "i32[512, 4096][4096, 1]cuda:0" = PlaceHolder[target=_int_mm_2]
#   %amin_2 : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amin_2]
#   %amax_2 : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amax_2]
#   %sum_3 : Tensor "i64[4096][1]cuda:0" = PlaceHolder[target=sum_3]
#   %arg10_1 : Tensor "bf16[4096, 1][1, 1]cuda:0" = PlaceHolder[target=arg10_1]
#   %convert_element_type_28 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0" = PlaceHolder[target=convert_element_type_28]
#   %mul_22 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0" = PlaceHolder[target=mul_22]
#   %amin_3 : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amin_3]
#   %amax_3 : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amax_3]
#   %full_default_8 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum_2 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%amin_2, %full_default_8), kwargs = {})
#   %neg_2 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%minimum_2,), kwargs = {})
#   %full_default_9 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum_4 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%amax_2, %full_default_9), kwargs = {})
#   %maximum_5 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%neg_2, %maximum_4), kwargs = {})
#   %div_2 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%maximum_5, 127.5), kwargs = {})
#   %convert_element_type_20 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.float32), kwargs = {})
#   %clamp_min_4 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_20, 1.1920928955078125e-07), kwargs = {})
#   %convert_element_type_21 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_4, torch.bfloat16), kwargs = {})
#   %view_38 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_21, [-1, 1]), kwargs = {})
#   %expand_2 : Tensor "bf16[512, 4096][1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_38, [512, 4096]), kwargs = {})
#   %mul_18 : Tensor "bf16[512, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_int_mm_2, %expand_2), kwargs = {})
#   %full_default_11 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([512, 1], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_40 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_21, [-1, 1]), kwargs = {})
#   %mul_19 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_11, %view_40), kwargs = {})
#   %convert_element_type_27 : Tensor "bf16[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_3, torch.bfloat16), kwargs = {})
#   %mul_20 : Tensor "bf16[512, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %convert_element_type_27), kwargs = {})
#   %sub_2 : Tensor "bf16[512, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_18, %mul_20), kwargs = {})
#   %view_41 : Tensor "bf16[4096][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg10_1, [4096]), kwargs = {})
#   %mul_21 : Tensor "bf16[512, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %view_41), kwargs = {})
#   %view_42 : Tensor "bf16[4, 128, 4096][524288, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_21, [4, 128, 4096]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_42, 2048, -1), kwargs = {})
#   %convert_element_type_28 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_12, torch.float32), kwargs = {})
#   %neg_3 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_28,), kwargs = {})
#   %exp : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_3,), kwargs = {})
#   %add_6 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div_3 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_28, %add_6), kwargs = {})
#   %convert_element_type_29 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_3, torch.bfloat16), kwargs = {})
#   %mul_22 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_29, %getitem_13), kwargs = {})
#   %amin_3 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amin.default](args = (%mul_22, [2], True), kwargs = {})
#   %full_default_12 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum_3 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%amin_3, %full_default_12), kwargs = {})
#   %neg_4 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%minimum_3,), kwargs = {})
#   %amax_3 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_22, [2], True), kwargs = {})
#   %full_default_13 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum_6 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%amax_3, %full_default_13), kwargs = {})
#   %maximum_7 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%neg_4, %maximum_6), kwargs = {})
#   %div_4 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%maximum_7, 127.5), kwargs = {})
#   %convert_element_type_30 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_4, torch.float32), kwargs = {})
#   %clamp_min_6 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_30, 1.1920928955078125e-07), kwargs = {})
#   %convert_element_type_31 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_6, torch.bfloat16), kwargs = {})
#   %reciprocal_3 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%convert_element_type_31,), kwargs = {})
#   %mul_23 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_3, 1.0), kwargs = {})
#   %mul_24 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %mul_23), kwargs = {})
#   %round_4 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.round.default](args = (%mul_24,), kwargs = {})
#   %convert_element_type_33 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%round_4, torch.float32), kwargs = {})
#   %clamp_min_7 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_33, -128), kwargs = {})
#   %clamp_max_3 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 127), kwargs = {})
#   %convert_element_type_34 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_3, torch.bfloat16), kwargs = {})
#   %convert_element_type_35 : Tensor "i8[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_34, torch.int8), kwargs = {})
#   return %convert_element_type_28,%mul_22,%amin_3,%amax_3,%convert_element_type_35
triton_red_fused__to_copy_add_amax_amin_clamp_div_expand_maximum_minimum_mul_neg_reciprocal_silu_split_sub_view_zeros_like_7 = async_compile.triton('triton_red_fused__to_copy_add_amax_amin_clamp_div_expand_maximum_minimum_mul_neg_reciprocal_silu_split_sub_view_zeros_like_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*i64', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'out_ptr4': '*i8', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_amax_amin_clamp_div_expand_maximum_minimum_mul_neg_reciprocal_silu_split_sub_view_zeros_like_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 4, 'num_reduction': 2, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 6144, 'r0_': 10526720}}
)
@triton.jit
def triton_red_fused__to_copy_add_amax_amin_clamp_div_expand_maximum_minimum_mul_neg_reciprocal_silu_split_sub_view_zeros_like_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    _tmp41 = tl.full([XBLOCK, R0_BLOCK], float("inf"), tl.float32)
    _tmp43 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr0 + (2048 + r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp33 = tl.load(in_ptr3 + (2048 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.load(in_ptr4 + (2048 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tl.full([1, 1], 0.0, tl.float32)
        tmp4 = triton_helpers.minimum(tmp2, tmp3)
        tmp5 = -tmp4
        tmp7 = triton_helpers.maximum(tmp6, tmp3)
        tmp8 = triton_helpers.maximum(tmp5, tmp7)
        tmp9 = tl.full([1, 1], 0.00784313725490196, tl.float32)
        tmp10 = tmp8 * tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.full([1, 1], 1.1920928955078125e-07, tl.float32)
        tmp13 = triton_helpers.maximum(tmp11, tmp12)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp1 * tmp14
        tmp16 = tmp3 * tmp14
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 * tmp18
        tmp20 = tmp15 - tmp19
        tmp22 = tmp20 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = -tmp23
        tmp25 = libdevice.exp(tmp24)
        tmp26 = tl.full([1, 1], 1.0, tl.float32)
        tmp27 = tmp25 + tmp26
        tmp28 = (tmp23 / tmp27)
        tmp29 = tmp28.to(tl.float32)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp31 * tmp14
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp16 * tmp34
        tmp36 = tmp32 - tmp35
        tmp38 = tmp36 * tmp37
        tmp39 = tmp29 * tmp38
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, R0_BLOCK])
        tmp42 = triton_helpers.minimum(_tmp41, tmp40)
        _tmp41 = tl.where(r0_mask & xmask, tmp42, _tmp41)
        tmp44 = triton_helpers.maximum(_tmp43, tmp40)
        _tmp43 = tl.where(r0_mask & xmask, tmp44, _tmp43)
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp39, r0_mask & xmask)
    tmp41 = triton_helpers.min2(_tmp41, 1)[:, None]
    tmp43 = triton_helpers.max2(_tmp43, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp41, xmask)
    tl.store(out_ptr3 + (x0), tmp43, xmask)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp45 = tl.load(out_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp46 = tl.full([1, 1], 0.0, tl.float32)
        tmp47 = triton_helpers.minimum(tmp41, tmp46)
        tmp48 = -tmp47
        tmp49 = triton_helpers.maximum(tmp43, tmp46)
        tmp50 = triton_helpers.maximum(tmp48, tmp49)
        tmp51 = tl.full([1, 1], 0.00784313725490196, tl.float32)
        tmp52 = tmp50 * tmp51
        tmp53 = tmp52.to(tl.float32)
        tmp54 = tl.full([1, 1], 1.1920928955078125e-07, tl.float32)
        tmp55 = triton_helpers.maximum(tmp53, tmp54)
        tmp56 = tmp55.to(tl.float32)
        tmp57 = tl.full([1, 1], 1.0, tl.float32)
        tmp58 = (tmp57 / tmp56)
        tmp59 = tmp58 * tmp57
        tmp60 = tmp45 * tmp59
        tmp61 = libdevice.nearbyint(tmp60)
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tl.full([1, 1], -128.0, tl.float32)
        tmp64 = triton_helpers.maximum(tmp62, tmp63)
        tmp65 = tl.full([1, 1], 127.0, tl.float32)
        tmp66 = triton_helpers.minimum(tmp64, tmp65)
        tmp67 = tmp66.to(tl.float32)
        tmp68 = tmp67.to(tl.int8)
        tl.store(out_ptr4 + (r0_1 + 2048*x0), tmp68, r0_mask & xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/os/cosh7tzeqbr3s5ffbhupougdnlcb2tqvhzpmvhyp6xlx55mw4ypb.py
# Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   linear_3 => sum_4
# Graph fragment:
#   %arg12_1 : Tensor "i8[768, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg12_1]
#   %sum_4 : Tensor "i64[768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%arg12_1, [-1]), kwargs = {})
#   return %sum_4
triton_red_fused_sum_8 = async_compile.triton('triton_red_fused_sum_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i8', 'out_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 12288, 'r0_': 1572864}}
)
@triton.jit
def triton_red_fused_sum_8(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 768
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.int64)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/7j/c7jjn4ny475elapocdfxituilm45d63wgtab7mwbbj4dh6afyvbl.py
# Topologically Sorted Source Nodes: [linear_3, add_1], Original ATen: [aten.zeros_like, aten.minimum, aten.neg, aten.maximum, aten.div, aten.clamp, aten.view, aten.expand, aten.mul, aten._to_copy, aten.sub, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_8
#   linear_3 => clamp_min_6, convert_element_type_30, convert_element_type_31, convert_element_type_37, div_4, expand_3, full_default_12, full_default_13, full_default_15, maximum_6, maximum_7, minimum_3, mul_25, mul_26, mul_27, mul_28, neg_4, sub_3, view_51, view_53, view_54, view_55
# Graph fragment:
#   %add_3 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_3]
#   %_int_mm_3 : Tensor "i32[512, 768][768, 1]cuda:0" = PlaceHolder[target=_int_mm_3]
#   %amin_3 : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amin_3]
#   %amax_3 : Tensor "bf16[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=amax_3]
#   %sum_4 : Tensor "i64[768][1]cuda:0" = PlaceHolder[target=sum_4]
#   %arg13_1 : Tensor "bf16[768, 1][1, 1]cuda:0" = PlaceHolder[target=arg13_1]
#   %full_default_12 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum_3 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%amin_3, %full_default_12), kwargs = {})
#   %neg_4 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%minimum_3,), kwargs = {})
#   %full_default_13 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 128, 1], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %maximum_6 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%amax_3, %full_default_13), kwargs = {})
#   %maximum_7 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%neg_4, %maximum_6), kwargs = {})
#   %div_4 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%maximum_7, 127.5), kwargs = {})
#   %convert_element_type_30 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_4, torch.float32), kwargs = {})
#   %clamp_min_6 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_30, 1.1920928955078125e-07), kwargs = {})
#   %convert_element_type_31 : Tensor "bf16[4, 128, 1][128, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_6, torch.bfloat16), kwargs = {})
#   %view_51 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_31, [-1, 1]), kwargs = {})
#   %expand_3 : Tensor "bf16[512, 768][1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_51, [512, 768]), kwargs = {})
#   %mul_25 : Tensor "bf16[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_int_mm_3, %expand_3), kwargs = {})
#   %full_default_15 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([512, 1], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_53 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_31, [-1, 1]), kwargs = {})
#   %mul_26 : Tensor "bf16[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default_15, %view_53), kwargs = {})
#   %convert_element_type_37 : Tensor "bf16[768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_4, torch.bfloat16), kwargs = {})
#   %mul_27 : Tensor "bf16[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %convert_element_type_37), kwargs = {})
#   %sub_3 : Tensor "bf16[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_25, %mul_27), kwargs = {})
#   %view_54 : Tensor "bf16[768][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg13_1, [768]), kwargs = {})
#   %mul_28 : Tensor "bf16[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %view_54), kwargs = {})
#   %view_55 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_28, [4, 128, 768]), kwargs = {})
#   %add_8 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_55), kwargs = {})
#   return %add_8
triton_poi_fused__to_copy_add_clamp_div_expand_maximum_minimum_mul_neg_sub_view_zeros_like_9 = async_compile.triton('triton_poi_fused__to_copy_add_clamp_div_expand_maximum_minimum_mul_neg_sub_view_zeros_like_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*i64', 'in_ptr4': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_div_expand_maximum_minimum_mul_neg_sub_view_zeros_like_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 3939840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clamp_div_expand_maximum_minimum_mul_neg_sub_view_zeros_like_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 768
    x0 = (xindex % 768)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tl.full([1], 0.0, tl.float32)
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = -tmp5
    tmp8 = triton_helpers.maximum(tmp7, tmp4)
    tmp9 = triton_helpers.maximum(tmp6, tmp8)
    tmp10 = tl.full([1], 0.00784313725490196, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.full([1], 1.1920928955078125e-07, tl.float32)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp2 * tmp15
    tmp17 = tmp4 * tmp15
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 * tmp19
    tmp21 = tmp16 - tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tmp0 + tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')

def partition_0(args):
    arg1_1, arg0_1, arg2_1, arg3_1, arg5_1, arg6_1, arg8_1, arg9_1, arg10_1, arg12_1, arg13_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_misaligned(arg1_1)
        buf1 = empty_strided_cuda((4, 128, 1), (128, 1, 512), torch.bfloat16)
        buf2 = empty_strided_cuda((4, 128, 1), (128, 1, 512), torch.bfloat16)
        buf4 = empty_strided_cuda((4, 128, 768), (98304, 768, 1), torch.int8)
        # Topologically Sorted Source Nodes: [rms_norm, linear], Original ATen: [aten._fused_rms_norm, aten.amin, aten.zeros_like, aten.minimum, aten.neg, aten.amax, aten.maximum, aten.div, aten.clamp, aten.reciprocal, aten.mul, aten.add, aten._to_copy]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused__fused_rms_norm__to_copy_add_amax_amin_clamp_div_maximum_minimum_mul_neg_reciprocal_zeros_like_0.run(arg1_1, arg0_1, buf1, buf2, buf4, 512, 768, stream=raw_stream0)
        del arg0_1
        buf5 = empty_strided_cuda((512, 2304), (2304, 1), torch.int32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.clamp, aten._to_copy, aten.view, aten.t, aten._int_mm]
        extern_kernels._int_mm(reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg2_1, (768, 2304), (1, 768), 0), out=buf5)
        buf6 = empty_strided_cuda((2304, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.sum]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_sum_1.run(arg2_1, buf6, 2304, 768, stream=raw_stream0)
        del arg2_1
        buf7 = empty_strided_cuda((4, 12, 128, 64), (98304, 64, 768, 1), torch.bfloat16)
        buf8 = empty_strided_cuda((4, 12, 128, 64), (98304, 64, 768, 1), torch.bfloat16)
        buf9 = empty_strided_cuda((4, 12, 128, 64), (98304, 64, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear, chunk, view, transpose, view_1, transpose_1, view_2, transpose_2, attended], Original ATen: [aten.zeros_like, aten.minimum, aten.neg, aten.maximum, aten.div, aten.clamp, aten.view, aten.expand, aten.mul, aten._to_copy, aten.sub, aten.split, aten.transpose, aten._scaled_dot_product_flash_attention]
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_flash_attention__to_copy_clamp_div_expand_maximum_minimum_mul_neg_split_sub_transpose_view_zeros_like_2.run(buf5, buf1, buf2, buf6, arg3_1, buf7, buf8, buf9, 393216, stream=raw_stream0)
        del arg3_1
        del buf5
        del buf6
        # Topologically Sorted Source Nodes: [linear, chunk, view, transpose, view_1, transpose_1, view_2, transpose_2, attended], Original ATen: [aten.zeros_like, aten.minimum, aten.neg, aten.maximum, aten.div, aten.clamp, aten.view, aten.expand, aten.mul, aten._to_copy, aten.sub, aten.split, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf10 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf7, buf8, buf9, 0.0, True, scale=0.125)
        del buf7
        del buf8
        del buf9
        buf11 = buf10[0]
        assert_size_stride(buf11, (4, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf11, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf10
        buf16 = buf2; del buf2  # reuse
        buf17 = buf1; del buf1  # reuse
        buf18 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [transpose_3, merged, linear_1], Original ATen: [aten.transpose, aten.view, aten.amin, aten.zeros_like, aten.minimum, aten.neg, aten.amax, aten.maximum, aten.div, aten.clamp, aten.reciprocal, aten.mul, aten.add, aten._to_copy]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_amax_amin_clamp_div_maximum_minimum_mul_neg_reciprocal_transpose_view_zeros_like_3.run(buf11, buf16, buf17, buf18, 512, 768, stream=raw_stream0)
        buf19 = empty_strided_cuda((512, 768), (768, 1), torch.int32)
        # Topologically Sorted Source Nodes: [transpose_3, merged, linear_1], Original ATen: [aten.transpose, aten.view, aten.zeros_like, aten.minimum, aten.neg, aten.maximum, aten.div, aten.clamp, aten.reciprocal, aten.mul, aten.add, aten._to_copy, aten.t, aten._int_mm]
        extern_kernels._int_mm(reinterpret_tensor(buf18, (512, 768), (768, 1), 0), reinterpret_tensor(arg5_1, (768, 768), (1, 768), 0), out=buf19)
        buf20 = empty_strided_cuda((768, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.sum]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_sum_4.run(arg5_1, buf20, 768, 768, stream=raw_stream0)
        del arg5_1
        buf21 = reinterpret_tensor(buf11, (4, 128, 768), (98304, 768, 1), 0); del buf11  # reuse
        buf23 = empty_strided_cuda((4, 128, 1), (128, 1, 512), torch.bfloat16)
        buf24 = empty_strided_cuda((4, 128, 1), (128, 1, 512), torch.bfloat16)
        buf26 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [linear_1, hidden, rms_norm_1, linear_2], Original ATen: [aten.zeros_like, aten.minimum, aten.neg, aten.maximum, aten.div, aten.clamp, aten.view, aten.expand, aten.mul, aten._to_copy, aten.sub, aten.add, aten._fused_rms_norm, aten.amin, aten.amax, aten.reciprocal]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused__fused_rms_norm__to_copy_add_amax_amin_clamp_div_expand_maximum_minimum_mul_neg_reciprocal_sub_view_zeros_like_5.run(arg1_1, buf19, buf16, buf17, buf20, arg6_1, arg8_1, buf21, buf23, buf24, buf26, 512, 768, stream=raw_stream0)
        del arg1_1
        del arg6_1
        del arg8_1
        del buf19
        del buf20
        buf27 = empty_strided_cuda((512, 4096), (4096, 1), torch.int32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.clamp, aten._to_copy, aten.view, aten.t, aten._int_mm]
        extern_kernels._int_mm(reinterpret_tensor(buf26, (512, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 4096), (1, 768), 0), out=buf27)
        del buf26
        buf28 = empty_strided_cuda((4096, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.sum]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_sum_6.run(arg9_1, buf28, 4096, 768, stream=raw_stream0)
        del arg9_1
        buf30 = empty_strided_cuda((4, 128, 2048), (262144, 2048, 1), torch.bfloat16)
        buf31 = buf17; del buf17  # reuse
        buf32 = buf16; del buf16  # reuse
        buf33 = empty_strided_cuda((4, 128, 2048), (262144, 2048, 1), torch.int8)
        # Topologically Sorted Source Nodes: [linear_2, chunk_1, silu, mul, linear_3], Original ATen: [aten.zeros_like, aten.minimum, aten.neg, aten.maximum, aten.div, aten.clamp, aten.view, aten.expand, aten.mul, aten._to_copy, aten.sub, aten.split, aten.silu, aten.amin, aten.amax, aten.reciprocal, aten.add]
        raw_stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_amax_amin_clamp_div_expand_maximum_minimum_mul_neg_reciprocal_silu_split_sub_view_zeros_like_7.run(buf27, buf23, buf24, buf28, arg10_1, buf30, buf31, buf32, buf33, 512, 2048, stream=raw_stream0)
        del arg10_1
        del buf23
        del buf24
        del buf27
        del buf28
        del buf30
        buf34 = empty_strided_cuda((512, 768), (768, 1), torch.int32)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.zeros_like, aten.minimum, aten.neg, aten.maximum, aten.div, aten.clamp, aten.reciprocal, aten.mul, aten.add, aten._to_copy, aten.view, aten.t, aten._int_mm]
        extern_kernels._int_mm(reinterpret_tensor(buf33, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg12_1, (2048, 768), (1, 2048), 0), out=buf34)
        del buf33
        buf35 = empty_strided_cuda((768, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.sum]
        raw_stream0 = get_raw_stream(0)
        triton_red_fused_sum_8.run(arg12_1, buf35, 768, 2048, stream=raw_stream0)
        del arg12_1
        buf36 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [linear_3, add_1], Original ATen: [aten.zeros_like, aten.minimum, aten.neg, aten.maximum, aten.div, aten.clamp, aten.view, aten.expand, aten.mul, aten._to_copy, aten.sub, aten.add]
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clamp_div_expand_maximum_minimum_mul_neg_sub_view_zeros_like_9.run(buf36, buf34, buf31, buf32, buf35, arg13_1, 393216, stream=raw_stream0)
        del arg13_1
        del buf31
        del buf32
        del buf34
        del buf35
    return (buf36, )


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1 = args
        args.clear()
        partition0_args = [arg1_1, arg0_1, arg2_1, arg3_1, arg5_1, arg6_1, arg8_1, arg9_1, arg10_1, arg12_1, arg13_1]
        del arg1_1, arg0_1, arg2_1, arg3_1, arg5_1, arg6_1, arg8_1, arg9_1, arg10_1, arg12_1, arg13_1
        (buf36,) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf36, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.int8)
    arg3_1 = rand_strided((2304, 1), (1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((2304, 1), (1, 1), device='cuda:0', dtype=torch.int8)
    arg5_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.int8)
    arg6_1 = rand_strided((768, 1), (1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((768, 1), (1, 1), device='cuda:0', dtype=torch.int8)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((4096, 768), (768, 1), device='cuda:0', dtype=torch.int8)
    arg10_1 = rand_strided((4096, 1), (1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((4096, 1), (1, 1), device='cuda:0', dtype=torch.int8)
    arg12_1 = rand_strided((768, 2048), (2048, 1), device='cuda:0', dtype=torch.int8)
    arg13_1 = rand_strided((768, 1), (1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((768, 1), (1, 1), device='cuda:0', dtype=torch.int8)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
