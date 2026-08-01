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



# kernel path: <inductor-cache>/sd/csdzsoc66675onzpjgfayskpilvmqtwzod5ttpxcxejktdlalcmj.py
# Topologically Sorted Source Nodes: [rms_norm, linear], Original ATen: [aten._fused_rms_norm, aten.abs, aten.amax]
# Source node to ATen node mapping:
#   linear => abs_1, amax
#   rms_norm => add, convert_element_type, convert_element_type_1, mean, mul, mul_1, pow_1, rsqrt
# Graph fragment:
#   %arg1_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %buf0 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=buf0]
#   %arg0_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg0_1), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %abs_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convert_element_type_1,), kwargs = {})
#   %amax : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_1, [0, 1, 2], True), kwargs = {})
#   return %buf0,%buf1
triton_per_fused__fused_rms_norm_abs_amax_0 = async_compile.triton('triton_per_fused__fused_rms_norm_abs_amax_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_abs_amax_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 8192, 'r0_': 787968}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_abs_amax_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp17 = tl_math.abs(tmp16)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(r0_mask & xmask, tmp18, float("-inf"))
    tmp21 = triton_helpers.max2(tmp20, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/oa/coadn6q7dau2uankjkqitnl6bm2h5f3a4qe6wfie5taxzjnen2kt.py
# Topologically Sorted Source Nodes: [rms_norm, linear], Original ATen: [aten._fused_rms_norm, aten._to_copy, aten.abs, aten.amax, aten.div, aten.clamp, aten.view, aten.t, aten._scaled_mm]
# Source node to ATen node mapping:
#   linear => _scaled_mm, abs_1, amax, clamp_max, clamp_min, convert_element_type_2, convert_element_type_3, convert_element_type_4, div, div_1, permute, permute_1, view_2, view_3
#   rms_norm => add, convert_element_type, convert_element_type_1, mean, mul, mul_1, pow_1, rsqrt
# Graph fragment:
#   %buf1 : Tensor "f32[1, 1, 1, 4, 128, 1][512, 512, 512, 128, 1, 512]cuda:0" = PlaceHolder[target=buf1]
#   %amax : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=amax]
#   %convert_element_type : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg0_1), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_1, torch.float32), kwargs = {})
#   %abs_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convert_element_type_1,), kwargs = {})
#   %amax : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_1, [0, 1, 2], True), kwargs = {})
#   %div : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%amax, 448.0), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float32), kwargs = {})
#   %div_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_3, %convert_element_type_2), kwargs = {})
#   %clamp_min : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_1, -448.0), kwargs = {})
#   %clamp_max : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 448.0), kwargs = {})
#   %convert_element_type_4 : Tensor "f8e4m3fn[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.float8_e4m3fn), kwargs = {})
#   %view_2 : Tensor "f8e4m3fn[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_4, [-1, 768]), kwargs = {})
#   %permute : Tensor "f8e4m3fn[768, 2304][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg2_1, [1, 0]), kwargs = {})
#   %view_3 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_2, [1, 1]), kwargs = {})
#   %permute_1 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg3_1, [1, 0]), kwargs = {})
#   %_scaled_mm : Tensor "bf16[512, 2304][2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_2, %permute, %view_3, %permute_1, None, None, torch.bfloat16, True), kwargs = {})
#   return %amax,%buf4
triton_per_fused__fused_rms_norm__scaled_mm__to_copy_abs_amax_clamp_div_t_view_1 = async_compile.triton('triton_per_fused__fused_rms_norm__scaled_mm__to_copy_abs_amax_clamp_div_t_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__scaled_mm__to_copy_abs_amax_clamp_div_t_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'r0_': 2048}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__scaled_mm__to_copy_abs_amax_clamp_div_t_view_1(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = triton_helpers.max2(tmp1, 1)[:, None].to(tl.float32)
    tmp4 = tl.full([1, 1], 0.002232142857142857, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr1 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp6, None)
    tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp3, None)
''', device_str='cuda')


# kernel path: <inductor-cache>/5q/c5qr5wfaxztymo2rhhnbwvwgcef2yfeaogrsmcrwq6pyfquk3qnh.py
# Topologically Sorted Source Nodes: [rms_norm, linear], Original ATen: [aten._fused_rms_norm, aten._to_copy, aten.div, aten.clamp, aten.view, aten.t, aten._scaled_mm]
# Source node to ATen node mapping:
#   linear => _scaled_mm, clamp_max, clamp_min, convert_element_type_2, convert_element_type_3, convert_element_type_4, div, div_1, permute, permute_1, view_2, view_3
#   rms_norm => add, convert_element_type, convert_element_type_1, mean, mul, mul_1, pow_1, rsqrt
# Graph fragment:
#   %arg1_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %buf0 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=buf0]
#   %arg0_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %amax : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=amax]
#   %convert_element_type : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [2], True), kwargs = {})
#   %add : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg0_1), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_1, torch.float32), kwargs = {})
#   %div : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%amax, 448.0), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float32), kwargs = {})
#   %div_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_3, %convert_element_type_2), kwargs = {})
#   %clamp_min : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_1, -448.0), kwargs = {})
#   %clamp_max : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 448.0), kwargs = {})
#   %convert_element_type_4 : Tensor "f8e4m3fn[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.float8_e4m3fn), kwargs = {})
#   %view_2 : Tensor "f8e4m3fn[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_4, [-1, 768]), kwargs = {})
#   %permute : Tensor "f8e4m3fn[768, 2304][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg2_1, [1, 0]), kwargs = {})
#   %view_3 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_2, [1, 1]), kwargs = {})
#   %permute_1 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg3_1, [1, 0]), kwargs = {})
#   %_scaled_mm : Tensor "bf16[512, 2304][2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_2, %permute, %view_3, %permute_1, None, None, torch.bfloat16, True), kwargs = {})
#   return %buf3
triton_poi_fused__fused_rms_norm__scaled_mm__to_copy_clamp_div_t_view_2 = async_compile.triton('triton_poi_fused__fused_rms_norm__scaled_mm__to_copy_clamp_div_t_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp8e4nv', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__fused_rms_norm__scaled_mm__to_copy_clamp_div_t_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 1574400}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__fused_rms_norm__scaled_mm__to_copy_clamp_div_t_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 768
    x0 = (xindex % 768)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr3 + (0)).to(tl.float32)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1], 768.0, tl.float32)
    tmp4 = (tmp2 / tmp3)
    tmp5 = tl.full([1], 1e-06, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.rsqrt(tmp6)
    tmp8 = tmp1 * tmp7
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp16 = tl.full([1], 0.002232142857142857, tl.float32)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = (tmp13 / tmp18)
    tmp20 = tl.full([1], -448.0, tl.float32)
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = tl.full([1], 448.0, tl.float32)
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tmp23.to(tl.float8e4nv)
    tl.store(out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: <inductor-cache>/zc/czcxtfl233d6vtb5kr53hkcweoim3tws3ttzw5enzwjmpus2g57f.py
# Topologically Sorted Source Nodes: [transpose_3, merged, linear_1], Original ATen: [aten.transpose, aten.view, aten.abs, aten.amax]
# Source node to ATen node mapping:
#   linear_1 => abs_2, amax_1
#   merged => view_8
#   transpose_3 => permute_5
# Graph fragment:
#   %getitem_3 : Tensor "bf16[4, 12, 128, 64][98304, 64, 768, 1]cuda:0" = PlaceHolder[target=getitem_3]
#   %permute_5 : Tensor "bf16[4, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_3, [0, 2, 1, 3]), kwargs = {})
#   %view_8 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_5, [4, 128, 768]), kwargs = {})
#   %abs_2 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%view_8,), kwargs = {})
#   %amax_1 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_2, [0, 1, 2], True), kwargs = {})
#   return %buf13
triton_red_fused_abs_amax_transpose_view_3 = async_compile.triton('triton_red_fused_abs_amax_transpose_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_amax_transpose_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 384, 'r0_': 786432}}
)
@triton.jit
def triton_red_fused_abs_amax_transpose_view_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 48
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl_math.abs(tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/7d/c7ddgrq7wp4nssuzjd2akzgusekoq3ctmzi7squhlmquch3vn5te.py
# Topologically Sorted Source Nodes: [transpose_3, merged, linear_1], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.abs, aten.amax, aten.div, aten.clamp, aten.t, aten._scaled_mm]
# Source node to ATen node mapping:
#   linear_1 => _scaled_mm_1, abs_2, amax_1, clamp_max_1, clamp_min_1, convert_element_type_5, convert_element_type_6, convert_element_type_7, div_2, div_3, permute_6, permute_7, view_11, view_12
#   merged => view_8
#   transpose_3 => permute_5
# Graph fragment:
#   %buf13 : Tensor "f32[1, 1, 1, 48][48, 48, 48, 1]cuda:0" = PlaceHolder[target=buf13]
#   %amax_1 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=amax_1]
#   %permute_5 : Tensor "bf16[4, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_3, [0, 2, 1, 3]), kwargs = {})
#   %view_8 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_5, [4, 128, 768]), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_8, torch.float32), kwargs = {})
#   %abs_2 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%view_8,), kwargs = {})
#   %amax_1 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_2, [0, 1, 2], True), kwargs = {})
#   %div_2 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%amax_1, 448.0), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.float32), kwargs = {})
#   %div_3 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_6, %convert_element_type_5), kwargs = {})
#   %clamp_min_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_3, -448.0), kwargs = {})
#   %clamp_max_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 448.0), kwargs = {})
#   %convert_element_type_7 : Tensor "f8e4m3fn[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_1, torch.float8_e4m3fn), kwargs = {})
#   %view_11 : Tensor "f8e4m3fn[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_7, [-1, 768]), kwargs = {})
#   %permute_6 : Tensor "f8e4m3fn[768, 768][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg4_1, [1, 0]), kwargs = {})
#   %view_12 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_5, [1, 1]), kwargs = {})
#   %permute_7 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg5_1, [1, 0]), kwargs = {})
#   %_scaled_mm_1 : Tensor "bf16[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_11, %permute_6, %view_12, %permute_7, None, None, torch.bfloat16, True), kwargs = {})
#   return %amax_1,%buf16
triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_div_t_transpose_view_4 = async_compile.triton('triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_div_t_transpose_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_div_t_transpose_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'r0_': 192}}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_div_t_transpose_view_4(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 48
    R0_BLOCK: tl.constexpr = 64
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = tl.full([1, 1], 0.002232142857142857, tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr1 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp7, None)
    tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp4, None)
''', device_str='cuda')


# kernel path: <inductor-cache>/xa/cxa6qkd4b47kwuqtc6aymoq6pzfto2kwdfiezgbsj3lbrp6gz7sz.py
# Topologically Sorted Source Nodes: [transpose_3, merged, linear_1], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.div, aten.clamp, aten.t, aten._scaled_mm]
# Source node to ATen node mapping:
#   linear_1 => _scaled_mm_1, clamp_max_1, clamp_min_1, convert_element_type_5, convert_element_type_6, convert_element_type_7, div_2, div_3, permute_6, permute_7, view_11, view_12
#   merged => view_8
#   transpose_3 => permute_5
# Graph fragment:
#   %getitem_3 : Tensor "bf16[4, 12, 128, 64][98304, 64, 768, 1]cuda:0" = PlaceHolder[target=getitem_3]
#   %amax_1 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=amax_1]
#   %permute_5 : Tensor "bf16[4, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_3, [0, 2, 1, 3]), kwargs = {})
#   %view_8 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_5, [4, 128, 768]), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_8, torch.float32), kwargs = {})
#   %div_2 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%amax_1, 448.0), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.float32), kwargs = {})
#   %div_3 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_6, %convert_element_type_5), kwargs = {})
#   %clamp_min_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_3, -448.0), kwargs = {})
#   %clamp_max_1 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 448.0), kwargs = {})
#   %convert_element_type_7 : Tensor "f8e4m3fn[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_1, torch.float8_e4m3fn), kwargs = {})
#   %view_11 : Tensor "f8e4m3fn[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_7, [-1, 768]), kwargs = {})
#   %permute_6 : Tensor "f8e4m3fn[768, 768][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg4_1, [1, 0]), kwargs = {})
#   %view_12 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_5, [1, 1]), kwargs = {})
#   %permute_7 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg5_1, [1, 0]), kwargs = {})
#   %_scaled_mm_1 : Tensor "bf16[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_11, %permute_6, %view_12, %permute_7, None, None, torch.bfloat16, True), kwargs = {})
#   return %buf15
triton_poi_fused__scaled_mm__to_copy_clamp_div_t_transpose_view_5 = async_compile.triton('triton_poi_fused__scaled_mm__to_copy_clamp_div_t_transpose_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp8e4nv', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_mm__to_copy_clamp_div_t_transpose_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 1572864}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__scaled_mm__to_copy_clamp_div_t_transpose_view_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tl.full([1], 0.002232142857142857, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = (tmp1 / tmp6)
    tmp8 = tl.full([1], -448.0, tl.float32)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = tl.full([1], 448.0, tl.float32)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = tmp11.to(tl.float8e4nv)
    tl.store(out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: <inductor-cache>/pt/cptqks2dppqmhix3yqlxpur7uzgpnpsm6ncdocfrq2vvntha5uuu.py
# Topologically Sorted Source Nodes: [linear_1, hidden, rms_norm_1, linear_2], Original ATen: [aten.view, aten.add, aten._fused_rms_norm, aten.abs, aten.amax]
# Source node to ATen node mapping:
#   hidden => add_1
#   linear_1 => view_13
#   linear_2 => abs_3, amax_2
#   rms_norm_1 => add_2, convert_element_type_8, convert_element_type_9, mean_1, mul_2, mul_3, pow_2, rsqrt_1
# Graph fragment:
#   %arg1_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %_scaled_mm_1 : Tensor "bf16[512, 768][768, 1]cuda:0" = PlaceHolder[target=_scaled_mm_1]
#   %buf18 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=buf18]
#   %arg6_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %view_13 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%_scaled_mm_1, [4, 128, 768]), kwargs = {})
#   %add_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %view_13), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_8, 2), kwargs = {})
#   %mean_1 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_2 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_2 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg6_1), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   %abs_3 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%convert_element_type_9,), kwargs = {})
#   %amax_2 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_3, [0, 1, 2], True), kwargs = {})
#   return %buf18,%buf19
triton_per_fused__fused_rms_norm_abs_add_amax_view_6 = async_compile.triton('triton_per_fused__fused_rms_norm_abs_add_amax_view_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_abs_add_amax_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 8192, 'r0_': 1574400}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_abs_add_amax_view_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([1, 1], 768.0, tl.float32)
    tmp10 = (tmp8 / tmp9)
    tmp11 = tl.full([1, 1], 1e-06, tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp3 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl_math.abs(tmp18)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(r0_mask & xmask, tmp20, float("-inf"))
    tmp23 = triton_helpers.max2(tmp22, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/n3/cn3l7cqshypjrc4hszw2e4h22hkukkqd4nopzijwme6jjjlbnq4r.py
# Topologically Sorted Source Nodes: [linear_1, hidden, rms_norm_1, linear_2], Original ATen: [aten.view, aten.add, aten._fused_rms_norm, aten._to_copy, aten.div, aten.clamp, aten.t, aten._scaled_mm]
# Source node to ATen node mapping:
#   hidden => add_1
#   linear_1 => view_13
#   linear_2 => _scaled_mm_2, clamp_max_2, clamp_min_2, convert_element_type_10, convert_element_type_11, convert_element_type_12, div_4, div_5, permute_8, permute_9, view_16, view_17
#   rms_norm_1 => add_2, convert_element_type_8, convert_element_type_9, mean_1, mul_2, mul_3, pow_2, rsqrt_1
# Graph fragment:
#   %arg1_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %_scaled_mm_1 : Tensor "bf16[512, 768][768, 1]cuda:0" = PlaceHolder[target=_scaled_mm_1]
#   %buf18 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=buf18]
#   %arg6_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %amax_2 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=amax_2]
#   %view_13 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%_scaled_mm_1, [4, 128, 768]), kwargs = {})
#   %add_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %view_13), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_8, 2), kwargs = {})
#   %mean_1 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_2 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_2 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg6_1), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   %convert_element_type_11 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_9, torch.float32), kwargs = {})
#   %div_4 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%amax_2, 448.0), kwargs = {})
#   %convert_element_type_10 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_4, torch.float32), kwargs = {})
#   %div_5 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_11, %convert_element_type_10), kwargs = {})
#   %clamp_min_2 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_5, -448.0), kwargs = {})
#   %clamp_max_2 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 448.0), kwargs = {})
#   %convert_element_type_12 : Tensor "f8e4m3fn[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_2, torch.float8_e4m3fn), kwargs = {})
#   %view_16 : Tensor "f8e4m3fn[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_12, [-1, 768]), kwargs = {})
#   %permute_8 : Tensor "f8e4m3fn[768, 4096][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg7_1, [1, 0]), kwargs = {})
#   %view_17 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_10, [1, 1]), kwargs = {})
#   %permute_9 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg8_1, [1, 0]), kwargs = {})
#   %_scaled_mm_2 : Tensor "bf16[512, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_16, %permute_8, %view_17, %permute_9, None, None, torch.bfloat16, True), kwargs = {})
#   return %buf21
triton_poi_fused__fused_rms_norm__scaled_mm__to_copy_add_clamp_div_t_view_7 = async_compile.triton('triton_poi_fused__fused_rms_norm__scaled_mm__to_copy_add_clamp_div_t_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*fp8e4nv', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__fused_rms_norm__scaled_mm__to_copy_add_clamp_div_t_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 2360832}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__fused_rms_norm__scaled_mm__to_copy_add_clamp_div_t_view_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x1 = xindex // 768
    x0 = (xindex % 768)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr4 + (0)).to(tl.float32)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tl.full([1], 768.0, tl.float32)
    tmp6 = (tmp4 / tmp5)
    tmp7 = tl.full([1], 1e-06, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp18 = tl.full([1], 0.002232142857142857, tl.float32)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = (tmp15 / tmp20)
    tmp22 = tl.full([1], -448.0, tl.float32)
    tmp23 = triton_helpers.maximum(tmp21, tmp22)
    tmp24 = tl.full([1], 448.0, tl.float32)
    tmp25 = triton_helpers.minimum(tmp23, tmp24)
    tmp26 = tmp25.to(tl.float8e4nv)
    tl.store(out_ptr0 + (x2), tmp26, None)
''', device_str='cuda')


# kernel path: <inductor-cache>/jy/cjyeqizze7353aarasexfgdtffho67bqei5yewt7radidpmrj5v4.py
# Topologically Sorted Source Nodes: [linear_2, chunk_1, silu, mul, linear_3], Original ATen: [aten.view, aten.split, aten.silu, aten.mul, aten.abs, aten.amax]
# Source node to ATen node mapping:
#   chunk_1 => split_1
#   linear_2 => view_18
#   linear_3 => abs_4, amax_3
#   mul => mul_4
#   silu => add_3, convert_element_type_13, convert_element_type_14, div_6, exp, neg
# Graph fragment:
#   %_scaled_mm_2 : Tensor "bf16[512, 4096][4096, 1]cuda:0" = PlaceHolder[target=_scaled_mm_2]
#   %view_18 : Tensor "bf16[4, 128, 4096][524288, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%_scaled_mm_2, [4, 128, 4096]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_18, 2048, -1), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_12, torch.float32), kwargs = {})
#   %neg : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_13,), kwargs = {})
#   %exp : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_3 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div_6 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_13, %add_3), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_6, torch.bfloat16), kwargs = {})
#   %mul_4 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_14, %getitem_13), kwargs = {})
#   %abs_4 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_4,), kwargs = {})
#   %amax_3 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_4, [0, 1, 2], True), kwargs = {})
#   return %buf25
triton_red_fused_abs_amax_mul_silu_split_view_8 = async_compile.triton('triton_red_fused_abs_amax_mul_silu_split_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_amax_mul_silu_split_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 512, 'r0_': 4194304}}
)
@triton.jit
def triton_red_fused_abs_amax_mul_silu_split_view_8(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (4096*(r0_1 // 2048) + 32768*x0 + ((r0_1 % 2048))), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr0 + (2048 + 4096*(r0_1 // 2048) + 32768*x0 + ((r0_1 % 2048))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = -tmp1
        tmp3 = libdevice.exp(tmp2)
        tmp4 = tl.full([1, 1], 1.0, tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = (tmp1 / tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = tl_math.abs(tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = triton_helpers.maximum(_tmp12, tmp11)
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = triton_helpers.max2(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/3q/c3qn37jmax53lunxtloff6xh5ori5l5bqkqln2le2xbg4f6qtukt.py
# Topologically Sorted Source Nodes: [linear_2, chunk_1, silu, mul, linear_3], Original ATen: [aten.view, aten.split, aten.silu, aten.mul, aten._to_copy, aten.abs, aten.amax, aten.div, aten.clamp, aten.t, aten._scaled_mm]
# Source node to ATen node mapping:
#   chunk_1 => split_1
#   linear_2 => view_18
#   linear_3 => _scaled_mm_3, abs_4, amax_3, clamp_max_3, clamp_min_3, convert_element_type_15, convert_element_type_16, convert_element_type_17, div_7, div_8, permute_10, permute_11, view_21, view_22
#   mul => mul_4
#   silu => add_3, convert_element_type_13, convert_element_type_14, div_6, exp, neg
# Graph fragment:
#   %buf25 : Tensor "f32[1, 1, 1, 64][64, 64, 64, 1]cuda:0" = PlaceHolder[target=buf25]
#   %amax_3 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=amax_3]
#   %view_18 : Tensor "bf16[4, 128, 4096][524288, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%_scaled_mm_2, [4, 128, 4096]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_18, 2048, -1), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_12, torch.float32), kwargs = {})
#   %neg : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_13,), kwargs = {})
#   %exp : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_3 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div_6 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_13, %add_3), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_6, torch.bfloat16), kwargs = {})
#   %mul_4 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_14, %getitem_13), kwargs = {})
#   %convert_element_type_16 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.float32), kwargs = {})
#   %abs_4 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_4,), kwargs = {})
#   %amax_3 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_4, [0, 1, 2], True), kwargs = {})
#   %div_7 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%amax_3, 448.0), kwargs = {})
#   %convert_element_type_15 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_7, torch.float32), kwargs = {})
#   %div_8 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_16, %convert_element_type_15), kwargs = {})
#   %clamp_min_3 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_8, -448.0), kwargs = {})
#   %clamp_max_3 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 448.0), kwargs = {})
#   %convert_element_type_17 : Tensor "f8e4m3fn[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_3, torch.float8_e4m3fn), kwargs = {})
#   %view_21 : Tensor "f8e4m3fn[512, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_17, [-1, 2048]), kwargs = {})
#   %permute_10 : Tensor "f8e4m3fn[2048, 768][1, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg9_1, [1, 0]), kwargs = {})
#   %view_22 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_15, [1, 1]), kwargs = {})
#   %permute_11 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg10_1, [1, 0]), kwargs = {})
#   %_scaled_mm_3 : Tensor "bf16[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_21, %permute_10, %view_22, %permute_11, None, None, torch.bfloat16, True), kwargs = {})
#   return %amax_3,%buf28
triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_div_mul_silu_split_t_view_9 = async_compile.triton('triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_div_mul_silu_split_t_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_div_mul_silu_split_t_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'r0_': 256}}
)
@triton.jit
def triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_div_mul_silu_split_t_view_9(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = triton_helpers.max2(tmp1, 1)[:, None].to(tl.float32)
    tmp4 = tl.full([1, 1], 0.002232142857142857, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tl.store(out_ptr1 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp6, None)
    tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp3, None)
''', device_str='cuda')


# kernel path: <inductor-cache>/mz/cmzd3lem6ldhkp6j5sexrettozpeiedbofc37gfwerzgzqfl3z7b.py
# Topologically Sorted Source Nodes: [linear_2, chunk_1, silu, mul, linear_3], Original ATen: [aten.view, aten.split, aten.silu, aten.mul, aten._to_copy, aten.div, aten.clamp, aten.t, aten._scaled_mm]
# Source node to ATen node mapping:
#   chunk_1 => split_1
#   linear_2 => view_18
#   linear_3 => _scaled_mm_3, clamp_max_3, clamp_min_3, convert_element_type_15, convert_element_type_16, convert_element_type_17, div_7, div_8, permute_10, permute_11, view_21, view_22
#   mul => mul_4
#   silu => add_3, convert_element_type_13, convert_element_type_14, div_6, exp, neg
# Graph fragment:
#   %_scaled_mm_2 : Tensor "bf16[512, 4096][4096, 1]cuda:0" = PlaceHolder[target=_scaled_mm_2]
#   %amax_3 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=amax_3]
#   %view_18 : Tensor "bf16[4, 128, 4096][524288, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%_scaled_mm_2, [4, 128, 4096]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_18, 2048, -1), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_12, torch.float32), kwargs = {})
#   %neg : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_13,), kwargs = {})
#   %exp : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_3 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div_6 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_13, %add_3), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_6, torch.bfloat16), kwargs = {})
#   %mul_4 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_14, %getitem_13), kwargs = {})
#   %convert_element_type_16 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.float32), kwargs = {})
#   %div_7 : Tensor "bf16[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%amax_3, 448.0), kwargs = {})
#   %convert_element_type_15 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_7, torch.float32), kwargs = {})
#   %div_8 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_16, %convert_element_type_15), kwargs = {})
#   %clamp_min_3 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_8, -448.0), kwargs = {})
#   %clamp_max_3 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 448.0), kwargs = {})
#   %convert_element_type_17 : Tensor "f8e4m3fn[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max_3, torch.float8_e4m3fn), kwargs = {})
#   %view_21 : Tensor "f8e4m3fn[512, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_17, [-1, 2048]), kwargs = {})
#   %permute_10 : Tensor "f8e4m3fn[2048, 768][1, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg9_1, [1, 0]), kwargs = {})
#   %view_22 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_15, [1, 1]), kwargs = {})
#   %permute_11 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg10_1, [1, 0]), kwargs = {})
#   %_scaled_mm_3 : Tensor "bf16[512, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._scaled_mm.default](args = (%view_21, %permute_10, %view_22, %permute_11, None, None, torch.bfloat16, True), kwargs = {})
#   return %buf27
triton_poi_fused__scaled_mm__to_copy_clamp_div_mul_silu_split_t_view_10 = async_compile.triton('triton_poi_fused__scaled_mm__to_copy_clamp_div_mul_silu_split_t_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp8e4nv', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_mm__to_copy_clamp_div_mul_silu_split_t_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 6291456}},
    min_elem_per_thread=2
)
@triton.jit
def triton_poi_fused__scaled_mm__to_copy_clamp_div_mul_silu_split_t_view_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096*x1), None).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (2048 + x0 + 4096*x1), None).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp13 = tl.full([1], 0.002232142857142857, tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = (tmp10 / tmp15)
    tmp17 = tl.full([1], -448.0, tl.float32)
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = tl.full([1], 448.0, tl.float32)
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp20.to(tl.float8e4nv)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: <inductor-cache>/zu/czunackjqzgypq3qhmoh4mdmy2hsjk6lohjnbgckp3o6xjc2yibb.py
# Topologically Sorted Source Nodes: [linear_1, hidden, linear_3, add_1], Original ATen: [aten.view, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_4
#   hidden => add_1
#   linear_1 => view_13
#   linear_3 => view_23
# Graph fragment:
#   %arg1_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %_scaled_mm_1 : Tensor "bf16[512, 768][768, 1]cuda:0" = PlaceHolder[target=_scaled_mm_1]
#   %_scaled_mm_3 : Tensor "bf16[512, 768][768, 1]cuda:0" = PlaceHolder[target=_scaled_mm_3]
#   %view_13 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%_scaled_mm_1, [4, 128, 768]), kwargs = {})
#   %add_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %view_13), kwargs = {})
#   %view_23 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%_scaled_mm_3, [4, 128, 768]), kwargs = {})
#   %add_4 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_23), kwargs = {})
#   return %add_4
triton_poi_fused_add_view_11 = async_compile.triton('triton_poi_fused_add_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': True, 'tiling_scores': {'x': 3932160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_view_11(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')

def partition_0(args):
    arg1_1, arg0_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_misaligned(arg1_1)
        buf0 = empty_strided_cuda((4, 128, 1), (128, 1, 512), torch.float32)
        buf1 = empty_strided_cuda((1, 1, 1, 4, 128, 1), (512, 512, 512, 128, 1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [rms_norm, linear], Original ATen: [aten._fused_rms_norm, aten.abs, aten.amax]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused__fused_rms_norm_abs_amax_0.run(arg1_1, arg0_1, buf0, buf1, 512, 768, stream=raw_stream0)
        buf2 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.bfloat16)
        buf4 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rms_norm, linear], Original ATen: [aten._fused_rms_norm, aten._to_copy, aten.abs, aten.amax, aten.div, aten.clamp, aten.view, aten.t, aten._scaled_mm]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused__fused_rms_norm__scaled_mm__to_copy_abs_amax_clamp_div_t_view_1.run(buf1, buf2, buf4, 1, 512, stream=raw_stream0)
        buf3 = empty_strided_cuda((512, 768), (768, 1), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [rms_norm, linear], Original ATen: [aten._fused_rms_norm, aten._to_copy, aten.div, aten.clamp, aten.view, aten.t, aten._scaled_mm]
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused__fused_rms_norm__scaled_mm__to_copy_clamp_div_t_view_2.run(arg1_1, buf0, arg0_1, buf2, buf3, 393216, stream=raw_stream0)
        del arg0_1
        buf5 = empty_strided_cuda((512, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rms_norm, linear], Original ATen: [aten._fused_rms_norm, aten._to_copy, aten.div, aten.clamp, aten.view, aten.t, aten._scaled_mm]
        extern_kernels._scaled_mm(buf3, reinterpret_tensor(arg2_1, (768, 2304), (1, 768), 0), buf4, arg3_1, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf5)
        del arg2_1
        del arg3_1
        # Topologically Sorted Source Nodes: [linear, chunk, view, transpose, view_1, transpose_1, view_2, transpose_2, attended], Original ATen: [aten.view, aten.split, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf6 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf5, (4, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf5, (4, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf5, (4, 12, 128, 64), (294912, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        del buf5
        buf7 = buf6[0]
        assert_size_stride(buf7, (4, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf7, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf6
        buf13 = empty_strided_cuda((1, 1, 1, 48), (48, 48, 48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [transpose_3, merged, linear_1], Original ATen: [aten.transpose, aten.view, aten.abs, aten.amax]
        raw_stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_transpose_view_3.run(buf7, buf13, 48, 8192, stream=raw_stream0)
        buf14 = buf2; del buf2  # reuse
        buf16 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [transpose_3, merged, linear_1], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.abs, aten.amax, aten.div, aten.clamp, aten.t, aten._scaled_mm]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_div_t_transpose_view_4.run(buf13, buf14, buf16, 1, 48, stream=raw_stream0)
        del buf13
        buf15 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [transpose_3, merged, linear_1], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.div, aten.clamp, aten.t, aten._scaled_mm]
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm__to_copy_clamp_div_t_transpose_view_5.run(buf7, buf14, buf15, 393216, stream=raw_stream0)
        buf17 = reinterpret_tensor(buf7, (512, 768), (768, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [transpose_3, merged, linear_1], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.div, aten.clamp, aten.t, aten._scaled_mm]
        extern_kernels._scaled_mm(buf15, reinterpret_tensor(arg4_1, (768, 768), (1, 768), 0), buf16, arg5_1, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf17)
        del arg4_1
        del arg5_1
        buf18 = buf0; del buf0  # reuse
        buf19 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [linear_1, hidden, rms_norm_1, linear_2], Original ATen: [aten.view, aten.add, aten._fused_rms_norm, aten.abs, aten.amax]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused__fused_rms_norm_abs_add_amax_view_6.run(arg1_1, buf17, arg6_1, buf18, buf19, 512, 768, stream=raw_stream0)
        buf20 = buf14; del buf14  # reuse
        buf22 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [linear_1, hidden, rms_norm_1, linear_2], Original ATen: [aten.view, aten.add, aten._fused_rms_norm, aten._to_copy, aten.abs, aten.amax, aten.div, aten.clamp, aten.t, aten._scaled_mm]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused__fused_rms_norm__scaled_mm__to_copy_abs_amax_clamp_div_t_view_1.run(buf19, buf20, buf22, 1, 512, stream=raw_stream0)
        del buf19
        buf21 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [linear_1, hidden, rms_norm_1, linear_2], Original ATen: [aten.view, aten.add, aten._fused_rms_norm, aten._to_copy, aten.div, aten.clamp, aten.t, aten._scaled_mm]
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused__fused_rms_norm__scaled_mm__to_copy_add_clamp_div_t_view_7.run(arg1_1, buf17, buf18, arg6_1, buf20, buf21, 393216, stream=raw_stream0)
        del arg6_1
        del buf18
        buf23 = empty_strided_cuda((512, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1, hidden, rms_norm_1, linear_2], Original ATen: [aten.view, aten.add, aten._fused_rms_norm, aten._to_copy, aten.div, aten.clamp, aten.t, aten._scaled_mm]
        extern_kernels._scaled_mm(buf21, reinterpret_tensor(arg7_1, (768, 4096), (1, 768), 0), buf22, arg8_1, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf23)
        del arg7_1
        del arg8_1
        del buf21
        buf25 = empty_strided_cuda((1, 1, 1, 64), (64, 64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2, chunk_1, silu, mul, linear_3], Original ATen: [aten.view, aten.split, aten.silu, aten.mul, aten.abs, aten.amax]
        raw_stream0 = get_raw_stream(0)
        triton_red_fused_abs_amax_mul_silu_split_view_8.run(buf23, buf25, 64, 16384, stream=raw_stream0)
        buf26 = buf20; del buf20  # reuse
        buf28 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [linear_2, chunk_1, silu, mul, linear_3], Original ATen: [aten.view, aten.split, aten.silu, aten.mul, aten._to_copy, aten.abs, aten.amax, aten.div, aten.clamp, aten.t, aten._scaled_mm]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused__scaled_mm__to_copy_abs_amax_clamp_div_mul_silu_split_t_view_9.run(buf25, buf26, buf28, 1, 64, stream=raw_stream0)
        del buf25
        buf27 = empty_strided_cuda((512, 2048), (2048, 1), torch.float8_e4m3fn)
        # Topologically Sorted Source Nodes: [linear_2, chunk_1, silu, mul, linear_3], Original ATen: [aten.view, aten.split, aten.silu, aten.mul, aten._to_copy, aten.div, aten.clamp, aten.t, aten._scaled_mm]
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_mm__to_copy_clamp_div_mul_silu_split_t_view_10.run(buf23, buf26, buf27, 1048576, stream=raw_stream0)
        del buf23
        del buf26
        buf29 = empty_strided_cuda((512, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2, chunk_1, silu, mul, linear_3], Original ATen: [aten.view, aten.split, aten.silu, aten.mul, aten._to_copy, aten.div, aten.clamp, aten.t, aten._scaled_mm]
        extern_kernels._scaled_mm(buf27, reinterpret_tensor(arg9_1, (2048, 768), (1, 2048), 0), buf28, arg10_1, out_dtype=torch.bfloat16, use_fast_accum=True, out=buf29)
        del arg10_1
        del arg9_1
        del buf27
        del buf28
        buf30 = reinterpret_tensor(buf17, (4, 128, 768), (98304, 768, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [linear_1, hidden, linear_3, add_1], Original ATen: [aten.view, aten.add]
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_view_11.run(buf30, arg1_1, buf29, 393216, stream=raw_stream0)
        del arg1_1
        del buf29
    return (buf30, )


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1 = args
        args.clear()
        partition0_args = [arg1_1, arg0_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1]
        del arg1_1, arg0_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1
        (buf30,) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf30, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg3_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg5_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((4096, 768), (768, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg8_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 2048), (2048, 1), device='cuda:0', dtype=torch.float8_e4m3fn)
    arg10_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
