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



# kernel path: <inductor-cache>/oy/coyasrtzonz4iu3d3574juxm2ljkteypzxi6zppasrqt5g74ntfh.py
# Topologically Sorted Source Nodes: [rms_norm], Original ATen: [aten._fused_rms_norm]
# Source node to ATen node mapping:
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
#   %convert_element_type_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   return %buf0,%convert_element_type_1
triton_per_fused__fused_rms_norm_0 = async_compile.triton('triton_per_fused__fused_rms_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 2360832}}
)
@triton.jit
def triton_per_fused__fused_rms_norm_0(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp16, r0_mask & xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/4j/c4j3xnjnmbtv4obfvqevfavvg5ovizgmcrskjnkgniwv3zjrqagw.py
# Topologically Sorted Source Nodes: [linear_1, hidden, rms_norm_1], Original ATen: [aten._unsafe_view, aten.add, aten._fused_rms_norm]
# Source node to ATen node mapping:
#   hidden => add_1
#   linear_1 => view_7
#   rms_norm_1 => add_2, convert_element_type_6, convert_element_type_7, mean_1, mul_2, mul_3, pow_2, rsqrt_1
# Graph fragment:
#   %arg1_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mm_1 : Tensor "bf16[512, 768][768, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %buf10 : Tensor "f32[4, 128, 1][128, 1, 512]cuda:0" = PlaceHolder[target=buf10]
#   %arg4_1 : Tensor "bf16[768][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %view_7 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [4, 128, 768]), kwargs = {})
#   %add_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %view_7), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_6, 2), kwargs = {})
#   %mean_1 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [2], True), kwargs = {})
#   %add_2 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[4, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_2 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg4_1), kwargs = {})
#   %convert_element_type_7 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   return %buf10,%convert_element_type_7
triton_per_fused__fused_rms_norm__unsafe_view_add_1 = async_compile.triton('triton_per_fused__fused_rms_norm__unsafe_view_add_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__fused_rms_norm__unsafe_view_add_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 3147264}}
)
@triton.jit
def triton_per_fused__fused_rms_norm__unsafe_view_add_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp18, r0_mask & xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/3g/c3goq2dqf5mf4saorhiefv7c7wfetktmx6yplei63c4va7h2wsps.py
# Topologically Sorted Source Nodes: [linear_2, chunk_1, silu, mul], Original ATen: [aten._unsafe_view, aten.split, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   chunk_1 => split_1
#   linear_2 => view_9
#   mul => mul_4
#   silu => add_3, convert_element_type_10, convert_element_type_11, div, exp, neg
# Graph fragment:
#   %mm_2 : Tensor "bf16[512, 4096][4096, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %view_9 : Tensor "bf16[4, 128, 4096][524288, 4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [4, 128, 4096]), kwargs = {})
#   %split_1 : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_9, 2048, -1), kwargs = {})
#   %convert_element_type_10 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_12, torch.float32), kwargs = {})
#   %neg : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_10,), kwargs = {})
#   %exp : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_3 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_10, %add_3), kwargs = {})
#   %convert_element_type_11 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %mul_4 : Tensor "bf16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_11, %getitem_13), kwargs = {})
#   return %mul_4
triton_poi_fused__unsafe_view_mul_silu_split_2 = async_compile.triton('triton_poi_fused__unsafe_view_mul_silu_split_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_mul_silu_split_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 8388608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_silu_split_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096*x1), None).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (2048 + x0 + 4096*x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, None)
''', device_str='cuda')


# kernel path: <inductor-cache>/22/c22v6lijw3jsmjyomluoi563xcbgqoqivrfu27upouvawsvpw6a2.py
# Topologically Sorted Source Nodes: [linear_1, hidden, linear_3, add_1], Original ATen: [aten._unsafe_view, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_4
#   hidden => add_1
#   linear_1 => view_7
#   linear_3 => view_11
# Graph fragment:
#   %arg1_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mm_1 : Tensor "bf16[512, 768][768, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %mm_3 : Tensor "bf16[512, 768][768, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %view_7 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [4, 128, 768]), kwargs = {})
#   %add_1 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %view_7), kwargs = {})
#   %view_11 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [4, 128, 768]), kwargs = {})
#   %add_4 : Tensor "bf16[4, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_11), kwargs = {})
#   return %add_4
triton_poi_fused__unsafe_view_add_3 = async_compile.triton('triton_poi_fused__unsafe_view_add_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3932160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_3(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    arg1_1, arg0_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_misaligned(arg1_1)
        buf1 = empty_strided_cuda((4, 128, 768), (98304, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rms_norm], Original ATen: [aten._fused_rms_norm]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused__fused_rms_norm_0.run(arg1_1, arg0_1, buf1, 512, 768, stream=raw_stream0)
        del arg0_1
        buf2 = empty_strided_cuda((512, 2304), (2304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rms_norm, linear], Original ATen: [aten._fused_rms_norm, aten.view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (512, 768), (768, 1), 0), reinterpret_tensor(arg2_1, (768, 2304), (1, 768), 0), out=buf2)
        del arg2_1
        # Topologically Sorted Source Nodes: [linear, chunk, view, transpose, view_1, transpose_1, view_2, transpose_2, attended], Original ATen: [aten._unsafe_view, aten.split, aten.view, aten.transpose, aten._scaled_dot_product_flash_attention]
        buf3 = torch.ops.aten._scaled_dot_product_flash_attention.default(reinterpret_tensor(buf2, (4, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf2, (4, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf2, (4, 12, 128, 64), (294912, 64, 2304, 1), 1536), 0.0, True, scale=0.125)
        del buf2
        buf4 = buf3[0]
        assert_size_stride(buf4, (4, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        assert_alignment(buf4, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
        del buf3
        buf9 = reinterpret_tensor(buf1, (512, 768), (768, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [transpose_3, merged, linear_1], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 768), (768, 1), 0), reinterpret_tensor(arg3_1, (768, 768), (1, 768), 0), out=buf9)
        del arg3_1
        buf11 = reinterpret_tensor(buf4, (4, 128, 768), (98304, 768, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [linear_1, hidden, rms_norm_1], Original ATen: [aten._unsafe_view, aten.add, aten._fused_rms_norm]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused__fused_rms_norm__unsafe_view_add_1.run(arg1_1, buf9, arg4_1, buf11, 512, 768, stream=raw_stream0)
        del arg4_1
        buf12 = empty_strided_cuda((512, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1, hidden, rms_norm_1, linear_2], Original ATen: [aten._unsafe_view, aten.add, aten._fused_rms_norm, aten.view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (512, 768), (768, 1), 0), reinterpret_tensor(arg5_1, (768, 4096), (1, 768), 0), out=buf12)
        del arg5_1
        del buf11
        buf13 = empty_strided_cuda((4, 128, 2048), (262144, 2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2, chunk_1, silu, mul], Original ATen: [aten._unsafe_view, aten.split, aten.silu, aten.mul]
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_mul_silu_split_2.run(buf12, buf13, 1048576, stream=raw_stream0)
        del buf12
        buf14 = empty_strided_cuda((512, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2, chunk_1, silu, mul, linear_3], Original ATen: [aten._unsafe_view, aten.split, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg6_1, (2048, 768), (1, 2048), 0), out=buf14)
        del arg6_1
        del buf13
        buf15 = reinterpret_tensor(buf9, (4, 128, 768), (98304, 768, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [linear_1, hidden, linear_3, add_1], Original ATen: [aten._unsafe_view, aten.add]
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_add_3.run(buf15, arg1_1, buf14, 393216, stream=raw_stream0)
        del arg1_1
        del buf14
    return (buf15, )


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
        args.clear()
        partition0_args = [arg1_1, arg0_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]
        del arg1_1, arg0_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1
        (buf15,) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf15, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((4096, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((768, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
