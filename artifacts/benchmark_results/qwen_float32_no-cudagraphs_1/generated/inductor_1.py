# AOT ID: ['3_inference']
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
from torch._C._dynamo.guards import copy_misaligned
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


# kernel path: <inductor-cache>/r4/cr4vrph5vnakal435l2agp2g6vy2is3sllenpjp3doaca2gs62iv.py
# Topologically Sorted Source Nodes: [inputs_embeds, pow_1, variance, rsqrt, hidden_states_1, hidden_states_2], Original ATen: [aten.embedding, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   hidden_states_1 => mul_26
#   hidden_states_2 => mul_27
#   inputs_embeds => embedding
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   variance => mean
# Graph fragment:
#   %arg0_1 : Tensor "i64[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[151936, 896][896, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg7_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %buf0 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf0]
#   %arg6_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg6_1]
#   %embedding : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %pow_1 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%embedding, 2), kwargs = {})
#   %mean : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.float32), kwargs = {})
#   %add_tensor : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %convert_element_type_default_4), kwargs = {})
#   %rsqrt : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_26 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, %rsqrt), kwargs = {})
#   %mul_27 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg7_1, %mul_26), kwargs = {})
#   return %buf0,%mul_27
triton_per_fused_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_per_fused_embedding_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 10752}}
)
@triton.jit
def triton_per_fused_embedding_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 896
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [1, 1])
    tmp13 = tl.load(in_ptr2 + (r0_0), r0_mask, other=0.0)
    tmp16 = in_ptr3
    tmp2 = tl.full([1, 1], 151936, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 151936), "index out of bounds: 0 <= tmp5 < 151936")
    tmp7 = tl.load(in_ptr1 + (r0_0 + 896*tmp5), r0_mask, other=0.0)
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp14 = tl.full([1, 1], 896.0, tl.float32)
    tmp15 = (tmp12 / tmp14)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp7 * tmp19
    tmp21 = tmp13 * tmp20
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp21, r0_mask)
''', device_str='cuda')


# kernel path: <inductor-cache>/f2/cf2ijha3zy2f5yidacf4ge7ho2g6d62huzrnkosr2qsifp7pkhh4.py
# Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, cos_3, sin, sin_3, linear_1, view_1, key_states, mul_6, x2_1, neg_1, x1_1, cat_2, mul_7, k_embed, keys], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
# Source node to ATen node mapping:
#   arange => iota
#   cat_2 => cat_1
#   cos => cos
#   cos_3 => unsqueeze_11
#   emb => clone, expand_4, unsqueeze_10, view_3
#   expand_1 => expand_1
#   freqs => permute
#   getitem_16 => unsqueeze_7, unsqueeze_8
#   getitem_17 => unsqueeze_9
#   k_embed => add_23
#   key_states => permute_4
#   keys => cat_2
#   linear_1 => view_8
#   matmul => mul_23
#   mul_6 => mul_30
#   mul_7 => mul_31
#   neg_1 => neg_1
#   position_ids => add
#   position_ids_1 => unsqueeze
#   position_ids_expanded => convert_element_type
#   sin => sin
#   sin_3 => unsqueeze_12
#   view_1 => view_9
#   x1_1 => slice_4
#   x2_1 => slice_5
# Graph fragment:
#   %arg3_1 : Tensor "f32[1, 2, s108, 64][128*s108, 64, 128, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %addmm_1 : Tensor "f32[1, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %arg4_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg5_1]
#   %unsqueeze_7 : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_8, [1, -1, 1]), kwargs = {})
#   %iota : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, %arg333_1), kwargs = {})
#   %unsqueeze : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_9, torch.float32), kwargs = {})
#   %mul_23 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, 1, 32][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_23, [0, 2, 1]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 1, 32][32, 1, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 1, 2, 32][32, 1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_10, [1, 1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, 1, 2, 32][64, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, %convert_element_type_default_1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor, 1), kwargs = {})
#   %sin : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, %convert_element_type_default_2), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor_1, 1), kwargs = {})
#   %view_8 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [1, 1, 128]), kwargs = {})
#   %view_9 : Tensor "f32[1, 1, 2, 64][128, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_8, [1, 1, -1, 64]), kwargs = {})
#   %permute_4 : Tensor "f32[1, 2, 1, 64][128, 64, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_9, [0, 2, 1, 3]), kwargs = {})
#   %mul_30 : Tensor "f32[1, 2, 1, 64][128, 64, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, %unsqueeze_11), kwargs = {})
#   %slice_5 : Tensor "f32[1, 2, 1, 32][128, 64, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_4, 3, 32, 9223372036854775807), kwargs = {})
#   %neg_1 : Tensor "f32[1, 2, 1, 32][64, 32, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_5,), kwargs = {})
#   %slice_4 : Tensor "f32[1, 2, 1, 32][128, 64, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_4, 3, 0, 32), kwargs = {})
#   %cat_1 : Tensor "f32[1, 2, 1, 64][128, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_4], -1), kwargs = {})
#   %mul_31 : Tensor "f32[1, 2, 1, 64][128, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %unsqueeze_12), kwargs = {})
#   %add_23 : Tensor "f32[1, 2, 1, 64][128, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %mul_31), kwargs = {})
#   %cat_2 : Tensor "f32[1, 2, s108 + 1, 64][128*s108 + 128, 64*s108 + 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%arg3_1, %add_23], -2), kwargs = {})
#   return %cat_2
triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1 = async_compile.triton('triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': 'fp64', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 51840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % ks0)
    x0 = (xindex % 64)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks1
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*x2 + 128*(x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = ks0
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + ((x3 % 32)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.broadcast_to(ks1, [XBLOCK])
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp14 = tl_math.cos(tmp13)
    tmp15 = in_ptr3
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp18 = tmp9 * tmp17
    tmp19 = x0
    tmp20 = tl.full([1], 0, tl.int64)
    tmp21 = tmp19 >= tmp20
    tmp22 = tl.full([1], 32, tl.int64)
    tmp23 = tmp19 < tmp22
    tmp24 = tmp23 & tmp6
    tmp25 = tl.load(in_ptr1 + (32 + 64*x2 + (x0)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = -tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp24, tmp26, tmp27)
    tmp29 = tmp19 >= tmp22
    tmp30 = tl.full([1], 64, tl.int64)
    tmp31 = tmp19 < tmp30
    tmp32 = tmp29 & tmp6
    tmp33 = tl.load(in_ptr1 + (64*x2 + ((-32) + x0)), tmp32 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.where(tmp23, tmp28, tmp33)
    tmp35 = tl_math.sin(tmp13)
    tmp36 = tmp35 * tmp16
    tmp37 = tmp34 * tmp36
    tmp38 = tmp18 + tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp6, tmp38, tmp39)
    tmp41 = tl.where(tmp4, tmp5, tmp40)
    tl.store(out_ptr0 + (x3), tmp41, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/3w/c3wir2pp22h2m6xv5dxfsgceluellbkikx44e2ebmouumirf5fbf.py
# Topologically Sorted Source Nodes: [linear_2, view_2, value_states, values], Original ATen: [aten.view, aten.transpose, aten.cat]
# Source node to ATen node mapping:
#   linear_2 => view_11
#   value_states => permute_6
#   values => cat_3
#   view_2 => view_12
# Graph fragment:
#   %arg15_1 : Tensor "f32[1, 2, s40, 64][128*s40, 64, 128, 1]cuda:0" = PlaceHolder[target=arg15_1]
#   %addmm_2 : Tensor "f32[1, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %view_11 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [1, 1, 128]), kwargs = {})
#   %view_12 : Tensor "f32[1, 1, 2, 64][128, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_11, [1, 1, -1, 64]), kwargs = {})
#   %permute_6 : Tensor "f32[1, 2, 1, 64][128, 64, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_12, [0, 2, 1, 3]), kwargs = {})
#   %cat_3 : Tensor "f32[1, 2, s40 + 1, 64][128*s40 + 128, 64*s40 + 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%arg15_1, %permute_6], -2), kwargs = {})
#   return %cat_3
triton_poi_fused_cat_transpose_view_2 = async_compile.triton('triton_poi_fused_cat_transpose_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_transpose_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 50688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_transpose_view_2(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 64) % ks0)
    x0 = (xindex % 64)
    x2 = xindex // ks2
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks1
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*x2 + 128*(x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = ks0
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 64*x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/2c/c2chpdz7l2ygaoyary4davf7ik5wnxu3oeyu7lxydjcev6c4vznm.py
# Topologically Sorted Source Nodes: [linear, view, query_states, getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_26, hidden_states_3, key, getitem_31, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   arange => iota
#   arange_3 => iota_3
#   arange_4 => iota_4
#   attention_mask => le_1
#   attention_mask_1 => expand
#   attn_output => _scaled_dot_product_efficient_attention, constant_pad_nd, expand_7, full_default, full_default_1, slice_8, where
#   cat_1 => cat
#   cos => cos
#   cos_3 => unsqueeze_11
#   emb => clone, expand_4, unsqueeze_10, view_3
#   expand_1 => expand_1
#   freqs => permute
#   getitem_16 => unsqueeze_7, unsqueeze_8
#   getitem_17 => unsqueeze_9
#   getitem_26 => unsqueeze_13
#   getitem_31 => unsqueeze_14
#   hidden_states_3 => expand_5
#   hidden_states_4 => expand_6
#   key => clone_2, view_13
#   kv_arange => add_6
#   kv_indices => unsqueeze_4, unsqueeze_5, unsqueeze_6
#   linear => view_5
#   matmul => mul_23
#   mul_4 => mul_28
#   mul_5 => mul_29
#   neg => neg
#   position_ids => add
#   position_ids_1 => unsqueeze
#   position_ids_expanded => convert_element_type
#   q_arange => add_2
#   q_embed => add_22
#   q_indices => unsqueeze_1, unsqueeze_2, unsqueeze_3
#   query_states => permute_2
#   sin => sin
#   sin_3 => unsqueeze_12
#   value => clone_3, view_14
#   view => view_6
#   x1 => slice_2
#   x2 => slice_3
# Graph fragment:
#   %addmm : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=addmm]
#   %arg4_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg5_1]
#   %view_5 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 1, 896]), kwargs = {})
#   %view_6 : Tensor "f32[1, 1, 14, 64][896, 896, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [1, 1, -1, 64]), kwargs = {})
#   %permute_2 : Tensor "f32[1, 14, 1, 64][896, 64, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_8, [1, -1, 1]), kwargs = {})
#   %iota : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, %arg333_1), kwargs = {})
#   %unsqueeze : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_9, torch.float32), kwargs = {})
#   %mul_23 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, 1, 32][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_23, [0, 2, 1]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 1, 32][32, 1, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 1, 2, 32][32, 1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_10, [1, 1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, 1, 2, 32][64, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, %convert_element_type_default_1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor, 1), kwargs = {})
#   %mul_28 : Tensor "f32[1, 14, 1, 64][896, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_11), kwargs = {})
#   %slice_3 : Tensor "f32[1, 14, 1, 32][896, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 32, 9223372036854775807), kwargs = {})
#   %neg : Tensor "f32[1, 14, 1, 32][448, 32, 448, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_3,), kwargs = {})
#   %slice_2 : Tensor "f32[1, 14, 1, 32][896, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 0, 32), kwargs = {})
#   %cat : Tensor "f32[1, 14, 1, 64][896, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_2], -1), kwargs = {})
#   %sin : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, %convert_element_type_default_2), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor_1, 1), kwargs = {})
#   %mul_29 : Tensor "f32[1, 14, 1, 64][896, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_12), kwargs = {})
#   %add_22 : Tensor "f32[1, 14, 1, 64][896, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %mul_29), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[1, 2, 1, s108 + 1, 64][128*s108 + 128, 64*s108 + 64, 64*s108 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_2, 2), kwargs = {})
#   %expand_5 : Tensor "f32[1, 2, 7, s108 + 1, 64][128*s108 + 128, 64*s108 + 64, 0, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_13, [1, 2, 7, %add_1, 64]), kwargs = {})
#   %clone_2 : Tensor "f32[1, 2, 7, s108 + 1, 64][896*s108 + 896, 448*s108 + 448, 64*s108 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_13 : Tensor "f32[1, 14, s108 + 1, 64][896*s108 + 896, 64*s108 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [1, 14, %add_1, 64]), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[1, 2, 1, s40 + 1, 64][128*s40 + 128, 64*s40 + 64, 64*s40 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_3, 2), kwargs = {})
#   %expand_6 : Tensor "f32[1, 2, 7, s40 + 1, 64][128*s40 + 128, 64*s40 + 64, 0, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_14, [1, 2, 7, %sym_sum, 64]), kwargs = {})
#   %clone_3 : Tensor "f32[1, 2, 7, s40 + 1, 64][896*s40 + 896, 448*s40 + 448, 64*s40 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_14 : Tensor "f32[1, 14, s40 + 1, 64][896*s40 + 896, 64*s40 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [1, 14, %sym_sum, 64]), kwargs = {})
#   %iota_4 : Tensor "i64[s108 + 1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%add_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_6 : Tensor "i64[s108 + 1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_4, 0), kwargs = {})
#   %unsqueeze_4 : Tensor "i64[1, s108 + 1][s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_6, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[1, 1, s108 + 1][s108 + 1, s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 1), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[1, 1, 1, s108 + 1][s108 + 1, s108 + 1, s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 2), kwargs = {})
#   %iota_3 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_2 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_3, %arg333_1), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_2, 0), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 1), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 3), kwargs = {})
#   %le_1 : Tensor "b8[1, 1, 1, s108 + 1][s108 + 1, s108 + 1, s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%unsqueeze_6, %unsqueeze_3), kwargs = {})
#   %expand : Tensor "b8[1, 1, 1, s108 + 1][s108 + 1, s108 + 1, s108 + 1, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.expand.default](args = (%le_1, [1, -1, 1, %add_1]), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[1, 1, 1, s108 + 1][s108 + 1, s108 + 1, s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_1, %full_default), kwargs = {})
#   %constant_pad_nd : Tensor "f32[1, 1, 1, s108 - (Mod(s108 + 1, 8)) + 9][Max(1, s108 - (Mod(s108 + 1, 8)) + 9), Max(1, s108 - (Mod(s108 + 1, 8)) + 9), Max(1, s108 - (Mod(s108 + 1, 8)) + 9), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, %sub_23], 0.0), kwargs = {})
#   %slice_8 : Tensor "f32[1, 1, 1, s108 + 1][Max(1, s108 - (Mod(s108 + 1, 8)) + 9), Max(1, s108 - (Mod(s108 + 1, 8)) + 9), Max(1, s108 - (Mod(s108 + 1, 8)) + 9), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, %add_1), kwargs = {})
#   %expand_7 : Tensor "f32[1, 14, 1, s108 + 1][Max(1, s108 - (Mod(s108 + 1, 8)) + 9), 0, Max(1, s108 - (Mod(s108 + 1, 8)) + 9), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_8, [1, 14, 1, %add_1]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%add_22, %view_13, %view_14, %expand_7, False), kwargs = {scale: 0.125})
#   return %buf7
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': 'fp64', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 18048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + ((x2 % 32)), xmask, eviction_policy='evict_last')
    tmp6 = in_ptr2
    tmp2 = ks0
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp5 = tl_math.cos(tmp4)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 * tmp7
    tmp9 = tmp0 * tmp8
    tmp10 = x0
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 32, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tl.load(in_ptr0 + (32 + 64*x1 + (x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = -tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp14, tmp16, tmp17)
    tmp19 = tmp10 >= tmp13
    tmp20 = tl.full([1], 64, tl.int64)
    tmp21 = tmp10 < tmp20
    tmp22 = tl.load(in_ptr0 + (64*x1 + ((-32) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.where(tmp14, tmp18, tmp22)
    tmp24 = tl_math.sin(tmp4)
    tmp25 = tmp24 * tmp7
    tmp26 = tmp23 * tmp25
    tmp27 = tmp9 + tmp26
    tl.store(out_ptr0 + (x2), tmp27, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/vl/cvleazfw7qpurfoyijdhcb4nt556yr4yhshithzxime3qlwk4wuo.py
# Topologically Sorted Source Nodes: [linear, view, query_states, getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_26, hidden_states_3, key, getitem_31, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   arange => iota
#   arange_3 => iota_3
#   arange_4 => iota_4
#   attention_mask => le_1
#   attention_mask_1 => expand
#   attn_output => _scaled_dot_product_efficient_attention, constant_pad_nd, expand_7, full_default, full_default_1, slice_8, where
#   cat_1 => cat
#   cos => cos
#   cos_3 => unsqueeze_11
#   emb => clone, expand_4, unsqueeze_10, view_3
#   expand_1 => expand_1
#   freqs => permute
#   getitem_16 => unsqueeze_7, unsqueeze_8
#   getitem_17 => unsqueeze_9
#   getitem_26 => unsqueeze_13
#   getitem_31 => unsqueeze_14
#   hidden_states_3 => expand_5
#   hidden_states_4 => expand_6
#   key => clone_2, view_13
#   kv_arange => add_6
#   kv_indices => unsqueeze_4, unsqueeze_5, unsqueeze_6
#   linear => view_5
#   matmul => mul_23
#   mul_4 => mul_28
#   mul_5 => mul_29
#   neg => neg
#   position_ids => add
#   position_ids_1 => unsqueeze
#   position_ids_expanded => convert_element_type
#   q_arange => add_2
#   q_embed => add_22
#   q_indices => unsqueeze_1, unsqueeze_2, unsqueeze_3
#   query_states => permute_2
#   sin => sin
#   sin_3 => unsqueeze_12
#   value => clone_3, view_14
#   view => view_6
#   x1 => slice_2
#   x2 => slice_3
# Graph fragment:
#   %cat_2 : Tensor "f32[1, 2, s108 + 1, 64][128*s108 + 128, 64*s108 + 64, 64, 1]cuda:0" = PlaceHolder[target=cat_2]
#   %view_5 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 1, 896]), kwargs = {})
#   %view_6 : Tensor "f32[1, 1, 14, 64][896, 896, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [1, 1, -1, 64]), kwargs = {})
#   %permute_2 : Tensor "f32[1, 14, 1, 64][896, 64, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_8, [1, -1, 1]), kwargs = {})
#   %iota : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, %arg333_1), kwargs = {})
#   %unsqueeze : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_9, torch.float32), kwargs = {})
#   %mul_23 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, 1, 32][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_23, [0, 2, 1]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 1, 32][32, 1, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 1, 2, 32][32, 1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_10, [1, 1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, 1, 2, 32][64, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, %convert_element_type_default_1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor, 1), kwargs = {})
#   %mul_28 : Tensor "f32[1, 14, 1, 64][896, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_11), kwargs = {})
#   %slice_3 : Tensor "f32[1, 14, 1, 32][896, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 32, 9223372036854775807), kwargs = {})
#   %neg : Tensor "f32[1, 14, 1, 32][448, 32, 448, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_3,), kwargs = {})
#   %slice_2 : Tensor "f32[1, 14, 1, 32][896, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 0, 32), kwargs = {})
#   %cat : Tensor "f32[1, 14, 1, 64][896, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_2], -1), kwargs = {})
#   %sin : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, %convert_element_type_default_2), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor_1, 1), kwargs = {})
#   %mul_29 : Tensor "f32[1, 14, 1, 64][896, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_12), kwargs = {})
#   %add_22 : Tensor "f32[1, 14, 1, 64][896, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %mul_29), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[1, 2, 1, s108 + 1, 64][128*s108 + 128, 64*s108 + 64, 64*s108 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_2, 2), kwargs = {})
#   %expand_5 : Tensor "f32[1, 2, 7, s108 + 1, 64][128*s108 + 128, 64*s108 + 64, 0, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_13, [1, 2, 7, %add_1, 64]), kwargs = {})
#   %clone_2 : Tensor "f32[1, 2, 7, s108 + 1, 64][896*s108 + 896, 448*s108 + 448, 64*s108 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_13 : Tensor "f32[1, 14, s108 + 1, 64][896*s108 + 896, 64*s108 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [1, 14, %add_1, 64]), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[1, 2, 1, s40 + 1, 64][128*s40 + 128, 64*s40 + 64, 64*s40 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_3, 2), kwargs = {})
#   %expand_6 : Tensor "f32[1, 2, 7, s40 + 1, 64][128*s40 + 128, 64*s40 + 64, 0, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_14, [1, 2, 7, %sym_sum, 64]), kwargs = {})
#   %clone_3 : Tensor "f32[1, 2, 7, s40 + 1, 64][896*s40 + 896, 448*s40 + 448, 64*s40 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_14 : Tensor "f32[1, 14, s40 + 1, 64][896*s40 + 896, 64*s40 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [1, 14, %sym_sum, 64]), kwargs = {})
#   %iota_4 : Tensor "i64[s108 + 1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%add_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_6 : Tensor "i64[s108 + 1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_4, 0), kwargs = {})
#   %unsqueeze_4 : Tensor "i64[1, s108 + 1][s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_6, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[1, 1, s108 + 1][s108 + 1, s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 1), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[1, 1, 1, s108 + 1][s108 + 1, s108 + 1, s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 2), kwargs = {})
#   %iota_3 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_2 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_3, %arg333_1), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_2, 0), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 1), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 3), kwargs = {})
#   %le_1 : Tensor "b8[1, 1, 1, s108 + 1][s108 + 1, s108 + 1, s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%unsqueeze_6, %unsqueeze_3), kwargs = {})
#   %expand : Tensor "b8[1, 1, 1, s108 + 1][s108 + 1, s108 + 1, s108 + 1, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.expand.default](args = (%le_1, [1, -1, 1, %add_1]), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[1, 1, 1, s108 + 1][s108 + 1, s108 + 1, s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_1, %full_default), kwargs = {})
#   %constant_pad_nd : Tensor "f32[1, 1, 1, s108 - (Mod(s108 + 1, 8)) + 9][Max(1, s108 - (Mod(s108 + 1, 8)) + 9), Max(1, s108 - (Mod(s108 + 1, 8)) + 9), Max(1, s108 - (Mod(s108 + 1, 8)) + 9), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, %sub_23], 0.0), kwargs = {})
#   %slice_8 : Tensor "f32[1, 1, 1, s108 + 1][Max(1, s108 - (Mod(s108 + 1, 8)) + 9), Max(1, s108 - (Mod(s108 + 1, 8)) + 9), Max(1, s108 - (Mod(s108 + 1, 8)) + 9), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, %add_1), kwargs = {})
#   %expand_7 : Tensor "f32[1, 14, 1, s108 + 1][Max(1, s108 - (Mod(s108 + 1, 8)) + 9), 0, Max(1, s108 - (Mod(s108 + 1, 8)) + 9), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_8, [1, 14, 1, %add_1]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%add_22, %view_13, %view_14, %expand_7, False), kwargs = {scale: 0.125})
#   return %buf8
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 253440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*(x1 // 7) + 64*ks1*(x1 // 7)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/ol/collbewnyoet2oh63gppdamrpvfh5eccdqr2f5p7md3sgdmkjicf.py
# Topologically Sorted Source Nodes: [linear, view, query_states, getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_26, hidden_states_3, key, getitem_31, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   arange => iota
#   arange_3 => iota_3
#   arange_4 => iota_4
#   attention_mask => le_1
#   attention_mask_1 => expand
#   attn_output => _scaled_dot_product_efficient_attention, constant_pad_nd, expand_7, full_default, full_default_1, slice_8, where
#   cat_1 => cat
#   cos => cos
#   cos_3 => unsqueeze_11
#   emb => clone, expand_4, unsqueeze_10, view_3
#   expand_1 => expand_1
#   freqs => permute
#   getitem_16 => unsqueeze_7, unsqueeze_8
#   getitem_17 => unsqueeze_9
#   getitem_26 => unsqueeze_13
#   getitem_31 => unsqueeze_14
#   hidden_states_3 => expand_5
#   hidden_states_4 => expand_6
#   key => clone_2, view_13
#   kv_arange => add_6
#   kv_indices => unsqueeze_4, unsqueeze_5, unsqueeze_6
#   linear => view_5
#   matmul => mul_23
#   mul_4 => mul_28
#   mul_5 => mul_29
#   neg => neg
#   position_ids => add
#   position_ids_1 => unsqueeze
#   position_ids_expanded => convert_element_type
#   q_arange => add_2
#   q_embed => add_22
#   q_indices => unsqueeze_1, unsqueeze_2, unsqueeze_3
#   query_states => permute_2
#   sin => sin
#   sin_3 => unsqueeze_12
#   value => clone_3, view_14
#   view => view_6
#   x1 => slice_2
#   x2 => slice_3
# Graph fragment:
#   %view_5 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 1, 896]), kwargs = {})
#   %view_6 : Tensor "f32[1, 1, 14, 64][896, 896, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [1, 1, -1, 64]), kwargs = {})
#   %permute_2 : Tensor "f32[1, 14, 1, 64][896, 64, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_8, [1, -1, 1]), kwargs = {})
#   %iota : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, %arg333_1), kwargs = {})
#   %unsqueeze : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_9, torch.float32), kwargs = {})
#   %mul_23 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, 1, 32][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_23, [0, 2, 1]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 1, 1, 32][32, 1, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 1, 2, 32][32, 1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_10, [1, 1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, 1, 2, 32][64, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, %convert_element_type_default_1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor, 1), kwargs = {})
#   %mul_28 : Tensor "f32[1, 14, 1, 64][896, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_11), kwargs = {})
#   %slice_3 : Tensor "f32[1, 14, 1, 32][896, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 32, 9223372036854775807), kwargs = {})
#   %neg : Tensor "f32[1, 14, 1, 32][448, 32, 448, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_3,), kwargs = {})
#   %slice_2 : Tensor "f32[1, 14, 1, 32][896, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 0, 32), kwargs = {})
#   %cat : Tensor "f32[1, 14, 1, 64][896, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_2], -1), kwargs = {})
#   %sin : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, %convert_element_type_default_2), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor_1, 1), kwargs = {})
#   %mul_29 : Tensor "f32[1, 14, 1, 64][896, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_12), kwargs = {})
#   %add_22 : Tensor "f32[1, 14, 1, 64][896, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %mul_29), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[1, 2, 1, s108 + 1, 64][128*s108 + 128, 64*s108 + 64, 64*s108 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_2, 2), kwargs = {})
#   %expand_5 : Tensor "f32[1, 2, 7, s108 + 1, 64][128*s108 + 128, 64*s108 + 64, 0, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_13, [1, 2, 7, %add_1, 64]), kwargs = {})
#   %clone_2 : Tensor "f32[1, 2, 7, s108 + 1, 64][896*s108 + 896, 448*s108 + 448, 64*s108 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_13 : Tensor "f32[1, 14, s108 + 1, 64][896*s108 + 896, 64*s108 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [1, 14, %add_1, 64]), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[1, 2, 1, s40 + 1, 64][128*s40 + 128, 64*s40 + 64, 64*s40 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_3, 2), kwargs = {})
#   %expand_6 : Tensor "f32[1, 2, 7, s40 + 1, 64][128*s40 + 128, 64*s40 + 64, 0, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_14, [1, 2, 7, %sym_sum, 64]), kwargs = {})
#   %clone_3 : Tensor "f32[1, 2, 7, s40 + 1, 64][896*s40 + 896, 448*s40 + 448, 64*s40 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_14 : Tensor "f32[1, 14, s40 + 1, 64][896*s40 + 896, 64*s40 + 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [1, 14, %sym_sum, 64]), kwargs = {})
#   %iota_4 : Tensor "i64[s108 + 1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%add_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_6 : Tensor "i64[s108 + 1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_4, 0), kwargs = {})
#   %unsqueeze_4 : Tensor "i64[1, s108 + 1][s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_6, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[1, 1, s108 + 1][s108 + 1, s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 1), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[1, 1, 1, s108 + 1][s108 + 1, s108 + 1, s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 2), kwargs = {})
#   %iota_3 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_2 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_3, %arg333_1), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_2, 0), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 1), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 3), kwargs = {})
#   %le_1 : Tensor "b8[1, 1, 1, s108 + 1][s108 + 1, s108 + 1, s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%unsqueeze_6, %unsqueeze_3), kwargs = {})
#   %expand : Tensor "b8[1, 1, 1, s108 + 1][s108 + 1, s108 + 1, s108 + 1, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.expand.default](args = (%le_1, [1, -1, 1, %add_1]), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[1, 1, 1, s108 + 1][s108 + 1, s108 + 1, s108 + 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_1, %full_default), kwargs = {})
#   %constant_pad_nd : Tensor "f32[1, 1, 1, s108 - (Mod(s108 + 1, 8)) + 9][Max(1, s108 - (Mod(s108 + 1, 8)) + 9), Max(1, s108 - (Mod(s108 + 1, 8)) + 9), Max(1, s108 - (Mod(s108 + 1, 8)) + 9), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, %sub_23], 0.0), kwargs = {})
#   %slice_8 : Tensor "f32[1, 1, 1, s108 + 1][Max(1, s108 - (Mod(s108 + 1, 8)) + 9), Max(1, s108 - (Mod(s108 + 1, 8)) + 9), Max(1, s108 - (Mod(s108 + 1, 8)) + 9), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, %add_1), kwargs = {})
#   %expand_7 : Tensor "f32[1, 14, 1, s108 + 1][Max(1, s108 - (Mod(s108 + 1, 8)) + 9), 0, Max(1, s108 - (Mod(s108 + 1, 8)) + 9), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_8, [1, 14, 1, %add_1]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%add_22, %view_13, %view_14, %expand_7, False), kwargs = {scale: 0.125})
#   return %buf10
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64},
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 264}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5(out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = ks0
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tl.broadcast_to(ks1, [XBLOCK])
    tmp5 = tmp3 <= tmp4
    tmp6 = tl.full([1], 0.0, tl.float32)
    tmp7 = tl.full([1], float("-inf"), tl.float32)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp2, tmp8, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/7v/c7vmvdpeir4wd4yr34ugxwcsjbcrduk6ztkiohppion56x64im4c.py
# Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, pow_2, variance_1, rsqrt_1, hidden_states_7, hidden_states_8], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   attn_output_3 => view_17
#   hidden_states_5 => add_88
#   hidden_states_7 => mul_145
#   hidden_states_8 => mul_146
#   inputs_embeds => embedding
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
#   variance_1 => mean_1
# Graph fragment:
#   %arg0_1 : Tensor "i64[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[151936, 896][896, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mm : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg18_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %buf17 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf17]
#   %arg17_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg17_1]
#   %embedding : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %view_17 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 1, 896]), kwargs = {})
#   %add_88 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_17), kwargs = {})
#   %pow_2 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_88, 2), kwargs = {})
#   %mean_1 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg17_1, torch.float32), kwargs = {})
#   %add_tensor_1 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, %convert_element_type_default_6), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_145 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_88, %rsqrt_1), kwargs = {})
#   %mul_146 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg18_1, %mul_145), kwargs = {})
#   return %buf17,%mul_146
triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_6 = async_compile.triton('triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 14336}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 896
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [1, 1])
    tmp8 = tl.load(in_ptr2 + (r0_0), r0_mask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (r0_0), r0_mask, other=0.0)
    tmp18 = in_ptr4
    tmp2 = tl.full([1, 1], 151936, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 151936), "index out of bounds: 0 <= tmp5 < 151936")
    tmp7 = tl.load(in_ptr1 + (r0_0 + 896*tmp5), r0_mask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([1, 1], 896.0, tl.float32)
    tmp17 = (tmp14 / tmp16)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp9 * tmp21
    tmp23 = tmp15 * tmp22
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp23, r0_mask)
''', device_str='cuda')


# kernel path: <inductor-cache>/bk/cbktqikvuna6b2wodggcwqewtlrlwj24upo7mmj6bdg4l7qaxrfw.py
# Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_10], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   linear_4 => view_19
#   linear_5 => view_21
#   mul_10 => mul_147
#   silu => add_90, div, exp, neg_2
# Graph fragment:
#   %mm_1 : Tensor "f32[1, 4864][4864, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %mm_2 : Tensor "f32[1, 4864][4864, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %view_19 : Tensor "f32[1, 1, 4864][4864, 4864, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [1, 1, 4864]), kwargs = {})
#   %neg_2 : Tensor "f32[1, 1, 4864][4864, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%view_19,), kwargs = {})
#   %exp : Tensor "f32[1, 1, 4864][4864, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %add_90 : Tensor "f32[1, 1, 4864][4864, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[1, 1, 4864][4864, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_19, %add_90), kwargs = {})
#   %view_21 : Tensor "f32[1, 1, 4864][4864, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [1, 1, 4864]), kwargs = {})
#   %mul_147 : Tensor "f32[1, 1, 4864][4864, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %view_21), kwargs = {})
#   return %mul_147
triton_poi_fused__unsafe_view_mul_silu_7 = async_compile.triton('triton_poi_fused__unsafe_view_mul_silu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_mul_silu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 77824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_silu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp6 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = -tmp0
    tmp2 = libdevice.exp(tmp1)
    tmp3 = tl.full([1], 1.0, tl.float32)
    tmp4 = tmp2 + tmp3
    tmp5 = (tmp0 / tmp4)
    tmp7 = tmp5 * tmp6
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/az/caz7k7cztrufjoqvmjvapvebqdvpdpnf4cl2g3nkku3ljqlvqygz.py
# Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, pow_3, variance_2, rsqrt_2, hidden_states_11, hidden_states_12], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   attn_output_3 => view_17
#   down_proj => view_23
#   hidden_states_11 => mul_148
#   hidden_states_12 => mul_149
#   hidden_states_5 => add_88
#   hidden_states_9 => add_91
#   inputs_embeds => embedding
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
#   variance_2 => mean_2
# Graph fragment:
#   %arg0_1 : Tensor "i64[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[151936, 896][896, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mm : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm]
#   %mm_3 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %arg23_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg23_1]
#   %buf23 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf23]
#   %arg22_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg22_1]
#   %embedding : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %view_17 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 1, 896]), kwargs = {})
#   %add_88 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_17), kwargs = {})
#   %view_23 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, 1, 896]), kwargs = {})
#   %add_91 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_88, %view_23), kwargs = {})
#   %pow_3 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_91, 2), kwargs = {})
#   %mean_2 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %convert_element_type_default_8 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg22_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, %convert_element_type_default_8), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_2,), kwargs = {})
#   %mul_148 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_91, %rsqrt_2), kwargs = {})
#   %mul_149 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg23_1, %mul_148), kwargs = {})
#   return %buf23,%mul_149
triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8 = async_compile.triton('triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 17920}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 896
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [1, 1])
    tmp8 = tl.load(in_ptr2 + (r0_0), r0_mask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r0_0), r0_mask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (r0_0), r0_mask, other=0.0)
    tmp20 = in_ptr5
    tmp2 = tl.full([1, 1], 151936, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 151936), "index out of bounds: 0 <= tmp5 < 151936")
    tmp7 = tl.load(in_ptr1 + (r0_0 + 896*tmp5), r0_mask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp18 = tl.full([1, 1], 896.0, tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp11 * tmp23
    tmp25 = tmp17 * tmp24
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: <inductor-cache>/7m/c7mgnxaqnx7j73c7225vk6wyb6rgzrgwpbwnpvuerkqf2364tqsk.py
# Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, attn_output_7, hidden_states_15, pow_4, variance_3, rsqrt_3, hidden_states_17, hidden_states_18], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   attn_output_3 => view_17
#   attn_output_7 => view_37
#   down_proj => view_23
#   hidden_states_15 => add_159
#   hidden_states_17 => mul_267
#   hidden_states_18 => mul_268
#   hidden_states_5 => add_88
#   hidden_states_9 => add_91
#   inputs_embeds => embedding
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
#   variance_3 => mean_3
# Graph fragment:
#   %arg0_1 : Tensor "i64[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[151936, 896][896, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mm : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm]
#   %mm_3 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %mm_4 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %add_159 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0" = PlaceHolder[target=add_159]
#   %arg35_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %buf41 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf41]
#   %arg34_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg34_1]
#   %embedding : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %view_17 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 1, 896]), kwargs = {})
#   %add_88 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_17), kwargs = {})
#   %view_23 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, 1, 896]), kwargs = {})
#   %add_91 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_88, %view_23), kwargs = {})
#   %view_37 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [1, 1, 896]), kwargs = {})
#   %add_159 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, %view_37), kwargs = {})
#   %pow_4 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_159, 2), kwargs = {})
#   %mean_3 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %convert_element_type_default_10 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg34_1, torch.float32), kwargs = {})
#   %add_tensor_3 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, %convert_element_type_default_10), kwargs = {})
#   %rsqrt_3 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_267 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_159, %rsqrt_3), kwargs = {})
#   %mul_268 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg35_1, %mul_267), kwargs = {})
#   return %add_159,%buf41,%mul_268
triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 28672}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 896
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [1, 1])
    tmp8 = tl.load(in_out_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r0_0), r0_mask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (r0_0), r0_mask, other=0.0)
    tmp19 = tl.load(in_ptr4 + (r0_0), r0_mask, other=0.0)
    tmp22 = in_ptr5
    tmp2 = tl.full([1, 1], 151936, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 151936), "index out of bounds: 0 <= tmp5 < 151936")
    tmp7 = tl.load(in_ptr1 + (r0_0 + 896*tmp5), r0_mask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp20 = tl.full([1, 1], 896.0, tl.float32)
    tmp21 = (tmp18 / tmp20)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = tmp13 * tmp25
    tmp27 = tmp19 * tmp26
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp13, r0_mask)
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp27, r0_mask)
''', device_str='cuda')


# kernel path: <inductor-cache>/js/cjsybibgfmjb4ydtk5xe3jnkc6php7q6dictyi7hdj4l6r2hya5c.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, pow_5, variance_4, rsqrt_4, hidden_states_21, hidden_states_22], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   down_proj_1 => view_43
#   hidden_states_19 => add_162
#   hidden_states_21 => mul_270
#   hidden_states_22 => mul_271
#   pow_5 => pow_5
#   rsqrt_4 => rsqrt_4
#   variance_4 => mean_4
# Graph fragment:
#   %add_159 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0" = PlaceHolder[target=add_159]
#   %mm_7 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_7]
#   %arg40_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg40_1]
#   %buf47 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf47]
#   %arg39_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg39_1]
#   %view_43 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [1, 1, 896]), kwargs = {})
#   %add_162 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_159, %view_43), kwargs = {})
#   %pow_5 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_162, 2), kwargs = {})
#   %mean_4 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [-1], True), kwargs = {})
#   %convert_element_type_default_12 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg39_1, torch.float32), kwargs = {})
#   %add_tensor_4 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, %convert_element_type_default_12), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_4,), kwargs = {})
#   %mul_270 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_162, %rsqrt_4), kwargs = {})
#   %mul_271 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg40_1, %mul_270), kwargs = {})
#   return %buf47,%mul_271
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 17920}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 896
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_0), r0_mask, other=0.0)
    tmp11 = in_ptr3
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([1, 1], 896.0, tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp2 * tmp14
    tmp16 = tmp8 * tmp15
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp16, r0_mask)
''', device_str='cuda')


# kernel path: <inductor-cache>/xo/cxotrkciubzdbwt7o4fx7evq3bknupvwvgknjksugnuq32v5cozh.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, pow_6, variance_5, rsqrt_5, hidden_states_27, hidden_states_28], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   attn_output_11 => view_57
#   down_proj_1 => view_43
#   hidden_states_19 => add_162
#   hidden_states_25 => add_230
#   hidden_states_27 => mul_389
#   hidden_states_28 => mul_390
#   pow_6 => pow_6
#   rsqrt_5 => rsqrt_5
#   variance_5 => mean_5
# Graph fragment:
#   %add_159 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0" = PlaceHolder[target=add_159]
#   %mm_7 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_7]
#   %mm_8 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %arg52_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg52_1]
#   %buf64 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf64]
#   %arg51_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg51_1]
#   %view_43 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [1, 1, 896]), kwargs = {})
#   %add_162 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_159, %view_43), kwargs = {})
#   %view_57 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [1, 1, 896]), kwargs = {})
#   %add_230 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_162, %view_57), kwargs = {})
#   %pow_6 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_230, 2), kwargs = {})
#   %mean_5 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_6, [-1], True), kwargs = {})
#   %convert_element_type_default_14 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg51_1, torch.float32), kwargs = {})
#   %add_tensor_5 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, %convert_element_type_default_14), kwargs = {})
#   %rsqrt_5 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_5,), kwargs = {})
#   %mul_389 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_230, %rsqrt_5), kwargs = {})
#   %mul_390 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg52_1, %mul_389), kwargs = {})
#   return %buf64,%mul_390
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 21504}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 896
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_0), r0_mask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r0_0), r0_mask, other=0.0)
    tmp13 = in_ptr4
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp11 = tl.full([1, 1], 896.0, tl.float32)
    tmp12 = (tmp9 / tmp11)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp4 * tmp16
    tmp18 = tmp10 * tmp17
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp18, r0_mask)
''', device_str='cuda')


# kernel path: <inductor-cache>/2u/c2uen5pohuwkyysjefixtrmevulfk7okqniiq4dicfxir4vibz5z.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, pow_7, variance_6, rsqrt_6, hidden_states_31, hidden_states_32], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   attn_output_11 => view_57
#   down_proj_1 => view_43
#   down_proj_2 => view_63
#   hidden_states_19 => add_162
#   hidden_states_25 => add_230
#   hidden_states_29 => add_233
#   hidden_states_31 => mul_392
#   hidden_states_32 => mul_393
#   pow_7 => pow_7
#   rsqrt_6 => rsqrt_6
#   variance_6 => mean_6
# Graph fragment:
#   %add_159 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0" = PlaceHolder[target=add_159]
#   %mm_7 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_7]
#   %mm_8 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %mm_11 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_11]
#   %arg57_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg57_1]
#   %buf70 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf70]
#   %arg56_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg56_1]
#   %view_43 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [1, 1, 896]), kwargs = {})
#   %add_162 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_159, %view_43), kwargs = {})
#   %view_57 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [1, 1, 896]), kwargs = {})
#   %add_230 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_162, %view_57), kwargs = {})
#   %view_63 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_11, [1, 1, 896]), kwargs = {})
#   %add_233 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_230, %view_63), kwargs = {})
#   %pow_7 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_233, 2), kwargs = {})
#   %mean_6 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [-1], True), kwargs = {})
#   %convert_element_type_default_16 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg56_1, torch.float32), kwargs = {})
#   %add_tensor_6 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_6, %convert_element_type_default_16), kwargs = {})
#   %rsqrt_6 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_6,), kwargs = {})
#   %mul_392 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_233, %rsqrt_6), kwargs = {})
#   %mul_393 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg57_1, %mul_392), kwargs = {})
#   return %buf70,%mul_393
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 25088}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 896
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_0), r0_mask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r0_0), r0_mask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (r0_0), r0_mask, other=0.0)
    tmp15 = in_ptr5
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp13 = tl.full([1, 1], 896.0, tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp6 * tmp18
    tmp20 = tmp12 * tmp19
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp20, r0_mask)
''', device_str='cuda')


# kernel path: <inductor-cache>/xu/cxuaf2yabpletu5s45vatgrhzyvcjos7cppjerresgcxje4ig3x3.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, attn_output_15, hidden_states_35, pow_8, variance_7, rsqrt_7, hidden_states_37, hidden_states_38], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   attn_output_11 => view_57
#   attn_output_15 => view_77
#   down_proj_1 => view_43
#   down_proj_2 => view_63
#   hidden_states_19 => add_162
#   hidden_states_25 => add_230
#   hidden_states_29 => add_233
#   hidden_states_35 => add_301
#   hidden_states_37 => mul_511
#   hidden_states_38 => mul_512
#   pow_8 => pow_8
#   rsqrt_7 => rsqrt_7
#   variance_7 => mean_7
# Graph fragment:
#   %add_159 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0" = PlaceHolder[target=add_159]
#   %mm_7 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_7]
#   %mm_8 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %mm_11 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_11]
#   %mm_12 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_12]
#   %add_301 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0" = PlaceHolder[target=add_301]
#   %arg69_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg69_1]
#   %buf88 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf88]
#   %arg68_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg68_1]
#   %view_43 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [1, 1, 896]), kwargs = {})
#   %add_162 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_159, %view_43), kwargs = {})
#   %view_57 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [1, 1, 896]), kwargs = {})
#   %add_230 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_162, %view_57), kwargs = {})
#   %view_63 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_11, [1, 1, 896]), kwargs = {})
#   %add_233 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_230, %view_63), kwargs = {})
#   %view_77 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_12, [1, 1, 896]), kwargs = {})
#   %add_301 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_233, %view_77), kwargs = {})
#   %pow_8 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_301, 2), kwargs = {})
#   %mean_7 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [-1], True), kwargs = {})
#   %convert_element_type_default_18 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg68_1, torch.float32), kwargs = {})
#   %add_tensor_7 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_7, %convert_element_type_default_18), kwargs = {})
#   %rsqrt_7 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_7,), kwargs = {})
#   %mul_511 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_301, %rsqrt_7), kwargs = {})
#   %mul_512 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg69_1, %mul_511), kwargs = {})
#   return %add_301,%buf88,%mul_512
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 35840}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 896
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
    r0_0 = r0_index
    tmp0 = tl.load(in_out_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_0), r0_mask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_0), r0_mask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r0_0), r0_mask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (r0_0), r0_mask, other=0.0)
    tmp17 = in_ptr5
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp15 = tl.full([1, 1], 896.0, tl.float32)
    tmp16 = (tmp13 / tmp15)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp8 * tmp20
    tmp22 = tmp14 * tmp21
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp8, r0_mask)
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp22, r0_mask)
''', device_str='cuda')


# kernel path: <inductor-cache>/dv/cdvj3kxaduryyryysvhxy6b2rkqz5ubn4636mx7m677cefnwrk2a.py
# Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, down_proj_8, hidden_states_89, attn_output_39, hidden_states_95, pow_20, variance_19, add_62, rsqrt_19, hidden_states_97, hidden_states_98], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_62 => add_728
#   attn_output_35 => view_177
#   attn_output_39 => view_197
#   down_proj_7 => view_163
#   down_proj_8 => view_183
#   hidden_states_79 => add_588
#   hidden_states_85 => add_656
#   hidden_states_89 => add_659
#   hidden_states_95 => add_727
#   hidden_states_97 => mul_1243
#   hidden_states_98 => mul_1244
#   pow_20 => pow_20
#   rsqrt_19 => rsqrt_19
#   variance_19 => mean_19
# Graph fragment:
#   %add_585 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0" = PlaceHolder[target=add_585]
#   %mm_31 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_31]
#   %mm_32 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_32]
#   %mm_35 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_35]
#   %mm_36 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_36]
#   %add_727 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0" = PlaceHolder[target=add_727]
#   %arg170_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg170_1]
#   %buf229 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf229]
#   %view_163 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_31, [1, 1, 896]), kwargs = {})
#   %add_588 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_585, %view_163), kwargs = {})
#   %view_177 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_32, [1, 1, 896]), kwargs = {})
#   %add_656 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_588, %view_177), kwargs = {})
#   %view_183 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_35, [1, 1, 896]), kwargs = {})
#   %add_659 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_656, %view_183), kwargs = {})
#   %view_197 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_36, [1, 1, 896]), kwargs = {})
#   %add_727 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_659, %view_197), kwargs = {})
#   %pow_20 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_727, 2), kwargs = {})
#   %mean_19 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_20, [-1], True), kwargs = {})
#   %add_728 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_19, 1e-06), kwargs = {})
#   %rsqrt_19 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_728,), kwargs = {})
#   %mul_1243 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_727, %rsqrt_19), kwargs = {})
#   %mul_1244 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg170_1, %mul_1243), kwargs = {})
#   return %add_727,%buf229,%mul_1244
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_14 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 35840}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 896
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
    r0_0 = r0_index
    tmp0 = tl.load(in_out_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_0), r0_mask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_0), r0_mask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r0_0), r0_mask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (r0_0), r0_mask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp15 = tl.full([1, 1], 896.0, tl.float32)
    tmp16 = (tmp13 / tmp15)
    tmp17 = tl.full([1, 1], 1e-06, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp8 * tmp19
    tmp21 = tmp14 * tmp20
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp8, r0_mask)
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp21, r0_mask)
''', device_str='cuda')


# kernel path: <inductor-cache>/bq/cbq5xze5i4wcv7b2imibfk5mfd2fktxika24dhenvn3ky7ozyhrv.py
# Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, pow_22, variance_21, add_68, rsqrt_21, hidden_states_107, hidden_states_108], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_68 => add_799
#   attn_output_43 => view_217
#   down_proj_9 => view_203
#   hidden_states_105 => add_798
#   hidden_states_107 => mul_1365
#   hidden_states_108 => mul_1366
#   hidden_states_99 => add_730
#   pow_22 => pow_22
#   rsqrt_21 => rsqrt_21
#   variance_21 => mean_21
# Graph fragment:
#   %add_727 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0" = PlaceHolder[target=add_727]
#   %mm_39 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_39]
#   %mm_40 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_40]
#   %arg186_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg186_1]
#   %buf252 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf252]
#   %view_203 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_39, [1, 1, 896]), kwargs = {})
#   %add_730 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_727, %view_203), kwargs = {})
#   %view_217 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_40, [1, 1, 896]), kwargs = {})
#   %add_798 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_730, %view_217), kwargs = {})
#   %pow_22 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_798, 2), kwargs = {})
#   %mean_21 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_22, [-1], True), kwargs = {})
#   %add_799 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_21, 1e-06), kwargs = {})
#   %rsqrt_21 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_799,), kwargs = {})
#   %mul_1365 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_798, %rsqrt_21), kwargs = {})
#   %mul_1366 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg186_1, %mul_1365), kwargs = {})
#   return %buf252,%mul_1366
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_15 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 21504}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 896
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_0), r0_mask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r0_0), r0_mask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp11 = tl.full([1, 1], 896.0, tl.float32)
    tmp12 = (tmp9 / tmp11)
    tmp13 = tl.full([1, 1], 1e-06, tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp4 * tmp15
    tmp17 = tmp10 * tmp16
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp17, r0_mask)
''', device_str='cuda')


# kernel path: <inductor-cache>/in/cindl2ayh3c76nxstsnalldljzc353eplj6s7end2phd5ja7q2pb.py
# Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, pow_29, variance_28, add_88, rsqrt_28, hidden_states_141, hidden_states_142], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_88 => add_1015
#   down_proj_13 => view_283
#   hidden_states_139 => add_1014
#   hidden_states_141 => mul_1734
#   hidden_states_142 => mul_1735
#   pow_29 => pow_29
#   rsqrt_28 => rsqrt_28
#   variance_28 => mean_28
# Graph fragment:
#   %add_1011 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0" = PlaceHolder[target=add_1011]
#   %mm_55 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_55]
#   %arg241_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg241_1]
#   %buf329 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf329]
#   %view_283 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_55, [1, 1, 896]), kwargs = {})
#   %add_1014 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1011, %view_283), kwargs = {})
#   %pow_29 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1014, 2), kwargs = {})
#   %mean_28 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_29, [-1], True), kwargs = {})
#   %add_1015 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_28, 1e-06), kwargs = {})
#   %rsqrt_28 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1015,), kwargs = {})
#   %mul_1734 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1014, %rsqrt_28), kwargs = {})
#   %mul_1735 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg241_1, %mul_1734), kwargs = {})
#   return %buf329,%mul_1735
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_16 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 17920}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_16(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 896
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_0), r0_mask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([1, 1], 896.0, tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tl.full([1, 1], 1e-06, tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp2 * tmp13
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp15, r0_mask)
''', device_str='cuda')


# kernel path: <inductor-cache>/ec/cecamdmmweh7sdelsqphys36mkp4uedk4iltvwmcjsm4pj66ofjt.py
# Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, pow_49, variance_48, rsqrt_48, hidden_states_241, hidden_states_242], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   down_proj_23 => view_483
#   hidden_states_239 => add_1724
#   hidden_states_241 => mul_2954
#   hidden_states_242 => mul_2955
#   pow_49 => pow_49
#   rsqrt_48 => rsqrt_48
#   variance_48 => mean_48
# Graph fragment:
#   %add_1721 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0" = PlaceHolder[target=add_1721]
#   %mm_95 : Tensor "f32[1, 896][896, 1]cuda:0" = PlaceHolder[target=mm_95]
#   %arg412_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg412_1]
#   %buf564 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf564]
#   %arg411_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg411_1]
#   %view_483 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_95, [1, 1, 896]), kwargs = {})
#   %add_1724 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1721, %view_483), kwargs = {})
#   %pow_49 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1724, 2), kwargs = {})
#   %mean_48 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_49, [-1], True), kwargs = {})
#   %convert_element_type_default_94 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg411_1, torch.float32), kwargs = {})
#   %add_tensor_45 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_48, %convert_element_type_default_94), kwargs = {})
#   %rsqrt_48 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_45,), kwargs = {})
#   %mul_2954 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1724, %rsqrt_48), kwargs = {})
#   %mul_2955 : Tensor "f32[1, 1, 896][896, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg412_1, %mul_2954), kwargs = {})
#   return %buf564,%mul_2955
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_17 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': 'fp64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 17920}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 896
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
    r0_0 = r0_index
    tmp0 = tl.load(in_out_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr1 + (r0_0), r0_mask, other=0.0)
    tmp11 = in_ptr2
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([1, 1], 896.0, tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp2 * tmp14
    tmp16 = tmp8 * tmp15
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp16, r0_mask)
''', device_str='cuda')


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1 = args
        args.clear()
        s53 = arg2_1
        s40 = arg14_1
        s6 = arg31_1
        s87 = arg48_1
        s19 = arg65_1
        s24 = arg82_1
        s98 = arg99_1
        s80 = arg116_1
        s2 = arg133_1
        s34 = arg150_1
        s4 = arg167_1
        s17 = arg183_1
        s31 = arg199_1
        s65 = arg216_1
        s45 = arg233_1
        s100 = arg249_1
        s102 = arg266_1
        s58 = arg283_1
        s75 = arg300_1
        s77 = arg317_1
        s108 = arg333_1
        s109 = arg335_1
        s9 = arg352_1
        s39 = arg369_1
        s117 = arg386_1
        s121 = arg403_1
        s125 = arg413_1
        assert_size_stride(arg0_1, (1, 1), (1, 1))
        assert_size_stride(arg1_1, (151936, 896), (896, 1))
        assert_size_stride(arg7_1, (896, ), (1, ))
        assert_size_stride(arg6_1, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_misaligned(arg0_1)
            buf1 = empty_strided_cuda((1, 1, 896), (896, 896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [inputs_embeds, pow_1, variance, rsqrt, hidden_states_1, hidden_states_2], Original ATen: [aten.embedding, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_embedding_mean_mul_pow_rsqrt_0.run(arg0_1, arg1_1, arg7_1, arg6_1.item(), buf1, 1, 896, stream=raw_stream0)
            del arg6_1
            del arg7_1
            assert_size_stride(arg9_1, (896, ), (1, ))
            assert_size_stride(arg8_1, (896, 896), (896, 1))
            buf2 = empty_strided_cuda((1, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg9_1, reinterpret_tensor(buf1, (1, 896), (896, 1), 0), reinterpret_tensor(arg8_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf2)
            del arg8_1
            del arg9_1
            assert_size_stride(arg11_1, (128, ), (1, ))
            assert_size_stride(arg10_1, (128, 896), (896, 1))
            buf3 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg11_1, reinterpret_tensor(buf1, (1, 896), (896, 1), 0), reinterpret_tensor(arg10_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf3)
            del arg10_1
            del arg11_1
            assert_size_stride(arg3_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            assert_size_stride(arg4_1, (32, ), (1, ))
            assert_size_stride(arg5_1, (), ())
            arg3_1 = copy_misaligned(arg3_1)
            ps0 = 1 + s108
            ps1 = 64 + 64*s108
            buf4 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, cos_3, sin, sin_3, linear_1, view_1, key_states, mul_6, x2_1, neg_1, x1_1, cat_2, mul_7, k_embed, keys], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg3_1, buf3, arg4_1, arg5_1.item(), buf4, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg3_1
            assert_size_stride(arg13_1, (128, ), (1, ))
            assert_size_stride(arg12_1, (128, 896), (896, 1))
            buf5 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg13_1, reinterpret_tensor(buf1, (1, 896), (896, 1), 0), reinterpret_tensor(arg12_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf5)
            del arg12_1
            del arg13_1
            assert_size_stride(arg15_1, (1, 2, s40, 64), (128*s40, 64, 128, 1))
            arg15_1 = copy_misaligned(arg15_1)
            ps2 = 1 + s40
            ps3 = 64 + 64*s40
            buf6 = empty_strided_cuda((1, 2, 1 + s40, 64), (128 + 128*s40, 64 + 64*s40, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2, view_2, value_states, values], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s40
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg15_1, buf5, buf6, ps2, s40, ps3, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg15_1
            buf7 = reinterpret_tensor(buf1, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf1  # reuse
            # Topologically Sorted Source Nodes: [linear, view, query_states, getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_26, hidden_states_3, key, getitem_31, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf2, arg4_1, arg5_1.item(), buf7, s108, 896, stream=raw_stream0)
            buf8 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, view, query_states, getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_26, hidden_states_3, key, getitem_31, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf4, buf8, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf9 = empty_strided_cuda((1, 14, 1 + s40, 64), (896 + 896*s40, 64 + 64*s40, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, view, query_states, getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_26, hidden_states_3, key, getitem_31, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s40
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf6, buf9, ps3, s40, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf10 = empty_strided_cuda((1, 1, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, view, query_states, getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_26, hidden_states_3, key, getitem_31, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf10, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [linear, view, query_states, getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_26, hidden_states_3, key, getitem_31, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf7, buf8, buf9, reinterpret_tensor(buf10, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf8
            del buf9
            buf12 = buf11[0]
            assert_size_stride(buf12, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf12, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf11
            assert_size_stride(arg16_1, (896, 896), (896, 1))
            buf16 = reinterpret_tensor(buf7, (1, 896), (896, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [transpose_4, reshape_2, attn_output_3], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf12, (1, 896), (896, 1), 0), reinterpret_tensor(arg16_1, (896, 896), (1, 896), 0), out=buf16)
            del arg16_1
            assert_size_stride(arg18_1, (896, ), (1, ))
            assert_size_stride(arg17_1, (), ())
            buf18 = reinterpret_tensor(buf12, (1, 1, 896), (896, 896, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, pow_2, variance_1, rsqrt_1, hidden_states_7, hidden_states_8], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_6.run(arg0_1, arg1_1, buf16, arg18_1, arg17_1.item(), buf18, 1, 896, stream=raw_stream0)
            del arg17_1
            del arg18_1
            assert_size_stride(arg19_1, (4864, 896), (896, 1))
            buf19 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf18, (1, 896), (896, 1), 0), reinterpret_tensor(arg19_1, (896, 4864), (1, 896), 0), out=buf19)
            del arg19_1
            assert_size_stride(arg20_1, (4864, 896), (896, 1))
            buf20 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf18, (1, 896), (896, 1), 0), reinterpret_tensor(arg20_1, (896, 4864), (1, 896), 0), out=buf20)
            del arg20_1
            buf21 = reinterpret_tensor(buf19, (1, 1, 4864), (4864, 4864, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_10], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf21, buf20, 4864, stream=raw_stream0)
            del buf20
            assert_size_stride(arg21_1, (896, 4864), (4864, 1))
            buf22 = reinterpret_tensor(buf18, (1, 896), (896, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_10, down_proj], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf21, (1, 4864), (0, 1), 0), reinterpret_tensor(arg21_1, (4864, 896), (1, 4864), 0), out=buf22)
            del arg21_1
            del buf21
            assert_size_stride(arg23_1, (896, ), (1, ))
            assert_size_stride(arg22_1, (), ())
            buf24 = reinterpret_tensor(buf2, (1, 1, 896), (896, 896, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, pow_3, variance_2, rsqrt_2, hidden_states_11, hidden_states_12], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8.run(arg0_1, arg1_1, buf16, buf22, arg23_1, arg22_1.item(), buf24, 1, 896, stream=raw_stream0)
            del arg22_1
            del arg23_1
            assert_size_stride(arg25_1, (896, ), (1, ))
            assert_size_stride(arg24_1, (896, 896), (896, 1))
            buf25 = empty_strided_cuda((1, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg25_1, reinterpret_tensor(buf24, (1, 896), (896, 1), 0), reinterpret_tensor(arg24_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf25)
            del arg24_1
            del arg25_1
            assert_size_stride(arg27_1, (128, ), (1, ))
            assert_size_stride(arg26_1, (128, 896), (896, 1))
            buf26 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg27_1, reinterpret_tensor(buf24, (1, 896), (896, 1), 0), reinterpret_tensor(arg26_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf26)
            del arg26_1
            del arg27_1
            assert_size_stride(arg30_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg30_1 = copy_misaligned(arg30_1)
            buf27 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_4, sin_4, linear_8, view_4, key_states_1, mul_15, x2_3, neg_3, x1_3, cat_6, mul_16, k_embed_1, keys_1], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg30_1, buf26, arg4_1, arg5_1.item(), buf27, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg30_1
            assert_size_stride(arg29_1, (128, ), (1, ))
            assert_size_stride(arg28_1, (128, 896), (896, 1))
            buf28 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg29_1, reinterpret_tensor(buf24, (1, 896), (896, 1), 0), reinterpret_tensor(arg28_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf28)
            del arg28_1
            del arg29_1
            assert_size_stride(arg32_1, (1, 2, s6, 64), (128*s6, 64, 128, 1))
            arg32_1 = copy_misaligned(arg32_1)
            ps4 = 1 + s6
            ps5 = 64 + 64*s6
            buf29 = empty_strided_cuda((1, 2, 1 + s6, 64), (128 + 128*s6, 64 + 64*s6, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_9, view_5, value_states_1, values_1], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s6
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg32_1, buf28, buf29, ps4, s6, ps5, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg32_1
            buf30 = reinterpret_tensor(buf24, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_7, view_3, query_states_1, cos_4, mul_13, x2_2, neg_2, x1_2, cat_5, sin_4, mul_14, q_embed_1, getitem_40, hidden_states_13, key_1, getitem_45, hidden_states_14, value_1, attn_output_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf25, arg4_1, arg5_1.item(), buf30, s108, 896, stream=raw_stream0)
            buf31 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_7, view_3, query_states_1, cos_4, mul_13, x2_2, neg_2, x1_2, cat_5, sin_4, mul_14, q_embed_1, getitem_40, hidden_states_13, key_1, getitem_45, hidden_states_14, value_1, attn_output_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf27, buf31, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf32 = empty_strided_cuda((1, 14, 1 + s6, 64), (896 + 896*s6, 64 + 64*s6, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_7, view_3, query_states_1, cos_4, mul_13, x2_2, neg_2, x1_2, cat_5, sin_4, mul_14, q_embed_1, getitem_40, hidden_states_13, key_1, getitem_45, hidden_states_14, value_1, attn_output_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s6
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf29, buf32, ps5, s6, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf33 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_7, view_3, query_states_1, cos_4, mul_13, x2_2, neg_2, x1_2, cat_5, sin_4, mul_14, q_embed_1, getitem_40, hidden_states_13, key_1, getitem_45, hidden_states_14, value_1, attn_output_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf33, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_7, view_3, query_states_1, cos_4, mul_13, x2_2, neg_2, x1_2, cat_5, sin_4, mul_14, q_embed_1, getitem_40, hidden_states_13, key_1, getitem_45, hidden_states_14, value_1, attn_output_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf34 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf30, buf31, buf32, reinterpret_tensor(buf33, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf31
            del buf32
            buf35 = buf34[0]
            assert_size_stride(buf35, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf35, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf34
            assert_size_stride(arg33_1, (896, 896), (896, 1))
            buf39 = reinterpret_tensor(buf30, (1, 896), (896, 1), 0); del buf30  # reuse
            # Topologically Sorted Source Nodes: [transpose_8, reshape_5, attn_output_7], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf35, (1, 896), (896, 1), 0), reinterpret_tensor(arg33_1, (896, 896), (1, 896), 0), out=buf39)
            del arg33_1
            assert_size_stride(arg35_1, (896, ), (1, ))
            assert_size_stride(arg34_1, (), ())
            buf40 = reinterpret_tensor(buf16, (1, 1, 896), (896, 896, 1), 0); del buf16  # reuse
            buf42 = reinterpret_tensor(buf35, (1, 1, 896), (896, 896, 1), 0); del buf35  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, attn_output_7, hidden_states_15, pow_4, variance_3, rsqrt_3, hidden_states_17, hidden_states_18], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9.run(buf40, arg0_1, arg1_1, buf22, buf39, arg35_1, arg34_1.item(), buf42, 1, 896, stream=raw_stream0)
            del arg0_1
            del arg34_1
            del arg35_1
            assert_size_stride(arg36_1, (4864, 896), (896, 1))
            buf43 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [pow_4, variance_3, rsqrt_3, hidden_states_17, hidden_states_18, linear_11], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf42, (1, 896), (0, 1), 0), reinterpret_tensor(arg36_1, (896, 4864), (1, 896), 0), out=buf43)
            del arg36_1
            assert_size_stride(arg37_1, (4864, 896), (896, 1))
            buf44 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf42, (1, 896), (896, 1), 0), reinterpret_tensor(arg37_1, (896, 4864), (1, 896), 0), out=buf44)
            del arg37_1
            buf45 = reinterpret_tensor(buf43, (1, 1, 4864), (4864, 4864, 1), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [linear_11, silu_1, linear_12, mul_19], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf45, buf44, 4864, stream=raw_stream0)
            del buf44
            assert_size_stride(arg38_1, (896, 4864), (4864, 1))
            buf46 = reinterpret_tensor(buf42, (1, 896), (896, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [linear_11, silu_1, linear_12, mul_19, down_proj_1], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf45, (1, 4864), (0, 1), 0), reinterpret_tensor(arg38_1, (4864, 896), (1, 4864), 0), out=buf46)
            del arg38_1
            del buf45
            assert_size_stride(arg40_1, (896, ), (1, ))
            assert_size_stride(arg39_1, (), ())
            buf48 = reinterpret_tensor(buf39, (1, 1, 896), (896, 896, 1), 0); del buf39  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, pow_5, variance_4, rsqrt_4, hidden_states_21, hidden_states_22], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf40, buf46, arg40_1, arg39_1.item(), buf48, 1, 896, stream=raw_stream0)
            del arg39_1
            del arg40_1
            assert_size_stride(arg42_1, (896, ), (1, ))
            assert_size_stride(arg41_1, (896, 896), (896, 1))
            buf49 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg42_1, reinterpret_tensor(buf48, (1, 896), (896, 1), 0), reinterpret_tensor(arg41_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf49)
            del arg41_1
            del arg42_1
            assert_size_stride(arg44_1, (128, ), (1, ))
            assert_size_stride(arg43_1, (128, 896), (896, 1))
            buf50 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg44_1, reinterpret_tensor(buf48, (1, 896), (896, 1), 0), reinterpret_tensor(arg43_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf50)
            del arg43_1
            del arg44_1
            assert_size_stride(arg47_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg47_1 = copy_misaligned(arg47_1)
            buf51 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_5, sin_5, linear_15, view_7, key_states_2, mul_24, x2_5, neg_5, x1_5, cat_10, mul_25, k_embed_2, keys_2], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg47_1, buf50, arg4_1, arg5_1.item(), buf51, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg47_1
            assert_size_stride(arg46_1, (128, ), (1, ))
            assert_size_stride(arg45_1, (128, 896), (896, 1))
            buf52 = buf50; del buf50  # reuse
            # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg46_1, reinterpret_tensor(buf48, (1, 896), (896, 1), 0), reinterpret_tensor(arg45_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf52)
            del arg45_1
            del arg46_1
            assert_size_stride(arg49_1, (1, 2, s87, 64), (128*s87, 64, 128, 1))
            arg49_1 = copy_misaligned(arg49_1)
            ps6 = 1 + s87
            ps7 = 64 + 64*s87
            buf53 = empty_strided_cuda((1, 2, 1 + s87, 64), (128 + 128*s87, 64 + 64*s87, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_16, view_8, value_states_2, values_2], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s87
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg49_1, buf52, buf53, ps6, s87, ps7, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg49_1
            buf54 = reinterpret_tensor(buf48, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf48  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_14, view_6, query_states_2, cos_5, mul_22, x2_4, neg_4, x1_4, cat_9, sin_5, mul_23, q_embed_2, getitem_54, hidden_states_23, key_2, getitem_59, hidden_states_24, value_2, attn_output_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf49, arg4_1, arg5_1.item(), buf54, s108, 896, stream=raw_stream0)
            buf55 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_14, view_6, query_states_2, cos_5, mul_22, x2_4, neg_4, x1_4, cat_9, sin_5, mul_23, q_embed_2, getitem_54, hidden_states_23, key_2, getitem_59, hidden_states_24, value_2, attn_output_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf51, buf55, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf56 = empty_strided_cuda((1, 14, 1 + s87, 64), (896 + 896*s87, 64 + 64*s87, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_14, view_6, query_states_2, cos_5, mul_22, x2_4, neg_4, x1_4, cat_9, sin_5, mul_23, q_embed_2, getitem_54, hidden_states_23, key_2, getitem_59, hidden_states_24, value_2, attn_output_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s87
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf53, buf56, ps7, s87, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf57 = buf33; del buf33  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_14, view_6, query_states_2, cos_5, mul_22, x2_4, neg_4, x1_4, cat_9, sin_5, mul_23, q_embed_2, getitem_54, hidden_states_23, key_2, getitem_59, hidden_states_24, value_2, attn_output_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf57, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_14, view_6, query_states_2, cos_5, mul_22, x2_4, neg_4, x1_4, cat_9, sin_5, mul_23, q_embed_2, getitem_54, hidden_states_23, key_2, getitem_59, hidden_states_24, value_2, attn_output_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf58 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf54, buf55, buf56, reinterpret_tensor(buf57, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf55
            del buf56
            buf59 = buf58[0]
            assert_size_stride(buf59, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf59, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf58
            assert_size_stride(arg50_1, (896, 896), (896, 1))
            buf63 = reinterpret_tensor(buf54, (1, 896), (896, 1), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [transpose_12, reshape_8, attn_output_11], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf59, (1, 896), (896, 1), 0), reinterpret_tensor(arg50_1, (896, 896), (1, 896), 0), out=buf63)
            del arg50_1
            assert_size_stride(arg52_1, (896, ), (1, ))
            assert_size_stride(arg51_1, (), ())
            buf65 = reinterpret_tensor(buf59, (1, 1, 896), (896, 896, 1), 0); del buf59  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, pow_6, variance_5, rsqrt_5, hidden_states_27, hidden_states_28], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf40, buf46, buf63, arg52_1, arg51_1.item(), buf65, 1, 896, stream=raw_stream0)
            del arg51_1
            del arg52_1
            assert_size_stride(arg53_1, (4864, 896), (896, 1))
            buf66 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf65, (1, 896), (896, 1), 0), reinterpret_tensor(arg53_1, (896, 4864), (1, 896), 0), out=buf66)
            del arg53_1
            assert_size_stride(arg54_1, (4864, 896), (896, 1))
            buf67 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf65, (1, 896), (896, 1), 0), reinterpret_tensor(arg54_1, (896, 4864), (1, 896), 0), out=buf67)
            del arg54_1
            buf68 = reinterpret_tensor(buf66, (1, 1, 4864), (4864, 4864, 1), 0); del buf66  # reuse
            # Topologically Sorted Source Nodes: [linear_18, silu_2, linear_19, mul_28], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf68, buf67, 4864, stream=raw_stream0)
            del buf67
            assert_size_stride(arg55_1, (896, 4864), (4864, 1))
            buf69 = reinterpret_tensor(buf65, (1, 896), (896, 1), 0); del buf65  # reuse
            # Topologically Sorted Source Nodes: [linear_18, silu_2, linear_19, mul_28, down_proj_2], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf68, (1, 4864), (0, 1), 0), reinterpret_tensor(arg55_1, (4864, 896), (1, 4864), 0), out=buf69)
            del arg55_1
            del buf68
            assert_size_stride(arg57_1, (896, ), (1, ))
            assert_size_stride(arg56_1, (), ())
            buf71 = reinterpret_tensor(buf49, (1, 1, 896), (896, 896, 1), 0); del buf49  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, pow_7, variance_6, rsqrt_6, hidden_states_31, hidden_states_32], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf40, buf46, buf63, buf69, arg57_1, arg56_1.item(), buf71, 1, 896, stream=raw_stream0)
            del arg56_1
            del arg57_1
            assert_size_stride(arg59_1, (896, ), (1, ))
            assert_size_stride(arg58_1, (896, 896), (896, 1))
            buf72 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg59_1, reinterpret_tensor(buf71, (1, 896), (896, 1), 0), reinterpret_tensor(arg58_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf72)
            del arg58_1
            del arg59_1
            assert_size_stride(arg61_1, (128, ), (1, ))
            assert_size_stride(arg60_1, (128, 896), (896, 1))
            buf73 = buf52; del buf52  # reuse
            # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg61_1, reinterpret_tensor(buf71, (1, 896), (896, 1), 0), reinterpret_tensor(arg60_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf73)
            del arg60_1
            del arg61_1
            assert_size_stride(arg64_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg64_1 = copy_misaligned(arg64_1)
            buf74 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_6, sin_6, linear_22, view_10, key_states_3, mul_33, x2_7, neg_7, x1_7, cat_14, mul_34, k_embed_3, keys_3], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg64_1, buf73, arg4_1, arg5_1.item(), buf74, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg64_1
            assert_size_stride(arg63_1, (128, ), (1, ))
            assert_size_stride(arg62_1, (128, 896), (896, 1))
            buf75 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg63_1, reinterpret_tensor(buf71, (1, 896), (896, 1), 0), reinterpret_tensor(arg62_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf75)
            del arg62_1
            del arg63_1
            assert_size_stride(arg66_1, (1, 2, s19, 64), (128*s19, 64, 128, 1))
            arg66_1 = copy_misaligned(arg66_1)
            ps8 = 1 + s19
            ps9 = 64 + 64*s19
            buf76 = empty_strided_cuda((1, 2, 1 + s19, 64), (128 + 128*s19, 64 + 64*s19, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_23, view_11, value_states_3, values_3], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s19
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg66_1, buf75, buf76, ps8, s19, ps9, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg66_1
            buf77 = reinterpret_tensor(buf71, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf71  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_21, view_9, query_states_3, cos_6, mul_31, x2_6, neg_6, x1_6, cat_13, sin_6, mul_32, q_embed_3, getitem_68, hidden_states_33, key_3, getitem_73, hidden_states_34, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf72, arg4_1, arg5_1.item(), buf77, s108, 896, stream=raw_stream0)
            del buf72
            buf78 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_21, view_9, query_states_3, cos_6, mul_31, x2_6, neg_6, x1_6, cat_13, sin_6, mul_32, q_embed_3, getitem_68, hidden_states_33, key_3, getitem_73, hidden_states_34, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf74, buf78, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf79 = empty_strided_cuda((1, 14, 1 + s19, 64), (896 + 896*s19, 64 + 64*s19, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_21, view_9, query_states_3, cos_6, mul_31, x2_6, neg_6, x1_6, cat_13, sin_6, mul_32, q_embed_3, getitem_68, hidden_states_33, key_3, getitem_73, hidden_states_34, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s19
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf76, buf79, ps9, s19, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf80 = buf57; del buf57  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_21, view_9, query_states_3, cos_6, mul_31, x2_6, neg_6, x1_6, cat_13, sin_6, mul_32, q_embed_3, getitem_68, hidden_states_33, key_3, getitem_73, hidden_states_34, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf80, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_21, view_9, query_states_3, cos_6, mul_31, x2_6, neg_6, x1_6, cat_13, sin_6, mul_32, q_embed_3, getitem_68, hidden_states_33, key_3, getitem_73, hidden_states_34, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf81 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf77, buf78, buf79, reinterpret_tensor(buf80, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf78
            del buf79
            buf82 = buf81[0]
            assert_size_stride(buf82, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf82, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf81
            assert_size_stride(arg67_1, (896, 896), (896, 1))
            buf86 = reinterpret_tensor(buf77, (1, 896), (896, 1), 0); del buf77  # reuse
            # Topologically Sorted Source Nodes: [transpose_16, reshape_11, attn_output_15], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf82, (1, 896), (896, 1), 0), reinterpret_tensor(arg67_1, (896, 896), (1, 896), 0), out=buf86)
            del arg67_1
            assert_size_stride(arg69_1, (896, ), (1, ))
            assert_size_stride(arg68_1, (), ())
            buf87 = buf40; del buf40  # reuse
            buf89 = reinterpret_tensor(buf82, (1, 1, 896), (896, 896, 1), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, attn_output_15, hidden_states_35, pow_8, variance_7, rsqrt_7, hidden_states_37, hidden_states_38], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf87, buf46, buf63, buf69, buf86, arg69_1, arg68_1.item(), buf89, 1, 896, stream=raw_stream0)
            del arg68_1
            del arg69_1
            del buf46
            assert_size_stride(arg70_1, (4864, 896), (896, 1))
            buf90 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [pow_8, variance_7, rsqrt_7, hidden_states_37, hidden_states_38, linear_25], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf89, (1, 896), (0, 1), 0), reinterpret_tensor(arg70_1, (896, 4864), (1, 896), 0), out=buf90)
            del arg70_1
            assert_size_stride(arg71_1, (4864, 896), (896, 1))
            buf91 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf89, (1, 896), (896, 1), 0), reinterpret_tensor(arg71_1, (896, 4864), (1, 896), 0), out=buf91)
            del arg71_1
            buf92 = reinterpret_tensor(buf90, (1, 1, 4864), (4864, 4864, 1), 0); del buf90  # reuse
            # Topologically Sorted Source Nodes: [linear_25, silu_3, linear_26, mul_37], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf92, buf91, 4864, stream=raw_stream0)
            del buf91
            assert_size_stride(arg72_1, (896, 4864), (4864, 1))
            buf93 = reinterpret_tensor(buf89, (1, 896), (896, 1), 0); del buf89  # reuse
            # Topologically Sorted Source Nodes: [linear_25, silu_3, linear_26, mul_37, down_proj_3], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf92, (1, 4864), (0, 1), 0), reinterpret_tensor(arg72_1, (4864, 896), (1, 4864), 0), out=buf93)
            del arg72_1
            del buf92
            assert_size_stride(arg74_1, (896, ), (1, ))
            assert_size_stride(arg73_1, (), ())
            buf95 = reinterpret_tensor(buf86, (1, 1, 896), (896, 896, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, pow_9, variance_8, rsqrt_8, hidden_states_41, hidden_states_42], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf87, buf93, arg74_1, arg73_1.item(), buf95, 1, 896, stream=raw_stream0)
            del arg73_1
            del arg74_1
            assert_size_stride(arg76_1, (896, ), (1, ))
            assert_size_stride(arg75_1, (896, 896), (896, 1))
            buf96 = buf69; del buf69  # reuse
            # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg76_1, reinterpret_tensor(buf95, (1, 896), (896, 1), 0), reinterpret_tensor(arg75_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf96)
            del arg75_1
            del arg76_1
            assert_size_stride(arg78_1, (128, ), (1, ))
            assert_size_stride(arg77_1, (128, 896), (896, 1))
            buf97 = buf75; del buf75  # reuse
            # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg78_1, reinterpret_tensor(buf95, (1, 896), (896, 1), 0), reinterpret_tensor(arg77_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf97)
            del arg77_1
            del arg78_1
            assert_size_stride(arg81_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg81_1 = copy_misaligned(arg81_1)
            buf98 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_7, sin_7, linear_29, view_13, key_states_4, mul_42, x2_9, neg_9, x1_9, cat_18, mul_43, k_embed_4, keys_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg81_1, buf97, arg4_1, arg5_1.item(), buf98, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg81_1
            assert_size_stride(arg80_1, (128, ), (1, ))
            assert_size_stride(arg79_1, (128, 896), (896, 1))
            buf99 = buf97; del buf97  # reuse
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg80_1, reinterpret_tensor(buf95, (1, 896), (896, 1), 0), reinterpret_tensor(arg79_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf99)
            del arg79_1
            del arg80_1
            assert_size_stride(arg83_1, (1, 2, s24, 64), (128*s24, 64, 128, 1))
            arg83_1 = copy_misaligned(arg83_1)
            ps10 = 1 + s24
            ps11 = 64 + 64*s24
            buf100 = empty_strided_cuda((1, 2, 1 + s24, 64), (128 + 128*s24, 64 + 64*s24, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_30, view_14, value_states_4, values_4], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s24
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg83_1, buf99, buf100, ps10, s24, ps11, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg83_1
            buf101 = reinterpret_tensor(buf95, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf95  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_28, view_12, query_states_4, cos_7, mul_40, x2_8, neg_8, x1_8, cat_17, sin_7, mul_41, q_embed_4, getitem_82, hidden_states_43, key_4, getitem_87, hidden_states_44, value_4, attn_output_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf96, arg4_1, arg5_1.item(), buf101, s108, 896, stream=raw_stream0)
            buf102 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_28, view_12, query_states_4, cos_7, mul_40, x2_8, neg_8, x1_8, cat_17, sin_7, mul_41, q_embed_4, getitem_82, hidden_states_43, key_4, getitem_87, hidden_states_44, value_4, attn_output_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf98, buf102, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf103 = empty_strided_cuda((1, 14, 1 + s24, 64), (896 + 896*s24, 64 + 64*s24, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_28, view_12, query_states_4, cos_7, mul_40, x2_8, neg_8, x1_8, cat_17, sin_7, mul_41, q_embed_4, getitem_82, hidden_states_43, key_4, getitem_87, hidden_states_44, value_4, attn_output_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s24
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf100, buf103, ps11, s24, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf104 = buf80; del buf80  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_28, view_12, query_states_4, cos_7, mul_40, x2_8, neg_8, x1_8, cat_17, sin_7, mul_41, q_embed_4, getitem_82, hidden_states_43, key_4, getitem_87, hidden_states_44, value_4, attn_output_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf104, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_28, view_12, query_states_4, cos_7, mul_40, x2_8, neg_8, x1_8, cat_17, sin_7, mul_41, q_embed_4, getitem_82, hidden_states_43, key_4, getitem_87, hidden_states_44, value_4, attn_output_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf105 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf101, buf102, buf103, reinterpret_tensor(buf104, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf102
            del buf103
            buf106 = buf105[0]
            assert_size_stride(buf106, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf106, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf105
            assert_size_stride(arg84_1, (896, 896), (896, 1))
            buf110 = reinterpret_tensor(buf101, (1, 896), (896, 1), 0); del buf101  # reuse
            # Topologically Sorted Source Nodes: [transpose_20, reshape_14, attn_output_19], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf106, (1, 896), (896, 1), 0), reinterpret_tensor(arg84_1, (896, 896), (1, 896), 0), out=buf110)
            del arg84_1
            assert_size_stride(arg86_1, (896, ), (1, ))
            assert_size_stride(arg85_1, (), ())
            buf112 = reinterpret_tensor(buf106, (1, 1, 896), (896, 896, 1), 0); del buf106  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, attn_output_19, hidden_states_45, pow_10, variance_9, rsqrt_9, hidden_states_47, hidden_states_48], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf87, buf93, buf110, arg86_1, arg85_1.item(), buf112, 1, 896, stream=raw_stream0)
            del arg85_1
            del arg86_1
            assert_size_stride(arg87_1, (4864, 896), (896, 1))
            buf113 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf112, (1, 896), (896, 1), 0), reinterpret_tensor(arg87_1, (896, 4864), (1, 896), 0), out=buf113)
            del arg87_1
            assert_size_stride(arg88_1, (4864, 896), (896, 1))
            buf114 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf112, (1, 896), (896, 1), 0), reinterpret_tensor(arg88_1, (896, 4864), (1, 896), 0), out=buf114)
            del arg88_1
            buf115 = reinterpret_tensor(buf113, (1, 1, 4864), (4864, 4864, 1), 0); del buf113  # reuse
            # Topologically Sorted Source Nodes: [linear_32, silu_4, linear_33, mul_46], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf115, buf114, 4864, stream=raw_stream0)
            del buf114
            assert_size_stride(arg89_1, (896, 4864), (4864, 1))
            buf116 = reinterpret_tensor(buf112, (1, 896), (896, 1), 0); del buf112  # reuse
            # Topologically Sorted Source Nodes: [linear_32, silu_4, linear_33, mul_46, down_proj_4], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf115, (1, 4864), (0, 1), 0), reinterpret_tensor(arg89_1, (4864, 896), (1, 4864), 0), out=buf116)
            del arg89_1
            del buf115
            assert_size_stride(arg91_1, (896, ), (1, ))
            assert_size_stride(arg90_1, (), ())
            buf118 = reinterpret_tensor(buf96, (1, 1, 896), (896, 896, 1), 0); del buf96  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, attn_output_19, hidden_states_45, down_proj_4, hidden_states_49, pow_11, variance_10, rsqrt_10, hidden_states_51, hidden_states_52], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf87, buf93, buf110, buf116, arg91_1, arg90_1.item(), buf118, 1, 896, stream=raw_stream0)
            del arg90_1
            del arg91_1
            assert_size_stride(arg93_1, (896, ), (1, ))
            assert_size_stride(arg92_1, (896, 896), (896, 1))
            buf119 = buf63; del buf63  # reuse
            # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg93_1, reinterpret_tensor(buf118, (1, 896), (896, 1), 0), reinterpret_tensor(arg92_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf119)
            del arg92_1
            del arg93_1
            assert_size_stride(arg95_1, (128, ), (1, ))
            assert_size_stride(arg94_1, (128, 896), (896, 1))
            buf120 = buf99; del buf99  # reuse
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg95_1, reinterpret_tensor(buf118, (1, 896), (896, 1), 0), reinterpret_tensor(arg94_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf120)
            del arg94_1
            del arg95_1
            assert_size_stride(arg98_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg98_1 = copy_misaligned(arg98_1)
            buf121 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_8, sin_8, linear_36, view_16, key_states_5, mul_51, x2_11, neg_11, x1_11, cat_22, mul_52, k_embed_5, keys_5], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg98_1, buf120, arg4_1, arg5_1.item(), buf121, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg98_1
            assert_size_stride(arg97_1, (128, ), (1, ))
            assert_size_stride(arg96_1, (128, 896), (896, 1))
            buf122 = buf120; del buf120  # reuse
            # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg97_1, reinterpret_tensor(buf118, (1, 896), (896, 1), 0), reinterpret_tensor(arg96_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf122)
            del arg96_1
            del arg97_1
            assert_size_stride(arg100_1, (1, 2, s98, 64), (128*s98, 64, 128, 1))
            arg100_1 = copy_misaligned(arg100_1)
            ps12 = 1 + s98
            ps13 = 64 + 64*s98
            buf123 = empty_strided_cuda((1, 2, 1 + s98, 64), (128 + 128*s98, 64 + 64*s98, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_37, view_17, value_states_5, values_5], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s98
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg100_1, buf122, buf123, ps12, s98, ps13, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg100_1
            buf124 = reinterpret_tensor(buf118, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf118  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_35, view_15, query_states_5, cos_8, mul_49, x2_10, neg_10, x1_10, cat_21, sin_8, mul_50, q_embed_5, getitem_96, hidden_states_53, key_5, getitem_101, hidden_states_54, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf119, arg4_1, arg5_1.item(), buf124, s108, 896, stream=raw_stream0)
            del buf119
            buf125 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_35, view_15, query_states_5, cos_8, mul_49, x2_10, neg_10, x1_10, cat_21, sin_8, mul_50, q_embed_5, getitem_96, hidden_states_53, key_5, getitem_101, hidden_states_54, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf121, buf125, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf126 = empty_strided_cuda((1, 14, 1 + s98, 64), (896 + 896*s98, 64 + 64*s98, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_35, view_15, query_states_5, cos_8, mul_49, x2_10, neg_10, x1_10, cat_21, sin_8, mul_50, q_embed_5, getitem_96, hidden_states_53, key_5, getitem_101, hidden_states_54, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s98
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf123, buf126, ps13, s98, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf127 = buf104; del buf104  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_35, view_15, query_states_5, cos_8, mul_49, x2_10, neg_10, x1_10, cat_21, sin_8, mul_50, q_embed_5, getitem_96, hidden_states_53, key_5, getitem_101, hidden_states_54, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf127, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_35, view_15, query_states_5, cos_8, mul_49, x2_10, neg_10, x1_10, cat_21, sin_8, mul_50, q_embed_5, getitem_96, hidden_states_53, key_5, getitem_101, hidden_states_54, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf128 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf124, buf125, buf126, reinterpret_tensor(buf127, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf125
            del buf126
            buf129 = buf128[0]
            assert_size_stride(buf129, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf129, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf128
            assert_size_stride(arg101_1, (896, 896), (896, 1))
            buf133 = reinterpret_tensor(buf124, (1, 896), (896, 1), 0); del buf124  # reuse
            # Topologically Sorted Source Nodes: [transpose_24, reshape_17, attn_output_23], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf129, (1, 896), (896, 1), 0), reinterpret_tensor(arg101_1, (896, 896), (1, 896), 0), out=buf133)
            del arg101_1
            assert_size_stride(arg103_1, (896, ), (1, ))
            assert_size_stride(arg102_1, (), ())
            buf134 = buf87; del buf87  # reuse
            buf136 = reinterpret_tensor(buf129, (1, 1, 896), (896, 896, 1), 0); del buf129  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, attn_output_19, hidden_states_45, down_proj_4, hidden_states_49, attn_output_23, hidden_states_55, pow_12, variance_11, rsqrt_11, hidden_states_57, hidden_states_58], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf134, buf93, buf110, buf116, buf133, arg103_1, arg102_1.item(), buf136, 1, 896, stream=raw_stream0)
            del arg102_1
            del arg103_1
            del buf110
            assert_size_stride(arg104_1, (4864, 896), (896, 1))
            buf137 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [pow_12, variance_11, rsqrt_11, hidden_states_57, hidden_states_58, linear_39], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf136, (1, 896), (0, 1), 0), reinterpret_tensor(arg104_1, (896, 4864), (1, 896), 0), out=buf137)
            del arg104_1
            assert_size_stride(arg105_1, (4864, 896), (896, 1))
            buf138 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf136, (1, 896), (896, 1), 0), reinterpret_tensor(arg105_1, (896, 4864), (1, 896), 0), out=buf138)
            del arg105_1
            buf139 = reinterpret_tensor(buf137, (1, 1, 4864), (4864, 4864, 1), 0); del buf137  # reuse
            # Topologically Sorted Source Nodes: [linear_39, silu_5, linear_40, mul_55], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf139, buf138, 4864, stream=raw_stream0)
            del buf138
            assert_size_stride(arg106_1, (896, 4864), (4864, 1))
            buf140 = reinterpret_tensor(buf136, (1, 896), (896, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [linear_39, silu_5, linear_40, mul_55, down_proj_5], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf139, (1, 4864), (0, 1), 0), reinterpret_tensor(arg106_1, (4864, 896), (1, 4864), 0), out=buf140)
            del arg106_1
            del buf139
            assert_size_stride(arg108_1, (896, ), (1, ))
            assert_size_stride(arg107_1, (), ())
            buf142 = reinterpret_tensor(buf93, (1, 1, 896), (896, 896, 1), 0); del buf93  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, pow_13, variance_12, rsqrt_12, hidden_states_61, hidden_states_62], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf134, buf140, arg108_1, arg107_1.item(), buf142, 1, 896, stream=raw_stream0)
            del arg107_1
            del arg108_1
            assert_size_stride(arg110_1, (896, ), (1, ))
            assert_size_stride(arg109_1, (896, 896), (896, 1))
            buf143 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg110_1, reinterpret_tensor(buf142, (1, 896), (896, 1), 0), reinterpret_tensor(arg109_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf143)
            del arg109_1
            del arg110_1
            assert_size_stride(arg112_1, (128, ), (1, ))
            assert_size_stride(arg111_1, (128, 896), (896, 1))
            buf144 = buf122; del buf122  # reuse
            # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg112_1, reinterpret_tensor(buf142, (1, 896), (896, 1), 0), reinterpret_tensor(arg111_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf144)
            del arg111_1
            del arg112_1
            assert_size_stride(arg115_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg115_1 = copy_misaligned(arg115_1)
            buf145 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_9, sin_9, linear_43, view_19, key_states_6, mul_60, x2_13, neg_13, x1_13, cat_26, mul_61, k_embed_6, keys_6], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg115_1, buf144, arg4_1, arg5_1.item(), buf145, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg115_1
            assert_size_stride(arg114_1, (128, ), (1, ))
            assert_size_stride(arg113_1, (128, 896), (896, 1))
            buf146 = buf144; del buf144  # reuse
            # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg114_1, reinterpret_tensor(buf142, (1, 896), (896, 1), 0), reinterpret_tensor(arg113_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf146)
            del arg113_1
            del arg114_1
            assert_size_stride(arg117_1, (1, 2, s80, 64), (128*s80, 64, 128, 1))
            arg117_1 = copy_misaligned(arg117_1)
            ps14 = 1 + s80
            ps15 = 64 + 64*s80
            buf147 = empty_strided_cuda((1, 2, 1 + s80, 64), (128 + 128*s80, 64 + 64*s80, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_44, view_20, value_states_6, values_6], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s80
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg117_1, buf146, buf147, ps14, s80, ps15, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg117_1
            buf148 = reinterpret_tensor(buf142, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf142  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_42, view_18, query_states_6, cos_9, mul_58, x2_12, neg_12, x1_12, cat_25, sin_9, mul_59, q_embed_6, getitem_110, hidden_states_63, key_6, getitem_115, hidden_states_64, value_6, attn_output_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf143, arg4_1, arg5_1.item(), buf148, s108, 896, stream=raw_stream0)
            buf149 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_42, view_18, query_states_6, cos_9, mul_58, x2_12, neg_12, x1_12, cat_25, sin_9, mul_59, q_embed_6, getitem_110, hidden_states_63, key_6, getitem_115, hidden_states_64, value_6, attn_output_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf145, buf149, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf150 = empty_strided_cuda((1, 14, 1 + s80, 64), (896 + 896*s80, 64 + 64*s80, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_42, view_18, query_states_6, cos_9, mul_58, x2_12, neg_12, x1_12, cat_25, sin_9, mul_59, q_embed_6, getitem_110, hidden_states_63, key_6, getitem_115, hidden_states_64, value_6, attn_output_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s80
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf147, buf150, ps15, s80, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf151 = buf127; del buf127  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_42, view_18, query_states_6, cos_9, mul_58, x2_12, neg_12, x1_12, cat_25, sin_9, mul_59, q_embed_6, getitem_110, hidden_states_63, key_6, getitem_115, hidden_states_64, value_6, attn_output_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf151, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_42, view_18, query_states_6, cos_9, mul_58, x2_12, neg_12, x1_12, cat_25, sin_9, mul_59, q_embed_6, getitem_110, hidden_states_63, key_6, getitem_115, hidden_states_64, value_6, attn_output_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf152 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf148, buf149, buf150, reinterpret_tensor(buf151, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf149
            del buf150
            buf153 = buf152[0]
            assert_size_stride(buf153, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf153, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf152
            assert_size_stride(arg118_1, (896, 896), (896, 1))
            buf157 = reinterpret_tensor(buf148, (1, 896), (896, 1), 0); del buf148  # reuse
            # Topologically Sorted Source Nodes: [transpose_28, reshape_20, attn_output_27], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf153, (1, 896), (896, 1), 0), reinterpret_tensor(arg118_1, (896, 896), (1, 896), 0), out=buf157)
            del arg118_1
            assert_size_stride(arg120_1, (896, ), (1, ))
            assert_size_stride(arg119_1, (), ())
            buf159 = reinterpret_tensor(buf153, (1, 1, 896), (896, 896, 1), 0); del buf153  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, attn_output_27, hidden_states_65, pow_14, variance_13, rsqrt_13, hidden_states_67, hidden_states_68], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf134, buf140, buf157, arg120_1, arg119_1.item(), buf159, 1, 896, stream=raw_stream0)
            del arg119_1
            del arg120_1
            assert_size_stride(arg121_1, (4864, 896), (896, 1))
            buf160 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf159, (1, 896), (896, 1), 0), reinterpret_tensor(arg121_1, (896, 4864), (1, 896), 0), out=buf160)
            del arg121_1
            assert_size_stride(arg122_1, (4864, 896), (896, 1))
            buf161 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf159, (1, 896), (896, 1), 0), reinterpret_tensor(arg122_1, (896, 4864), (1, 896), 0), out=buf161)
            del arg122_1
            buf162 = reinterpret_tensor(buf160, (1, 1, 4864), (4864, 4864, 1), 0); del buf160  # reuse
            # Topologically Sorted Source Nodes: [linear_46, silu_6, linear_47, mul_64], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf162, buf161, 4864, stream=raw_stream0)
            del buf161
            assert_size_stride(arg123_1, (896, 4864), (4864, 1))
            buf163 = reinterpret_tensor(buf159, (1, 896), (896, 1), 0); del buf159  # reuse
            # Topologically Sorted Source Nodes: [linear_46, silu_6, linear_47, mul_64, down_proj_6], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf162, (1, 4864), (0, 1), 0), reinterpret_tensor(arg123_1, (4864, 896), (1, 4864), 0), out=buf163)
            del arg123_1
            del buf162
            assert_size_stride(arg125_1, (896, ), (1, ))
            assert_size_stride(arg124_1, (), ())
            buf165 = reinterpret_tensor(buf143, (1, 1, 896), (896, 896, 1), 0); del buf143  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, attn_output_27, hidden_states_65, down_proj_6, hidden_states_69, pow_15, variance_14, rsqrt_14, hidden_states_71, hidden_states_72], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf134, buf140, buf157, buf163, arg125_1, arg124_1.item(), buf165, 1, 896, stream=raw_stream0)
            del arg124_1
            del arg125_1
            assert_size_stride(arg127_1, (896, ), (1, ))
            assert_size_stride(arg126_1, (896, 896), (896, 1))
            buf166 = buf116; del buf116  # reuse
            # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg127_1, reinterpret_tensor(buf165, (1, 896), (896, 1), 0), reinterpret_tensor(arg126_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf166)
            del arg126_1
            del arg127_1
            assert_size_stride(arg129_1, (128, ), (1, ))
            assert_size_stride(arg128_1, (128, 896), (896, 1))
            buf167 = buf146; del buf146  # reuse
            # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg129_1, reinterpret_tensor(buf165, (1, 896), (896, 1), 0), reinterpret_tensor(arg128_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf167)
            del arg128_1
            del arg129_1
            assert_size_stride(arg132_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg132_1 = copy_misaligned(arg132_1)
            buf168 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_10, sin_10, linear_50, view_22, key_states_7, mul_69, x2_15, neg_15, x1_15, cat_30, mul_70, k_embed_7, keys_7], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg132_1, buf167, arg4_1, arg5_1.item(), buf168, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg132_1
            assert_size_stride(arg131_1, (128, ), (1, ))
            assert_size_stride(arg130_1, (128, 896), (896, 1))
            buf169 = buf167; del buf167  # reuse
            # Topologically Sorted Source Nodes: [linear_51], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg131_1, reinterpret_tensor(buf165, (1, 896), (896, 1), 0), reinterpret_tensor(arg130_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf169)
            del arg130_1
            del arg131_1
            assert_size_stride(arg134_1, (1, 2, s2, 64), (128*s2, 64, 128, 1))
            arg134_1 = copy_misaligned(arg134_1)
            ps16 = 1 + s2
            ps17 = 64 + 64*s2
            buf170 = empty_strided_cuda((1, 2, 1 + s2, 64), (128 + 128*s2, 64 + 64*s2, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_51, view_23, value_states_7, values_7], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s2
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg134_1, buf169, buf170, ps16, s2, ps17, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg134_1
            buf171 = reinterpret_tensor(buf165, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf165  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_49, view_21, query_states_7, cos_10, mul_67, x2_14, neg_14, x1_14, cat_29, sin_10, mul_68, q_embed_7, getitem_124, hidden_states_73, key_7, getitem_129, hidden_states_74, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf166, arg4_1, arg5_1.item(), buf171, s108, 896, stream=raw_stream0)
            del buf166
            buf172 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_49, view_21, query_states_7, cos_10, mul_67, x2_14, neg_14, x1_14, cat_29, sin_10, mul_68, q_embed_7, getitem_124, hidden_states_73, key_7, getitem_129, hidden_states_74, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf168, buf172, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf173 = empty_strided_cuda((1, 14, 1 + s2, 64), (896 + 896*s2, 64 + 64*s2, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_49, view_21, query_states_7, cos_10, mul_67, x2_14, neg_14, x1_14, cat_29, sin_10, mul_68, q_embed_7, getitem_124, hidden_states_73, key_7, getitem_129, hidden_states_74, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s2
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf170, buf173, ps17, s2, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf174 = buf151; del buf151  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_49, view_21, query_states_7, cos_10, mul_67, x2_14, neg_14, x1_14, cat_29, sin_10, mul_68, q_embed_7, getitem_124, hidden_states_73, key_7, getitem_129, hidden_states_74, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf174, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_49, view_21, query_states_7, cos_10, mul_67, x2_14, neg_14, x1_14, cat_29, sin_10, mul_68, q_embed_7, getitem_124, hidden_states_73, key_7, getitem_129, hidden_states_74, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf175 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf171, buf172, buf173, reinterpret_tensor(buf174, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf172
            del buf173
            buf176 = buf175[0]
            assert_size_stride(buf176, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf176, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf175
            assert_size_stride(arg135_1, (896, 896), (896, 1))
            buf180 = reinterpret_tensor(buf171, (1, 896), (896, 1), 0); del buf171  # reuse
            # Topologically Sorted Source Nodes: [transpose_32, reshape_23, attn_output_31], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf176, (1, 896), (896, 1), 0), reinterpret_tensor(arg135_1, (896, 896), (1, 896), 0), out=buf180)
            del arg135_1
            assert_size_stride(arg137_1, (896, ), (1, ))
            assert_size_stride(arg136_1, (), ())
            buf181 = buf134; del buf134  # reuse
            buf183 = reinterpret_tensor(buf176, (1, 1, 896), (896, 896, 1), 0); del buf176  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, attn_output_27, hidden_states_65, down_proj_6, hidden_states_69, attn_output_31, hidden_states_75, pow_16, variance_15, rsqrt_15, hidden_states_77, hidden_states_78], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf181, buf140, buf157, buf163, buf180, arg137_1, arg136_1.item(), buf183, 1, 896, stream=raw_stream0)
            del arg136_1
            del arg137_1
            del buf140
            assert_size_stride(arg138_1, (4864, 896), (896, 1))
            buf184 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [pow_16, variance_15, rsqrt_15, hidden_states_77, hidden_states_78, linear_53], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf183, (1, 896), (0, 1), 0), reinterpret_tensor(arg138_1, (896, 4864), (1, 896), 0), out=buf184)
            del arg138_1
            assert_size_stride(arg139_1, (4864, 896), (896, 1))
            buf185 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf183, (1, 896), (896, 1), 0), reinterpret_tensor(arg139_1, (896, 4864), (1, 896), 0), out=buf185)
            del arg139_1
            buf186 = reinterpret_tensor(buf184, (1, 1, 4864), (4864, 4864, 1), 0); del buf184  # reuse
            # Topologically Sorted Source Nodes: [linear_53, silu_7, linear_54, mul_73], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf186, buf185, 4864, stream=raw_stream0)
            del buf185
            assert_size_stride(arg140_1, (896, 4864), (4864, 1))
            buf187 = reinterpret_tensor(buf183, (1, 896), (896, 1), 0); del buf183  # reuse
            # Topologically Sorted Source Nodes: [linear_53, silu_7, linear_54, mul_73, down_proj_7], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf186, (1, 4864), (0, 1), 0), reinterpret_tensor(arg140_1, (4864, 896), (1, 4864), 0), out=buf187)
            del arg140_1
            del buf186
            assert_size_stride(arg142_1, (896, ), (1, ))
            assert_size_stride(arg141_1, (), ())
            buf189 = reinterpret_tensor(buf180, (1, 1, 896), (896, 896, 1), 0); del buf180  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, pow_17, variance_16, rsqrt_16, hidden_states_81, hidden_states_82], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf181, buf187, arg142_1, arg141_1.item(), buf189, 1, 896, stream=raw_stream0)
            del arg141_1
            del arg142_1
            assert_size_stride(arg144_1, (896, ), (1, ))
            assert_size_stride(arg143_1, (896, 896), (896, 1))
            buf190 = buf163; del buf163  # reuse
            # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg144_1, reinterpret_tensor(buf189, (1, 896), (896, 1), 0), reinterpret_tensor(arg143_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf190)
            del arg143_1
            del arg144_1
            assert_size_stride(arg146_1, (128, ), (1, ))
            assert_size_stride(arg145_1, (128, 896), (896, 1))
            buf191 = buf169; del buf169  # reuse
            # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg146_1, reinterpret_tensor(buf189, (1, 896), (896, 1), 0), reinterpret_tensor(arg145_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf191)
            del arg145_1
            del arg146_1
            assert_size_stride(arg149_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg149_1 = copy_misaligned(arg149_1)
            buf192 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_11, sin_11, linear_57, view_25, key_states_8, mul_78, x2_17, neg_17, x1_17, cat_34, mul_79, k_embed_8, keys_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg149_1, buf191, arg4_1, arg5_1.item(), buf192, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg149_1
            assert_size_stride(arg148_1, (128, ), (1, ))
            assert_size_stride(arg147_1, (128, 896), (896, 1))
            buf193 = buf191; del buf191  # reuse
            # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg148_1, reinterpret_tensor(buf189, (1, 896), (896, 1), 0), reinterpret_tensor(arg147_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf193)
            del arg147_1
            del arg148_1
            assert_size_stride(arg151_1, (1, 2, s34, 64), (128*s34, 64, 128, 1))
            arg151_1 = copy_misaligned(arg151_1)
            ps18 = 1 + s34
            ps19 = 64 + 64*s34
            buf194 = empty_strided_cuda((1, 2, 1 + s34, 64), (128 + 128*s34, 64 + 64*s34, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_58, view_26, value_states_8, values_8], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s34
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg151_1, buf193, buf194, ps18, s34, ps19, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg151_1
            buf195 = reinterpret_tensor(buf189, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf189  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_56, view_24, query_states_8, cos_11, mul_76, x2_16, neg_16, x1_16, cat_33, sin_11, mul_77, q_embed_8, getitem_138, hidden_states_83, key_8, getitem_143, hidden_states_84, value_8, attn_output_32], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf190, arg4_1, arg5_1.item(), buf195, s108, 896, stream=raw_stream0)
            buf196 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_56, view_24, query_states_8, cos_11, mul_76, x2_16, neg_16, x1_16, cat_33, sin_11, mul_77, q_embed_8, getitem_138, hidden_states_83, key_8, getitem_143, hidden_states_84, value_8, attn_output_32], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf192, buf196, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf197 = empty_strided_cuda((1, 14, 1 + s34, 64), (896 + 896*s34, 64 + 64*s34, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_56, view_24, query_states_8, cos_11, mul_76, x2_16, neg_16, x1_16, cat_33, sin_11, mul_77, q_embed_8, getitem_138, hidden_states_83, key_8, getitem_143, hidden_states_84, value_8, attn_output_32], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s34
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf194, buf197, ps19, s34, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf198 = buf174; del buf174  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_56, view_24, query_states_8, cos_11, mul_76, x2_16, neg_16, x1_16, cat_33, sin_11, mul_77, q_embed_8, getitem_138, hidden_states_83, key_8, getitem_143, hidden_states_84, value_8, attn_output_32], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf198, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_56, view_24, query_states_8, cos_11, mul_76, x2_16, neg_16, x1_16, cat_33, sin_11, mul_77, q_embed_8, getitem_138, hidden_states_83, key_8, getitem_143, hidden_states_84, value_8, attn_output_32], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf199 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf195, buf196, buf197, reinterpret_tensor(buf198, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf196
            del buf197
            buf200 = buf199[0]
            assert_size_stride(buf200, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf200, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf199
            assert_size_stride(arg152_1, (896, 896), (896, 1))
            buf204 = reinterpret_tensor(buf195, (1, 896), (896, 1), 0); del buf195  # reuse
            # Topologically Sorted Source Nodes: [transpose_36, reshape_26, attn_output_35], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf200, (1, 896), (896, 1), 0), reinterpret_tensor(arg152_1, (896, 896), (1, 896), 0), out=buf204)
            del arg152_1
            assert_size_stride(arg154_1, (896, ), (1, ))
            assert_size_stride(arg153_1, (), ())
            buf206 = reinterpret_tensor(buf200, (1, 1, 896), (896, 896, 1), 0); del buf200  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, pow_18, variance_17, rsqrt_17, hidden_states_87, hidden_states_88], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf181, buf187, buf204, arg154_1, arg153_1.item(), buf206, 1, 896, stream=raw_stream0)
            del arg153_1
            del arg154_1
            assert_size_stride(arg155_1, (4864, 896), (896, 1))
            buf207 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf206, (1, 896), (896, 1), 0), reinterpret_tensor(arg155_1, (896, 4864), (1, 896), 0), out=buf207)
            del arg155_1
            assert_size_stride(arg156_1, (4864, 896), (896, 1))
            buf208 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf206, (1, 896), (896, 1), 0), reinterpret_tensor(arg156_1, (896, 4864), (1, 896), 0), out=buf208)
            del arg156_1
            buf209 = reinterpret_tensor(buf207, (1, 1, 4864), (4864, 4864, 1), 0); del buf207  # reuse
            # Topologically Sorted Source Nodes: [linear_60, silu_8, linear_61, mul_82], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf209, buf208, 4864, stream=raw_stream0)
            del buf208
            assert_size_stride(arg157_1, (896, 4864), (4864, 1))
            buf210 = reinterpret_tensor(buf206, (1, 896), (896, 1), 0); del buf206  # reuse
            # Topologically Sorted Source Nodes: [linear_60, silu_8, linear_61, mul_82, down_proj_8], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf209, (1, 4864), (0, 1), 0), reinterpret_tensor(arg157_1, (4864, 896), (1, 4864), 0), out=buf210)
            del arg157_1
            del buf209
            assert_size_stride(arg159_1, (896, ), (1, ))
            assert_size_stride(arg158_1, (), ())
            buf212 = reinterpret_tensor(buf190, (1, 1, 896), (896, 896, 1), 0); del buf190  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, down_proj_8, hidden_states_89, pow_19, variance_18, rsqrt_18, hidden_states_91, hidden_states_92], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf181, buf187, buf204, buf210, arg159_1, arg158_1.item(), buf212, 1, 896, stream=raw_stream0)
            del arg158_1
            del arg159_1
            assert_size_stride(arg161_1, (896, ), (1, ))
            assert_size_stride(arg160_1, (896, 896), (896, 1))
            buf213 = buf157; del buf157  # reuse
            # Topologically Sorted Source Nodes: [linear_63], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg161_1, reinterpret_tensor(buf212, (1, 896), (896, 1), 0), reinterpret_tensor(arg160_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf213)
            del arg160_1
            del arg161_1
            assert_size_stride(arg163_1, (128, ), (1, ))
            assert_size_stride(arg162_1, (128, 896), (896, 1))
            buf214 = buf193; del buf193  # reuse
            # Topologically Sorted Source Nodes: [linear_64], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg163_1, reinterpret_tensor(buf212, (1, 896), (896, 1), 0), reinterpret_tensor(arg162_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf214)
            del arg162_1
            del arg163_1
            assert_size_stride(arg166_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg166_1 = copy_misaligned(arg166_1)
            buf215 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_12, sin_12, linear_64, view_28, key_states_9, mul_87, x2_19, neg_19, x1_19, cat_38, mul_88, k_embed_9, keys_9], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg166_1, buf214, arg4_1, arg5_1.item(), buf215, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg166_1
            assert_size_stride(arg165_1, (128, ), (1, ))
            assert_size_stride(arg164_1, (128, 896), (896, 1))
            buf216 = buf214; del buf214  # reuse
            # Topologically Sorted Source Nodes: [linear_65], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg165_1, reinterpret_tensor(buf212, (1, 896), (896, 1), 0), reinterpret_tensor(arg164_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf216)
            del arg164_1
            del arg165_1
            assert_size_stride(arg168_1, (1, 2, s4, 64), (128*s4, 64, 128, 1))
            arg168_1 = copy_misaligned(arg168_1)
            ps20 = 1 + s4
            ps21 = 64 + 64*s4
            buf217 = empty_strided_cuda((1, 2, 1 + s4, 64), (128 + 128*s4, 64 + 64*s4, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_65, view_29, value_states_9, values_9], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s4
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg168_1, buf216, buf217, ps20, s4, ps21, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg168_1
            buf218 = reinterpret_tensor(buf212, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf212  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_63, view_27, query_states_9, cos_12, mul_85, x2_18, neg_18, x1_18, cat_37, sin_12, mul_86, q_embed_9, getitem_152, hidden_states_93, key_9, getitem_157, hidden_states_94, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf213, arg4_1, arg5_1.item(), buf218, s108, 896, stream=raw_stream0)
            del buf213
            buf219 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_63, view_27, query_states_9, cos_12, mul_85, x2_18, neg_18, x1_18, cat_37, sin_12, mul_86, q_embed_9, getitem_152, hidden_states_93, key_9, getitem_157, hidden_states_94, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf215, buf219, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf220 = empty_strided_cuda((1, 14, 1 + s4, 64), (896 + 896*s4, 64 + 64*s4, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_63, view_27, query_states_9, cos_12, mul_85, x2_18, neg_18, x1_18, cat_37, sin_12, mul_86, q_embed_9, getitem_152, hidden_states_93, key_9, getitem_157, hidden_states_94, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s4
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf217, buf220, ps21, s4, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf221 = buf198; del buf198  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_63, view_27, query_states_9, cos_12, mul_85, x2_18, neg_18, x1_18, cat_37, sin_12, mul_86, q_embed_9, getitem_152, hidden_states_93, key_9, getitem_157, hidden_states_94, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf221, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_63, view_27, query_states_9, cos_12, mul_85, x2_18, neg_18, x1_18, cat_37, sin_12, mul_86, q_embed_9, getitem_152, hidden_states_93, key_9, getitem_157, hidden_states_94, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf222 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf218, buf219, buf220, reinterpret_tensor(buf221, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf219
            del buf220
            buf223 = buf222[0]
            assert_size_stride(buf223, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf223, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf222
            assert_size_stride(arg169_1, (896, 896), (896, 1))
            buf227 = reinterpret_tensor(buf218, (1, 896), (896, 1), 0); del buf218  # reuse
            # Topologically Sorted Source Nodes: [transpose_40, reshape_29, attn_output_39], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf223, (1, 896), (896, 1), 0), reinterpret_tensor(arg169_1, (896, 896), (1, 896), 0), out=buf227)
            del arg169_1
            assert_size_stride(arg170_1, (896, ), (1, ))
            buf228 = buf181; del buf181  # reuse
            buf230 = reinterpret_tensor(buf223, (1, 1, 896), (896, 896, 1), 0); del buf223  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, down_proj_8, hidden_states_89, attn_output_39, hidden_states_95, pow_20, variance_19, add_62, rsqrt_19, hidden_states_97, hidden_states_98], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf228, buf187, buf204, buf210, buf227, arg170_1, buf230, 1, 896, stream=raw_stream0)
            del arg170_1
            del buf187
            assert_size_stride(arg171_1, (4864, 896), (896, 1))
            buf231 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [pow_20, variance_19, add_62, rsqrt_19, hidden_states_97, hidden_states_98, linear_67], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf230, (1, 896), (0, 1), 0), reinterpret_tensor(arg171_1, (896, 4864), (1, 896), 0), out=buf231)
            del arg171_1
            assert_size_stride(arg172_1, (4864, 896), (896, 1))
            buf232 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf230, (1, 896), (896, 1), 0), reinterpret_tensor(arg172_1, (896, 4864), (1, 896), 0), out=buf232)
            del arg172_1
            buf233 = reinterpret_tensor(buf231, (1, 1, 4864), (4864, 4864, 1), 0); del buf231  # reuse
            # Topologically Sorted Source Nodes: [linear_67, silu_9, linear_68, mul_91], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf233, buf232, 4864, stream=raw_stream0)
            del buf232
            assert_size_stride(arg173_1, (896, 4864), (4864, 1))
            buf234 = reinterpret_tensor(buf230, (1, 896), (896, 1), 0); del buf230  # reuse
            # Topologically Sorted Source Nodes: [linear_67, silu_9, linear_68, mul_91, down_proj_9], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf233, (1, 4864), (0, 1), 0), reinterpret_tensor(arg173_1, (4864, 896), (1, 4864), 0), out=buf234)
            del arg173_1
            del buf233
            assert_size_stride(arg175_1, (896, ), (1, ))
            assert_size_stride(arg174_1, (), ())
            buf236 = reinterpret_tensor(buf227, (1, 1, 896), (896, 896, 1), 0); del buf227  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, pow_21, variance_20, rsqrt_20, hidden_states_101, hidden_states_102], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf228, buf234, arg175_1, arg174_1.item(), buf236, 1, 896, stream=raw_stream0)
            del arg174_1
            del arg175_1
            assert_size_stride(arg177_1, (896, ), (1, ))
            assert_size_stride(arg176_1, (896, 896), (896, 1))
            buf237 = buf210; del buf210  # reuse
            # Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg177_1, reinterpret_tensor(buf236, (1, 896), (896, 1), 0), reinterpret_tensor(arg176_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf237)
            del arg176_1
            del arg177_1
            assert_size_stride(arg179_1, (128, ), (1, ))
            assert_size_stride(arg178_1, (128, 896), (896, 1))
            buf238 = buf216; del buf216  # reuse
            # Topologically Sorted Source Nodes: [linear_71], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg179_1, reinterpret_tensor(buf236, (1, 896), (896, 1), 0), reinterpret_tensor(arg178_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf238)
            del arg178_1
            del arg179_1
            assert_size_stride(arg182_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg182_1 = copy_misaligned(arg182_1)
            buf239 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_13, sin_13, linear_71, view_31, key_states_10, mul_96, x2_21, neg_21, x1_21, cat_42, mul_97, k_embed_10, keys_10], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg182_1, buf238, arg4_1, arg5_1.item(), buf239, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg182_1
            assert_size_stride(arg181_1, (128, ), (1, ))
            assert_size_stride(arg180_1, (128, 896), (896, 1))
            buf240 = buf238; del buf238  # reuse
            # Topologically Sorted Source Nodes: [linear_72], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg181_1, reinterpret_tensor(buf236, (1, 896), (896, 1), 0), reinterpret_tensor(arg180_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf240)
            del arg180_1
            del arg181_1
            assert_size_stride(arg184_1, (1, 2, s17, 64), (128*s17, 64, 128, 1))
            arg184_1 = copy_misaligned(arg184_1)
            ps22 = 1 + s17
            ps23 = 64 + 64*s17
            buf241 = empty_strided_cuda((1, 2, 1 + s17, 64), (128 + 128*s17, 64 + 64*s17, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_72, view_32, value_states_10, values_10], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s17
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg184_1, buf240, buf241, ps22, s17, ps23, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg184_1
            buf242 = reinterpret_tensor(buf236, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf236  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_70, view_30, query_states_10, cos_13, mul_94, x2_20, neg_20, x1_20, cat_41, sin_13, mul_95, q_embed_10, getitem_166, hidden_states_103, key_10, getitem_171, hidden_states_104, value_10, attn_output_40], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf237, arg4_1, arg5_1.item(), buf242, s108, 896, stream=raw_stream0)
            buf243 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_70, view_30, query_states_10, cos_13, mul_94, x2_20, neg_20, x1_20, cat_41, sin_13, mul_95, q_embed_10, getitem_166, hidden_states_103, key_10, getitem_171, hidden_states_104, value_10, attn_output_40], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf239, buf243, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf244 = empty_strided_cuda((1, 14, 1 + s17, 64), (896 + 896*s17, 64 + 64*s17, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_70, view_30, query_states_10, cos_13, mul_94, x2_20, neg_20, x1_20, cat_41, sin_13, mul_95, q_embed_10, getitem_166, hidden_states_103, key_10, getitem_171, hidden_states_104, value_10, attn_output_40], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s17
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf241, buf244, ps23, s17, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf245 = buf221; del buf221  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_70, view_30, query_states_10, cos_13, mul_94, x2_20, neg_20, x1_20, cat_41, sin_13, mul_95, q_embed_10, getitem_166, hidden_states_103, key_10, getitem_171, hidden_states_104, value_10, attn_output_40], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf245, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_70, view_30, query_states_10, cos_13, mul_94, x2_20, neg_20, x1_20, cat_41, sin_13, mul_95, q_embed_10, getitem_166, hidden_states_103, key_10, getitem_171, hidden_states_104, value_10, attn_output_40], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf246 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf242, buf243, buf244, reinterpret_tensor(buf245, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf243
            del buf244
            buf247 = buf246[0]
            assert_size_stride(buf247, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf247, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf246
            assert_size_stride(arg185_1, (896, 896), (896, 1))
            buf251 = reinterpret_tensor(buf242, (1, 896), (896, 1), 0); del buf242  # reuse
            # Topologically Sorted Source Nodes: [transpose_44, reshape_32, attn_output_43], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf247, (1, 896), (896, 1), 0), reinterpret_tensor(arg185_1, (896, 896), (1, 896), 0), out=buf251)
            del arg185_1
            assert_size_stride(arg186_1, (896, ), (1, ))
            buf253 = reinterpret_tensor(buf247, (1, 1, 896), (896, 896, 1), 0); del buf247  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, pow_22, variance_21, add_68, rsqrt_21, hidden_states_107, hidden_states_108], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf228, buf234, buf251, arg186_1, buf253, 1, 896, stream=raw_stream0)
            del arg186_1
            assert_size_stride(arg187_1, (4864, 896), (896, 1))
            buf254 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_74], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf253, (1, 896), (896, 1), 0), reinterpret_tensor(arg187_1, (896, 4864), (1, 896), 0), out=buf254)
            del arg187_1
            assert_size_stride(arg188_1, (4864, 896), (896, 1))
            buf255 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_75], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf253, (1, 896), (896, 1), 0), reinterpret_tensor(arg188_1, (896, 4864), (1, 896), 0), out=buf255)
            del arg188_1
            buf256 = reinterpret_tensor(buf254, (1, 1, 4864), (4864, 4864, 1), 0); del buf254  # reuse
            # Topologically Sorted Source Nodes: [linear_74, silu_10, linear_75, mul_100], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf256, buf255, 4864, stream=raw_stream0)
            del buf255
            assert_size_stride(arg189_1, (896, 4864), (4864, 1))
            buf257 = reinterpret_tensor(buf253, (1, 896), (896, 1), 0); del buf253  # reuse
            # Topologically Sorted Source Nodes: [linear_74, silu_10, linear_75, mul_100, down_proj_10], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf256, (1, 4864), (0, 1), 0), reinterpret_tensor(arg189_1, (4864, 896), (1, 4864), 0), out=buf257)
            del arg189_1
            del buf256
            assert_size_stride(arg191_1, (896, ), (1, ))
            assert_size_stride(arg190_1, (), ())
            buf259 = reinterpret_tensor(buf237, (1, 1, 896), (896, 896, 1), 0); del buf237  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, down_proj_10, hidden_states_109, pow_23, variance_22, rsqrt_22, hidden_states_111, hidden_states_112], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf228, buf234, buf251, buf257, arg191_1, arg190_1.item(), buf259, 1, 896, stream=raw_stream0)
            del arg190_1
            del arg191_1
            assert_size_stride(arg193_1, (896, ), (1, ))
            assert_size_stride(arg192_1, (896, 896), (896, 1))
            buf260 = buf204; del buf204  # reuse
            # Topologically Sorted Source Nodes: [linear_77], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg193_1, reinterpret_tensor(buf259, (1, 896), (896, 1), 0), reinterpret_tensor(arg192_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf260)
            del arg192_1
            del arg193_1
            assert_size_stride(arg195_1, (128, ), (1, ))
            assert_size_stride(arg194_1, (128, 896), (896, 1))
            buf261 = buf240; del buf240  # reuse
            # Topologically Sorted Source Nodes: [linear_78], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg195_1, reinterpret_tensor(buf259, (1, 896), (896, 1), 0), reinterpret_tensor(arg194_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf261)
            del arg194_1
            del arg195_1
            assert_size_stride(arg198_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg198_1 = copy_misaligned(arg198_1)
            buf262 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_14, sin_14, linear_78, view_34, key_states_11, mul_105, x2_23, neg_23, x1_23, cat_46, mul_106, k_embed_11, keys_11], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg198_1, buf261, arg4_1, arg5_1.item(), buf262, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg198_1
            assert_size_stride(arg197_1, (128, ), (1, ))
            assert_size_stride(arg196_1, (128, 896), (896, 1))
            buf263 = buf261; del buf261  # reuse
            # Topologically Sorted Source Nodes: [linear_79], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg197_1, reinterpret_tensor(buf259, (1, 896), (896, 1), 0), reinterpret_tensor(arg196_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf263)
            del arg196_1
            del arg197_1
            assert_size_stride(arg200_1, (1, 2, s31, 64), (128*s31, 64, 128, 1))
            arg200_1 = copy_misaligned(arg200_1)
            ps24 = 1 + s31
            ps25 = 64 + 64*s31
            buf264 = empty_strided_cuda((1, 2, 1 + s31, 64), (128 + 128*s31, 64 + 64*s31, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_79, view_35, value_states_11, values_11], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s31
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg200_1, buf263, buf264, ps24, s31, ps25, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg200_1
            buf265 = reinterpret_tensor(buf259, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf259  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_77, view_33, query_states_11, cos_14, mul_103, x2_22, neg_22, x1_22, cat_45, sin_14, mul_104, q_embed_11, getitem_180, hidden_states_113, key_11, getitem_185, hidden_states_114, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf260, arg4_1, arg5_1.item(), buf265, s108, 896, stream=raw_stream0)
            del buf260
            buf266 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_77, view_33, query_states_11, cos_14, mul_103, x2_22, neg_22, x1_22, cat_45, sin_14, mul_104, q_embed_11, getitem_180, hidden_states_113, key_11, getitem_185, hidden_states_114, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf262, buf266, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf267 = empty_strided_cuda((1, 14, 1 + s31, 64), (896 + 896*s31, 64 + 64*s31, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_77, view_33, query_states_11, cos_14, mul_103, x2_22, neg_22, x1_22, cat_45, sin_14, mul_104, q_embed_11, getitem_180, hidden_states_113, key_11, getitem_185, hidden_states_114, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s31
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf264, buf267, ps25, s31, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf268 = buf245; del buf245  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_77, view_33, query_states_11, cos_14, mul_103, x2_22, neg_22, x1_22, cat_45, sin_14, mul_104, q_embed_11, getitem_180, hidden_states_113, key_11, getitem_185, hidden_states_114, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf268, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_77, view_33, query_states_11, cos_14, mul_103, x2_22, neg_22, x1_22, cat_45, sin_14, mul_104, q_embed_11, getitem_180, hidden_states_113, key_11, getitem_185, hidden_states_114, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf269 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf265, buf266, buf267, reinterpret_tensor(buf268, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf266
            del buf267
            buf270 = buf269[0]
            assert_size_stride(buf270, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf270, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf269
            assert_size_stride(arg201_1, (896, 896), (896, 1))
            buf274 = reinterpret_tensor(buf265, (1, 896), (896, 1), 0); del buf265  # reuse
            # Topologically Sorted Source Nodes: [transpose_48, reshape_35, attn_output_47], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf270, (1, 896), (896, 1), 0), reinterpret_tensor(arg201_1, (896, 896), (1, 896), 0), out=buf274)
            del arg201_1
            assert_size_stride(arg203_1, (896, ), (1, ))
            assert_size_stride(arg202_1, (), ())
            buf275 = buf228; del buf228  # reuse
            buf277 = reinterpret_tensor(buf270, (1, 1, 896), (896, 896, 1), 0); del buf270  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, down_proj_10, hidden_states_109, attn_output_47, hidden_states_115, pow_24, variance_23, rsqrt_23, hidden_states_117, hidden_states_118], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf275, buf234, buf251, buf257, buf274, arg203_1, arg202_1.item(), buf277, 1, 896, stream=raw_stream0)
            del arg202_1
            del arg203_1
            del buf234
            assert_size_stride(arg204_1, (4864, 896), (896, 1))
            buf278 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [pow_24, variance_23, rsqrt_23, hidden_states_117, hidden_states_118, linear_81], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf277, (1, 896), (0, 1), 0), reinterpret_tensor(arg204_1, (896, 4864), (1, 896), 0), out=buf278)
            del arg204_1
            assert_size_stride(arg205_1, (4864, 896), (896, 1))
            buf279 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf277, (1, 896), (896, 1), 0), reinterpret_tensor(arg205_1, (896, 4864), (1, 896), 0), out=buf279)
            del arg205_1
            buf280 = reinterpret_tensor(buf278, (1, 1, 4864), (4864, 4864, 1), 0); del buf278  # reuse
            # Topologically Sorted Source Nodes: [linear_81, silu_11, linear_82, mul_109], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf280, buf279, 4864, stream=raw_stream0)
            del buf279
            assert_size_stride(arg206_1, (896, 4864), (4864, 1))
            buf281 = reinterpret_tensor(buf277, (1, 896), (896, 1), 0); del buf277  # reuse
            # Topologically Sorted Source Nodes: [linear_81, silu_11, linear_82, mul_109, down_proj_11], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf280, (1, 4864), (0, 1), 0), reinterpret_tensor(arg206_1, (4864, 896), (1, 4864), 0), out=buf281)
            del arg206_1
            del buf280
            assert_size_stride(arg208_1, (896, ), (1, ))
            assert_size_stride(arg207_1, (), ())
            buf283 = reinterpret_tensor(buf274, (1, 1, 896), (896, 896, 1), 0); del buf274  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, pow_25, variance_24, rsqrt_24, hidden_states_121, hidden_states_122], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf275, buf281, arg208_1, arg207_1.item(), buf283, 1, 896, stream=raw_stream0)
            del arg207_1
            del arg208_1
            assert_size_stride(arg210_1, (896, ), (1, ))
            assert_size_stride(arg209_1, (896, 896), (896, 1))
            buf284 = buf257; del buf257  # reuse
            # Topologically Sorted Source Nodes: [linear_84], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg210_1, reinterpret_tensor(buf283, (1, 896), (896, 1), 0), reinterpret_tensor(arg209_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf284)
            del arg209_1
            del arg210_1
            assert_size_stride(arg212_1, (128, ), (1, ))
            assert_size_stride(arg211_1, (128, 896), (896, 1))
            buf285 = buf263; del buf263  # reuse
            # Topologically Sorted Source Nodes: [linear_85], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg212_1, reinterpret_tensor(buf283, (1, 896), (896, 1), 0), reinterpret_tensor(arg211_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf285)
            del arg211_1
            del arg212_1
            assert_size_stride(arg215_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg215_1 = copy_misaligned(arg215_1)
            buf286 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_15, sin_15, linear_85, view_37, key_states_12, mul_114, x2_25, neg_25, x1_25, cat_50, mul_115, k_embed_12, keys_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg215_1, buf285, arg4_1, arg5_1.item(), buf286, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg215_1
            assert_size_stride(arg214_1, (128, ), (1, ))
            assert_size_stride(arg213_1, (128, 896), (896, 1))
            buf287 = buf285; del buf285  # reuse
            # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg214_1, reinterpret_tensor(buf283, (1, 896), (896, 1), 0), reinterpret_tensor(arg213_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf287)
            del arg213_1
            del arg214_1
            assert_size_stride(arg217_1, (1, 2, s65, 64), (128*s65, 64, 128, 1))
            arg217_1 = copy_misaligned(arg217_1)
            ps26 = 1 + s65
            ps27 = 64 + 64*s65
            buf288 = empty_strided_cuda((1, 2, 1 + s65, 64), (128 + 128*s65, 64 + 64*s65, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_86, view_38, value_states_12, values_12], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s65
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg217_1, buf287, buf288, ps26, s65, ps27, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg217_1
            buf289 = reinterpret_tensor(buf283, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf283  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_84, view_36, query_states_12, cos_15, mul_112, x2_24, neg_24, x1_24, cat_49, sin_15, mul_113, q_embed_12, getitem_194, hidden_states_123, key_12, getitem_199, hidden_states_124, value_12, attn_output_48], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf284, arg4_1, arg5_1.item(), buf289, s108, 896, stream=raw_stream0)
            buf290 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_84, view_36, query_states_12, cos_15, mul_112, x2_24, neg_24, x1_24, cat_49, sin_15, mul_113, q_embed_12, getitem_194, hidden_states_123, key_12, getitem_199, hidden_states_124, value_12, attn_output_48], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf286, buf290, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf291 = empty_strided_cuda((1, 14, 1 + s65, 64), (896 + 896*s65, 64 + 64*s65, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_84, view_36, query_states_12, cos_15, mul_112, x2_24, neg_24, x1_24, cat_49, sin_15, mul_113, q_embed_12, getitem_194, hidden_states_123, key_12, getitem_199, hidden_states_124, value_12, attn_output_48], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s65
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf288, buf291, ps27, s65, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf292 = buf268; del buf268  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_84, view_36, query_states_12, cos_15, mul_112, x2_24, neg_24, x1_24, cat_49, sin_15, mul_113, q_embed_12, getitem_194, hidden_states_123, key_12, getitem_199, hidden_states_124, value_12, attn_output_48], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf292, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_84, view_36, query_states_12, cos_15, mul_112, x2_24, neg_24, x1_24, cat_49, sin_15, mul_113, q_embed_12, getitem_194, hidden_states_123, key_12, getitem_199, hidden_states_124, value_12, attn_output_48], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf293 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf289, buf290, buf291, reinterpret_tensor(buf292, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf290
            del buf291
            buf294 = buf293[0]
            assert_size_stride(buf294, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf294, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf293
            assert_size_stride(arg218_1, (896, 896), (896, 1))
            buf298 = reinterpret_tensor(buf289, (1, 896), (896, 1), 0); del buf289  # reuse
            # Topologically Sorted Source Nodes: [transpose_52, reshape_38, attn_output_51], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf294, (1, 896), (896, 1), 0), reinterpret_tensor(arg218_1, (896, 896), (1, 896), 0), out=buf298)
            del arg218_1
            assert_size_stride(arg220_1, (896, ), (1, ))
            assert_size_stride(arg219_1, (), ())
            buf300 = reinterpret_tensor(buf294, (1, 1, 896), (896, 896, 1), 0); del buf294  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, attn_output_51, hidden_states_125, pow_26, variance_25, rsqrt_25, hidden_states_127, hidden_states_128], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf275, buf281, buf298, arg220_1, arg219_1.item(), buf300, 1, 896, stream=raw_stream0)
            del arg219_1
            del arg220_1
            assert_size_stride(arg221_1, (4864, 896), (896, 1))
            buf301 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf300, (1, 896), (896, 1), 0), reinterpret_tensor(arg221_1, (896, 4864), (1, 896), 0), out=buf301)
            del arg221_1
            assert_size_stride(arg222_1, (4864, 896), (896, 1))
            buf302 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf300, (1, 896), (896, 1), 0), reinterpret_tensor(arg222_1, (896, 4864), (1, 896), 0), out=buf302)
            del arg222_1
            buf303 = reinterpret_tensor(buf301, (1, 1, 4864), (4864, 4864, 1), 0); del buf301  # reuse
            # Topologically Sorted Source Nodes: [linear_88, silu_12, linear_89, mul_118], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf303, buf302, 4864, stream=raw_stream0)
            del buf302
            assert_size_stride(arg223_1, (896, 4864), (4864, 1))
            buf304 = reinterpret_tensor(buf300, (1, 896), (896, 1), 0); del buf300  # reuse
            # Topologically Sorted Source Nodes: [linear_88, silu_12, linear_89, mul_118, down_proj_12], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf303, (1, 4864), (0, 1), 0), reinterpret_tensor(arg223_1, (4864, 896), (1, 4864), 0), out=buf304)
            del arg223_1
            del buf303
            assert_size_stride(arg225_1, (896, ), (1, ))
            assert_size_stride(arg224_1, (), ())
            buf306 = reinterpret_tensor(buf284, (1, 1, 896), (896, 896, 1), 0); del buf284  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, attn_output_51, hidden_states_125, down_proj_12, hidden_states_129, pow_27, variance_26, rsqrt_26, hidden_states_131, hidden_states_132], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf275, buf281, buf298, buf304, arg225_1, arg224_1.item(), buf306, 1, 896, stream=raw_stream0)
            del arg224_1
            del arg225_1
            assert_size_stride(arg227_1, (896, ), (1, ))
            assert_size_stride(arg226_1, (896, 896), (896, 1))
            buf307 = buf251; del buf251  # reuse
            # Topologically Sorted Source Nodes: [linear_91], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg227_1, reinterpret_tensor(buf306, (1, 896), (896, 1), 0), reinterpret_tensor(arg226_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf307)
            del arg226_1
            del arg227_1
            assert_size_stride(arg229_1, (128, ), (1, ))
            assert_size_stride(arg228_1, (128, 896), (896, 1))
            buf308 = buf287; del buf287  # reuse
            # Topologically Sorted Source Nodes: [linear_92], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg229_1, reinterpret_tensor(buf306, (1, 896), (896, 1), 0), reinterpret_tensor(arg228_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf308)
            del arg228_1
            del arg229_1
            assert_size_stride(arg232_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg232_1 = copy_misaligned(arg232_1)
            buf309 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_16, sin_16, linear_92, view_40, key_states_13, mul_123, x2_27, neg_27, x1_27, cat_54, mul_124, k_embed_13, keys_13], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg232_1, buf308, arg4_1, arg5_1.item(), buf309, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg232_1
            assert_size_stride(arg231_1, (128, ), (1, ))
            assert_size_stride(arg230_1, (128, 896), (896, 1))
            buf310 = buf308; del buf308  # reuse
            # Topologically Sorted Source Nodes: [linear_93], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg231_1, reinterpret_tensor(buf306, (1, 896), (896, 1), 0), reinterpret_tensor(arg230_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf310)
            del arg230_1
            del arg231_1
            assert_size_stride(arg234_1, (1, 2, s45, 64), (128*s45, 64, 128, 1))
            arg234_1 = copy_misaligned(arg234_1)
            ps28 = 1 + s45
            ps29 = 64 + 64*s45
            buf311 = empty_strided_cuda((1, 2, 1 + s45, 64), (128 + 128*s45, 64 + 64*s45, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_93, view_41, value_states_13, values_13], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s45
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg234_1, buf310, buf311, ps28, s45, ps29, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg234_1
            buf312 = reinterpret_tensor(buf306, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf306  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_91, view_39, query_states_13, cos_16, mul_121, x2_26, neg_26, x1_26, cat_53, sin_16, mul_122, q_embed_13, getitem_208, hidden_states_133, key_13, getitem_213, hidden_states_134, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf307, arg4_1, arg5_1.item(), buf312, s108, 896, stream=raw_stream0)
            del buf307
            buf313 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_91, view_39, query_states_13, cos_16, mul_121, x2_26, neg_26, x1_26, cat_53, sin_16, mul_122, q_embed_13, getitem_208, hidden_states_133, key_13, getitem_213, hidden_states_134, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf309, buf313, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf314 = empty_strided_cuda((1, 14, 1 + s45, 64), (896 + 896*s45, 64 + 64*s45, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_91, view_39, query_states_13, cos_16, mul_121, x2_26, neg_26, x1_26, cat_53, sin_16, mul_122, q_embed_13, getitem_208, hidden_states_133, key_13, getitem_213, hidden_states_134, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s45
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf311, buf314, ps29, s45, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf315 = buf292; del buf292  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_91, view_39, query_states_13, cos_16, mul_121, x2_26, neg_26, x1_26, cat_53, sin_16, mul_122, q_embed_13, getitem_208, hidden_states_133, key_13, getitem_213, hidden_states_134, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf315, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_91, view_39, query_states_13, cos_16, mul_121, x2_26, neg_26, x1_26, cat_53, sin_16, mul_122, q_embed_13, getitem_208, hidden_states_133, key_13, getitem_213, hidden_states_134, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf316 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf312, buf313, buf314, reinterpret_tensor(buf315, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf313
            del buf314
            buf317 = buf316[0]
            assert_size_stride(buf317, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf317, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf316
            assert_size_stride(arg235_1, (896, 896), (896, 1))
            buf321 = reinterpret_tensor(buf312, (1, 896), (896, 1), 0); del buf312  # reuse
            # Topologically Sorted Source Nodes: [transpose_56, reshape_41, attn_output_55], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf317, (1, 896), (896, 1), 0), reinterpret_tensor(arg235_1, (896, 896), (1, 896), 0), out=buf321)
            del arg235_1
            assert_size_stride(arg237_1, (896, ), (1, ))
            assert_size_stride(arg236_1, (), ())
            buf322 = buf275; del buf275  # reuse
            buf324 = reinterpret_tensor(buf317, (1, 1, 896), (896, 896, 1), 0); del buf317  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, attn_output_51, hidden_states_125, down_proj_12, hidden_states_129, attn_output_55, hidden_states_135, pow_28, variance_27, rsqrt_27, hidden_states_137, hidden_states_138], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf322, buf281, buf298, buf304, buf321, arg237_1, arg236_1.item(), buf324, 1, 896, stream=raw_stream0)
            del arg236_1
            del arg237_1
            del buf281
            assert_size_stride(arg238_1, (4864, 896), (896, 1))
            buf325 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [pow_28, variance_27, rsqrt_27, hidden_states_137, hidden_states_138, linear_95], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf324, (1, 896), (0, 1), 0), reinterpret_tensor(arg238_1, (896, 4864), (1, 896), 0), out=buf325)
            del arg238_1
            assert_size_stride(arg239_1, (4864, 896), (896, 1))
            buf326 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_96], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf324, (1, 896), (896, 1), 0), reinterpret_tensor(arg239_1, (896, 4864), (1, 896), 0), out=buf326)
            del arg239_1
            buf327 = reinterpret_tensor(buf325, (1, 1, 4864), (4864, 4864, 1), 0); del buf325  # reuse
            # Topologically Sorted Source Nodes: [linear_95, silu_13, linear_96, mul_127], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf327, buf326, 4864, stream=raw_stream0)
            del buf326
            assert_size_stride(arg240_1, (896, 4864), (4864, 1))
            buf328 = reinterpret_tensor(buf324, (1, 896), (896, 1), 0); del buf324  # reuse
            # Topologically Sorted Source Nodes: [linear_95, silu_13, linear_96, mul_127, down_proj_13], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf327, (1, 4864), (0, 1), 0), reinterpret_tensor(arg240_1, (4864, 896), (1, 4864), 0), out=buf328)
            del arg240_1
            del buf327
            assert_size_stride(arg241_1, (896, ), (1, ))
            buf330 = reinterpret_tensor(buf321, (1, 1, 896), (896, 896, 1), 0); del buf321  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, pow_29, variance_28, add_88, rsqrt_28, hidden_states_141, hidden_states_142], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_16.run(buf322, buf328, arg241_1, buf330, 1, 896, stream=raw_stream0)
            del arg241_1
            assert_size_stride(arg243_1, (896, ), (1, ))
            assert_size_stride(arg242_1, (896, 896), (896, 1))
            buf331 = buf304; del buf304  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, pow_29, variance_28, add_88, rsqrt_28, hidden_states_141, hidden_states_142, linear_98], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg243_1, reinterpret_tensor(buf330, (1, 896), (0, 1), 0), reinterpret_tensor(arg242_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf331)
            del arg242_1
            del arg243_1
            assert_size_stride(arg245_1, (128, ), (1, ))
            assert_size_stride(arg244_1, (128, 896), (896, 1))
            buf332 = buf310; del buf310  # reuse
            # Topologically Sorted Source Nodes: [linear_99], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg245_1, reinterpret_tensor(buf330, (1, 896), (896, 1), 0), reinterpret_tensor(arg244_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf332)
            del arg244_1
            del arg245_1
            assert_size_stride(arg248_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg248_1 = copy_misaligned(arg248_1)
            buf333 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_17, sin_17, linear_99, view_43, key_states_14, mul_132, x2_29, neg_29, x1_29, cat_58, mul_133, k_embed_14, keys_14], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg248_1, buf332, arg4_1, arg5_1.item(), buf333, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg248_1
            assert_size_stride(arg247_1, (128, ), (1, ))
            assert_size_stride(arg246_1, (128, 896), (896, 1))
            buf334 = buf332; del buf332  # reuse
            # Topologically Sorted Source Nodes: [linear_100], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg247_1, reinterpret_tensor(buf330, (1, 896), (896, 1), 0), reinterpret_tensor(arg246_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf334)
            del arg246_1
            del arg247_1
            assert_size_stride(arg250_1, (1, 2, s100, 64), (128*s100, 64, 128, 1))
            arg250_1 = copy_misaligned(arg250_1)
            ps30 = 1 + s100
            ps31 = 64 + 64*s100
            buf335 = empty_strided_cuda((1, 2, 1 + s100, 64), (128 + 128*s100, 64 + 64*s100, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_100, view_44, value_states_14, values_14], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s100
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg250_1, buf334, buf335, ps30, s100, ps31, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg250_1
            buf336 = reinterpret_tensor(buf330, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf330  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_98, view_42, query_states_14, cos_17, mul_130, x2_28, neg_28, x1_28, cat_57, sin_17, mul_131, q_embed_14, getitem_222, hidden_states_143, key_14, getitem_227, hidden_states_144, value_14, attn_output_56], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf331, arg4_1, arg5_1.item(), buf336, s108, 896, stream=raw_stream0)
            buf337 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_98, view_42, query_states_14, cos_17, mul_130, x2_28, neg_28, x1_28, cat_57, sin_17, mul_131, q_embed_14, getitem_222, hidden_states_143, key_14, getitem_227, hidden_states_144, value_14, attn_output_56], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf333, buf337, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf338 = empty_strided_cuda((1, 14, 1 + s100, 64), (896 + 896*s100, 64 + 64*s100, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_98, view_42, query_states_14, cos_17, mul_130, x2_28, neg_28, x1_28, cat_57, sin_17, mul_131, q_embed_14, getitem_222, hidden_states_143, key_14, getitem_227, hidden_states_144, value_14, attn_output_56], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s100
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf335, buf338, ps31, s100, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf339 = buf315; del buf315  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_98, view_42, query_states_14, cos_17, mul_130, x2_28, neg_28, x1_28, cat_57, sin_17, mul_131, q_embed_14, getitem_222, hidden_states_143, key_14, getitem_227, hidden_states_144, value_14, attn_output_56], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf339, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_98, view_42, query_states_14, cos_17, mul_130, x2_28, neg_28, x1_28, cat_57, sin_17, mul_131, q_embed_14, getitem_222, hidden_states_143, key_14, getitem_227, hidden_states_144, value_14, attn_output_56], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf340 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf336, buf337, buf338, reinterpret_tensor(buf339, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf337
            del buf338
            buf341 = buf340[0]
            assert_size_stride(buf341, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf341, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf340
            assert_size_stride(arg251_1, (896, 896), (896, 1))
            buf345 = reinterpret_tensor(buf336, (1, 896), (896, 1), 0); del buf336  # reuse
            # Topologically Sorted Source Nodes: [transpose_60, reshape_44, attn_output_59], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf341, (1, 896), (896, 1), 0), reinterpret_tensor(arg251_1, (896, 896), (1, 896), 0), out=buf345)
            del arg251_1
            assert_size_stride(arg253_1, (896, ), (1, ))
            assert_size_stride(arg252_1, (), ())
            buf347 = reinterpret_tensor(buf341, (1, 1, 896), (896, 896, 1), 0); del buf341  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, attn_output_59, hidden_states_145, pow_30, variance_29, rsqrt_29, hidden_states_147, hidden_states_148], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf322, buf328, buf345, arg253_1, arg252_1.item(), buf347, 1, 896, stream=raw_stream0)
            del arg252_1
            del arg253_1
            assert_size_stride(arg254_1, (4864, 896), (896, 1))
            buf348 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_102], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf347, (1, 896), (896, 1), 0), reinterpret_tensor(arg254_1, (896, 4864), (1, 896), 0), out=buf348)
            del arg254_1
            assert_size_stride(arg255_1, (4864, 896), (896, 1))
            buf349 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_103], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf347, (1, 896), (896, 1), 0), reinterpret_tensor(arg255_1, (896, 4864), (1, 896), 0), out=buf349)
            del arg255_1
            buf350 = reinterpret_tensor(buf348, (1, 1, 4864), (4864, 4864, 1), 0); del buf348  # reuse
            # Topologically Sorted Source Nodes: [linear_102, silu_14, linear_103, mul_136], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf350, buf349, 4864, stream=raw_stream0)
            del buf349
            assert_size_stride(arg256_1, (896, 4864), (4864, 1))
            buf351 = reinterpret_tensor(buf347, (1, 896), (896, 1), 0); del buf347  # reuse
            # Topologically Sorted Source Nodes: [linear_102, silu_14, linear_103, mul_136, down_proj_14], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf350, (1, 4864), (0, 1), 0), reinterpret_tensor(arg256_1, (4864, 896), (1, 4864), 0), out=buf351)
            del arg256_1
            del buf350
            assert_size_stride(arg258_1, (896, ), (1, ))
            assert_size_stride(arg257_1, (), ())
            buf353 = reinterpret_tensor(buf331, (1, 1, 896), (896, 896, 1), 0); del buf331  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, attn_output_59, hidden_states_145, down_proj_14, hidden_states_149, pow_31, variance_30, rsqrt_30, hidden_states_151, hidden_states_152], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf322, buf328, buf345, buf351, arg258_1, arg257_1.item(), buf353, 1, 896, stream=raw_stream0)
            del arg257_1
            del arg258_1
            assert_size_stride(arg260_1, (896, ), (1, ))
            assert_size_stride(arg259_1, (896, 896), (896, 1))
            buf354 = buf298; del buf298  # reuse
            # Topologically Sorted Source Nodes: [linear_105], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg260_1, reinterpret_tensor(buf353, (1, 896), (896, 1), 0), reinterpret_tensor(arg259_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf354)
            del arg259_1
            del arg260_1
            assert_size_stride(arg262_1, (128, ), (1, ))
            assert_size_stride(arg261_1, (128, 896), (896, 1))
            buf355 = buf334; del buf334  # reuse
            # Topologically Sorted Source Nodes: [linear_106], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg262_1, reinterpret_tensor(buf353, (1, 896), (896, 1), 0), reinterpret_tensor(arg261_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf355)
            del arg261_1
            del arg262_1
            assert_size_stride(arg265_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg265_1 = copy_misaligned(arg265_1)
            buf356 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_18, sin_18, linear_106, view_46, key_states_15, mul_141, x2_31, neg_31, x1_31, cat_62, mul_142, k_embed_15, keys_15], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg265_1, buf355, arg4_1, arg5_1.item(), buf356, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg265_1
            assert_size_stride(arg264_1, (128, ), (1, ))
            assert_size_stride(arg263_1, (128, 896), (896, 1))
            buf357 = buf355; del buf355  # reuse
            # Topologically Sorted Source Nodes: [linear_107], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg264_1, reinterpret_tensor(buf353, (1, 896), (896, 1), 0), reinterpret_tensor(arg263_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf357)
            del arg263_1
            del arg264_1
            assert_size_stride(arg267_1, (1, 2, s102, 64), (128*s102, 64, 128, 1))
            arg267_1 = copy_misaligned(arg267_1)
            ps32 = 1 + s102
            ps33 = 64 + 64*s102
            buf358 = empty_strided_cuda((1, 2, 1 + s102, 64), (128 + 128*s102, 64 + 64*s102, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_107, view_47, value_states_15, values_15], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s102
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg267_1, buf357, buf358, ps32, s102, ps33, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg267_1
            buf359 = reinterpret_tensor(buf353, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf353  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_105, view_45, query_states_15, cos_18, mul_139, x2_30, neg_30, x1_30, cat_61, sin_18, mul_140, q_embed_15, getitem_236, hidden_states_153, key_15, getitem_241, hidden_states_154, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf354, arg4_1, arg5_1.item(), buf359, s108, 896, stream=raw_stream0)
            del buf354
            buf360 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_105, view_45, query_states_15, cos_18, mul_139, x2_30, neg_30, x1_30, cat_61, sin_18, mul_140, q_embed_15, getitem_236, hidden_states_153, key_15, getitem_241, hidden_states_154, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf356, buf360, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf361 = empty_strided_cuda((1, 14, 1 + s102, 64), (896 + 896*s102, 64 + 64*s102, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_105, view_45, query_states_15, cos_18, mul_139, x2_30, neg_30, x1_30, cat_61, sin_18, mul_140, q_embed_15, getitem_236, hidden_states_153, key_15, getitem_241, hidden_states_154, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s102
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf358, buf361, ps33, s102, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf362 = buf339; del buf339  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_105, view_45, query_states_15, cos_18, mul_139, x2_30, neg_30, x1_30, cat_61, sin_18, mul_140, q_embed_15, getitem_236, hidden_states_153, key_15, getitem_241, hidden_states_154, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf362, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_105, view_45, query_states_15, cos_18, mul_139, x2_30, neg_30, x1_30, cat_61, sin_18, mul_140, q_embed_15, getitem_236, hidden_states_153, key_15, getitem_241, hidden_states_154, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf363 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf359, buf360, buf361, reinterpret_tensor(buf362, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf360
            del buf361
            buf364 = buf363[0]
            assert_size_stride(buf364, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf364, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf363
            assert_size_stride(arg268_1, (896, 896), (896, 1))
            buf368 = reinterpret_tensor(buf359, (1, 896), (896, 1), 0); del buf359  # reuse
            # Topologically Sorted Source Nodes: [transpose_64, reshape_47, attn_output_63], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf364, (1, 896), (896, 1), 0), reinterpret_tensor(arg268_1, (896, 896), (1, 896), 0), out=buf368)
            del arg268_1
            assert_size_stride(arg270_1, (896, ), (1, ))
            assert_size_stride(arg269_1, (), ())
            buf369 = buf322; del buf322  # reuse
            buf371 = reinterpret_tensor(buf364, (1, 1, 896), (896, 896, 1), 0); del buf364  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, attn_output_59, hidden_states_145, down_proj_14, hidden_states_149, attn_output_63, hidden_states_155, pow_32, variance_31, rsqrt_31, hidden_states_157, hidden_states_158], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf369, buf328, buf345, buf351, buf368, arg270_1, arg269_1.item(), buf371, 1, 896, stream=raw_stream0)
            del arg269_1
            del arg270_1
            del buf328
            assert_size_stride(arg271_1, (4864, 896), (896, 1))
            buf372 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [pow_32, variance_31, rsqrt_31, hidden_states_157, hidden_states_158, linear_109], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf371, (1, 896), (0, 1), 0), reinterpret_tensor(arg271_1, (896, 4864), (1, 896), 0), out=buf372)
            del arg271_1
            assert_size_stride(arg272_1, (4864, 896), (896, 1))
            buf373 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_110], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf371, (1, 896), (896, 1), 0), reinterpret_tensor(arg272_1, (896, 4864), (1, 896), 0), out=buf373)
            del arg272_1
            buf374 = reinterpret_tensor(buf372, (1, 1, 4864), (4864, 4864, 1), 0); del buf372  # reuse
            # Topologically Sorted Source Nodes: [linear_109, silu_15, linear_110, mul_145], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf374, buf373, 4864, stream=raw_stream0)
            del buf373
            assert_size_stride(arg273_1, (896, 4864), (4864, 1))
            buf375 = reinterpret_tensor(buf371, (1, 896), (896, 1), 0); del buf371  # reuse
            # Topologically Sorted Source Nodes: [linear_109, silu_15, linear_110, mul_145, down_proj_15], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf374, (1, 4864), (0, 1), 0), reinterpret_tensor(arg273_1, (4864, 896), (1, 4864), 0), out=buf375)
            del arg273_1
            del buf374
            assert_size_stride(arg275_1, (896, ), (1, ))
            assert_size_stride(arg274_1, (), ())
            buf377 = reinterpret_tensor(buf368, (1, 1, 896), (896, 896, 1), 0); del buf368  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, pow_33, variance_32, rsqrt_32, hidden_states_161, hidden_states_162], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf369, buf375, arg275_1, arg274_1.item(), buf377, 1, 896, stream=raw_stream0)
            del arg274_1
            del arg275_1
            assert_size_stride(arg277_1, (896, ), (1, ))
            assert_size_stride(arg276_1, (896, 896), (896, 1))
            buf378 = buf351; del buf351  # reuse
            # Topologically Sorted Source Nodes: [linear_112], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg277_1, reinterpret_tensor(buf377, (1, 896), (896, 1), 0), reinterpret_tensor(arg276_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf378)
            del arg276_1
            del arg277_1
            assert_size_stride(arg279_1, (128, ), (1, ))
            assert_size_stride(arg278_1, (128, 896), (896, 1))
            buf379 = buf357; del buf357  # reuse
            # Topologically Sorted Source Nodes: [linear_113], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg279_1, reinterpret_tensor(buf377, (1, 896), (896, 1), 0), reinterpret_tensor(arg278_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf379)
            del arg278_1
            del arg279_1
            assert_size_stride(arg282_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg282_1 = copy_misaligned(arg282_1)
            buf380 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_19, sin_19, linear_113, view_49, key_states_16, mul_150, x2_33, neg_33, x1_33, cat_66, mul_151, k_embed_16, keys_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg282_1, buf379, arg4_1, arg5_1.item(), buf380, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg282_1
            assert_size_stride(arg281_1, (128, ), (1, ))
            assert_size_stride(arg280_1, (128, 896), (896, 1))
            buf381 = buf379; del buf379  # reuse
            # Topologically Sorted Source Nodes: [linear_114], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg281_1, reinterpret_tensor(buf377, (1, 896), (896, 1), 0), reinterpret_tensor(arg280_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf381)
            del arg280_1
            del arg281_1
            assert_size_stride(arg284_1, (1, 2, s58, 64), (128*s58, 64, 128, 1))
            arg284_1 = copy_misaligned(arg284_1)
            ps34 = 1 + s58
            ps35 = 64 + 64*s58
            buf382 = empty_strided_cuda((1, 2, 1 + s58, 64), (128 + 128*s58, 64 + 64*s58, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_114, view_50, value_states_16, values_16], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s58
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg284_1, buf381, buf382, ps34, s58, ps35, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg284_1
            buf383 = reinterpret_tensor(buf377, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf377  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_112, view_48, query_states_16, cos_19, mul_148, x2_32, neg_32, x1_32, cat_65, sin_19, mul_149, q_embed_16, getitem_250, hidden_states_163, key_16, getitem_255, hidden_states_164, value_16, attn_output_64], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf378, arg4_1, arg5_1.item(), buf383, s108, 896, stream=raw_stream0)
            buf384 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_112, view_48, query_states_16, cos_19, mul_148, x2_32, neg_32, x1_32, cat_65, sin_19, mul_149, q_embed_16, getitem_250, hidden_states_163, key_16, getitem_255, hidden_states_164, value_16, attn_output_64], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf380, buf384, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf385 = empty_strided_cuda((1, 14, 1 + s58, 64), (896 + 896*s58, 64 + 64*s58, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_112, view_48, query_states_16, cos_19, mul_148, x2_32, neg_32, x1_32, cat_65, sin_19, mul_149, q_embed_16, getitem_250, hidden_states_163, key_16, getitem_255, hidden_states_164, value_16, attn_output_64], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s58
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf382, buf385, ps35, s58, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf386 = buf362; del buf362  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_112, view_48, query_states_16, cos_19, mul_148, x2_32, neg_32, x1_32, cat_65, sin_19, mul_149, q_embed_16, getitem_250, hidden_states_163, key_16, getitem_255, hidden_states_164, value_16, attn_output_64], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf386, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_112, view_48, query_states_16, cos_19, mul_148, x2_32, neg_32, x1_32, cat_65, sin_19, mul_149, q_embed_16, getitem_250, hidden_states_163, key_16, getitem_255, hidden_states_164, value_16, attn_output_64], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf387 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf383, buf384, buf385, reinterpret_tensor(buf386, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf384
            del buf385
            buf388 = buf387[0]
            assert_size_stride(buf388, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf388, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf387
            assert_size_stride(arg285_1, (896, 896), (896, 1))
            buf392 = reinterpret_tensor(buf383, (1, 896), (896, 1), 0); del buf383  # reuse
            # Topologically Sorted Source Nodes: [transpose_68, reshape_50, attn_output_67], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf388, (1, 896), (896, 1), 0), reinterpret_tensor(arg285_1, (896, 896), (1, 896), 0), out=buf392)
            del arg285_1
            assert_size_stride(arg287_1, (896, ), (1, ))
            assert_size_stride(arg286_1, (), ())
            buf394 = reinterpret_tensor(buf388, (1, 1, 896), (896, 896, 1), 0); del buf388  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, attn_output_67, hidden_states_165, pow_34, variance_33, rsqrt_33, hidden_states_167, hidden_states_168], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf369, buf375, buf392, arg287_1, arg286_1.item(), buf394, 1, 896, stream=raw_stream0)
            del arg286_1
            del arg287_1
            assert_size_stride(arg288_1, (4864, 896), (896, 1))
            buf395 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_116], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf394, (1, 896), (896, 1), 0), reinterpret_tensor(arg288_1, (896, 4864), (1, 896), 0), out=buf395)
            del arg288_1
            assert_size_stride(arg289_1, (4864, 896), (896, 1))
            buf396 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_117], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf394, (1, 896), (896, 1), 0), reinterpret_tensor(arg289_1, (896, 4864), (1, 896), 0), out=buf396)
            del arg289_1
            buf397 = reinterpret_tensor(buf395, (1, 1, 4864), (4864, 4864, 1), 0); del buf395  # reuse
            # Topologically Sorted Source Nodes: [linear_116, silu_16, linear_117, mul_154], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf397, buf396, 4864, stream=raw_stream0)
            del buf396
            assert_size_stride(arg290_1, (896, 4864), (4864, 1))
            buf398 = reinterpret_tensor(buf394, (1, 896), (896, 1), 0); del buf394  # reuse
            # Topologically Sorted Source Nodes: [linear_116, silu_16, linear_117, mul_154, down_proj_16], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf397, (1, 4864), (0, 1), 0), reinterpret_tensor(arg290_1, (4864, 896), (1, 4864), 0), out=buf398)
            del arg290_1
            del buf397
            assert_size_stride(arg292_1, (896, ), (1, ))
            assert_size_stride(arg291_1, (), ())
            buf400 = reinterpret_tensor(buf378, (1, 1, 896), (896, 896, 1), 0); del buf378  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, attn_output_67, hidden_states_165, down_proj_16, hidden_states_169, pow_35, variance_34, rsqrt_34, hidden_states_171, hidden_states_172], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf369, buf375, buf392, buf398, arg292_1, arg291_1.item(), buf400, 1, 896, stream=raw_stream0)
            del arg291_1
            del arg292_1
            assert_size_stride(arg294_1, (896, ), (1, ))
            assert_size_stride(arg293_1, (896, 896), (896, 1))
            buf401 = buf345; del buf345  # reuse
            # Topologically Sorted Source Nodes: [linear_119], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg294_1, reinterpret_tensor(buf400, (1, 896), (896, 1), 0), reinterpret_tensor(arg293_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf401)
            del arg293_1
            del arg294_1
            assert_size_stride(arg296_1, (128, ), (1, ))
            assert_size_stride(arg295_1, (128, 896), (896, 1))
            buf402 = buf381; del buf381  # reuse
            # Topologically Sorted Source Nodes: [linear_120], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg296_1, reinterpret_tensor(buf400, (1, 896), (896, 1), 0), reinterpret_tensor(arg295_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf402)
            del arg295_1
            del arg296_1
            assert_size_stride(arg299_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg299_1 = copy_misaligned(arg299_1)
            buf403 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_20, sin_20, linear_120, view_52, key_states_17, mul_159, x2_35, neg_35, x1_35, cat_70, mul_160, k_embed_17, keys_17], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg299_1, buf402, arg4_1, arg5_1.item(), buf403, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg299_1
            assert_size_stride(arg298_1, (128, ), (1, ))
            assert_size_stride(arg297_1, (128, 896), (896, 1))
            buf404 = buf402; del buf402  # reuse
            # Topologically Sorted Source Nodes: [linear_121], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg298_1, reinterpret_tensor(buf400, (1, 896), (896, 1), 0), reinterpret_tensor(arg297_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf404)
            del arg297_1
            del arg298_1
            assert_size_stride(arg301_1, (1, 2, s75, 64), (128*s75, 64, 128, 1))
            arg301_1 = copy_misaligned(arg301_1)
            ps36 = 1 + s75
            ps37 = 64 + 64*s75
            buf405 = empty_strided_cuda((1, 2, 1 + s75, 64), (128 + 128*s75, 64 + 64*s75, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_121, view_53, value_states_17, values_17], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s75
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg301_1, buf404, buf405, ps36, s75, ps37, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg301_1
            buf406 = reinterpret_tensor(buf400, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf400  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_119, view_51, query_states_17, cos_20, mul_157, x2_34, neg_34, x1_34, cat_69, sin_20, mul_158, q_embed_17, getitem_264, hidden_states_173, key_17, getitem_269, hidden_states_174, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf401, arg4_1, arg5_1.item(), buf406, s108, 896, stream=raw_stream0)
            del buf401
            buf407 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_119, view_51, query_states_17, cos_20, mul_157, x2_34, neg_34, x1_34, cat_69, sin_20, mul_158, q_embed_17, getitem_264, hidden_states_173, key_17, getitem_269, hidden_states_174, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf403, buf407, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf408 = empty_strided_cuda((1, 14, 1 + s75, 64), (896 + 896*s75, 64 + 64*s75, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_119, view_51, query_states_17, cos_20, mul_157, x2_34, neg_34, x1_34, cat_69, sin_20, mul_158, q_embed_17, getitem_264, hidden_states_173, key_17, getitem_269, hidden_states_174, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s75
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf405, buf408, ps37, s75, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf409 = buf386; del buf386  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_119, view_51, query_states_17, cos_20, mul_157, x2_34, neg_34, x1_34, cat_69, sin_20, mul_158, q_embed_17, getitem_264, hidden_states_173, key_17, getitem_269, hidden_states_174, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf409, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_119, view_51, query_states_17, cos_20, mul_157, x2_34, neg_34, x1_34, cat_69, sin_20, mul_158, q_embed_17, getitem_264, hidden_states_173, key_17, getitem_269, hidden_states_174, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf410 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf406, buf407, buf408, reinterpret_tensor(buf409, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf407
            del buf408
            buf411 = buf410[0]
            assert_size_stride(buf411, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf411, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf410
            assert_size_stride(arg302_1, (896, 896), (896, 1))
            buf415 = reinterpret_tensor(buf406, (1, 896), (896, 1), 0); del buf406  # reuse
            # Topologically Sorted Source Nodes: [transpose_72, reshape_53, attn_output_71], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf411, (1, 896), (896, 1), 0), reinterpret_tensor(arg302_1, (896, 896), (1, 896), 0), out=buf415)
            del arg302_1
            assert_size_stride(arg304_1, (896, ), (1, ))
            assert_size_stride(arg303_1, (), ())
            buf416 = buf369; del buf369  # reuse
            buf418 = reinterpret_tensor(buf411, (1, 1, 896), (896, 896, 1), 0); del buf411  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, attn_output_67, hidden_states_165, down_proj_16, hidden_states_169, attn_output_71, hidden_states_175, pow_36, variance_35, rsqrt_35, hidden_states_177, hidden_states_178], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf416, buf375, buf392, buf398, buf415, arg304_1, arg303_1.item(), buf418, 1, 896, stream=raw_stream0)
            del arg303_1
            del arg304_1
            del buf375
            assert_size_stride(arg305_1, (4864, 896), (896, 1))
            buf419 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [pow_36, variance_35, rsqrt_35, hidden_states_177, hidden_states_178, linear_123], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf418, (1, 896), (0, 1), 0), reinterpret_tensor(arg305_1, (896, 4864), (1, 896), 0), out=buf419)
            del arg305_1
            assert_size_stride(arg306_1, (4864, 896), (896, 1))
            buf420 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_124], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf418, (1, 896), (896, 1), 0), reinterpret_tensor(arg306_1, (896, 4864), (1, 896), 0), out=buf420)
            del arg306_1
            buf421 = reinterpret_tensor(buf419, (1, 1, 4864), (4864, 4864, 1), 0); del buf419  # reuse
            # Topologically Sorted Source Nodes: [linear_123, silu_17, linear_124, mul_163], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf421, buf420, 4864, stream=raw_stream0)
            del buf420
            assert_size_stride(arg307_1, (896, 4864), (4864, 1))
            buf422 = reinterpret_tensor(buf418, (1, 896), (896, 1), 0); del buf418  # reuse
            # Topologically Sorted Source Nodes: [linear_123, silu_17, linear_124, mul_163, down_proj_17], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf421, (1, 4864), (0, 1), 0), reinterpret_tensor(arg307_1, (4864, 896), (1, 4864), 0), out=buf422)
            del arg307_1
            del buf421
            assert_size_stride(arg309_1, (896, ), (1, ))
            assert_size_stride(arg308_1, (), ())
            buf424 = reinterpret_tensor(buf415, (1, 1, 896), (896, 896, 1), 0); del buf415  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, pow_37, variance_36, rsqrt_36, hidden_states_181, hidden_states_182], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf416, buf422, arg309_1, arg308_1.item(), buf424, 1, 896, stream=raw_stream0)
            del arg308_1
            del arg309_1
            assert_size_stride(arg311_1, (896, ), (1, ))
            assert_size_stride(arg310_1, (896, 896), (896, 1))
            buf425 = buf398; del buf398  # reuse
            # Topologically Sorted Source Nodes: [linear_126], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg311_1, reinterpret_tensor(buf424, (1, 896), (896, 1), 0), reinterpret_tensor(arg310_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf425)
            del arg310_1
            del arg311_1
            assert_size_stride(arg313_1, (128, ), (1, ))
            assert_size_stride(arg312_1, (128, 896), (896, 1))
            buf426 = buf404; del buf404  # reuse
            # Topologically Sorted Source Nodes: [linear_127], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg313_1, reinterpret_tensor(buf424, (1, 896), (896, 1), 0), reinterpret_tensor(arg312_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf426)
            del arg312_1
            del arg313_1
            assert_size_stride(arg316_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg316_1 = copy_misaligned(arg316_1)
            buf427 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_21, sin_21, linear_127, view_55, key_states_18, mul_168, x2_37, neg_37, x1_37, cat_74, mul_169, k_embed_18, keys_18], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg316_1, buf426, arg4_1, arg5_1.item(), buf427, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg316_1
            assert_size_stride(arg315_1, (128, ), (1, ))
            assert_size_stride(arg314_1, (128, 896), (896, 1))
            buf428 = buf426; del buf426  # reuse
            # Topologically Sorted Source Nodes: [linear_128], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg315_1, reinterpret_tensor(buf424, (1, 896), (896, 1), 0), reinterpret_tensor(arg314_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf428)
            del arg314_1
            del arg315_1
            assert_size_stride(arg318_1, (1, 2, s77, 64), (128*s77, 64, 128, 1))
            arg318_1 = copy_misaligned(arg318_1)
            ps38 = 1 + s77
            ps39 = 64 + 64*s77
            buf429 = empty_strided_cuda((1, 2, 1 + s77, 64), (128 + 128*s77, 64 + 64*s77, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_128, view_56, value_states_18, values_18], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s77
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg318_1, buf428, buf429, ps38, s77, ps39, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg318_1
            buf430 = reinterpret_tensor(buf424, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf424  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_126, view_54, query_states_18, cos_21, mul_166, x2_36, neg_36, x1_36, cat_73, sin_21, mul_167, q_embed_18, getitem_278, hidden_states_183, key_18, getitem_283, hidden_states_184, value_18, attn_output_72], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf425, arg4_1, arg5_1.item(), buf430, s108, 896, stream=raw_stream0)
            buf431 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_126, view_54, query_states_18, cos_21, mul_166, x2_36, neg_36, x1_36, cat_73, sin_21, mul_167, q_embed_18, getitem_278, hidden_states_183, key_18, getitem_283, hidden_states_184, value_18, attn_output_72], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf427, buf431, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf432 = empty_strided_cuda((1, 14, 1 + s77, 64), (896 + 896*s77, 64 + 64*s77, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_126, view_54, query_states_18, cos_21, mul_166, x2_36, neg_36, x1_36, cat_73, sin_21, mul_167, q_embed_18, getitem_278, hidden_states_183, key_18, getitem_283, hidden_states_184, value_18, attn_output_72], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s77
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf429, buf432, ps39, s77, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf433 = buf409; del buf409  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_126, view_54, query_states_18, cos_21, mul_166, x2_36, neg_36, x1_36, cat_73, sin_21, mul_167, q_embed_18, getitem_278, hidden_states_183, key_18, getitem_283, hidden_states_184, value_18, attn_output_72], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf433, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_126, view_54, query_states_18, cos_21, mul_166, x2_36, neg_36, x1_36, cat_73, sin_21, mul_167, q_embed_18, getitem_278, hidden_states_183, key_18, getitem_283, hidden_states_184, value_18, attn_output_72], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf434 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf430, buf431, buf432, reinterpret_tensor(buf433, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf431
            del buf432
            buf435 = buf434[0]
            assert_size_stride(buf435, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf435, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf434
            assert_size_stride(arg319_1, (896, 896), (896, 1))
            buf439 = reinterpret_tensor(buf430, (1, 896), (896, 1), 0); del buf430  # reuse
            # Topologically Sorted Source Nodes: [transpose_76, reshape_56, attn_output_75], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf435, (1, 896), (896, 1), 0), reinterpret_tensor(arg319_1, (896, 896), (1, 896), 0), out=buf439)
            del arg319_1
            assert_size_stride(arg321_1, (896, ), (1, ))
            assert_size_stride(arg320_1, (), ())
            buf441 = reinterpret_tensor(buf435, (1, 1, 896), (896, 896, 1), 0); del buf435  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, attn_output_75, hidden_states_185, pow_38, variance_37, rsqrt_37, hidden_states_187, hidden_states_188], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf416, buf422, buf439, arg321_1, arg320_1.item(), buf441, 1, 896, stream=raw_stream0)
            del arg320_1
            del arg321_1
            assert_size_stride(arg322_1, (4864, 896), (896, 1))
            buf442 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_130], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf441, (1, 896), (896, 1), 0), reinterpret_tensor(arg322_1, (896, 4864), (1, 896), 0), out=buf442)
            del arg322_1
            assert_size_stride(arg323_1, (4864, 896), (896, 1))
            buf443 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_131], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf441, (1, 896), (896, 1), 0), reinterpret_tensor(arg323_1, (896, 4864), (1, 896), 0), out=buf443)
            del arg323_1
            buf444 = reinterpret_tensor(buf442, (1, 1, 4864), (4864, 4864, 1), 0); del buf442  # reuse
            # Topologically Sorted Source Nodes: [linear_130, silu_18, linear_131, mul_172], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf444, buf443, 4864, stream=raw_stream0)
            del buf443
            assert_size_stride(arg324_1, (896, 4864), (4864, 1))
            buf445 = reinterpret_tensor(buf441, (1, 896), (896, 1), 0); del buf441  # reuse
            # Topologically Sorted Source Nodes: [linear_130, silu_18, linear_131, mul_172, down_proj_18], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf444, (1, 4864), (0, 1), 0), reinterpret_tensor(arg324_1, (4864, 896), (1, 4864), 0), out=buf445)
            del arg324_1
            del buf444
            assert_size_stride(arg326_1, (896, ), (1, ))
            assert_size_stride(arg325_1, (), ())
            buf447 = reinterpret_tensor(buf425, (1, 1, 896), (896, 896, 1), 0); del buf425  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, attn_output_75, hidden_states_185, down_proj_18, hidden_states_189, pow_39, variance_38, rsqrt_38, hidden_states_191, hidden_states_192], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf416, buf422, buf439, buf445, arg326_1, arg325_1.item(), buf447, 1, 896, stream=raw_stream0)
            del arg325_1
            del arg326_1
            assert_size_stride(arg328_1, (896, ), (1, ))
            assert_size_stride(arg327_1, (896, 896), (896, 1))
            buf448 = buf392; del buf392  # reuse
            # Topologically Sorted Source Nodes: [linear_133], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg328_1, reinterpret_tensor(buf447, (1, 896), (896, 1), 0), reinterpret_tensor(arg327_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf448)
            del arg327_1
            del arg328_1
            assert_size_stride(arg330_1, (128, ), (1, ))
            assert_size_stride(arg329_1, (128, 896), (896, 1))
            buf449 = buf428; del buf428  # reuse
            # Topologically Sorted Source Nodes: [linear_134], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg330_1, reinterpret_tensor(buf447, (1, 896), (896, 1), 0), reinterpret_tensor(arg329_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf449)
            del arg329_1
            del arg330_1
            assert_size_stride(arg334_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg334_1 = copy_misaligned(arg334_1)
            buf450 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_22, sin_22, linear_134, view_58, key_states_19, mul_177, x2_39, neg_39, x1_39, cat_78, mul_178, k_embed_19, keys_19], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg334_1, buf449, arg4_1, arg5_1.item(), buf450, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg334_1
            assert_size_stride(arg332_1, (128, ), (1, ))
            assert_size_stride(arg331_1, (128, 896), (896, 1))
            buf451 = buf449; del buf449  # reuse
            # Topologically Sorted Source Nodes: [linear_135], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg332_1, reinterpret_tensor(buf447, (1, 896), (896, 1), 0), reinterpret_tensor(arg331_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf451)
            del arg331_1
            del arg332_1
            assert_size_stride(arg336_1, (1, 2, s109, 64), (128*s109, 64, 128, 1))
            arg336_1 = copy_misaligned(arg336_1)
            ps40 = 1 + s109
            ps41 = 64 + 64*s109
            buf452 = empty_strided_cuda((1, 2, 1 + s109, 64), (128 + 128*s109, 64 + 64*s109, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_135, view_59, value_states_19, values_19], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s109
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg336_1, buf451, buf452, ps40, s109, ps41, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg336_1
            buf453 = reinterpret_tensor(buf447, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf447  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_133, view_57, query_states_19, cos_22, mul_175, x2_38, neg_38, x1_38, cat_77, sin_22, mul_176, q_embed_19, getitem_292, hidden_states_193, key_19, getitem_297, hidden_states_194, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf448, arg4_1, arg5_1.item(), buf453, s108, 896, stream=raw_stream0)
            del buf448
            buf454 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_133, view_57, query_states_19, cos_22, mul_175, x2_38, neg_38, x1_38, cat_77, sin_22, mul_176, q_embed_19, getitem_292, hidden_states_193, key_19, getitem_297, hidden_states_194, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf450, buf454, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf455 = empty_strided_cuda((1, 14, 1 + s109, 64), (896 + 896*s109, 64 + 64*s109, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_133, view_57, query_states_19, cos_22, mul_175, x2_38, neg_38, x1_38, cat_77, sin_22, mul_176, q_embed_19, getitem_292, hidden_states_193, key_19, getitem_297, hidden_states_194, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s109
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf452, buf455, ps41, s109, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf456 = buf433; del buf433  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_133, view_57, query_states_19, cos_22, mul_175, x2_38, neg_38, x1_38, cat_77, sin_22, mul_176, q_embed_19, getitem_292, hidden_states_193, key_19, getitem_297, hidden_states_194, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf456, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_133, view_57, query_states_19, cos_22, mul_175, x2_38, neg_38, x1_38, cat_77, sin_22, mul_176, q_embed_19, getitem_292, hidden_states_193, key_19, getitem_297, hidden_states_194, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf457 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf453, buf454, buf455, reinterpret_tensor(buf456, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf454
            del buf455
            buf458 = buf457[0]
            assert_size_stride(buf458, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf458, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf457
            assert_size_stride(arg337_1, (896, 896), (896, 1))
            buf462 = reinterpret_tensor(buf453, (1, 896), (896, 1), 0); del buf453  # reuse
            # Topologically Sorted Source Nodes: [transpose_80, reshape_59, attn_output_79], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf458, (1, 896), (896, 1), 0), reinterpret_tensor(arg337_1, (896, 896), (1, 896), 0), out=buf462)
            del arg337_1
            assert_size_stride(arg339_1, (896, ), (1, ))
            assert_size_stride(arg338_1, (), ())
            buf463 = buf416; del buf416  # reuse
            buf465 = reinterpret_tensor(buf458, (1, 1, 896), (896, 896, 1), 0); del buf458  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, attn_output_75, hidden_states_185, down_proj_18, hidden_states_189, attn_output_79, hidden_states_195, pow_40, variance_39, rsqrt_39, hidden_states_197, hidden_states_198], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf463, buf422, buf439, buf445, buf462, arg339_1, arg338_1.item(), buf465, 1, 896, stream=raw_stream0)
            del arg338_1
            del arg339_1
            del buf422
            assert_size_stride(arg340_1, (4864, 896), (896, 1))
            buf466 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [pow_40, variance_39, rsqrt_39, hidden_states_197, hidden_states_198, linear_137], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf465, (1, 896), (0, 1), 0), reinterpret_tensor(arg340_1, (896, 4864), (1, 896), 0), out=buf466)
            del arg340_1
            assert_size_stride(arg341_1, (4864, 896), (896, 1))
            buf467 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_138], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf465, (1, 896), (896, 1), 0), reinterpret_tensor(arg341_1, (896, 4864), (1, 896), 0), out=buf467)
            del arg341_1
            buf468 = reinterpret_tensor(buf466, (1, 1, 4864), (4864, 4864, 1), 0); del buf466  # reuse
            # Topologically Sorted Source Nodes: [linear_137, silu_19, linear_138, mul_181], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf468, buf467, 4864, stream=raw_stream0)
            del buf467
            assert_size_stride(arg342_1, (896, 4864), (4864, 1))
            buf469 = reinterpret_tensor(buf465, (1, 896), (896, 1), 0); del buf465  # reuse
            # Topologically Sorted Source Nodes: [linear_137, silu_19, linear_138, mul_181, down_proj_19], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf468, (1, 4864), (0, 1), 0), reinterpret_tensor(arg342_1, (4864, 896), (1, 4864), 0), out=buf469)
            del arg342_1
            del buf468
            assert_size_stride(arg344_1, (896, ), (1, ))
            assert_size_stride(arg343_1, (), ())
            buf471 = reinterpret_tensor(buf462, (1, 1, 896), (896, 896, 1), 0); del buf462  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, pow_41, variance_40, rsqrt_40, hidden_states_201, hidden_states_202], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf463, buf469, arg344_1, arg343_1.item(), buf471, 1, 896, stream=raw_stream0)
            del arg343_1
            del arg344_1
            assert_size_stride(arg346_1, (896, ), (1, ))
            assert_size_stride(arg345_1, (896, 896), (896, 1))
            buf472 = buf445; del buf445  # reuse
            # Topologically Sorted Source Nodes: [linear_140], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg346_1, reinterpret_tensor(buf471, (1, 896), (896, 1), 0), reinterpret_tensor(arg345_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf472)
            del arg345_1
            del arg346_1
            assert_size_stride(arg348_1, (128, ), (1, ))
            assert_size_stride(arg347_1, (128, 896), (896, 1))
            buf473 = buf451; del buf451  # reuse
            # Topologically Sorted Source Nodes: [linear_141], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg348_1, reinterpret_tensor(buf471, (1, 896), (896, 1), 0), reinterpret_tensor(arg347_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf473)
            del arg347_1
            del arg348_1
            assert_size_stride(arg351_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg351_1 = copy_misaligned(arg351_1)
            buf474 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_23, sin_23, linear_141, view_61, key_states_20, mul_186, x2_41, neg_41, x1_41, cat_82, mul_187, k_embed_20, keys_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg351_1, buf473, arg4_1, arg5_1.item(), buf474, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg351_1
            assert_size_stride(arg350_1, (128, ), (1, ))
            assert_size_stride(arg349_1, (128, 896), (896, 1))
            buf475 = buf473; del buf473  # reuse
            # Topologically Sorted Source Nodes: [linear_142], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg350_1, reinterpret_tensor(buf471, (1, 896), (896, 1), 0), reinterpret_tensor(arg349_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf475)
            del arg349_1
            del arg350_1
            assert_size_stride(arg353_1, (1, 2, s9, 64), (128*s9, 64, 128, 1))
            arg353_1 = copy_misaligned(arg353_1)
            ps42 = 1 + s9
            ps43 = 64 + 64*s9
            buf476 = empty_strided_cuda((1, 2, 1 + s9, 64), (128 + 128*s9, 64 + 64*s9, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_142, view_62, value_states_20, values_20], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s9
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg353_1, buf475, buf476, ps42, s9, ps43, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg353_1
            buf477 = reinterpret_tensor(buf471, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf471  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_140, view_60, query_states_20, cos_23, mul_184, x2_40, neg_40, x1_40, cat_81, sin_23, mul_185, q_embed_20, getitem_306, hidden_states_203, key_20, getitem_311, hidden_states_204, value_20, attn_output_80], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf472, arg4_1, arg5_1.item(), buf477, s108, 896, stream=raw_stream0)
            buf478 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_140, view_60, query_states_20, cos_23, mul_184, x2_40, neg_40, x1_40, cat_81, sin_23, mul_185, q_embed_20, getitem_306, hidden_states_203, key_20, getitem_311, hidden_states_204, value_20, attn_output_80], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf474, buf478, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf479 = empty_strided_cuda((1, 14, 1 + s9, 64), (896 + 896*s9, 64 + 64*s9, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_140, view_60, query_states_20, cos_23, mul_184, x2_40, neg_40, x1_40, cat_81, sin_23, mul_185, q_embed_20, getitem_306, hidden_states_203, key_20, getitem_311, hidden_states_204, value_20, attn_output_80], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s9
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf476, buf479, ps43, s9, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf480 = buf456; del buf456  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_140, view_60, query_states_20, cos_23, mul_184, x2_40, neg_40, x1_40, cat_81, sin_23, mul_185, q_embed_20, getitem_306, hidden_states_203, key_20, getitem_311, hidden_states_204, value_20, attn_output_80], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf480, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_140, view_60, query_states_20, cos_23, mul_184, x2_40, neg_40, x1_40, cat_81, sin_23, mul_185, q_embed_20, getitem_306, hidden_states_203, key_20, getitem_311, hidden_states_204, value_20, attn_output_80], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf481 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf477, buf478, buf479, reinterpret_tensor(buf480, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf478
            del buf479
            buf482 = buf481[0]
            assert_size_stride(buf482, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf482, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf481
            assert_size_stride(arg354_1, (896, 896), (896, 1))
            buf486 = reinterpret_tensor(buf477, (1, 896), (896, 1), 0); del buf477  # reuse
            # Topologically Sorted Source Nodes: [transpose_84, reshape_62, attn_output_83], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf482, (1, 896), (896, 1), 0), reinterpret_tensor(arg354_1, (896, 896), (1, 896), 0), out=buf486)
            del arg354_1
            assert_size_stride(arg356_1, (896, ), (1, ))
            assert_size_stride(arg355_1, (), ())
            buf488 = reinterpret_tensor(buf482, (1, 1, 896), (896, 896, 1), 0); del buf482  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, attn_output_83, hidden_states_205, pow_42, variance_41, rsqrt_41, hidden_states_207, hidden_states_208], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf463, buf469, buf486, arg356_1, arg355_1.item(), buf488, 1, 896, stream=raw_stream0)
            del arg355_1
            del arg356_1
            assert_size_stride(arg357_1, (4864, 896), (896, 1))
            buf489 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_144], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf488, (1, 896), (896, 1), 0), reinterpret_tensor(arg357_1, (896, 4864), (1, 896), 0), out=buf489)
            del arg357_1
            assert_size_stride(arg358_1, (4864, 896), (896, 1))
            buf490 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_145], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf488, (1, 896), (896, 1), 0), reinterpret_tensor(arg358_1, (896, 4864), (1, 896), 0), out=buf490)
            del arg358_1
            buf491 = reinterpret_tensor(buf489, (1, 1, 4864), (4864, 4864, 1), 0); del buf489  # reuse
            # Topologically Sorted Source Nodes: [linear_144, silu_20, linear_145, mul_190], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf491, buf490, 4864, stream=raw_stream0)
            del buf490
            assert_size_stride(arg359_1, (896, 4864), (4864, 1))
            buf492 = reinterpret_tensor(buf488, (1, 896), (896, 1), 0); del buf488  # reuse
            # Topologically Sorted Source Nodes: [linear_144, silu_20, linear_145, mul_190, down_proj_20], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf491, (1, 4864), (0, 1), 0), reinterpret_tensor(arg359_1, (4864, 896), (1, 4864), 0), out=buf492)
            del arg359_1
            del buf491
            assert_size_stride(arg361_1, (896, ), (1, ))
            assert_size_stride(arg360_1, (), ())
            buf494 = reinterpret_tensor(buf472, (1, 1, 896), (896, 896, 1), 0); del buf472  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, attn_output_83, hidden_states_205, down_proj_20, hidden_states_209, pow_43, variance_42, rsqrt_42, hidden_states_211, hidden_states_212], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf463, buf469, buf486, buf492, arg361_1, arg360_1.item(), buf494, 1, 896, stream=raw_stream0)
            del arg360_1
            del arg361_1
            assert_size_stride(arg363_1, (896, ), (1, ))
            assert_size_stride(arg362_1, (896, 896), (896, 1))
            buf495 = buf439; del buf439  # reuse
            # Topologically Sorted Source Nodes: [linear_147], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg363_1, reinterpret_tensor(buf494, (1, 896), (896, 1), 0), reinterpret_tensor(arg362_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf495)
            del arg362_1
            del arg363_1
            assert_size_stride(arg365_1, (128, ), (1, ))
            assert_size_stride(arg364_1, (128, 896), (896, 1))
            buf496 = buf475; del buf475  # reuse
            # Topologically Sorted Source Nodes: [linear_148], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg365_1, reinterpret_tensor(buf494, (1, 896), (896, 1), 0), reinterpret_tensor(arg364_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf496)
            del arg364_1
            del arg365_1
            assert_size_stride(arg368_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg368_1 = copy_misaligned(arg368_1)
            buf497 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_24, sin_24, linear_148, view_64, key_states_21, mul_195, x2_43, neg_43, x1_43, cat_86, mul_196, k_embed_21, keys_21], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg368_1, buf496, arg4_1, arg5_1.item(), buf497, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg368_1
            assert_size_stride(arg367_1, (128, ), (1, ))
            assert_size_stride(arg366_1, (128, 896), (896, 1))
            buf498 = buf496; del buf496  # reuse
            # Topologically Sorted Source Nodes: [linear_149], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg367_1, reinterpret_tensor(buf494, (1, 896), (896, 1), 0), reinterpret_tensor(arg366_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf498)
            del arg366_1
            del arg367_1
            assert_size_stride(arg370_1, (1, 2, s39, 64), (128*s39, 64, 128, 1))
            arg370_1 = copy_misaligned(arg370_1)
            ps44 = 1 + s39
            ps45 = 64 + 64*s39
            buf499 = empty_strided_cuda((1, 2, 1 + s39, 64), (128 + 128*s39, 64 + 64*s39, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_149, view_65, value_states_21, values_21], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s39
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg370_1, buf498, buf499, ps44, s39, ps45, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg370_1
            buf500 = reinterpret_tensor(buf494, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf494  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_147, view_63, query_states_21, cos_24, mul_193, x2_42, neg_42, x1_42, cat_85, sin_24, mul_194, q_embed_21, getitem_320, hidden_states_213, key_21, getitem_325, hidden_states_214, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf495, arg4_1, arg5_1.item(), buf500, s108, 896, stream=raw_stream0)
            del buf495
            buf501 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_147, view_63, query_states_21, cos_24, mul_193, x2_42, neg_42, x1_42, cat_85, sin_24, mul_194, q_embed_21, getitem_320, hidden_states_213, key_21, getitem_325, hidden_states_214, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf497, buf501, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf502 = empty_strided_cuda((1, 14, 1 + s39, 64), (896 + 896*s39, 64 + 64*s39, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_147, view_63, query_states_21, cos_24, mul_193, x2_42, neg_42, x1_42, cat_85, sin_24, mul_194, q_embed_21, getitem_320, hidden_states_213, key_21, getitem_325, hidden_states_214, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s39
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf499, buf502, ps45, s39, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf503 = buf480; del buf480  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_147, view_63, query_states_21, cos_24, mul_193, x2_42, neg_42, x1_42, cat_85, sin_24, mul_194, q_embed_21, getitem_320, hidden_states_213, key_21, getitem_325, hidden_states_214, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf503, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_147, view_63, query_states_21, cos_24, mul_193, x2_42, neg_42, x1_42, cat_85, sin_24, mul_194, q_embed_21, getitem_320, hidden_states_213, key_21, getitem_325, hidden_states_214, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf504 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf500, buf501, buf502, reinterpret_tensor(buf503, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf501
            del buf502
            buf505 = buf504[0]
            assert_size_stride(buf505, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf505, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf504
            assert_size_stride(arg371_1, (896, 896), (896, 1))
            buf509 = reinterpret_tensor(buf500, (1, 896), (896, 1), 0); del buf500  # reuse
            # Topologically Sorted Source Nodes: [transpose_88, reshape_65, attn_output_87], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf505, (1, 896), (896, 1), 0), reinterpret_tensor(arg371_1, (896, 896), (1, 896), 0), out=buf509)
            del arg371_1
            assert_size_stride(arg373_1, (896, ), (1, ))
            assert_size_stride(arg372_1, (), ())
            buf510 = buf463; del buf463  # reuse
            buf512 = reinterpret_tensor(buf505, (1, 1, 896), (896, 896, 1), 0); del buf505  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, attn_output_83, hidden_states_205, down_proj_20, hidden_states_209, attn_output_87, hidden_states_215, pow_44, variance_43, rsqrt_43, hidden_states_217, hidden_states_218], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf510, buf469, buf486, buf492, buf509, arg373_1, arg372_1.item(), buf512, 1, 896, stream=raw_stream0)
            del arg372_1
            del arg373_1
            del buf469
            assert_size_stride(arg374_1, (4864, 896), (896, 1))
            buf513 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [pow_44, variance_43, rsqrt_43, hidden_states_217, hidden_states_218, linear_151], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf512, (1, 896), (0, 1), 0), reinterpret_tensor(arg374_1, (896, 4864), (1, 896), 0), out=buf513)
            del arg374_1
            assert_size_stride(arg375_1, (4864, 896), (896, 1))
            buf514 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_152], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf512, (1, 896), (896, 1), 0), reinterpret_tensor(arg375_1, (896, 4864), (1, 896), 0), out=buf514)
            del arg375_1
            buf515 = reinterpret_tensor(buf513, (1, 1, 4864), (4864, 4864, 1), 0); del buf513  # reuse
            # Topologically Sorted Source Nodes: [linear_151, silu_21, linear_152, mul_199], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf515, buf514, 4864, stream=raw_stream0)
            del buf514
            assert_size_stride(arg376_1, (896, 4864), (4864, 1))
            buf516 = reinterpret_tensor(buf512, (1, 896), (896, 1), 0); del buf512  # reuse
            # Topologically Sorted Source Nodes: [linear_151, silu_21, linear_152, mul_199, down_proj_21], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf515, (1, 4864), (0, 1), 0), reinterpret_tensor(arg376_1, (4864, 896), (1, 4864), 0), out=buf516)
            del arg376_1
            del buf515
            assert_size_stride(arg378_1, (896, ), (1, ))
            assert_size_stride(arg377_1, (), ())
            buf518 = reinterpret_tensor(buf509, (1, 1, 896), (896, 896, 1), 0); del buf509  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, pow_45, variance_44, rsqrt_44, hidden_states_221, hidden_states_222], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf510, buf516, arg378_1, arg377_1.item(), buf518, 1, 896, stream=raw_stream0)
            del arg377_1
            del arg378_1
            assert_size_stride(arg380_1, (896, ), (1, ))
            assert_size_stride(arg379_1, (896, 896), (896, 1))
            buf519 = buf492; del buf492  # reuse
            # Topologically Sorted Source Nodes: [linear_154], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg380_1, reinterpret_tensor(buf518, (1, 896), (896, 1), 0), reinterpret_tensor(arg379_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf519)
            del arg379_1
            del arg380_1
            assert_size_stride(arg382_1, (128, ), (1, ))
            assert_size_stride(arg381_1, (128, 896), (896, 1))
            buf520 = buf498; del buf498  # reuse
            # Topologically Sorted Source Nodes: [linear_155], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg382_1, reinterpret_tensor(buf518, (1, 896), (896, 1), 0), reinterpret_tensor(arg381_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf520)
            del arg381_1
            del arg382_1
            assert_size_stride(arg385_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg385_1 = copy_misaligned(arg385_1)
            buf521 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_25, sin_25, linear_155, view_67, key_states_22, mul_204, x2_45, neg_45, x1_45, cat_90, mul_205, k_embed_22, keys_22], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg385_1, buf520, arg4_1, arg5_1.item(), buf521, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg385_1
            assert_size_stride(arg384_1, (128, ), (1, ))
            assert_size_stride(arg383_1, (128, 896), (896, 1))
            buf522 = buf520; del buf520  # reuse
            # Topologically Sorted Source Nodes: [linear_156], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg384_1, reinterpret_tensor(buf518, (1, 896), (896, 1), 0), reinterpret_tensor(arg383_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf522)
            del arg383_1
            del arg384_1
            assert_size_stride(arg387_1, (1, 2, s117, 64), (128*s117, 64, 128, 1))
            arg387_1 = copy_misaligned(arg387_1)
            ps46 = 1 + s117
            ps47 = 64 + 64*s117
            buf523 = empty_strided_cuda((1, 2, 1 + s117, 64), (128 + 128*s117, 64 + 64*s117, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_156, view_68, value_states_22, values_22], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s117
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg387_1, buf522, buf523, ps46, s117, ps47, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg387_1
            buf524 = reinterpret_tensor(buf518, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf518  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_154, view_66, query_states_22, cos_25, mul_202, x2_44, neg_44, x1_44, cat_89, sin_25, mul_203, q_embed_22, getitem_334, hidden_states_223, key_22, getitem_339, hidden_states_224, value_22, attn_output_88], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf519, arg4_1, arg5_1.item(), buf524, s108, 896, stream=raw_stream0)
            buf525 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_154, view_66, query_states_22, cos_25, mul_202, x2_44, neg_44, x1_44, cat_89, sin_25, mul_203, q_embed_22, getitem_334, hidden_states_223, key_22, getitem_339, hidden_states_224, value_22, attn_output_88], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf521, buf525, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf526 = empty_strided_cuda((1, 14, 1 + s117, 64), (896 + 896*s117, 64 + 64*s117, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_154, view_66, query_states_22, cos_25, mul_202, x2_44, neg_44, x1_44, cat_89, sin_25, mul_203, q_embed_22, getitem_334, hidden_states_223, key_22, getitem_339, hidden_states_224, value_22, attn_output_88], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s117
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf523, buf526, ps47, s117, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf527 = buf503; del buf503  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_154, view_66, query_states_22, cos_25, mul_202, x2_44, neg_44, x1_44, cat_89, sin_25, mul_203, q_embed_22, getitem_334, hidden_states_223, key_22, getitem_339, hidden_states_224, value_22, attn_output_88], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf527, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_154, view_66, query_states_22, cos_25, mul_202, x2_44, neg_44, x1_44, cat_89, sin_25, mul_203, q_embed_22, getitem_334, hidden_states_223, key_22, getitem_339, hidden_states_224, value_22, attn_output_88], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf528 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf524, buf525, buf526, reinterpret_tensor(buf527, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf525
            del buf526
            buf529 = buf528[0]
            assert_size_stride(buf529, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf529, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf528
            assert_size_stride(arg388_1, (896, 896), (896, 1))
            buf533 = reinterpret_tensor(buf524, (1, 896), (896, 1), 0); del buf524  # reuse
            # Topologically Sorted Source Nodes: [transpose_92, reshape_68, attn_output_91], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf529, (1, 896), (896, 1), 0), reinterpret_tensor(arg388_1, (896, 896), (1, 896), 0), out=buf533)
            del arg388_1
            assert_size_stride(arg390_1, (896, ), (1, ))
            assert_size_stride(arg389_1, (), ())
            buf535 = reinterpret_tensor(buf529, (1, 1, 896), (896, 896, 1), 0); del buf529  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, attn_output_91, hidden_states_225, pow_46, variance_45, rsqrt_45, hidden_states_227, hidden_states_228], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf510, buf516, buf533, arg390_1, arg389_1.item(), buf535, 1, 896, stream=raw_stream0)
            del arg389_1
            del arg390_1
            assert_size_stride(arg391_1, (4864, 896), (896, 1))
            buf536 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_158], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf535, (1, 896), (896, 1), 0), reinterpret_tensor(arg391_1, (896, 4864), (1, 896), 0), out=buf536)
            del arg391_1
            assert_size_stride(arg392_1, (4864, 896), (896, 1))
            buf537 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_159], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf535, (1, 896), (896, 1), 0), reinterpret_tensor(arg392_1, (896, 4864), (1, 896), 0), out=buf537)
            del arg392_1
            buf538 = reinterpret_tensor(buf536, (1, 1, 4864), (4864, 4864, 1), 0); del buf536  # reuse
            # Topologically Sorted Source Nodes: [linear_158, silu_22, linear_159, mul_208], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf538, buf537, 4864, stream=raw_stream0)
            del buf537
            assert_size_stride(arg393_1, (896, 4864), (4864, 1))
            buf539 = reinterpret_tensor(buf535, (1, 896), (896, 1), 0); del buf535  # reuse
            # Topologically Sorted Source Nodes: [linear_158, silu_22, linear_159, mul_208, down_proj_22], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf538, (1, 4864), (0, 1), 0), reinterpret_tensor(arg393_1, (4864, 896), (1, 4864), 0), out=buf539)
            del arg393_1
            del buf538
            assert_size_stride(arg395_1, (896, ), (1, ))
            assert_size_stride(arg394_1, (), ())
            buf541 = reinterpret_tensor(buf519, (1, 1, 896), (896, 896, 1), 0); del buf519  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, attn_output_91, hidden_states_225, down_proj_22, hidden_states_229, pow_47, variance_46, rsqrt_46, hidden_states_231, hidden_states_232], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf510, buf516, buf533, buf539, arg395_1, arg394_1.item(), buf541, 1, 896, stream=raw_stream0)
            del arg394_1
            del arg395_1
            assert_size_stride(arg397_1, (896, ), (1, ))
            assert_size_stride(arg396_1, (896, 896), (896, 1))
            buf542 = buf486; del buf486  # reuse
            # Topologically Sorted Source Nodes: [linear_161], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg397_1, reinterpret_tensor(buf541, (1, 896), (896, 1), 0), reinterpret_tensor(arg396_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf542)
            del arg396_1
            del arg397_1
            assert_size_stride(arg399_1, (128, ), (1, ))
            assert_size_stride(arg398_1, (128, 896), (896, 1))
            buf543 = buf522; del buf522  # reuse
            # Topologically Sorted Source Nodes: [linear_162], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg399_1, reinterpret_tensor(buf541, (1, 896), (896, 1), 0), reinterpret_tensor(arg398_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf543)
            del arg398_1
            del arg399_1
            assert_size_stride(arg402_1, (1, 2, s108, 64), (128*s108, 64, 128, 1))
            arg402_1 = copy_misaligned(arg402_1)
            buf544 = empty_strided_cuda((1, 2, 1 + s108, 64), (128 + 128*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_26, sin_26, linear_162, view_70, key_states_23, mul_213, x2_47, neg_47, x1_47, cat_94, mul_214, k_embed_23, keys_23], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128 + 128*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(arg402_1, buf543, arg4_1, arg5_1.item(), buf544, ps0, s108, ps1, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            del arg402_1
            assert_size_stride(arg401_1, (128, ), (1, ))
            assert_size_stride(arg400_1, (128, 896), (896, 1))
            buf545 = buf543; del buf543  # reuse
            # Topologically Sorted Source Nodes: [linear_163], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg401_1, reinterpret_tensor(buf541, (1, 896), (896, 1), 0), reinterpret_tensor(arg400_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf545)
            del arg400_1
            del arg401_1
            assert_size_stride(arg404_1, (1, 2, s121, 64), (128*s121, 64, 128, 1))
            arg404_1 = copy_misaligned(arg404_1)
            ps48 = 1 + s121
            ps49 = 64 + 64*s121
            buf546 = empty_strided_cuda((1, 2, 1 + s121, 64), (128 + 128*s121, 64 + 64*s121, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_163, view_71, value_states_23, values_23], Original ATen: [aten.view, aten.transpose, aten.cat]
            triton_poi_fused_cat_transpose_view_2_xnumel = 128 + 128*s121
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_transpose_view_2.run(arg404_1, buf545, buf546, ps48, s121, ps49, triton_poi_fused_cat_transpose_view_2_xnumel, stream=raw_stream0)
            del arg404_1
            del buf545
            buf547 = reinterpret_tensor(buf541, (1, 14, 1, 64), (896, 64, 64, 1), 0); del buf541  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_161, view_69, query_states_23, cos_26, mul_211, x2_46, neg_46, x1_46, cat_93, sin_26, mul_212, q_embed_23, getitem_348, hidden_states_233, key_23, getitem_353, hidden_states_234, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf542, arg4_1, arg5_1.item(), buf547, s108, 896, stream=raw_stream0)
            del arg4_1
            del arg5_1
            del buf542
            buf548 = empty_strided_cuda((1, 14, 1 + s108, 64), (896 + 896*s108, 64 + 64*s108, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_161, view_69, query_states_23, cos_26, mul_211, x2_46, neg_46, x1_46, cat_93, sin_26, mul_212, q_embed_23, getitem_348, hidden_states_233, key_23, getitem_353, hidden_states_234, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf544, buf548, ps1, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf549 = empty_strided_cuda((1, 14, 1 + s121, 64), (896 + 896*s121, 64 + 64*s121, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_161, view_69, query_states_23, cos_26, mul_211, x2_46, neg_46, x1_46, cat_93, sin_26, mul_212, q_embed_23, getitem_348, hidden_states_233, key_23, getitem_353, hidden_states_234, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = 896 + 896*s121
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf546, buf549, ps49, s121, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            buf550 = buf527; del buf527  # reuse
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_161, view_69, query_states_23, cos_26, mul_211, x2_46, neg_46, x1_46, cat_93, sin_26, mul_212, q_embed_23, getitem_348, hidden_states_233, key_23, getitem_353, hidden_states_234, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel = 1 + s108
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5.run(buf550, ps0, s108, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_5_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_16, expand_1, arange, position_ids, position_ids_1, getitem_17, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_161, view_69, query_states_23, cos_26, mul_211, x2_46, neg_46, x1_46, cat_93, sin_26, mul_212, q_embed_23, getitem_348, hidden_states_233, key_23, getitem_353, hidden_states_234, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf551 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf547, buf548, buf549, reinterpret_tensor(buf550, (1, 14, 1, 1 + s108), (8 + 8*(s108 // 8), 0, 8 + 8*(s108 // 8), 1), 0), False, scale=0.125)
            del buf548
            del buf549
            del buf550
            buf552 = buf551[0]
            assert_size_stride(buf552, (1, 14, 1, 64), (896, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf552, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf551
            assert_size_stride(arg405_1, (896, 896), (896, 1))
            buf556 = reinterpret_tensor(buf547, (1, 896), (896, 1), 0); del buf547  # reuse
            # Topologically Sorted Source Nodes: [transpose_96, reshape_71, attn_output_95], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf552, (1, 896), (896, 1), 0), reinterpret_tensor(arg405_1, (896, 896), (1, 896), 0), out=buf556)
            del arg405_1
            assert_size_stride(arg407_1, (896, ), (1, ))
            assert_size_stride(arg406_1, (), ())
            buf557 = buf510; del buf510  # reuse
            buf559 = reinterpret_tensor(buf552, (1, 1, 896), (896, 896, 1), 0); del buf552  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, attn_output_91, hidden_states_225, down_proj_22, hidden_states_229, attn_output_95, hidden_states_235, pow_48, variance_47, rsqrt_47, hidden_states_237, hidden_states_238], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf557, buf516, buf533, buf539, buf556, arg407_1, arg406_1.item(), buf559, 1, 896, stream=raw_stream0)
            del arg406_1
            del arg407_1
            del buf516
            del buf533
            del buf539
            del buf556
            assert_size_stride(arg408_1, (4864, 896), (896, 1))
            buf560 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [pow_48, variance_47, rsqrt_47, hidden_states_237, hidden_states_238, linear_165], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf559, (1, 896), (0, 1), 0), reinterpret_tensor(arg408_1, (896, 4864), (1, 896), 0), out=buf560)
            del arg408_1
            assert_size_stride(arg409_1, (4864, 896), (896, 1))
            buf561 = empty_strided_cuda((1, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_166], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf559, (1, 896), (896, 1), 0), reinterpret_tensor(arg409_1, (896, 4864), (1, 896), 0), out=buf561)
            del arg409_1
            buf562 = reinterpret_tensor(buf560, (1, 1, 4864), (4864, 4864, 1), 0); del buf560  # reuse
            # Topologically Sorted Source Nodes: [linear_165, silu_23, linear_166, mul_217], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_7.run(buf562, buf561, 4864, stream=raw_stream0)
            del buf561
            assert_size_stride(arg410_1, (896, 4864), (4864, 1))
            buf563 = reinterpret_tensor(buf559, (1, 896), (896, 1), 0); del buf559  # reuse
            # Topologically Sorted Source Nodes: [linear_165, silu_23, linear_166, mul_217, down_proj_23], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf562, (1, 4864), (0, 1), 0), reinterpret_tensor(arg410_1, (4864, 896), (1, 4864), 0), out=buf563)
            del arg410_1
            del buf562
            assert_size_stride(arg412_1, (896, ), (1, ))
            assert_size_stride(arg411_1, (), ())
            buf565 = buf557; del buf557  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, pow_49, variance_48, rsqrt_48, hidden_states_241, hidden_states_242], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_17.run(buf565, buf563, arg412_1, arg411_1.item(), 1, 896, stream=raw_stream0)
            del arg411_1
            del arg412_1
            del buf563
            buf566 = empty_strided_cuda((1, 151936), (151936, 1), torch.float32)
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, pow_49, variance_48, rsqrt_48, hidden_states_241, hidden_states_242, getitem_354, logits], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.slice, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf565, (1, 896), (0, 1), 0), reinterpret_tensor(arg1_1, (896, 151936), (1, 896), 0), out=buf566)
            del arg1_1
            del buf565
        return (buf6, buf4, buf29, buf27, buf53, buf51, buf76, buf74, buf100, buf98, buf123, buf121, buf147, buf145, buf170, buf168, buf194, buf192, buf217, buf215, buf241, buf239, buf264, buf262, buf288, buf286, buf311, buf309, buf335, buf333, buf358, buf356, buf382, buf380, buf405, buf403, buf429, buf427, buf452, buf450, buf476, buf474, buf499, buf497, buf523, buf521, buf546, buf544, reinterpret_tensor(buf566, (1, 1, 151936), (151936, 151936, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((151936, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = 32
    arg3_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg6_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg7_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = 32
    arg15_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg18_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg23_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = 32
    arg32_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg35_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg40_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = 32
    arg49_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg52_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg57_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = 32
    arg66_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg69_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg74_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = 32
    arg83_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg86_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg91_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = 32
    arg100_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg103_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg108_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = 32
    arg117_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg120_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg125_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = 32
    arg134_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg137_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg142_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = 32
    arg151_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg154_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg159_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = 32
    arg168_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg175_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = 32
    arg184_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg191_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = 32
    arg200_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg203_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg208_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = 32
    arg217_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg220_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg225_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = 32
    arg234_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg237_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = 32
    arg250_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg253_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg258_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = 32
    arg267_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg270_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg275_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = 32
    arg284_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg287_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg292_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = 32
    arg301_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg304_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg309_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = 32
    arg318_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg321_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg326_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = 32
    arg334_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = 32
    arg336_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg339_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg344_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = 32
    arg353_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg356_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg361_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg369_1 = 32
    arg370_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg373_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg378_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg386_1 = 32
    arg387_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg390_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg395_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg403_1 = 32
    arg404_1 = rand_strided((1, 2, 32, 64), (4096, 64, 128, 1), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg407_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg412_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = 1
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
