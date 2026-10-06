# AOT ID: ['1_inference']
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


# kernel path: <inductor-cache>/y2/cy2q64wpncbl2n3nxk44utcayvu2lmfxomhg6k66rkgdmvcsf2vs.py
# Topologically Sorted Source Nodes: [inputs_embeds, pow_1, variance, rsqrt, hidden_states_1, hidden_states_2], Original ATen: [aten.embedding, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   hidden_states_1 => mul_99
#   hidden_states_2 => mul_103
#   inputs_embeds => embedding
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   variance => mean
# Graph fragment:
#   %arg2_1 : Tensor "i64[1, s50][s47, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[151936, 896][896, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg7_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %buf0 : Tensor "f32[1, s50, 1][s50, 1, s50]cuda:0" = PlaceHolder[target=buf0]
#   %arg6_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg6_1]
#   %embedding : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg2_1), kwargs = {})
#   %pow_1 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%embedding, 2), kwargs = {})
#   %mean : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %convert_element_type_default_4 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg6_1, torch.float32), kwargs = {})
#   %add_tensor : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %convert_element_type_default_4), kwargs = {})
#   %rsqrt : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor,), kwargs = {})
#   %mul_99 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, %rsqrt), kwargs = {})
#   %mul_103 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg7_1, %mul_99), kwargs = {})
#   return %buf0,%mul_103
triton_red_fused_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_red_fused_embedding_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 256, 'r0_': 232960}}
)
@triton.jit
def triton_red_fused_embedding_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 896
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tl.full([1, 1], 151936, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 151936")
        tmp6 = tl.load(in_ptr1 + (r0_1 + 896*tmp4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp6 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp20 = in_ptr3
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp11 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full([1, 1], 151936, tl.int32)
        tmp13 = tmp0 + tmp12
        tmp14 = tmp0 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp0)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp15 < 151936")
        tmp17 = tl.load(in_ptr1 + (r0_1 + 896*tmp15), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.full([1, 1], 896.0, tl.float32)
        tmp19 = (tmp9 / tmp18)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp19 + tmp21
        tmp23 = libdevice.rsqrt(tmp22)
        tmp24 = tmp17 * tmp23
        tmp25 = tmp11 * tmp24
        tl.store(out_ptr1 + (r0_1 + 896*x0), tmp25, r0_mask & xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/g5/cg5copvqtyqququq3632negnixmeavuiulubdbjhks55bjui4cfg.py
# Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, cos_3, sin, sin_3, linear_1, view_1, key_states, mul_6, x2_1, neg_1, x1_1, cat_2, mul_7, k_embed], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
# Source node to ATen node mapping:
#   arange => iota
#   cat_2 => cat_1
#   cos => cos
#   cos_3 => unsqueeze_11
#   emb => clone, expand_4, unsqueeze_10, view_3
#   expand_1 => expand_1
#   freqs => permute
#   getitem_13 => unsqueeze_7, unsqueeze_8
#   getitem_16 => unsqueeze_9
#   k_embed => add_197
#   key_states => permute_4
#   linear_1 => view_8
#   matmul => mul_56
#   mul_6 => mul_182
#   mul_7 => mul_198
#   neg_1 => neg_1
#   position_ids => add_6
#   position_ids_1 => unsqueeze
#   position_ids_expanded => convert_element_type
#   sin => sin
#   sin_3 => unsqueeze_12
#   view_1 => view_9
#   x1_1 => slice_6
#   x2_1 => slice_7
# Graph fragment:
#   %addmm_1 : Tensor "f32[s50, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %arg4_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg5_1]
#   %unsqueeze_7 : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_8, [1, -1, 1]), kwargs = {})
#   %iota : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg0_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_6 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 0), kwargs = {})
#   %unsqueeze : Tensor "i64[1, s50][s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_6, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "i64[1, 1, s50][s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, s50][s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_9, torch.float32), kwargs = {})
#   %mul_56 : Tensor "f32[1, 32, s50][32*s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, s50, 32][32*s50, 1, s50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_56, [0, 2, 1]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, s50, 1, 32][32*s50, 1, 32*s50, s50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, s50, 2, 32][32*s50, 1, 0, s50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_10, [1, %arg0_1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, s50, 2, 32][64*s50, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, %arg0_1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, %convert_element_type_default_1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 1, s50, 64][64*s50, 64*s50, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor, 1), kwargs = {})
#   %sin : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, %convert_element_type_default_2), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 1, s50, 64][64*s50, 64*s50, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor_1, 1), kwargs = {})
#   %view_8 : Tensor "f32[1, s50, 128][128*s50, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [1, %arg0_1, 128]), kwargs = {})
#   %view_9 : Tensor "f32[1, s50, 2, 64][128*s50, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_8, [1, %arg0_1, -1, 64]), kwargs = {})
#   %permute_4 : Tensor "f32[1, 2, s50, 64][128*s50, 64, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_9, [0, 2, 1, 3]), kwargs = {})
#   %mul_182 : Tensor "f32[1, 2, s50, 64][128*s50, 64, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, %unsqueeze_11), kwargs = {})
#   %slice_7 : Tensor "f32[1, 2, s50, 32][128*s50, 64, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_4, 3, 32, 9223372036854775807), kwargs = {})
#   %neg_1 : Tensor "f32[1, 2, s50, 32][64*s50, 32, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_7,), kwargs = {})
#   %slice_6 : Tensor "f32[1, 2, s50, 32][128*s50, 64, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_4, 3, 0, 32), kwargs = {})
#   %cat_1 : Tensor "f32[1, 2, s50, 64][128*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_6], -1), kwargs = {})
#   %mul_198 : Tensor "f32[1, 2, s50, 64][128*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %unsqueeze_12), kwargs = {})
#   %add_197 : Tensor "f32[1, 2, s50, 64][128*s50, 64, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_182, %mul_198), kwargs = {})
#   return %add_197
triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1 = async_compile.triton('triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': 'fp64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 82048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = xindex // 128
    x0 = (xindex % 64)
    x4 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + ((x3 % 32)), xmask, eviction_policy='evict_last')
    tmp6 = in_ptr2
    tmp2 = x2
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
    tmp15 = tl.load(in_ptr0 + (32 + 64*x4 + (x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = -tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp14, tmp16, tmp17)
    tmp19 = tmp10 >= tmp13
    tmp20 = tl.full([1], 64, tl.int64)
    tmp21 = tmp10 < tmp20
    tmp22 = tl.load(in_ptr0 + (64*x4 + ((-32) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.where(tmp14, tmp18, tmp22)
    tmp24 = tl_math.sin(tmp4)
    tmp25 = tmp24 * tmp7
    tmp26 = tmp23 * tmp25
    tmp27 = tmp9 + tmp26
    tl.store(out_ptr0 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/gq/cgq5wdlj3p2x5drwovegigylqq25arl3nnzyycf3v3ipmxl7o2be.py
# Topologically Sorted Source Nodes: [linear, view, query_states, getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_44, hidden_states_3, key, linear_2, view_2, value_states, getitem_49, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   arange => iota
#   arange_3 => iota_3
#   arange_4 => iota_4
#   attention_mask => le_2
#   attention_mask_1 => expand
#   attn_output => _scaled_dot_product_efficient_attention, constant_pad_nd, expand_7, full_default, full_default_1, slice_10, where
#   cat_1 => cat
#   cos => cos
#   cos_3 => unsqueeze_11
#   emb => clone, expand_4, unsqueeze_10, view_3
#   expand_1 => expand_1
#   freqs => permute
#   getitem_13 => unsqueeze_7, unsqueeze_8
#   getitem_16 => unsqueeze_9
#   getitem_44 => unsqueeze_13
#   getitem_49 => unsqueeze_14
#   hidden_states_3 => expand_5
#   hidden_states_4 => expand_6
#   key => clone_4, view_13
#   kv_arange => add_20
#   kv_indices => unsqueeze_4, unsqueeze_5, unsqueeze_6
#   linear => view_5
#   linear_2 => view_11
#   matmul => mul_56
#   mul_4 => mul_157
#   mul_5 => mul_174
#   neg => neg
#   position_ids => add_6
#   position_ids_1 => unsqueeze
#   position_ids_expanded => convert_element_type
#   q_arange => add_14
#   q_embed => add_173
#   q_indices => unsqueeze_1, unsqueeze_2, unsqueeze_3
#   query_states => permute_2
#   sin => sin
#   sin_3 => unsqueeze_12
#   value => clone_5, view_14
#   value_states => permute_6
#   view => view_6
#   view_2 => view_12
#   x1 => slice_4
#   x2 => slice_5
# Graph fragment:
#   %addmm : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=addmm]
#   %arg4_1 : Tensor "f32[32][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg5_1]
#   %view_5 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, %arg0_1, 896]), kwargs = {})
#   %view_6 : Tensor "f32[1, s50, 14, 64][896*s50, 896, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [1, %arg0_1, -1, 64]), kwargs = {})
#   %permute_2 : Tensor "f32[1, 14, s50, 64][896*s50, 64, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_8, [1, -1, 1]), kwargs = {})
#   %iota : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg0_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_6 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 0), kwargs = {})
#   %unsqueeze : Tensor "i64[1, s50][s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_6, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "i64[1, 1, s50][s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, s50][s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_9, torch.float32), kwargs = {})
#   %mul_56 : Tensor "f32[1, 32, s50][32*s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, s50, 32][32*s50, 1, s50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_56, [0, 2, 1]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, s50, 1, 32][32*s50, 1, 32*s50, s50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, s50, 2, 32][32*s50, 1, 0, s50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_10, [1, %arg0_1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, s50, 2, 32][64*s50, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, %arg0_1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, %convert_element_type_default_1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 1, s50, 64][64*s50, 64*s50, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor, 1), kwargs = {})
#   %mul_157 : Tensor "f32[1, 14, s50, 64][896*s50, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_11), kwargs = {})
#   %slice_5 : Tensor "f32[1, 14, s50, 32][896*s50, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 32, 9223372036854775807), kwargs = {})
#   %neg : Tensor "f32[1, 14, s50, 32][448*s50, 32, 448, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_5,), kwargs = {})
#   %slice_4 : Tensor "f32[1, 14, s50, 32][896*s50, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 0, 32), kwargs = {})
#   %cat : Tensor "f32[1, 14, s50, 64][896*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_4], -1), kwargs = {})
#   %sin : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, %convert_element_type_default_2), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 1, s50, 64][64*s50, 64*s50, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor_1, 1), kwargs = {})
#   %mul_174 : Tensor "f32[1, 14, s50, 64][896*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_12), kwargs = {})
#   %add_173 : Tensor "f32[1, 14, s50, 64][896*s50, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_157, %mul_174), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[1, 2, 1, s50, 64][128*s50, 64, 128*s50, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_197, 2), kwargs = {})
#   %expand_5 : Tensor "f32[1, 2, 7, s50, 64][128*s50, 64, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_13, [1, 2, 7, %arg0_1, 64]), kwargs = {})
#   %clone_4 : Tensor "f32[1, 2, 7, s50, 64][896*s50, 448*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_13 : Tensor "f32[1, 14, s50, 64][896*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [1, 14, %arg0_1, 64]), kwargs = {})
#   %view_11 : Tensor "f32[1, s50, 128][128*s50, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [1, %arg0_1, 128]), kwargs = {})
#   %view_12 : Tensor "f32[1, s50, 2, 64][128*s50, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_11, [1, %arg0_1, -1, 64]), kwargs = {})
#   %permute_6 : Tensor "f32[1, 2, s50, 64][128*s50, 64, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_12, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[1, 2, 1, s50, 64][128*s50, 64, 128*s50, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_6, 2), kwargs = {})
#   %expand_6 : Tensor "f32[1, 2, 7, s50, 64][128*s50, 64, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_14, [1, 2, 7, %arg0_1, 64]), kwargs = {})
#   %clone_5 : Tensor "f32[1, 2, 7, s50, 64][896*s50, 448*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_14 : Tensor "f32[1, 14, s50, 64][896*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_5, [1, 14, %arg0_1, 64]), kwargs = {})
#   %iota_4 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg0_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_20 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_4, 0), kwargs = {})
#   %unsqueeze_4 : Tensor "i64[1, s50][s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_20, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[1, 1, s50][s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 1), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[1, 1, 1, s50][s50, s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 2), kwargs = {})
#   %iota_3 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg0_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_14 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_3, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[1, s50][s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_14, 0), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[1, 1, s50][s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 1), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, s50, 1][s50, s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 3), kwargs = {})
#   %le_2 : Tensor "b8[1, 1, s50, s50][s50**2, s50**2, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%unsqueeze_6, %unsqueeze_3), kwargs = {})
#   %expand : Tensor "b8[1, 1, s50, s50][s50**2, s50**2, s50, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.expand.default](args = (%le_2, [1, -1, %arg0_1, %arg0_1]), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[1, 1, s50, s50][s50**2, s50**2, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_1, %full_default), kwargs = {})
#   %constant_pad_nd : Tensor "f32[1, 1, s50, s50 - (Mod(s50, 8)) + 8][s50*Max(1, s50 - (Mod(s50, 8)) + 8), s50*Max(1, s50 - (Mod(s50, 8)) + 8), Max(1, s50 - (Mod(s50, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, %sub_90], 0.0), kwargs = {})
#   %slice_10 : Tensor "f32[1, 1, s50, s50][s50*Max(1, s50 - (Mod(s50, 8)) + 8), s50*Max(1, s50 - (Mod(s50, 8)) + 8), Max(1, s50 - (Mod(s50, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, %arg0_1), kwargs = {})
#   %expand_7 : Tensor "f32[1, 14, s50, s50][s50*Max(1, s50 - (Mod(s50, 8)) + 8), 0, Max(1, s50 - (Mod(s50, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_10, [1, 14, %arg0_1, %arg0_1]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%add_173, %view_13, %view_14, %expand_7, False), kwargs = {scale: 0.125})
#   return %buf6
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': 'fp64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 573568}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = xindex // 896
    x0 = (xindex % 64)
    x4 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + ((x3 % 32)), xmask, eviction_policy='evict_last')
    tmp6 = in_ptr2
    tmp2 = x2
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
    tmp15 = tl.load(in_ptr0 + (32 + 64*x4 + (x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = -tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp14, tmp16, tmp17)
    tmp19 = tmp10 >= tmp13
    tmp20 = tl.full([1], 64, tl.int64)
    tmp21 = tmp10 < tmp20
    tmp22 = tl.load(in_ptr0 + (64*x4 + ((-32) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.where(tmp14, tmp18, tmp22)
    tmp24 = tl_math.sin(tmp4)
    tmp25 = tmp24 * tmp7
    tmp26 = tmp23 * tmp25
    tmp27 = tmp9 + tmp26
    tl.store(out_ptr0 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/tp/ctpmm3sld5z7mlnqb2iimz6eqkmghrzyioxxsbjojn5twzvo5i6v.py
# Topologically Sorted Source Nodes: [linear, view, query_states, getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_44, hidden_states_3, key, linear_2, view_2, value_states, getitem_49, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   arange => iota
#   arange_3 => iota_3
#   arange_4 => iota_4
#   attention_mask => le_2
#   attention_mask_1 => expand
#   attn_output => _scaled_dot_product_efficient_attention, constant_pad_nd, expand_7, full_default, full_default_1, slice_10, where
#   cat_1 => cat
#   cos => cos
#   cos_3 => unsqueeze_11
#   emb => clone, expand_4, unsqueeze_10, view_3
#   expand_1 => expand_1
#   freqs => permute
#   getitem_13 => unsqueeze_7, unsqueeze_8
#   getitem_16 => unsqueeze_9
#   getitem_44 => unsqueeze_13
#   getitem_49 => unsqueeze_14
#   hidden_states_3 => expand_5
#   hidden_states_4 => expand_6
#   key => clone_4, view_13
#   kv_arange => add_20
#   kv_indices => unsqueeze_4, unsqueeze_5, unsqueeze_6
#   linear => view_5
#   linear_2 => view_11
#   matmul => mul_56
#   mul_4 => mul_157
#   mul_5 => mul_174
#   neg => neg
#   position_ids => add_6
#   position_ids_1 => unsqueeze
#   position_ids_expanded => convert_element_type
#   q_arange => add_14
#   q_embed => add_173
#   q_indices => unsqueeze_1, unsqueeze_2, unsqueeze_3
#   query_states => permute_2
#   sin => sin
#   sin_3 => unsqueeze_12
#   value => clone_5, view_14
#   value_states => permute_6
#   view => view_6
#   view_2 => view_12
#   x1 => slice_4
#   x2 => slice_5
# Graph fragment:
#   %add_197 : Tensor "f32[1, 2, s50, 64][128*s50, 64, 128, 1]cuda:0" = PlaceHolder[target=add_197]
#   %view_5 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, %arg0_1, 896]), kwargs = {})
#   %view_6 : Tensor "f32[1, s50, 14, 64][896*s50, 896, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [1, %arg0_1, -1, 64]), kwargs = {})
#   %permute_2 : Tensor "f32[1, 14, s50, 64][896*s50, 64, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_8, [1, -1, 1]), kwargs = {})
#   %iota : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg0_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_6 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 0), kwargs = {})
#   %unsqueeze : Tensor "i64[1, s50][s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_6, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "i64[1, 1, s50][s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, s50][s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_9, torch.float32), kwargs = {})
#   %mul_56 : Tensor "f32[1, 32, s50][32*s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, s50, 32][32*s50, 1, s50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_56, [0, 2, 1]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, s50, 1, 32][32*s50, 1, 32*s50, s50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, s50, 2, 32][32*s50, 1, 0, s50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_10, [1, %arg0_1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, s50, 2, 32][64*s50, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, %arg0_1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, %convert_element_type_default_1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 1, s50, 64][64*s50, 64*s50, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor, 1), kwargs = {})
#   %mul_157 : Tensor "f32[1, 14, s50, 64][896*s50, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_11), kwargs = {})
#   %slice_5 : Tensor "f32[1, 14, s50, 32][896*s50, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 32, 9223372036854775807), kwargs = {})
#   %neg : Tensor "f32[1, 14, s50, 32][448*s50, 32, 448, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_5,), kwargs = {})
#   %slice_4 : Tensor "f32[1, 14, s50, 32][896*s50, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 0, 32), kwargs = {})
#   %cat : Tensor "f32[1, 14, s50, 64][896*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_4], -1), kwargs = {})
#   %sin : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, %convert_element_type_default_2), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 1, s50, 64][64*s50, 64*s50, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor_1, 1), kwargs = {})
#   %mul_174 : Tensor "f32[1, 14, s50, 64][896*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_12), kwargs = {})
#   %add_173 : Tensor "f32[1, 14, s50, 64][896*s50, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_157, %mul_174), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[1, 2, 1, s50, 64][128*s50, 64, 128*s50, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_197, 2), kwargs = {})
#   %expand_5 : Tensor "f32[1, 2, 7, s50, 64][128*s50, 64, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_13, [1, 2, 7, %arg0_1, 64]), kwargs = {})
#   %clone_4 : Tensor "f32[1, 2, 7, s50, 64][896*s50, 448*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_13 : Tensor "f32[1, 14, s50, 64][896*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [1, 14, %arg0_1, 64]), kwargs = {})
#   %view_11 : Tensor "f32[1, s50, 128][128*s50, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [1, %arg0_1, 128]), kwargs = {})
#   %view_12 : Tensor "f32[1, s50, 2, 64][128*s50, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_11, [1, %arg0_1, -1, 64]), kwargs = {})
#   %permute_6 : Tensor "f32[1, 2, s50, 64][128*s50, 64, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_12, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[1, 2, 1, s50, 64][128*s50, 64, 128*s50, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_6, 2), kwargs = {})
#   %expand_6 : Tensor "f32[1, 2, 7, s50, 64][128*s50, 64, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_14, [1, 2, 7, %arg0_1, 64]), kwargs = {})
#   %clone_5 : Tensor "f32[1, 2, 7, s50, 64][896*s50, 448*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_14 : Tensor "f32[1, 14, s50, 64][896*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_5, [1, 14, %arg0_1, 64]), kwargs = {})
#   %iota_4 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg0_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_20 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_4, 0), kwargs = {})
#   %unsqueeze_4 : Tensor "i64[1, s50][s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_20, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[1, 1, s50][s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 1), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[1, 1, 1, s50][s50, s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 2), kwargs = {})
#   %iota_3 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg0_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_14 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_3, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[1, s50][s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_14, 0), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[1, 1, s50][s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 1), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, s50, 1][s50, s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 3), kwargs = {})
#   %le_2 : Tensor "b8[1, 1, s50, s50][s50**2, s50**2, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%unsqueeze_6, %unsqueeze_3), kwargs = {})
#   %expand : Tensor "b8[1, 1, s50, s50][s50**2, s50**2, s50, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.expand.default](args = (%le_2, [1, -1, %arg0_1, %arg0_1]), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[1, 1, s50, s50][s50**2, s50**2, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_1, %full_default), kwargs = {})
#   %constant_pad_nd : Tensor "f32[1, 1, s50, s50 - (Mod(s50, 8)) + 8][s50*Max(1, s50 - (Mod(s50, 8)) + 8), s50*Max(1, s50 - (Mod(s50, 8)) + 8), Max(1, s50 - (Mod(s50, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, %sub_90], 0.0), kwargs = {})
#   %slice_10 : Tensor "f32[1, 1, s50, s50][s50*Max(1, s50 - (Mod(s50, 8)) + 8), s50*Max(1, s50 - (Mod(s50, 8)) + 8), Max(1, s50 - (Mod(s50, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, %arg0_1), kwargs = {})
#   %expand_7 : Tensor "f32[1, 14, s50, s50][s50*Max(1, s50 - (Mod(s50, 8)) + 8), 0, Max(1, s50 - (Mod(s50, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_10, [1, 14, %arg0_1, %arg0_1]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%add_173, %view_13, %view_14, %expand_7, False), kwargs = {scale: 0.125})
#   return %buf7
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 245760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % ks0)
    x2 = xindex // ks1
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*(x2 // 7) + 128*x1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/yp/cypiqw3o35wv7bh6mjttyrmuppgcytiziq7gxac2ibhxeym3wlwv.py
# Topologically Sorted Source Nodes: [linear, view, query_states, getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_44, hidden_states_3, key, linear_2, view_2, value_states, getitem_49, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   arange => iota
#   arange_3 => iota_3
#   arange_4 => iota_4
#   attention_mask => le_2
#   attention_mask_1 => expand
#   attn_output => _scaled_dot_product_efficient_attention, constant_pad_nd, expand_7, full_default, full_default_1, slice_10, where
#   cat_1 => cat
#   cos => cos
#   cos_3 => unsqueeze_11
#   emb => clone, expand_4, unsqueeze_10, view_3
#   expand_1 => expand_1
#   freqs => permute
#   getitem_13 => unsqueeze_7, unsqueeze_8
#   getitem_16 => unsqueeze_9
#   getitem_44 => unsqueeze_13
#   getitem_49 => unsqueeze_14
#   hidden_states_3 => expand_5
#   hidden_states_4 => expand_6
#   key => clone_4, view_13
#   kv_arange => add_20
#   kv_indices => unsqueeze_4, unsqueeze_5, unsqueeze_6
#   linear => view_5
#   linear_2 => view_11
#   matmul => mul_56
#   mul_4 => mul_157
#   mul_5 => mul_174
#   neg => neg
#   position_ids => add_6
#   position_ids_1 => unsqueeze
#   position_ids_expanded => convert_element_type
#   q_arange => add_14
#   q_embed => add_173
#   q_indices => unsqueeze_1, unsqueeze_2, unsqueeze_3
#   query_states => permute_2
#   sin => sin
#   sin_3 => unsqueeze_12
#   value => clone_5, view_14
#   value_states => permute_6
#   view => view_6
#   view_2 => view_12
#   x1 => slice_4
#   x2 => slice_5
# Graph fragment:
#   %view_5 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, %arg0_1, 896]), kwargs = {})
#   %view_6 : Tensor "f32[1, s50, 14, 64][896*s50, 896, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [1, %arg0_1, -1, 64]), kwargs = {})
#   %permute_2 : Tensor "f32[1, 14, s50, 64][896*s50, 64, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_8, [1, -1, 1]), kwargs = {})
#   %iota : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg0_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_6 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 0), kwargs = {})
#   %unsqueeze : Tensor "i64[1, s50][s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_6, 0), kwargs = {})
#   %unsqueeze_9 : Tensor "i64[1, 1, s50][s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, s50][s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_9, torch.float32), kwargs = {})
#   %mul_56 : Tensor "f32[1, 32, s50][32*s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, s50, 32][32*s50, 1, s50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_56, [0, 2, 1]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, s50, 1, 32][32*s50, 1, 32*s50, s50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, s50, 2, 32][32*s50, 1, 0, s50]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_10, [1, %arg0_1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, s50, 2, 32][64*s50, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, %arg0_1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, %convert_element_type_default_1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 1, s50, 64][64*s50, 64*s50, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor, 1), kwargs = {})
#   %mul_157 : Tensor "f32[1, 14, s50, 64][896*s50, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_11), kwargs = {})
#   %slice_5 : Tensor "f32[1, 14, s50, 32][896*s50, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 32, 9223372036854775807), kwargs = {})
#   %neg : Tensor "f32[1, 14, s50, 32][448*s50, 32, 448, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_5,), kwargs = {})
#   %slice_4 : Tensor "f32[1, 14, s50, 32][896*s50, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 0, 32), kwargs = {})
#   %cat : Tensor "f32[1, 14, s50, 64][896*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_4], -1), kwargs = {})
#   %sin : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.float32), kwargs = {})
#   %mul_tensor_1 : Tensor "f32[1, s50, 64][64*s50, 64, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, %convert_element_type_default_2), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 1, s50, 64][64*s50, 64*s50, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_tensor_1, 1), kwargs = {})
#   %mul_174 : Tensor "f32[1, 14, s50, 64][896*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_12), kwargs = {})
#   %add_173 : Tensor "f32[1, 14, s50, 64][896*s50, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_157, %mul_174), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[1, 2, 1, s50, 64][128*s50, 64, 128*s50, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_197, 2), kwargs = {})
#   %expand_5 : Tensor "f32[1, 2, 7, s50, 64][128*s50, 64, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_13, [1, 2, 7, %arg0_1, 64]), kwargs = {})
#   %clone_4 : Tensor "f32[1, 2, 7, s50, 64][896*s50, 448*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_13 : Tensor "f32[1, 14, s50, 64][896*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [1, 14, %arg0_1, 64]), kwargs = {})
#   %view_11 : Tensor "f32[1, s50, 128][128*s50, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [1, %arg0_1, 128]), kwargs = {})
#   %view_12 : Tensor "f32[1, s50, 2, 64][128*s50, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_11, [1, %arg0_1, -1, 64]), kwargs = {})
#   %permute_6 : Tensor "f32[1, 2, s50, 64][128*s50, 64, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_12, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[1, 2, 1, s50, 64][128*s50, 64, 128*s50, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_6, 2), kwargs = {})
#   %expand_6 : Tensor "f32[1, 2, 7, s50, 64][128*s50, 64, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_14, [1, 2, 7, %arg0_1, 64]), kwargs = {})
#   %clone_5 : Tensor "f32[1, 2, 7, s50, 64][896*s50, 448*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_14 : Tensor "f32[1, 14, s50, 64][896*s50, 64*s50, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_5, [1, 14, %arg0_1, 64]), kwargs = {})
#   %iota_4 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg0_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_20 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_4, 0), kwargs = {})
#   %unsqueeze_4 : Tensor "i64[1, s50][s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_20, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[1, 1, s50][s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 1), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[1, 1, 1, s50][s50, s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 2), kwargs = {})
#   %iota_3 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg0_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_14 : Tensor "i64[s50][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_3, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[1, s50][s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_14, 0), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[1, 1, s50][s50, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 1), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, s50, 1][s50, s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 3), kwargs = {})
#   %le_2 : Tensor "b8[1, 1, s50, s50][s50**2, s50**2, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%unsqueeze_6, %unsqueeze_3), kwargs = {})
#   %expand : Tensor "b8[1, 1, s50, s50][s50**2, s50**2, s50, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.expand.default](args = (%le_2, [1, -1, %arg0_1, %arg0_1]), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[1, 1, s50, s50][s50**2, s50**2, s50, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_1, %full_default), kwargs = {})
#   %constant_pad_nd : Tensor "f32[1, 1, s50, s50 - (Mod(s50, 8)) + 8][s50*Max(1, s50 - (Mod(s50, 8)) + 8), s50*Max(1, s50 - (Mod(s50, 8)) + 8), Max(1, s50 - (Mod(s50, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, %sub_90], 0.0), kwargs = {})
#   %slice_10 : Tensor "f32[1, 1, s50, s50][s50*Max(1, s50 - (Mod(s50, 8)) + 8), s50*Max(1, s50 - (Mod(s50, 8)) + 8), Max(1, s50 - (Mod(s50, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, %arg0_1), kwargs = {})
#   %expand_7 : Tensor "f32[1, 14, s50, s50][s50*Max(1, s50 - (Mod(s50, 8)) + 8), 0, Max(1, s50 - (Mod(s50, 8)) + 8), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_10, [1, 14, %arg0_1, %arg0_1]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%add_173, %view_13, %view_14, %expand_7, False), kwargs = {scale: 0.125})
#   return %buf9
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024},
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8192}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4(out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    tmp0 = x0
    tmp1 = ks0
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = x1
    tmp5 = tmp3 <= tmp4
    tmp6 = tl.full([1], 0.0, tl.float32)
    tmp7 = tl.full([1], float("-inf"), tl.float32)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp2, tmp8, tmp9)
    tl.store(out_ptr0 + (x0 + 8*x1*((7 + ks0) // 8)), tmp10, xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/li/clilrru4aszwdcqiw5taqj4l44e6rmv3m7z2cf2g22dn4wd66cf4.py
# Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, pow_2, variance_1, rsqrt_1, hidden_states_7, hidden_states_8], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   attn_output_3 => view_17
#   hidden_states_5 => add_274
#   hidden_states_7 => mul_352
#   hidden_states_8 => mul_356
#   inputs_embeds => embedding
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
#   variance_1 => mean_1
# Graph fragment:
#   %arg2_1 : Tensor "i64[1, s50][s47, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[151936, 896][896, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg16_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %buf16 : Tensor "f32[1, s50, 1][s50, 1, s50]cuda:0" = PlaceHolder[target=buf16]
#   %arg15_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg15_1]
#   %embedding : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg2_1), kwargs = {})
#   %view_17 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, %arg0_1, 896]), kwargs = {})
#   %add_274 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_17), kwargs = {})
#   %pow_2 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_274, 2), kwargs = {})
#   %mean_1 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg15_1, torch.float32), kwargs = {})
#   %add_tensor_1 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, %convert_element_type_default_6), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_352 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_274, %rsqrt_1), kwargs = {})
#   %mul_356 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg16_1, %mul_352), kwargs = {})
#   return %buf16,%mul_356
triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_5 = async_compile.triton('triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 256, 'r0_': 347648}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 896
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp17 = in_ptr4
    tmp1 = tl.full([1, 1], 151936, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 151936")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 896*tmp4), r0_mask & xmask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp15 = tl.full([1, 1], 896.0, tl.float32)
    tmp16 = (tmp13 / tmp15)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp8 * tmp20
    tmp22 = tmp14 * tmp21
    tl.store(out_ptr1 + (r0_1 + 896*x0), tmp22, r0_mask & xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/cr/ccrpukacv2zfzmvejlu6mops5vq4kztxdtyjdn2ahynau5xz76ib.py
# Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_10], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   linear_4 => view_19
#   linear_5 => view_21
#   mul_10 => mul_390
#   silu => add_306, div, exp, neg_2
# Graph fragment:
#   %mm_1 : Tensor "f32[s50, 4864][4864, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %mm_2 : Tensor "f32[s50, 4864][4864, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %view_19 : Tensor "f32[1, s50, 4864][4864*s50, 4864, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [1, %arg0_1, 4864]), kwargs = {})
#   %neg_2 : Tensor "f32[1, s50, 4864][4864*s50, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%view_19,), kwargs = {})
#   %exp : Tensor "f32[1, s50, 4864][4864*s50, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %add_306 : Tensor "f32[1, s50, 4864][4864*s50, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[1, s50, 4864][4864*s50, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_19, %add_306), kwargs = {})
#   %view_21 : Tensor "f32[1, s50, 4864][4864*s50, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [1, %arg0_1, 4864]), kwargs = {})
#   %mul_390 : Tensor "f32[1, s50, 4864][4864*s50, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %view_21), kwargs = {})
#   return %mul_390
triton_poi_fused__unsafe_view_mul_silu_6 = async_compile.triton('triton_poi_fused__unsafe_view_mul_silu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_mul_silu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2490368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_silu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: <inductor-cache>/sg/csg4qzj3gyjy7nd3lg5dr53224f2cylmtvizwnherbla3jwkyfh4.py
# Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, pow_3, variance_2, rsqrt_2, hidden_states_11, hidden_states_12], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   attn_output_3 => view_17
#   down_proj => view_23
#   hidden_states_11 => mul_421
#   hidden_states_12 => mul_425
#   hidden_states_5 => add_274
#   hidden_states_9 => add_331
#   inputs_embeds => embedding
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
#   variance_2 => mean_2
# Graph fragment:
#   %arg2_1 : Tensor "i64[1, s50][s47, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[151936, 896][896, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm]
#   %mm_3 : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %arg21_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg21_1]
#   %buf22 : Tensor "f32[1, s50, 1][s50, 1, s50]cuda:0" = PlaceHolder[target=buf22]
#   %arg20_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg20_1]
#   %embedding : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg2_1), kwargs = {})
#   %view_17 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, %arg0_1, 896]), kwargs = {})
#   %add_274 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_17), kwargs = {})
#   %view_23 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, %arg0_1, 896]), kwargs = {})
#   %add_331 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_274, %view_23), kwargs = {})
#   %pow_3 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_331, 2), kwargs = {})
#   %mean_2 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %convert_element_type_default_8 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg20_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, %convert_element_type_default_8), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_2,), kwargs = {})
#   %mul_421 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_331, %rsqrt_2), kwargs = {})
#   %mul_425 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg21_1, %mul_421), kwargs = {})
#   return %buf22,%mul_425
triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 256, 'r0_': 462336}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 896
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp19 = in_ptr5
    tmp1 = tl.full([1, 1], 151936, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 151936")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 896*tmp4), r0_mask & xmask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp17 = tl.full([1, 1], 896.0, tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp10 * tmp22
    tmp24 = tmp16 * tmp23
    tl.store(out_ptr1 + (r0_1 + 896*x0), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/o6/co6bagpfuxqfhkcz7vz5yp3mzbwjc62y2ttrmsaw7kijxf655sjq.py
# Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, attn_output_7, hidden_states_15, pow_4, variance_3, rsqrt_3, hidden_states_17, hidden_states_18], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   attn_output_3 => view_17
#   attn_output_7 => view_37
#   down_proj => view_23
#   hidden_states_15 => add_529
#   hidden_states_17 => mul_674
#   hidden_states_18 => mul_678
#   hidden_states_5 => add_274
#   hidden_states_9 => add_331
#   inputs_embeds => embedding
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
#   variance_3 => mean_3
# Graph fragment:
#   %arg2_1 : Tensor "i64[1, s50][s47, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[151936, 896][896, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %mm : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm]
#   %mm_3 : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %mm_4 : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %add_529 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0" = PlaceHolder[target=add_529]
#   %arg30_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg30_1]
#   %buf39 : Tensor "f32[1, s50, 1][s50, 1, s50]cuda:0" = PlaceHolder[target=buf39]
#   %arg29_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg29_1]
#   %embedding : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg2_1), kwargs = {})
#   %view_17 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, %arg0_1, 896]), kwargs = {})
#   %add_274 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_17), kwargs = {})
#   %view_23 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, %arg0_1, 896]), kwargs = {})
#   %add_331 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_274, %view_23), kwargs = {})
#   %view_37 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [1, %arg0_1, 896]), kwargs = {})
#   %add_529 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_331, %view_37), kwargs = {})
#   %pow_4 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_529, 2), kwargs = {})
#   %mean_3 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %convert_element_type_default_10 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg29_1, torch.float32), kwargs = {})
#   %add_tensor_3 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, %convert_element_type_default_10), kwargs = {})
#   %rsqrt_3 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_674 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_529, %rsqrt_3), kwargs = {})
#   %mul_678 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg30_1, %mul_674), kwargs = {})
#   return %add_529,%buf39,%mul_678
triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8 = async_compile.triton('triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 256, 'r0_': 806400}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 896
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp21 = in_ptr5
    tmp1 = tl.full([1, 1], 151936, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 151936")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 896*tmp4), r0_mask & xmask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp19 = tl.full([1, 1], 896.0, tl.float32)
    tmp20 = (tmp17 / tmp19)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp12 * tmp24
    tmp26 = tmp18 * tmp25
    tl.store(in_out_ptr0 + (r0_1 + 896*x0), tmp12, r0_mask & xmask)
    tl.store(out_ptr1 + (r0_1 + 896*x0), tmp26, r0_mask & xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/op/cop5dbjd4rmllcn4cdoezccuv6xbbmhv4dwvjkgsftbpbdsb5l4u.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, pow_5, variance_4, rsqrt_4, hidden_states_21, hidden_states_22], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   down_proj_1 => view_43
#   hidden_states_19 => add_586
#   hidden_states_21 => mul_743
#   hidden_states_22 => mul_747
#   pow_5 => pow_5
#   rsqrt_4 => rsqrt_4
#   variance_4 => mean_4
# Graph fragment:
#   %add_529 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0" = PlaceHolder[target=add_529]
#   %mm_7 : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm_7]
#   %arg35_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %buf45 : Tensor "f32[1, s50, 1][s50, 1, s50]cuda:0" = PlaceHolder[target=buf45]
#   %arg34_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg34_1]
#   %view_43 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [1, %arg0_1, 896]), kwargs = {})
#   %add_586 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_529, %view_43), kwargs = {})
#   %pow_5 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_586, 2), kwargs = {})
#   %mean_4 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [-1], True), kwargs = {})
#   %convert_element_type_default_12 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg34_1, torch.float32), kwargs = {})
#   %add_tensor_4 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, %convert_element_type_default_12), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_4,), kwargs = {})
#   %mul_743 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_586, %rsqrt_4), kwargs = {})
#   %mul_747 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg35_1, %mul_743), kwargs = {})
#   return %buf45,%mul_747
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 462336}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 896
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp11 = in_ptr3
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([1, 1], 896.0, tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp2 * tmp14
    tmp16 = tmp8 * tmp15
    tl.store(out_ptr1 + (r0_1 + 896*x0), tmp16, r0_mask & xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/hx/chx5mow7ck6hfrkkn7w6u6464dnsomq2e46jh3exx63x4ahlmsnt.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, pow_6, variance_5, rsqrt_5, hidden_states_27, hidden_states_28], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   attn_output_11 => view_57
#   down_proj_1 => view_43
#   hidden_states_19 => add_586
#   hidden_states_25 => add_784
#   hidden_states_27 => mul_996
#   hidden_states_28 => mul_1000
#   pow_6 => pow_6
#   rsqrt_5 => rsqrt_5
#   variance_5 => mean_5
# Graph fragment:
#   %add_529 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0" = PlaceHolder[target=add_529]
#   %mm_7 : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm_7]
#   %mm_8 : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %arg44_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg44_1]
#   %buf61 : Tensor "f32[1, s50, 1][s50, 1, s50]cuda:0" = PlaceHolder[target=buf61]
#   %arg43_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg43_1]
#   %view_43 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [1, %arg0_1, 896]), kwargs = {})
#   %add_586 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_529, %view_43), kwargs = {})
#   %view_57 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [1, %arg0_1, 896]), kwargs = {})
#   %add_784 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_586, %view_57), kwargs = {})
#   %pow_6 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_784, 2), kwargs = {})
#   %mean_5 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_6, [-1], True), kwargs = {})
#   %convert_element_type_default_14 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg43_1, torch.float32), kwargs = {})
#   %add_tensor_5 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, %convert_element_type_default_14), kwargs = {})
#   %rsqrt_5 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_5,), kwargs = {})
#   %mul_996 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_784, %rsqrt_5), kwargs = {})
#   %mul_1000 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg44_1, %mul_996), kwargs = {})
#   return %buf61,%mul_1000
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 577024}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 896
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp13 = in_ptr4
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp11 = tl.full([1, 1], 896.0, tl.float32)
    tmp12 = (tmp9 / tmp11)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp4 * tmp16
    tmp18 = tmp10 * tmp17
    tl.store(out_ptr1 + (r0_1 + 896*x0), tmp18, r0_mask & xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/gx/cgx3zueyot4r7lbrekhu2stxh7qhdjvznk46wgzsecrf5lxr5qav.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, pow_7, variance_6, rsqrt_6, hidden_states_31, hidden_states_32], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   attn_output_11 => view_57
#   down_proj_1 => view_43
#   down_proj_2 => view_63
#   hidden_states_19 => add_586
#   hidden_states_25 => add_784
#   hidden_states_29 => add_841
#   hidden_states_31 => mul_1065
#   hidden_states_32 => mul_1069
#   pow_7 => pow_7
#   rsqrt_6 => rsqrt_6
#   variance_6 => mean_6
# Graph fragment:
#   %add_529 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0" = PlaceHolder[target=add_529]
#   %mm_7 : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm_7]
#   %mm_8 : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %mm_11 : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm_11]
#   %arg49_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg49_1]
#   %buf67 : Tensor "f32[1, s50, 1][s50, 1, s50]cuda:0" = PlaceHolder[target=buf67]
#   %arg48_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg48_1]
#   %view_43 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [1, %arg0_1, 896]), kwargs = {})
#   %add_586 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_529, %view_43), kwargs = {})
#   %view_57 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [1, %arg0_1, 896]), kwargs = {})
#   %add_784 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_586, %view_57), kwargs = {})
#   %view_63 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_11, [1, %arg0_1, 896]), kwargs = {})
#   %add_841 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_784, %view_63), kwargs = {})
#   %pow_7 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_841, 2), kwargs = {})
#   %mean_6 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [-1], True), kwargs = {})
#   %convert_element_type_default_16 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg48_1, torch.float32), kwargs = {})
#   %add_tensor_6 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_6, %convert_element_type_default_16), kwargs = {})
#   %rsqrt_6 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_6,), kwargs = {})
#   %mul_1065 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_841, %rsqrt_6), kwargs = {})
#   %mul_1069 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg49_1, %mul_1065), kwargs = {})
#   return %buf67,%mul_1069
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 691712}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 896
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp15 = in_ptr5
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp13 = tl.full([1, 1], 896.0, tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp6 * tmp18
    tmp20 = tmp12 * tmp19
    tl.store(out_ptr1 + (r0_1 + 896*x0), tmp20, r0_mask & xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/jk/cjkjogptzmdrz22vqd5spq4lftlrchr55kdqoqy32ov72zlw3ajg.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, attn_output_15, hidden_states_35, pow_8, variance_7, rsqrt_7, hidden_states_37, hidden_states_38], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   attn_output_11 => view_57
#   attn_output_15 => view_77
#   down_proj_1 => view_43
#   down_proj_2 => view_63
#   hidden_states_19 => add_586
#   hidden_states_25 => add_784
#   hidden_states_29 => add_841
#   hidden_states_35 => add_1039
#   hidden_states_37 => mul_1318
#   hidden_states_38 => mul_1322
#   pow_8 => pow_8
#   rsqrt_7 => rsqrt_7
#   variance_7 => mean_7
# Graph fragment:
#   %add_529 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0" = PlaceHolder[target=add_529]
#   %mm_7 : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm_7]
#   %mm_8 : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %mm_11 : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm_11]
#   %mm_12 : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm_12]
#   %add_1039 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0" = PlaceHolder[target=add_1039]
#   %arg58_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg58_1]
#   %buf84 : Tensor "f32[1, s50, 1][s50, 1, s50]cuda:0" = PlaceHolder[target=buf84]
#   %arg57_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg57_1]
#   %view_43 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [1, %arg0_1, 896]), kwargs = {})
#   %add_586 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_529, %view_43), kwargs = {})
#   %view_57 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [1, %arg0_1, 896]), kwargs = {})
#   %add_784 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_586, %view_57), kwargs = {})
#   %view_63 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_11, [1, %arg0_1, 896]), kwargs = {})
#   %add_841 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_784, %view_63), kwargs = {})
#   %view_77 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_12, [1, %arg0_1, 896]), kwargs = {})
#   %add_1039 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_841, %view_77), kwargs = {})
#   %pow_8 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1039, 2), kwargs = {})
#   %mean_7 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [-1], True), kwargs = {})
#   %convert_element_type_default_18 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg57_1, torch.float32), kwargs = {})
#   %add_tensor_7 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_7, %convert_element_type_default_18), kwargs = {})
#   %rsqrt_7 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_7,), kwargs = {})
#   %mul_1318 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1039, %rsqrt_7), kwargs = {})
#   %mul_1322 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg58_1, %mul_1318), kwargs = {})
#   return %add_1039,%buf84,%mul_1322
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': 'fp64', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 1035776}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 896
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp17 = in_ptr5
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp15 = tl.full([1, 1], 896.0, tl.float32)
    tmp16 = (tmp13 / tmp15)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp8 * tmp20
    tmp22 = tmp14 * tmp21
    tl.store(in_out_ptr0 + (r0_1 + 896*x0), tmp8, r0_mask & xmask)
    tl.store(out_ptr1 + (r0_1 + 896*x0), tmp22, r0_mask & xmask)
''', device_str='cuda')


# kernel path: <inductor-cache>/qb/cqbfeeauf3izc646kxbg3ekjs67umauttktsx7gbtdkyusnlfeyk.py
# Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, pow_49, variance_48, rsqrt_48, hidden_states_241, hidden_states_242], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   down_proj_23 => view_483
#   hidden_states_239 => add_6196
#   hidden_states_241 => mul_7827
#   hidden_states_242 => mul_7831
#   pow_49 => pow_49
#   rsqrt_48 => rsqrt_48
#   variance_48 => mean_48
# Graph fragment:
#   %add_6139 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0" = PlaceHolder[target=add_6139]
#   %mm_95 : Tensor "f32[s50, 896][896, 1]cuda:0" = PlaceHolder[target=mm_95]
#   %arg343_1 : Tensor "f32[896][1]cuda:0" = PlaceHolder[target=arg343_1]
#   %buf540 : Tensor "f32[1, s50, 1][s50, 1, s50]cuda:0" = PlaceHolder[target=buf540]
#   %arg342_1 : Tensor "f64[][]cpu" = PlaceHolder[target=arg342_1]
#   %view_483 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_95, [1, %arg0_1, 896]), kwargs = {})
#   %add_6196 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6139, %view_483), kwargs = {})
#   %pow_49 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_6196, 2), kwargs = {})
#   %mean_48 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_49, [-1], True), kwargs = {})
#   %convert_element_type_default_100 : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg342_1, torch.float32), kwargs = {})
#   %add_tensor_48 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_48, %convert_element_type_default_100), kwargs = {})
#   %rsqrt_48 : Tensor "f32[1, s50, 1][s50, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_48,), kwargs = {})
#   %mul_7827 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6196, %rsqrt_48), kwargs = {})
#   %mul_7831 : Tensor "f32[1, s50, 896][896*s50, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg343_1, %mul_7827), kwargs = {})
#   return %buf540,%mul_7831
triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': 'fp64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 462336}}
)
@triton.jit
def triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 896
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp11 = in_ptr2
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([1, 1], 896.0, tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp2 * tmp14
    tmp16 = tmp8 * tmp15
    tl.store(in_out_ptr0 + (r0_1 + 896*x0), tmp16, r0_mask & xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1 = args
        args.clear()
        s50 = arg0_1
        s47 = arg1_1
        s35 = arg344_1
        assert_size_stride(arg2_1, (1, s50), (s47, 1))
        assert_size_stride(arg3_1, (151936, 896), (896, 1))
        assert_size_stride(arg7_1, (896, ), (1, ))
        assert_size_stride(arg6_1, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg2_1 = copy_misaligned(arg2_1)
            buf1 = empty_strided_cuda((1, s50, 896), (896*s50, 896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [inputs_embeds, pow_1, variance, rsqrt, hidden_states_1, hidden_states_2], Original ATen: [aten.embedding, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_embedding_mean_mul_pow_rsqrt_0.run(arg2_1, arg3_1, arg7_1, arg6_1.item(), buf1, s50, 896, stream=raw_stream0)
            del arg6_1
            del arg7_1
            assert_size_stride(arg9_1, (896, ), (1, ))
            assert_size_stride(arg8_1, (896, 896), (896, 1))
            buf2 = empty_strided_cuda((s50, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg9_1, reinterpret_tensor(buf1, (s50, 896), (896, 1), 0), reinterpret_tensor(arg8_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf2)
            del arg8_1
            del arg9_1
            assert_size_stride(arg11_1, (128, ), (1, ))
            assert_size_stride(arg10_1, (128, 896), (896, 1))
            buf3 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg11_1, reinterpret_tensor(buf1, (s50, 896), (896, 1), 0), reinterpret_tensor(arg10_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf3)
            del arg10_1
            del arg11_1
            assert_size_stride(arg4_1, (32, ), (1, ))
            assert_size_stride(arg5_1, (), ())
            buf4 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, cos_3, sin, sin_3, linear_1, view_1, key_states, mul_6, x2_1, neg_1, x1_1, cat_2, mul_7, k_embed], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf3, arg4_1, arg5_1.item(), buf4, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg13_1, (128, ), (1, ))
            assert_size_stride(arg12_1, (128, 896), (896, 1))
            buf5 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg13_1, reinterpret_tensor(buf1, (s50, 896), (896, 1), 0), reinterpret_tensor(arg12_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf5)
            del arg12_1
            del arg13_1
            buf6 = reinterpret_tensor(buf1, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf1  # reuse
            # Topologically Sorted Source Nodes: [linear, view, query_states, getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_44, hidden_states_3, key, linear_2, view_2, value_states, getitem_49, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf2, arg4_1, arg5_1.item(), buf6, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            ps0 = 64*s50
            buf7 = reinterpret_tensor(buf2, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [linear, view, query_states, getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_44, hidden_states_3, key, linear_2, view_2, value_states, getitem_49, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf4, buf7, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf8 = empty_strided_cuda((1, 14, s50, 64), (896*s50, 64*s50, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, view, query_states, getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_44, hidden_states_3, key, linear_2, view_2, value_states, getitem_49, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf5, buf8, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf9 = empty_strided_cuda((1, 1, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, view, query_states, getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_44, hidden_states_3, key, linear_2, view_2, value_states, getitem_49, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf9, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [linear, view, query_states, getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, cos_3, mul_4, x2, neg, x1, cat_1, sin, sin_3, mul_5, q_embed, getitem_44, hidden_states_3, key, linear_2, view_2, value_states, getitem_49, hidden_states_4, value, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, attn_output], Original ATen: [aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.cat, aten.cos, aten.mul, aten.slice, aten.neg, aten.sin, aten.clone, aten._unsafe_view, aten.le, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf6, buf7, buf8, reinterpret_tensor(buf9, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            buf11 = buf10[0]
            assert_size_stride(buf11, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf11, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf10
            assert_size_stride(arg14_1, (896, 896), (896, 1))
            buf15 = reinterpret_tensor(buf8, (s50, 896), (896, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [transpose_4, reshape_2, attn_output_3], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf11, (s50, 896), (896, 1), 0), reinterpret_tensor(arg14_1, (896, 896), (1, 896), 0), out=buf15)
            del arg14_1
            assert_size_stride(arg16_1, (896, ), (1, ))
            assert_size_stride(arg15_1, (), ())
            buf17 = reinterpret_tensor(buf11, (1, s50, 896), (896*s50, 896, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, pow_2, variance_1, rsqrt_1, hidden_states_7, hidden_states_8], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_5.run(arg2_1, arg3_1, buf15, arg16_1, arg15_1.item(), buf17, s50, 896, stream=raw_stream0)
            del arg15_1
            del arg16_1
            assert_size_stride(arg17_1, (4864, 896), (896, 1))
            buf18 = empty_strided_cuda((s50, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf17, (s50, 896), (896, 1), 0), reinterpret_tensor(arg17_1, (896, 4864), (1, 896), 0), out=buf18)
            del arg17_1
            assert_size_stride(arg18_1, (4864, 896), (896, 1))
            buf19 = empty_strided_cuda((s50, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf17, (s50, 896), (896, 1), 0), reinterpret_tensor(arg18_1, (896, 4864), (1, 896), 0), out=buf19)
            del arg18_1
            buf20 = reinterpret_tensor(buf18, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_10], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf20, buf19, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg19_1, (896, 4864), (4864, 1))
            buf21 = reinterpret_tensor(buf17, (s50, 896), (896, 1), 0); del buf17  # reuse
            # Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_10, down_proj], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf20, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg19_1, (4864, 896), (1, 4864), 0), out=buf21)
            del arg19_1
            assert_size_stride(arg21_1, (896, ), (1, ))
            assert_size_stride(arg20_1, (), ())
            buf23 = reinterpret_tensor(buf7, (1, s50, 896), (896*s50, 896, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, pow_3, variance_2, rsqrt_2, hidden_states_11, hidden_states_12], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7.run(arg2_1, arg3_1, buf15, buf21, arg21_1, arg20_1.item(), buf23, s50, 896, stream=raw_stream0)
            del arg20_1
            del arg21_1
            assert_size_stride(arg23_1, (896, ), (1, ))
            assert_size_stride(arg22_1, (896, 896), (896, 1))
            buf24 = reinterpret_tensor(buf6, (s50, 896), (896, 1), 0); del buf6  # reuse
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg23_1, reinterpret_tensor(buf23, (s50, 896), (896, 1), 0), reinterpret_tensor(arg22_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf24)
            del arg22_1
            del arg23_1
            assert_size_stride(arg25_1, (128, ), (1, ))
            assert_size_stride(arg24_1, (128, 896), (896, 1))
            buf25 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg25_1, reinterpret_tensor(buf23, (s50, 896), (896, 1), 0), reinterpret_tensor(arg24_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf25)
            del arg24_1
            del arg25_1
            buf26 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_4, sin_4, linear_8, view_4, key_states_1, mul_15, x2_3, neg_3, x1_3, cat_6, mul_16, k_embed_1], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf25, arg4_1, arg5_1.item(), buf26, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg27_1, (128, ), (1, ))
            assert_size_stride(arg26_1, (128, 896), (896, 1))
            buf27 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg27_1, reinterpret_tensor(buf23, (s50, 896), (896, 1), 0), reinterpret_tensor(arg26_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf27)
            del arg26_1
            del arg27_1
            buf28 = reinterpret_tensor(buf23, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf23  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_7, view_3, query_states_1, cos_4, mul_13, x2_2, neg_2, x1_2, cat_5, sin_4, mul_14, q_embed_1, getitem_81, hidden_states_13, key_1, linear_9, view_5, value_states_1, getitem_86, hidden_states_14, value_1, attn_output_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf24, arg4_1, arg5_1.item(), buf28, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf29 = reinterpret_tensor(buf24, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_7, view_3, query_states_1, cos_4, mul_13, x2_2, neg_2, x1_2, cat_5, sin_4, mul_14, q_embed_1, getitem_81, hidden_states_13, key_1, linear_9, view_5, value_states_1, getitem_86, hidden_states_14, value_1, attn_output_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf26, buf29, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf30 = empty_strided_cuda((1, 14, s50, 64), (896*s50, 64*s50, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_7, view_3, query_states_1, cos_4, mul_13, x2_2, neg_2, x1_2, cat_5, sin_4, mul_14, q_embed_1, getitem_81, hidden_states_13, key_1, linear_9, view_5, value_states_1, getitem_86, hidden_states_14, value_1, attn_output_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf27, buf30, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf31 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_7, view_3, query_states_1, cos_4, mul_13, x2_2, neg_2, x1_2, cat_5, sin_4, mul_14, q_embed_1, getitem_81, hidden_states_13, key_1, linear_9, view_5, value_states_1, getitem_86, hidden_states_14, value_1, attn_output_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf31, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_7, view_3, query_states_1, cos_4, mul_13, x2_2, neg_2, x1_2, cat_5, sin_4, mul_14, q_embed_1, getitem_81, hidden_states_13, key_1, linear_9, view_5, value_states_1, getitem_86, hidden_states_14, value_1, attn_output_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf32 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf28, buf29, buf30, reinterpret_tensor(buf31, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            buf33 = buf32[0]
            assert_size_stride(buf33, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf33, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf32
            assert_size_stride(arg28_1, (896, 896), (896, 1))
            buf37 = reinterpret_tensor(buf30, (s50, 896), (896, 1), 0); del buf30  # reuse
            # Topologically Sorted Source Nodes: [transpose_8, reshape_5, attn_output_7], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf33, (s50, 896), (896, 1), 0), reinterpret_tensor(arg28_1, (896, 896), (1, 896), 0), out=buf37)
            del arg28_1
            assert_size_stride(arg30_1, (896, ), (1, ))
            assert_size_stride(arg29_1, (), ())
            buf38 = reinterpret_tensor(buf15, (1, s50, 896), (896*s50, 896, 1), 0); del buf15  # reuse
            buf40 = reinterpret_tensor(buf33, (1, s50, 896), (896*s50, 896, 1), 0); del buf33  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, attn_output_7, hidden_states_15, pow_4, variance_3, rsqrt_3, hidden_states_17, hidden_states_18], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8.run(buf38, arg2_1, arg3_1, buf21, buf37, arg30_1, arg29_1.item(), buf40, s50, 896, stream=raw_stream0)
            del arg29_1
            del arg2_1
            del arg30_1
            assert_size_stride(arg31_1, (4864, 896), (896, 1))
            buf41 = reinterpret_tensor(buf20, (s50, 4864), (4864, 1), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [pow_4, variance_3, rsqrt_3, hidden_states_17, hidden_states_18, linear_11], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (s50, 896), (896, 1), 0), reinterpret_tensor(arg31_1, (896, 4864), (1, 896), 0), out=buf41)
            del arg31_1
            assert_size_stride(arg32_1, (4864, 896), (896, 1))
            buf42 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (s50, 896), (896, 1), 0), reinterpret_tensor(arg32_1, (896, 4864), (1, 896), 0), out=buf42)
            del arg32_1
            buf43 = reinterpret_tensor(buf41, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf41  # reuse
            # Topologically Sorted Source Nodes: [linear_11, silu_1, linear_12, mul_19], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf43, buf42, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg33_1, (896, 4864), (4864, 1))
            buf44 = reinterpret_tensor(buf40, (s50, 896), (896, 1), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [linear_11, silu_1, linear_12, mul_19, down_proj_1], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf43, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg33_1, (4864, 896), (1, 4864), 0), out=buf44)
            del arg33_1
            assert_size_stride(arg35_1, (896, ), (1, ))
            assert_size_stride(arg34_1, (), ())
            buf46 = reinterpret_tensor(buf37, (1, s50, 896), (896*s50, 896, 1), 0); del buf37  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, pow_5, variance_4, rsqrt_4, hidden_states_21, hidden_states_22], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf38, buf44, arg35_1, arg34_1.item(), buf46, s50, 896, stream=raw_stream0)
            del arg34_1
            del arg35_1
            assert_size_stride(arg37_1, (896, ), (1, ))
            assert_size_stride(arg36_1, (896, 896), (896, 1))
            buf47 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg37_1, reinterpret_tensor(buf46, (s50, 896), (896, 1), 0), reinterpret_tensor(arg36_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf47)
            del arg36_1
            del arg37_1
            assert_size_stride(arg39_1, (128, ), (1, ))
            assert_size_stride(arg38_1, (128, 896), (896, 1))
            buf48 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg39_1, reinterpret_tensor(buf46, (s50, 896), (896, 1), 0), reinterpret_tensor(arg38_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf48)
            del arg38_1
            del arg39_1
            buf49 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_5, sin_5, linear_15, view_7, key_states_2, mul_24, x2_5, neg_5, x1_5, cat_10, mul_25, k_embed_2], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf48, arg4_1, arg5_1.item(), buf49, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg41_1, (128, ), (1, ))
            assert_size_stride(arg40_1, (128, 896), (896, 1))
            buf50 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg41_1, reinterpret_tensor(buf46, (s50, 896), (896, 1), 0), reinterpret_tensor(arg40_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf50)
            del arg40_1
            del arg41_1
            buf51 = reinterpret_tensor(buf46, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf46  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_14, view_6, query_states_2, cos_5, mul_22, x2_4, neg_4, x1_4, cat_9, sin_5, mul_23, q_embed_2, getitem_118, hidden_states_23, key_2, linear_16, view_8, value_states_2, getitem_123, hidden_states_24, value_2, attn_output_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf47, arg4_1, arg5_1.item(), buf51, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf52 = reinterpret_tensor(buf47, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf47  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_14, view_6, query_states_2, cos_5, mul_22, x2_4, neg_4, x1_4, cat_9, sin_5, mul_23, q_embed_2, getitem_118, hidden_states_23, key_2, linear_16, view_8, value_states_2, getitem_123, hidden_states_24, value_2, attn_output_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf49, buf52, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf53 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_14, view_6, query_states_2, cos_5, mul_22, x2_4, neg_4, x1_4, cat_9, sin_5, mul_23, q_embed_2, getitem_118, hidden_states_23, key_2, linear_16, view_8, value_states_2, getitem_123, hidden_states_24, value_2, attn_output_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf50, buf53, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf54 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_14, view_6, query_states_2, cos_5, mul_22, x2_4, neg_4, x1_4, cat_9, sin_5, mul_23, q_embed_2, getitem_118, hidden_states_23, key_2, linear_16, view_8, value_states_2, getitem_123, hidden_states_24, value_2, attn_output_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf54, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_14, view_6, query_states_2, cos_5, mul_22, x2_4, neg_4, x1_4, cat_9, sin_5, mul_23, q_embed_2, getitem_118, hidden_states_23, key_2, linear_16, view_8, value_states_2, getitem_123, hidden_states_24, value_2, attn_output_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf55 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf51, buf52, buf53, reinterpret_tensor(buf54, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            buf56 = buf55[0]
            assert_size_stride(buf56, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf56, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf55
            assert_size_stride(arg42_1, (896, 896), (896, 1))
            buf60 = reinterpret_tensor(buf53, (s50, 896), (896, 1), 0); del buf53  # reuse
            # Topologically Sorted Source Nodes: [transpose_12, reshape_8, attn_output_11], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf56, (s50, 896), (896, 1), 0), reinterpret_tensor(arg42_1, (896, 896), (1, 896), 0), out=buf60)
            del arg42_1
            assert_size_stride(arg44_1, (896, ), (1, ))
            assert_size_stride(arg43_1, (), ())
            buf62 = reinterpret_tensor(buf56, (1, s50, 896), (896*s50, 896, 1), 0); del buf56  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, pow_6, variance_5, rsqrt_5, hidden_states_27, hidden_states_28], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf38, buf44, buf60, arg44_1, arg43_1.item(), buf62, s50, 896, stream=raw_stream0)
            del arg43_1
            del arg44_1
            assert_size_stride(arg45_1, (4864, 896), (896, 1))
            buf63 = reinterpret_tensor(buf43, (s50, 4864), (4864, 1), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf62, (s50, 896), (896, 1), 0), reinterpret_tensor(arg45_1, (896, 4864), (1, 896), 0), out=buf63)
            del arg45_1
            assert_size_stride(arg46_1, (4864, 896), (896, 1))
            buf64 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf62, (s50, 896), (896, 1), 0), reinterpret_tensor(arg46_1, (896, 4864), (1, 896), 0), out=buf64)
            del arg46_1
            buf65 = reinterpret_tensor(buf63, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [linear_18, silu_2, linear_19, mul_28], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf65, buf64, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg47_1, (896, 4864), (4864, 1))
            buf66 = reinterpret_tensor(buf62, (s50, 896), (896, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [linear_18, silu_2, linear_19, mul_28, down_proj_2], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf65, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg47_1, (4864, 896), (1, 4864), 0), out=buf66)
            del arg47_1
            assert_size_stride(arg49_1, (896, ), (1, ))
            assert_size_stride(arg48_1, (), ())
            buf68 = reinterpret_tensor(buf52, (1, s50, 896), (896*s50, 896, 1), 0); del buf52  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, pow_7, variance_6, rsqrt_6, hidden_states_31, hidden_states_32], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf38, buf44, buf60, buf66, arg49_1, arg48_1.item(), buf68, s50, 896, stream=raw_stream0)
            del arg48_1
            del arg49_1
            assert_size_stride(arg51_1, (896, ), (1, ))
            assert_size_stride(arg50_1, (896, 896), (896, 1))
            buf69 = reinterpret_tensor(buf51, (s50, 896), (896, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg51_1, reinterpret_tensor(buf68, (s50, 896), (896, 1), 0), reinterpret_tensor(arg50_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf69)
            del arg50_1
            del arg51_1
            assert_size_stride(arg53_1, (128, ), (1, ))
            assert_size_stride(arg52_1, (128, 896), (896, 1))
            buf70 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg53_1, reinterpret_tensor(buf68, (s50, 896), (896, 1), 0), reinterpret_tensor(arg52_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf70)
            del arg52_1
            del arg53_1
            buf71 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_6, sin_6, linear_22, view_10, key_states_3, mul_33, x2_7, neg_7, x1_7, cat_14, mul_34, k_embed_3], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf70, arg4_1, arg5_1.item(), buf71, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg55_1, (128, ), (1, ))
            assert_size_stride(arg54_1, (128, 896), (896, 1))
            buf72 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg55_1, reinterpret_tensor(buf68, (s50, 896), (896, 1), 0), reinterpret_tensor(arg54_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf72)
            del arg54_1
            del arg55_1
            buf73 = reinterpret_tensor(buf68, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf68  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_21, view_9, query_states_3, cos_6, mul_31, x2_6, neg_6, x1_6, cat_13, sin_6, mul_32, q_embed_3, getitem_155, hidden_states_33, key_3, linear_23, view_11, value_states_3, getitem_160, hidden_states_34, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf69, arg4_1, arg5_1.item(), buf73, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf74 = reinterpret_tensor(buf69, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf69  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_21, view_9, query_states_3, cos_6, mul_31, x2_6, neg_6, x1_6, cat_13, sin_6, mul_32, q_embed_3, getitem_155, hidden_states_33, key_3, linear_23, view_11, value_states_3, getitem_160, hidden_states_34, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf71, buf74, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf75 = reinterpret_tensor(buf28, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_21, view_9, query_states_3, cos_6, mul_31, x2_6, neg_6, x1_6, cat_13, sin_6, mul_32, q_embed_3, getitem_155, hidden_states_33, key_3, linear_23, view_11, value_states_3, getitem_160, hidden_states_34, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf72, buf75, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf76 = buf54; del buf54  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_21, view_9, query_states_3, cos_6, mul_31, x2_6, neg_6, x1_6, cat_13, sin_6, mul_32, q_embed_3, getitem_155, hidden_states_33, key_3, linear_23, view_11, value_states_3, getitem_160, hidden_states_34, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf76, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_21, view_9, query_states_3, cos_6, mul_31, x2_6, neg_6, x1_6, cat_13, sin_6, mul_32, q_embed_3, getitem_155, hidden_states_33, key_3, linear_23, view_11, value_states_3, getitem_160, hidden_states_34, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf77 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf73, buf74, buf75, reinterpret_tensor(buf76, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf73
            del buf74
            buf78 = buf77[0]
            assert_size_stride(buf78, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf78, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf77
            assert_size_stride(arg56_1, (896, 896), (896, 1))
            buf82 = reinterpret_tensor(buf75, (s50, 896), (896, 1), 0); del buf75  # reuse
            # Topologically Sorted Source Nodes: [transpose_16, reshape_11, attn_output_15], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf78, (s50, 896), (896, 1), 0), reinterpret_tensor(arg56_1, (896, 896), (1, 896), 0), out=buf82)
            del arg56_1
            assert_size_stride(arg58_1, (896, ), (1, ))
            assert_size_stride(arg57_1, (), ())
            buf83 = buf38; del buf38  # reuse
            buf85 = reinterpret_tensor(buf78, (1, s50, 896), (896*s50, 896, 1), 0); del buf78  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, attn_output_15, hidden_states_35, pow_8, variance_7, rsqrt_7, hidden_states_37, hidden_states_38], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf83, buf44, buf60, buf66, buf82, arg58_1, arg57_1.item(), buf85, s50, 896, stream=raw_stream0)
            del arg57_1
            del arg58_1
            assert_size_stride(arg59_1, (4864, 896), (896, 1))
            buf86 = reinterpret_tensor(buf65, (s50, 4864), (4864, 1), 0); del buf65  # reuse
            # Topologically Sorted Source Nodes: [pow_8, variance_7, rsqrt_7, hidden_states_37, hidden_states_38, linear_25], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf85, (s50, 896), (896, 1), 0), reinterpret_tensor(arg59_1, (896, 4864), (1, 896), 0), out=buf86)
            del arg59_1
            assert_size_stride(arg60_1, (4864, 896), (896, 1))
            buf87 = buf64; del buf64  # reuse
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf85, (s50, 896), (896, 1), 0), reinterpret_tensor(arg60_1, (896, 4864), (1, 896), 0), out=buf87)
            del arg60_1
            buf88 = reinterpret_tensor(buf86, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [linear_25, silu_3, linear_26, mul_37], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf88, buf87, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg61_1, (896, 4864), (4864, 1))
            buf89 = reinterpret_tensor(buf85, (s50, 896), (896, 1), 0); del buf85  # reuse
            # Topologically Sorted Source Nodes: [linear_25, silu_3, linear_26, mul_37, down_proj_3], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf88, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg61_1, (4864, 896), (1, 4864), 0), out=buf89)
            del arg61_1
            assert_size_stride(arg63_1, (896, ), (1, ))
            assert_size_stride(arg62_1, (), ())
            buf91 = reinterpret_tensor(buf82, (1, s50, 896), (896*s50, 896, 1), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, pow_9, variance_8, rsqrt_8, hidden_states_41, hidden_states_42], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf83, buf89, arg63_1, arg62_1.item(), buf91, s50, 896, stream=raw_stream0)
            del arg62_1
            del arg63_1
            assert_size_stride(arg65_1, (896, ), (1, ))
            assert_size_stride(arg64_1, (896, 896), (896, 1))
            buf92 = buf66; del buf66  # reuse
            # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg65_1, reinterpret_tensor(buf91, (s50, 896), (896, 1), 0), reinterpret_tensor(arg64_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf92)
            del arg64_1
            del arg65_1
            assert_size_stride(arg67_1, (128, ), (1, ))
            assert_size_stride(arg66_1, (128, 896), (896, 1))
            buf93 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg67_1, reinterpret_tensor(buf91, (s50, 896), (896, 1), 0), reinterpret_tensor(arg66_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf93)
            del arg66_1
            del arg67_1
            buf94 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_7, sin_7, linear_29, view_13, key_states_4, mul_42, x2_9, neg_9, x1_9, cat_18, mul_43, k_embed_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf93, arg4_1, arg5_1.item(), buf94, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg69_1, (128, ), (1, ))
            assert_size_stride(arg68_1, (128, 896), (896, 1))
            buf95 = buf93; del buf93  # reuse
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg69_1, reinterpret_tensor(buf91, (s50, 896), (896, 1), 0), reinterpret_tensor(arg68_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf95)
            del arg68_1
            del arg69_1
            buf96 = reinterpret_tensor(buf91, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf91  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_28, view_12, query_states_4, cos_7, mul_40, x2_8, neg_8, x1_8, cat_17, sin_7, mul_41, q_embed_4, getitem_192, hidden_states_43, key_4, linear_30, view_14, value_states_4, getitem_197, hidden_states_44, value_4, attn_output_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf92, arg4_1, arg5_1.item(), buf96, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf97 = reinterpret_tensor(buf92, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_28, view_12, query_states_4, cos_7, mul_40, x2_8, neg_8, x1_8, cat_17, sin_7, mul_41, q_embed_4, getitem_192, hidden_states_43, key_4, linear_30, view_14, value_states_4, getitem_197, hidden_states_44, value_4, attn_output_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf94, buf97, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf98 = reinterpret_tensor(buf60, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_28, view_12, query_states_4, cos_7, mul_40, x2_8, neg_8, x1_8, cat_17, sin_7, mul_41, q_embed_4, getitem_192, hidden_states_43, key_4, linear_30, view_14, value_states_4, getitem_197, hidden_states_44, value_4, attn_output_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf95, buf98, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf99 = buf76; del buf76  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_28, view_12, query_states_4, cos_7, mul_40, x2_8, neg_8, x1_8, cat_17, sin_7, mul_41, q_embed_4, getitem_192, hidden_states_43, key_4, linear_30, view_14, value_states_4, getitem_197, hidden_states_44, value_4, attn_output_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf99, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_28, view_12, query_states_4, cos_7, mul_40, x2_8, neg_8, x1_8, cat_17, sin_7, mul_41, q_embed_4, getitem_192, hidden_states_43, key_4, linear_30, view_14, value_states_4, getitem_197, hidden_states_44, value_4, attn_output_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf100 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf96, buf97, buf98, reinterpret_tensor(buf99, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            buf101 = buf100[0]
            assert_size_stride(buf101, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf101, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf100
            assert_size_stride(arg70_1, (896, 896), (896, 1))
            buf105 = reinterpret_tensor(buf98, (s50, 896), (896, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [transpose_20, reshape_14, attn_output_19], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf101, (s50, 896), (896, 1), 0), reinterpret_tensor(arg70_1, (896, 896), (1, 896), 0), out=buf105)
            del arg70_1
            assert_size_stride(arg72_1, (896, ), (1, ))
            assert_size_stride(arg71_1, (), ())
            buf107 = reinterpret_tensor(buf101, (1, s50, 896), (896*s50, 896, 1), 0); del buf101  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, attn_output_19, hidden_states_45, pow_10, variance_9, rsqrt_9, hidden_states_47, hidden_states_48], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf83, buf89, buf105, arg72_1, arg71_1.item(), buf107, s50, 896, stream=raw_stream0)
            del arg71_1
            del arg72_1
            assert_size_stride(arg73_1, (4864, 896), (896, 1))
            buf108 = reinterpret_tensor(buf88, (s50, 4864), (4864, 1), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf107, (s50, 896), (896, 1), 0), reinterpret_tensor(arg73_1, (896, 4864), (1, 896), 0), out=buf108)
            del arg73_1
            assert_size_stride(arg74_1, (4864, 896), (896, 1))
            buf109 = buf87; del buf87  # reuse
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf107, (s50, 896), (896, 1), 0), reinterpret_tensor(arg74_1, (896, 4864), (1, 896), 0), out=buf109)
            del arg74_1
            buf110 = reinterpret_tensor(buf108, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf108  # reuse
            # Topologically Sorted Source Nodes: [linear_32, silu_4, linear_33, mul_46], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf110, buf109, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg75_1, (896, 4864), (4864, 1))
            buf111 = reinterpret_tensor(buf107, (s50, 896), (896, 1), 0); del buf107  # reuse
            # Topologically Sorted Source Nodes: [linear_32, silu_4, linear_33, mul_46, down_proj_4], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf110, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg75_1, (4864, 896), (1, 4864), 0), out=buf111)
            del arg75_1
            assert_size_stride(arg77_1, (896, ), (1, ))
            assert_size_stride(arg76_1, (), ())
            buf113 = reinterpret_tensor(buf97, (1, s50, 896), (896*s50, 896, 1), 0); del buf97  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, attn_output_19, hidden_states_45, down_proj_4, hidden_states_49, pow_11, variance_10, rsqrt_10, hidden_states_51, hidden_states_52], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf83, buf89, buf105, buf111, arg77_1, arg76_1.item(), buf113, s50, 896, stream=raw_stream0)
            del arg76_1
            del arg77_1
            assert_size_stride(arg79_1, (896, ), (1, ))
            assert_size_stride(arg78_1, (896, 896), (896, 1))
            buf114 = reinterpret_tensor(buf96, (s50, 896), (896, 1), 0); del buf96  # reuse
            # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg79_1, reinterpret_tensor(buf113, (s50, 896), (896, 1), 0), reinterpret_tensor(arg78_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf114)
            del arg78_1
            del arg79_1
            assert_size_stride(arg81_1, (128, ), (1, ))
            assert_size_stride(arg80_1, (128, 896), (896, 1))
            buf115 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg81_1, reinterpret_tensor(buf113, (s50, 896), (896, 1), 0), reinterpret_tensor(arg80_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf115)
            del arg80_1
            del arg81_1
            buf116 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_8, sin_8, linear_36, view_16, key_states_5, mul_51, x2_11, neg_11, x1_11, cat_22, mul_52, k_embed_5], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf115, arg4_1, arg5_1.item(), buf116, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg83_1, (128, ), (1, ))
            assert_size_stride(arg82_1, (128, 896), (896, 1))
            buf117 = buf115; del buf115  # reuse
            # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg83_1, reinterpret_tensor(buf113, (s50, 896), (896, 1), 0), reinterpret_tensor(arg82_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf117)
            del arg82_1
            del arg83_1
            buf118 = reinterpret_tensor(buf113, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf113  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_35, view_15, query_states_5, cos_8, mul_49, x2_10, neg_10, x1_10, cat_21, sin_8, mul_50, q_embed_5, getitem_229, hidden_states_53, key_5, linear_37, view_17, value_states_5, getitem_234, hidden_states_54, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf114, arg4_1, arg5_1.item(), buf118, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf119 = reinterpret_tensor(buf114, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf114  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_35, view_15, query_states_5, cos_8, mul_49, x2_10, neg_10, x1_10, cat_21, sin_8, mul_50, q_embed_5, getitem_229, hidden_states_53, key_5, linear_37, view_17, value_states_5, getitem_234, hidden_states_54, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf116, buf119, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf120 = reinterpret_tensor(buf44, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_35, view_15, query_states_5, cos_8, mul_49, x2_10, neg_10, x1_10, cat_21, sin_8, mul_50, q_embed_5, getitem_229, hidden_states_53, key_5, linear_37, view_17, value_states_5, getitem_234, hidden_states_54, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf117, buf120, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf121 = buf99; del buf99  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_35, view_15, query_states_5, cos_8, mul_49, x2_10, neg_10, x1_10, cat_21, sin_8, mul_50, q_embed_5, getitem_229, hidden_states_53, key_5, linear_37, view_17, value_states_5, getitem_234, hidden_states_54, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf121, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_35, view_15, query_states_5, cos_8, mul_49, x2_10, neg_10, x1_10, cat_21, sin_8, mul_50, q_embed_5, getitem_229, hidden_states_53, key_5, linear_37, view_17, value_states_5, getitem_234, hidden_states_54, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf122 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf118, buf119, buf120, reinterpret_tensor(buf121, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf118
            del buf119
            buf123 = buf122[0]
            assert_size_stride(buf123, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf123, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf122
            assert_size_stride(arg84_1, (896, 896), (896, 1))
            buf127 = reinterpret_tensor(buf120, (s50, 896), (896, 1), 0); del buf120  # reuse
            # Topologically Sorted Source Nodes: [transpose_24, reshape_17, attn_output_23], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf123, (s50, 896), (896, 1), 0), reinterpret_tensor(arg84_1, (896, 896), (1, 896), 0), out=buf127)
            del arg84_1
            assert_size_stride(arg86_1, (896, ), (1, ))
            assert_size_stride(arg85_1, (), ())
            buf128 = buf83; del buf83  # reuse
            buf130 = reinterpret_tensor(buf123, (1, s50, 896), (896*s50, 896, 1), 0); del buf123  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, attn_output_19, hidden_states_45, down_proj_4, hidden_states_49, attn_output_23, hidden_states_55, pow_12, variance_11, rsqrt_11, hidden_states_57, hidden_states_58], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf128, buf89, buf105, buf111, buf127, arg86_1, arg85_1.item(), buf130, s50, 896, stream=raw_stream0)
            del arg85_1
            del arg86_1
            assert_size_stride(arg87_1, (4864, 896), (896, 1))
            buf131 = reinterpret_tensor(buf110, (s50, 4864), (4864, 1), 0); del buf110  # reuse
            # Topologically Sorted Source Nodes: [pow_12, variance_11, rsqrt_11, hidden_states_57, hidden_states_58, linear_39], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf130, (s50, 896), (896, 1), 0), reinterpret_tensor(arg87_1, (896, 4864), (1, 896), 0), out=buf131)
            del arg87_1
            assert_size_stride(arg88_1, (4864, 896), (896, 1))
            buf132 = buf109; del buf109  # reuse
            # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf130, (s50, 896), (896, 1), 0), reinterpret_tensor(arg88_1, (896, 4864), (1, 896), 0), out=buf132)
            del arg88_1
            buf133 = reinterpret_tensor(buf131, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf131  # reuse
            # Topologically Sorted Source Nodes: [linear_39, silu_5, linear_40, mul_55], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf133, buf132, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg89_1, (896, 4864), (4864, 1))
            buf134 = reinterpret_tensor(buf130, (s50, 896), (896, 1), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [linear_39, silu_5, linear_40, mul_55, down_proj_5], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf133, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg89_1, (4864, 896), (1, 4864), 0), out=buf134)
            del arg89_1
            assert_size_stride(arg91_1, (896, ), (1, ))
            assert_size_stride(arg90_1, (), ())
            buf136 = reinterpret_tensor(buf89, (1, s50, 896), (896*s50, 896, 1), 0); del buf89  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, pow_13, variance_12, rsqrt_12, hidden_states_61, hidden_states_62], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf128, buf134, arg91_1, arg90_1.item(), buf136, s50, 896, stream=raw_stream0)
            del arg90_1
            del arg91_1
            assert_size_stride(arg93_1, (896, ), (1, ))
            assert_size_stride(arg92_1, (896, 896), (896, 1))
            buf137 = buf127; del buf127  # reuse
            # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg93_1, reinterpret_tensor(buf136, (s50, 896), (896, 1), 0), reinterpret_tensor(arg92_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf137)
            del arg92_1
            del arg93_1
            assert_size_stride(arg95_1, (128, ), (1, ))
            assert_size_stride(arg94_1, (128, 896), (896, 1))
            buf138 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg95_1, reinterpret_tensor(buf136, (s50, 896), (896, 1), 0), reinterpret_tensor(arg94_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf138)
            del arg94_1
            del arg95_1
            buf139 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_9, sin_9, linear_43, view_19, key_states_6, mul_60, x2_13, neg_13, x1_13, cat_26, mul_61, k_embed_6], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf138, arg4_1, arg5_1.item(), buf139, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg97_1, (128, ), (1, ))
            assert_size_stride(arg96_1, (128, 896), (896, 1))
            buf140 = buf138; del buf138  # reuse
            # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg97_1, reinterpret_tensor(buf136, (s50, 896), (896, 1), 0), reinterpret_tensor(arg96_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf140)
            del arg96_1
            del arg97_1
            buf141 = reinterpret_tensor(buf136, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_42, view_18, query_states_6, cos_9, mul_58, x2_12, neg_12, x1_12, cat_25, sin_9, mul_59, q_embed_6, getitem_266, hidden_states_63, key_6, linear_44, view_20, value_states_6, getitem_271, hidden_states_64, value_6, attn_output_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf137, arg4_1, arg5_1.item(), buf141, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf142 = reinterpret_tensor(buf137, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf137  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_42, view_18, query_states_6, cos_9, mul_58, x2_12, neg_12, x1_12, cat_25, sin_9, mul_59, q_embed_6, getitem_266, hidden_states_63, key_6, linear_44, view_20, value_states_6, getitem_271, hidden_states_64, value_6, attn_output_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf139, buf142, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf143 = reinterpret_tensor(buf111, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf111  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_42, view_18, query_states_6, cos_9, mul_58, x2_12, neg_12, x1_12, cat_25, sin_9, mul_59, q_embed_6, getitem_266, hidden_states_63, key_6, linear_44, view_20, value_states_6, getitem_271, hidden_states_64, value_6, attn_output_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf140, buf143, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf144 = buf121; del buf121  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_42, view_18, query_states_6, cos_9, mul_58, x2_12, neg_12, x1_12, cat_25, sin_9, mul_59, q_embed_6, getitem_266, hidden_states_63, key_6, linear_44, view_20, value_states_6, getitem_271, hidden_states_64, value_6, attn_output_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf144, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_42, view_18, query_states_6, cos_9, mul_58, x2_12, neg_12, x1_12, cat_25, sin_9, mul_59, q_embed_6, getitem_266, hidden_states_63, key_6, linear_44, view_20, value_states_6, getitem_271, hidden_states_64, value_6, attn_output_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf145 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf141, buf142, buf143, reinterpret_tensor(buf144, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            buf146 = buf145[0]
            assert_size_stride(buf146, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf146, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf145
            assert_size_stride(arg98_1, (896, 896), (896, 1))
            buf150 = reinterpret_tensor(buf143, (s50, 896), (896, 1), 0); del buf143  # reuse
            # Topologically Sorted Source Nodes: [transpose_28, reshape_20, attn_output_27], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf146, (s50, 896), (896, 1), 0), reinterpret_tensor(arg98_1, (896, 896), (1, 896), 0), out=buf150)
            del arg98_1
            assert_size_stride(arg100_1, (896, ), (1, ))
            assert_size_stride(arg99_1, (), ())
            buf152 = reinterpret_tensor(buf146, (1, s50, 896), (896*s50, 896, 1), 0); del buf146  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, attn_output_27, hidden_states_65, pow_14, variance_13, rsqrt_13, hidden_states_67, hidden_states_68], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf128, buf134, buf150, arg100_1, arg99_1.item(), buf152, s50, 896, stream=raw_stream0)
            del arg100_1
            del arg99_1
            assert_size_stride(arg101_1, (4864, 896), (896, 1))
            buf153 = reinterpret_tensor(buf133, (s50, 4864), (4864, 1), 0); del buf133  # reuse
            # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf152, (s50, 896), (896, 1), 0), reinterpret_tensor(arg101_1, (896, 4864), (1, 896), 0), out=buf153)
            del arg101_1
            assert_size_stride(arg102_1, (4864, 896), (896, 1))
            buf154 = buf132; del buf132  # reuse
            # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf152, (s50, 896), (896, 1), 0), reinterpret_tensor(arg102_1, (896, 4864), (1, 896), 0), out=buf154)
            del arg102_1
            buf155 = reinterpret_tensor(buf153, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf153  # reuse
            # Topologically Sorted Source Nodes: [linear_46, silu_6, linear_47, mul_64], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf155, buf154, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg103_1, (896, 4864), (4864, 1))
            buf156 = reinterpret_tensor(buf152, (s50, 896), (896, 1), 0); del buf152  # reuse
            # Topologically Sorted Source Nodes: [linear_46, silu_6, linear_47, mul_64, down_proj_6], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf155, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg103_1, (4864, 896), (1, 4864), 0), out=buf156)
            del arg103_1
            assert_size_stride(arg105_1, (896, ), (1, ))
            assert_size_stride(arg104_1, (), ())
            buf158 = reinterpret_tensor(buf142, (1, s50, 896), (896*s50, 896, 1), 0); del buf142  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, attn_output_27, hidden_states_65, down_proj_6, hidden_states_69, pow_15, variance_14, rsqrt_14, hidden_states_71, hidden_states_72], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf128, buf134, buf150, buf156, arg105_1, arg104_1.item(), buf158, s50, 896, stream=raw_stream0)
            del arg104_1
            del arg105_1
            assert_size_stride(arg107_1, (896, ), (1, ))
            assert_size_stride(arg106_1, (896, 896), (896, 1))
            buf159 = reinterpret_tensor(buf141, (s50, 896), (896, 1), 0); del buf141  # reuse
            # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg107_1, reinterpret_tensor(buf158, (s50, 896), (896, 1), 0), reinterpret_tensor(arg106_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf159)
            del arg106_1
            del arg107_1
            assert_size_stride(arg109_1, (128, ), (1, ))
            assert_size_stride(arg108_1, (128, 896), (896, 1))
            buf160 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg109_1, reinterpret_tensor(buf158, (s50, 896), (896, 1), 0), reinterpret_tensor(arg108_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf160)
            del arg108_1
            del arg109_1
            buf161 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_10, sin_10, linear_50, view_22, key_states_7, mul_69, x2_15, neg_15, x1_15, cat_30, mul_70, k_embed_7], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf160, arg4_1, arg5_1.item(), buf161, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg111_1, (128, ), (1, ))
            assert_size_stride(arg110_1, (128, 896), (896, 1))
            buf162 = buf160; del buf160  # reuse
            # Topologically Sorted Source Nodes: [linear_51], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg111_1, reinterpret_tensor(buf158, (s50, 896), (896, 1), 0), reinterpret_tensor(arg110_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf162)
            del arg110_1
            del arg111_1
            buf163 = reinterpret_tensor(buf158, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_49, view_21, query_states_7, cos_10, mul_67, x2_14, neg_14, x1_14, cat_29, sin_10, mul_68, q_embed_7, getitem_303, hidden_states_73, key_7, linear_51, view_23, value_states_7, getitem_308, hidden_states_74, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf159, arg4_1, arg5_1.item(), buf163, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf164 = reinterpret_tensor(buf159, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf159  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_49, view_21, query_states_7, cos_10, mul_67, x2_14, neg_14, x1_14, cat_29, sin_10, mul_68, q_embed_7, getitem_303, hidden_states_73, key_7, linear_51, view_23, value_states_7, getitem_308, hidden_states_74, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf161, buf164, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf165 = reinterpret_tensor(buf105, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf105  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_49, view_21, query_states_7, cos_10, mul_67, x2_14, neg_14, x1_14, cat_29, sin_10, mul_68, q_embed_7, getitem_303, hidden_states_73, key_7, linear_51, view_23, value_states_7, getitem_308, hidden_states_74, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf162, buf165, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf166 = buf144; del buf144  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_49, view_21, query_states_7, cos_10, mul_67, x2_14, neg_14, x1_14, cat_29, sin_10, mul_68, q_embed_7, getitem_303, hidden_states_73, key_7, linear_51, view_23, value_states_7, getitem_308, hidden_states_74, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf166, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_49, view_21, query_states_7, cos_10, mul_67, x2_14, neg_14, x1_14, cat_29, sin_10, mul_68, q_embed_7, getitem_303, hidden_states_73, key_7, linear_51, view_23, value_states_7, getitem_308, hidden_states_74, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf167 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf163, buf164, buf165, reinterpret_tensor(buf166, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf163
            del buf164
            buf168 = buf167[0]
            assert_size_stride(buf168, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf168, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf167
            assert_size_stride(arg112_1, (896, 896), (896, 1))
            buf172 = reinterpret_tensor(buf165, (s50, 896), (896, 1), 0); del buf165  # reuse
            # Topologically Sorted Source Nodes: [transpose_32, reshape_23, attn_output_31], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf168, (s50, 896), (896, 1), 0), reinterpret_tensor(arg112_1, (896, 896), (1, 896), 0), out=buf172)
            del arg112_1
            assert_size_stride(arg114_1, (896, ), (1, ))
            assert_size_stride(arg113_1, (), ())
            buf173 = buf128; del buf128  # reuse
            buf175 = reinterpret_tensor(buf168, (1, s50, 896), (896*s50, 896, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, attn_output_27, hidden_states_65, down_proj_6, hidden_states_69, attn_output_31, hidden_states_75, pow_16, variance_15, rsqrt_15, hidden_states_77, hidden_states_78], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf173, buf134, buf150, buf156, buf172, arg114_1, arg113_1.item(), buf175, s50, 896, stream=raw_stream0)
            del arg113_1
            del arg114_1
            assert_size_stride(arg115_1, (4864, 896), (896, 1))
            buf176 = reinterpret_tensor(buf155, (s50, 4864), (4864, 1), 0); del buf155  # reuse
            # Topologically Sorted Source Nodes: [pow_16, variance_15, rsqrt_15, hidden_states_77, hidden_states_78, linear_53], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf175, (s50, 896), (896, 1), 0), reinterpret_tensor(arg115_1, (896, 4864), (1, 896), 0), out=buf176)
            del arg115_1
            assert_size_stride(arg116_1, (4864, 896), (896, 1))
            buf177 = buf154; del buf154  # reuse
            # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf175, (s50, 896), (896, 1), 0), reinterpret_tensor(arg116_1, (896, 4864), (1, 896), 0), out=buf177)
            del arg116_1
            buf178 = reinterpret_tensor(buf176, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf176  # reuse
            # Topologically Sorted Source Nodes: [linear_53, silu_7, linear_54, mul_73], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf178, buf177, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg117_1, (896, 4864), (4864, 1))
            buf179 = reinterpret_tensor(buf175, (s50, 896), (896, 1), 0); del buf175  # reuse
            # Topologically Sorted Source Nodes: [linear_53, silu_7, linear_54, mul_73, down_proj_7], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf178, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg117_1, (4864, 896), (1, 4864), 0), out=buf179)
            del arg117_1
            assert_size_stride(arg119_1, (896, ), (1, ))
            assert_size_stride(arg118_1, (), ())
            buf181 = reinterpret_tensor(buf172, (1, s50, 896), (896*s50, 896, 1), 0); del buf172  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, pow_17, variance_16, rsqrt_16, hidden_states_81, hidden_states_82], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf173, buf179, arg119_1, arg118_1.item(), buf181, s50, 896, stream=raw_stream0)
            del arg118_1
            del arg119_1
            assert_size_stride(arg121_1, (896, ), (1, ))
            assert_size_stride(arg120_1, (896, 896), (896, 1))
            buf182 = buf156; del buf156  # reuse
            # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg121_1, reinterpret_tensor(buf181, (s50, 896), (896, 1), 0), reinterpret_tensor(arg120_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf182)
            del arg120_1
            del arg121_1
            assert_size_stride(arg123_1, (128, ), (1, ))
            assert_size_stride(arg122_1, (128, 896), (896, 1))
            buf183 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg123_1, reinterpret_tensor(buf181, (s50, 896), (896, 1), 0), reinterpret_tensor(arg122_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf183)
            del arg122_1
            del arg123_1
            buf184 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_11, sin_11, linear_57, view_25, key_states_8, mul_78, x2_17, neg_17, x1_17, cat_34, mul_79, k_embed_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf183, arg4_1, arg5_1.item(), buf184, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg125_1, (128, ), (1, ))
            assert_size_stride(arg124_1, (128, 896), (896, 1))
            buf185 = buf183; del buf183  # reuse
            # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg125_1, reinterpret_tensor(buf181, (s50, 896), (896, 1), 0), reinterpret_tensor(arg124_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf185)
            del arg124_1
            del arg125_1
            buf186 = reinterpret_tensor(buf181, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf181  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_56, view_24, query_states_8, cos_11, mul_76, x2_16, neg_16, x1_16, cat_33, sin_11, mul_77, q_embed_8, getitem_340, hidden_states_83, key_8, linear_58, view_26, value_states_8, getitem_345, hidden_states_84, value_8, attn_output_32], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf182, arg4_1, arg5_1.item(), buf186, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf187 = reinterpret_tensor(buf182, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf182  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_56, view_24, query_states_8, cos_11, mul_76, x2_16, neg_16, x1_16, cat_33, sin_11, mul_77, q_embed_8, getitem_340, hidden_states_83, key_8, linear_58, view_26, value_states_8, getitem_345, hidden_states_84, value_8, attn_output_32], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf184, buf187, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf188 = reinterpret_tensor(buf150, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf150  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_56, view_24, query_states_8, cos_11, mul_76, x2_16, neg_16, x1_16, cat_33, sin_11, mul_77, q_embed_8, getitem_340, hidden_states_83, key_8, linear_58, view_26, value_states_8, getitem_345, hidden_states_84, value_8, attn_output_32], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf185, buf188, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf189 = buf166; del buf166  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_56, view_24, query_states_8, cos_11, mul_76, x2_16, neg_16, x1_16, cat_33, sin_11, mul_77, q_embed_8, getitem_340, hidden_states_83, key_8, linear_58, view_26, value_states_8, getitem_345, hidden_states_84, value_8, attn_output_32], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf189, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_56, view_24, query_states_8, cos_11, mul_76, x2_16, neg_16, x1_16, cat_33, sin_11, mul_77, q_embed_8, getitem_340, hidden_states_83, key_8, linear_58, view_26, value_states_8, getitem_345, hidden_states_84, value_8, attn_output_32], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf190 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf186, buf187, buf188, reinterpret_tensor(buf189, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf189
            buf191 = buf190[0]
            assert_size_stride(buf191, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf191, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf190
            assert_size_stride(arg126_1, (896, 896), (896, 1))
            buf195 = reinterpret_tensor(buf188, (s50, 896), (896, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [transpose_36, reshape_26, attn_output_35], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf191, (s50, 896), (896, 1), 0), reinterpret_tensor(arg126_1, (896, 896), (1, 896), 0), out=buf195)
            del arg126_1
            assert_size_stride(arg128_1, (896, ), (1, ))
            assert_size_stride(arg127_1, (), ())
            buf197 = reinterpret_tensor(buf191, (1, s50, 896), (896*s50, 896, 1), 0); del buf191  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, pow_18, variance_17, rsqrt_17, hidden_states_87, hidden_states_88], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf173, buf179, buf195, arg128_1, arg127_1.item(), buf197, s50, 896, stream=raw_stream0)
            del arg127_1
            del arg128_1
            assert_size_stride(arg129_1, (4864, 896), (896, 1))
            buf198 = reinterpret_tensor(buf178, (s50, 4864), (4864, 1), 0); del buf178  # reuse
            # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf197, (s50, 896), (896, 1), 0), reinterpret_tensor(arg129_1, (896, 4864), (1, 896), 0), out=buf198)
            del arg129_1
            assert_size_stride(arg130_1, (4864, 896), (896, 1))
            buf199 = buf177; del buf177  # reuse
            # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf197, (s50, 896), (896, 1), 0), reinterpret_tensor(arg130_1, (896, 4864), (1, 896), 0), out=buf199)
            del arg130_1
            buf200 = reinterpret_tensor(buf198, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf198  # reuse
            # Topologically Sorted Source Nodes: [linear_60, silu_8, linear_61, mul_82], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf200, buf199, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg131_1, (896, 4864), (4864, 1))
            buf201 = reinterpret_tensor(buf197, (s50, 896), (896, 1), 0); del buf197  # reuse
            # Topologically Sorted Source Nodes: [linear_60, silu_8, linear_61, mul_82, down_proj_8], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf200, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg131_1, (4864, 896), (1, 4864), 0), out=buf201)
            del arg131_1
            assert_size_stride(arg133_1, (896, ), (1, ))
            assert_size_stride(arg132_1, (), ())
            buf203 = reinterpret_tensor(buf187, (1, s50, 896), (896*s50, 896, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, down_proj_8, hidden_states_89, pow_19, variance_18, rsqrt_18, hidden_states_91, hidden_states_92], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf173, buf179, buf195, buf201, arg133_1, arg132_1.item(), buf203, s50, 896, stream=raw_stream0)
            del arg132_1
            del arg133_1
            assert_size_stride(arg135_1, (896, ), (1, ))
            assert_size_stride(arg134_1, (896, 896), (896, 1))
            buf204 = reinterpret_tensor(buf186, (s50, 896), (896, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [linear_63], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg135_1, reinterpret_tensor(buf203, (s50, 896), (896, 1), 0), reinterpret_tensor(arg134_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf204)
            del arg134_1
            del arg135_1
            assert_size_stride(arg137_1, (128, ), (1, ))
            assert_size_stride(arg136_1, (128, 896), (896, 1))
            buf205 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_64], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg137_1, reinterpret_tensor(buf203, (s50, 896), (896, 1), 0), reinterpret_tensor(arg136_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf205)
            del arg136_1
            del arg137_1
            buf206 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_12, sin_12, linear_64, view_28, key_states_9, mul_87, x2_19, neg_19, x1_19, cat_38, mul_88, k_embed_9], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf205, arg4_1, arg5_1.item(), buf206, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg139_1, (128, ), (1, ))
            assert_size_stride(arg138_1, (128, 896), (896, 1))
            buf207 = buf205; del buf205  # reuse
            # Topologically Sorted Source Nodes: [linear_65], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg139_1, reinterpret_tensor(buf203, (s50, 896), (896, 1), 0), reinterpret_tensor(arg138_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf207)
            del arg138_1
            del arg139_1
            buf208 = reinterpret_tensor(buf203, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf203  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_63, view_27, query_states_9, cos_12, mul_85, x2_18, neg_18, x1_18, cat_37, sin_12, mul_86, q_embed_9, getitem_377, hidden_states_93, key_9, linear_65, view_29, value_states_9, getitem_382, hidden_states_94, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf204, arg4_1, arg5_1.item(), buf208, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf209 = reinterpret_tensor(buf204, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf204  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_63, view_27, query_states_9, cos_12, mul_85, x2_18, neg_18, x1_18, cat_37, sin_12, mul_86, q_embed_9, getitem_377, hidden_states_93, key_9, linear_65, view_29, value_states_9, getitem_382, hidden_states_94, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf206, buf209, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf210 = reinterpret_tensor(buf134, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf134  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_63, view_27, query_states_9, cos_12, mul_85, x2_18, neg_18, x1_18, cat_37, sin_12, mul_86, q_embed_9, getitem_377, hidden_states_93, key_9, linear_65, view_29, value_states_9, getitem_382, hidden_states_94, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf207, buf210, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf211 = empty_strided_cuda((1, 1, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_63, view_27, query_states_9, cos_12, mul_85, x2_18, neg_18, x1_18, cat_37, sin_12, mul_86, q_embed_9, getitem_377, hidden_states_93, key_9, linear_65, view_29, value_states_9, getitem_382, hidden_states_94, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf211, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_63, view_27, query_states_9, cos_12, mul_85, x2_18, neg_18, x1_18, cat_37, sin_12, mul_86, q_embed_9, getitem_377, hidden_states_93, key_9, linear_65, view_29, value_states_9, getitem_382, hidden_states_94, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf212 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf208, buf209, buf210, reinterpret_tensor(buf211, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf208
            del buf209
            buf213 = buf212[0]
            assert_size_stride(buf213, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf213, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf212
            assert_size_stride(arg140_1, (896, 896), (896, 1))
            buf217 = reinterpret_tensor(buf210, (s50, 896), (896, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [transpose_40, reshape_29, attn_output_39], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf213, (s50, 896), (896, 1), 0), reinterpret_tensor(arg140_1, (896, 896), (1, 896), 0), out=buf217)
            del arg140_1
            assert_size_stride(arg142_1, (896, ), (1, ))
            assert_size_stride(arg141_1, (), ())
            buf218 = buf173; del buf173  # reuse
            buf220 = reinterpret_tensor(buf213, (1, s50, 896), (896*s50, 896, 1), 0); del buf213  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, down_proj_8, hidden_states_89, attn_output_39, hidden_states_95, pow_20, variance_19, rsqrt_19, hidden_states_97, hidden_states_98], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf218, buf179, buf195, buf201, buf217, arg142_1, arg141_1.item(), buf220, s50, 896, stream=raw_stream0)
            del arg141_1
            del arg142_1
            del buf179
            assert_size_stride(arg143_1, (4864, 896), (896, 1))
            buf221 = reinterpret_tensor(buf200, (s50, 4864), (4864, 1), 0); del buf200  # reuse
            # Topologically Sorted Source Nodes: [pow_20, variance_19, rsqrt_19, hidden_states_97, hidden_states_98, linear_67], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf220, (s50, 896), (896, 1), 0), reinterpret_tensor(arg143_1, (896, 4864), (1, 896), 0), out=buf221)
            del arg143_1
            assert_size_stride(arg144_1, (4864, 896), (896, 1))
            buf222 = buf199; del buf199  # reuse
            # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf220, (s50, 896), (896, 1), 0), reinterpret_tensor(arg144_1, (896, 4864), (1, 896), 0), out=buf222)
            del arg144_1
            buf223 = reinterpret_tensor(buf221, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf221  # reuse
            # Topologically Sorted Source Nodes: [linear_67, silu_9, linear_68, mul_91], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf223, buf222, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg145_1, (896, 4864), (4864, 1))
            buf224 = reinterpret_tensor(buf220, (s50, 896), (896, 1), 0); del buf220  # reuse
            # Topologically Sorted Source Nodes: [linear_67, silu_9, linear_68, mul_91, down_proj_9], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf223, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg145_1, (4864, 896), (1, 4864), 0), out=buf224)
            del arg145_1
            assert_size_stride(arg147_1, (896, ), (1, ))
            assert_size_stride(arg146_1, (), ())
            buf226 = reinterpret_tensor(buf217, (1, s50, 896), (896*s50, 896, 1), 0); del buf217  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, pow_21, variance_20, rsqrt_20, hidden_states_101, hidden_states_102], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf218, buf224, arg147_1, arg146_1.item(), buf226, s50, 896, stream=raw_stream0)
            del arg146_1
            del arg147_1
            assert_size_stride(arg149_1, (896, ), (1, ))
            assert_size_stride(arg148_1, (896, 896), (896, 1))
            buf227 = buf201; del buf201  # reuse
            # Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg149_1, reinterpret_tensor(buf226, (s50, 896), (896, 1), 0), reinterpret_tensor(arg148_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf227)
            del arg148_1
            del arg149_1
            assert_size_stride(arg151_1, (128, ), (1, ))
            assert_size_stride(arg150_1, (128, 896), (896, 1))
            buf228 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_71], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg151_1, reinterpret_tensor(buf226, (s50, 896), (896, 1), 0), reinterpret_tensor(arg150_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf228)
            del arg150_1
            del arg151_1
            buf229 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_13, sin_13, linear_71, view_31, key_states_10, mul_96, x2_21, neg_21, x1_21, cat_42, mul_97, k_embed_10], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf228, arg4_1, arg5_1.item(), buf229, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg153_1, (128, ), (1, ))
            assert_size_stride(arg152_1, (128, 896), (896, 1))
            buf230 = buf228; del buf228  # reuse
            # Topologically Sorted Source Nodes: [linear_72], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg153_1, reinterpret_tensor(buf226, (s50, 896), (896, 1), 0), reinterpret_tensor(arg152_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf230)
            del arg152_1
            del arg153_1
            buf231 = reinterpret_tensor(buf226, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf226  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_70, view_30, query_states_10, cos_13, mul_94, x2_20, neg_20, x1_20, cat_41, sin_13, mul_95, q_embed_10, getitem_414, hidden_states_103, key_10, linear_72, view_32, value_states_10, getitem_419, hidden_states_104, value_10, attn_output_40], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf227, arg4_1, arg5_1.item(), buf231, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf232 = reinterpret_tensor(buf227, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf227  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_70, view_30, query_states_10, cos_13, mul_94, x2_20, neg_20, x1_20, cat_41, sin_13, mul_95, q_embed_10, getitem_414, hidden_states_103, key_10, linear_72, view_32, value_states_10, getitem_419, hidden_states_104, value_10, attn_output_40], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf229, buf232, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf233 = reinterpret_tensor(buf195, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf195  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_70, view_30, query_states_10, cos_13, mul_94, x2_20, neg_20, x1_20, cat_41, sin_13, mul_95, q_embed_10, getitem_414, hidden_states_103, key_10, linear_72, view_32, value_states_10, getitem_419, hidden_states_104, value_10, attn_output_40], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf230, buf233, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf234 = buf211; del buf211  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_70, view_30, query_states_10, cos_13, mul_94, x2_20, neg_20, x1_20, cat_41, sin_13, mul_95, q_embed_10, getitem_414, hidden_states_103, key_10, linear_72, view_32, value_states_10, getitem_419, hidden_states_104, value_10, attn_output_40], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf234, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_70, view_30, query_states_10, cos_13, mul_94, x2_20, neg_20, x1_20, cat_41, sin_13, mul_95, q_embed_10, getitem_414, hidden_states_103, key_10, linear_72, view_32, value_states_10, getitem_419, hidden_states_104, value_10, attn_output_40], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf235 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf231, buf232, buf233, reinterpret_tensor(buf234, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            buf236 = buf235[0]
            assert_size_stride(buf236, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf236, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf235
            assert_size_stride(arg154_1, (896, 896), (896, 1))
            buf240 = reinterpret_tensor(buf233, (s50, 896), (896, 1), 0); del buf233  # reuse
            # Topologically Sorted Source Nodes: [transpose_44, reshape_32, attn_output_43], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf236, (s50, 896), (896, 1), 0), reinterpret_tensor(arg154_1, (896, 896), (1, 896), 0), out=buf240)
            del arg154_1
            assert_size_stride(arg156_1, (896, ), (1, ))
            assert_size_stride(arg155_1, (), ())
            buf242 = reinterpret_tensor(buf236, (1, s50, 896), (896*s50, 896, 1), 0); del buf236  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, pow_22, variance_21, rsqrt_21, hidden_states_107, hidden_states_108], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf218, buf224, buf240, arg156_1, arg155_1.item(), buf242, s50, 896, stream=raw_stream0)
            del arg155_1
            del arg156_1
            assert_size_stride(arg157_1, (4864, 896), (896, 1))
            buf243 = reinterpret_tensor(buf223, (s50, 4864), (4864, 1), 0); del buf223  # reuse
            # Topologically Sorted Source Nodes: [linear_74], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf242, (s50, 896), (896, 1), 0), reinterpret_tensor(arg157_1, (896, 4864), (1, 896), 0), out=buf243)
            del arg157_1
            assert_size_stride(arg158_1, (4864, 896), (896, 1))
            buf244 = buf222; del buf222  # reuse
            # Topologically Sorted Source Nodes: [linear_75], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf242, (s50, 896), (896, 1), 0), reinterpret_tensor(arg158_1, (896, 4864), (1, 896), 0), out=buf244)
            del arg158_1
            buf245 = reinterpret_tensor(buf243, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf243  # reuse
            # Topologically Sorted Source Nodes: [linear_74, silu_10, linear_75, mul_100], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf245, buf244, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg159_1, (896, 4864), (4864, 1))
            buf246 = reinterpret_tensor(buf242, (s50, 896), (896, 1), 0); del buf242  # reuse
            # Topologically Sorted Source Nodes: [linear_74, silu_10, linear_75, mul_100, down_proj_10], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf245, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg159_1, (4864, 896), (1, 4864), 0), out=buf246)
            del arg159_1
            assert_size_stride(arg161_1, (896, ), (1, ))
            assert_size_stride(arg160_1, (), ())
            buf248 = reinterpret_tensor(buf232, (1, s50, 896), (896*s50, 896, 1), 0); del buf232  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, down_proj_10, hidden_states_109, pow_23, variance_22, rsqrt_22, hidden_states_111, hidden_states_112], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf218, buf224, buf240, buf246, arg161_1, arg160_1.item(), buf248, s50, 896, stream=raw_stream0)
            del arg160_1
            del arg161_1
            assert_size_stride(arg163_1, (896, ), (1, ))
            assert_size_stride(arg162_1, (896, 896), (896, 1))
            buf249 = reinterpret_tensor(buf231, (s50, 896), (896, 1), 0); del buf231  # reuse
            # Topologically Sorted Source Nodes: [linear_77], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg163_1, reinterpret_tensor(buf248, (s50, 896), (896, 1), 0), reinterpret_tensor(arg162_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf249)
            del arg162_1
            del arg163_1
            assert_size_stride(arg165_1, (128, ), (1, ))
            assert_size_stride(arg164_1, (128, 896), (896, 1))
            buf250 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_78], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg165_1, reinterpret_tensor(buf248, (s50, 896), (896, 1), 0), reinterpret_tensor(arg164_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf250)
            del arg164_1
            del arg165_1
            buf251 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_14, sin_14, linear_78, view_34, key_states_11, mul_105, x2_23, neg_23, x1_23, cat_46, mul_106, k_embed_11], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf250, arg4_1, arg5_1.item(), buf251, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg167_1, (128, ), (1, ))
            assert_size_stride(arg166_1, (128, 896), (896, 1))
            buf252 = buf250; del buf250  # reuse
            # Topologically Sorted Source Nodes: [linear_79], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg167_1, reinterpret_tensor(buf248, (s50, 896), (896, 1), 0), reinterpret_tensor(arg166_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf252)
            del arg166_1
            del arg167_1
            buf253 = reinterpret_tensor(buf248, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf248  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_77, view_33, query_states_11, cos_14, mul_103, x2_22, neg_22, x1_22, cat_45, sin_14, mul_104, q_embed_11, getitem_451, hidden_states_113, key_11, linear_79, view_35, value_states_11, getitem_456, hidden_states_114, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf249, arg4_1, arg5_1.item(), buf253, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf254 = reinterpret_tensor(buf249, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf249  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_77, view_33, query_states_11, cos_14, mul_103, x2_22, neg_22, x1_22, cat_45, sin_14, mul_104, q_embed_11, getitem_451, hidden_states_113, key_11, linear_79, view_35, value_states_11, getitem_456, hidden_states_114, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf251, buf254, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf255 = empty_strided_cuda((1, 14, s50, 64), (896*s50, 64*s50, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_77, view_33, query_states_11, cos_14, mul_103, x2_22, neg_22, x1_22, cat_45, sin_14, mul_104, q_embed_11, getitem_451, hidden_states_113, key_11, linear_79, view_35, value_states_11, getitem_456, hidden_states_114, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf252, buf255, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf256 = buf234; del buf234  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_77, view_33, query_states_11, cos_14, mul_103, x2_22, neg_22, x1_22, cat_45, sin_14, mul_104, q_embed_11, getitem_451, hidden_states_113, key_11, linear_79, view_35, value_states_11, getitem_456, hidden_states_114, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf256, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_77, view_33, query_states_11, cos_14, mul_103, x2_22, neg_22, x1_22, cat_45, sin_14, mul_104, q_embed_11, getitem_451, hidden_states_113, key_11, linear_79, view_35, value_states_11, getitem_456, hidden_states_114, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf257 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf253, buf254, buf255, reinterpret_tensor(buf256, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf253
            del buf254
            buf258 = buf257[0]
            assert_size_stride(buf258, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf258, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf257
            assert_size_stride(arg168_1, (896, 896), (896, 1))
            buf262 = reinterpret_tensor(buf255, (s50, 896), (896, 1), 0); del buf255  # reuse
            # Topologically Sorted Source Nodes: [transpose_48, reshape_35, attn_output_47], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf258, (s50, 896), (896, 1), 0), reinterpret_tensor(arg168_1, (896, 896), (1, 896), 0), out=buf262)
            del arg168_1
            assert_size_stride(arg170_1, (896, ), (1, ))
            assert_size_stride(arg169_1, (), ())
            buf263 = buf218; del buf218  # reuse
            buf265 = reinterpret_tensor(buf258, (1, s50, 896), (896*s50, 896, 1), 0); del buf258  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, down_proj_10, hidden_states_109, attn_output_47, hidden_states_115, pow_24, variance_23, rsqrt_23, hidden_states_117, hidden_states_118], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf263, buf224, buf240, buf246, buf262, arg170_1, arg169_1.item(), buf265, s50, 896, stream=raw_stream0)
            del arg169_1
            del arg170_1
            del buf224
            assert_size_stride(arg171_1, (4864, 896), (896, 1))
            buf266 = reinterpret_tensor(buf245, (s50, 4864), (4864, 1), 0); del buf245  # reuse
            # Topologically Sorted Source Nodes: [pow_24, variance_23, rsqrt_23, hidden_states_117, hidden_states_118, linear_81], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf265, (s50, 896), (896, 1), 0), reinterpret_tensor(arg171_1, (896, 4864), (1, 896), 0), out=buf266)
            del arg171_1
            assert_size_stride(arg172_1, (4864, 896), (896, 1))
            buf267 = buf244; del buf244  # reuse
            # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf265, (s50, 896), (896, 1), 0), reinterpret_tensor(arg172_1, (896, 4864), (1, 896), 0), out=buf267)
            del arg172_1
            buf268 = reinterpret_tensor(buf266, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf266  # reuse
            # Topologically Sorted Source Nodes: [linear_81, silu_11, linear_82, mul_109], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf268, buf267, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg173_1, (896, 4864), (4864, 1))
            buf269 = reinterpret_tensor(buf265, (s50, 896), (896, 1), 0); del buf265  # reuse
            # Topologically Sorted Source Nodes: [linear_81, silu_11, linear_82, mul_109, down_proj_11], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf268, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg173_1, (4864, 896), (1, 4864), 0), out=buf269)
            del arg173_1
            assert_size_stride(arg175_1, (896, ), (1, ))
            assert_size_stride(arg174_1, (), ())
            buf271 = reinterpret_tensor(buf262, (1, s50, 896), (896*s50, 896, 1), 0); del buf262  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, pow_25, variance_24, rsqrt_24, hidden_states_121, hidden_states_122], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf263, buf269, arg175_1, arg174_1.item(), buf271, s50, 896, stream=raw_stream0)
            del arg174_1
            del arg175_1
            assert_size_stride(arg177_1, (896, ), (1, ))
            assert_size_stride(arg176_1, (896, 896), (896, 1))
            buf272 = buf246; del buf246  # reuse
            # Topologically Sorted Source Nodes: [linear_84], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg177_1, reinterpret_tensor(buf271, (s50, 896), (896, 1), 0), reinterpret_tensor(arg176_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf272)
            del arg176_1
            del arg177_1
            assert_size_stride(arg179_1, (128, ), (1, ))
            assert_size_stride(arg178_1, (128, 896), (896, 1))
            buf273 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_85], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg179_1, reinterpret_tensor(buf271, (s50, 896), (896, 1), 0), reinterpret_tensor(arg178_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf273)
            del arg178_1
            del arg179_1
            buf274 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_15, sin_15, linear_85, view_37, key_states_12, mul_114, x2_25, neg_25, x1_25, cat_50, mul_115, k_embed_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf273, arg4_1, arg5_1.item(), buf274, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg181_1, (128, ), (1, ))
            assert_size_stride(arg180_1, (128, 896), (896, 1))
            buf275 = buf273; del buf273  # reuse
            # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg181_1, reinterpret_tensor(buf271, (s50, 896), (896, 1), 0), reinterpret_tensor(arg180_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf275)
            del arg180_1
            del arg181_1
            buf276 = reinterpret_tensor(buf271, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf271  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_84, view_36, query_states_12, cos_15, mul_112, x2_24, neg_24, x1_24, cat_49, sin_15, mul_113, q_embed_12, getitem_488, hidden_states_123, key_12, linear_86, view_38, value_states_12, getitem_493, hidden_states_124, value_12, attn_output_48], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf272, arg4_1, arg5_1.item(), buf276, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf277 = reinterpret_tensor(buf272, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf272  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_84, view_36, query_states_12, cos_15, mul_112, x2_24, neg_24, x1_24, cat_49, sin_15, mul_113, q_embed_12, getitem_488, hidden_states_123, key_12, linear_86, view_38, value_states_12, getitem_493, hidden_states_124, value_12, attn_output_48], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf274, buf277, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf278 = reinterpret_tensor(buf240, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf240  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_84, view_36, query_states_12, cos_15, mul_112, x2_24, neg_24, x1_24, cat_49, sin_15, mul_113, q_embed_12, getitem_488, hidden_states_123, key_12, linear_86, view_38, value_states_12, getitem_493, hidden_states_124, value_12, attn_output_48], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf275, buf278, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf279 = buf256; del buf256  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_84, view_36, query_states_12, cos_15, mul_112, x2_24, neg_24, x1_24, cat_49, sin_15, mul_113, q_embed_12, getitem_488, hidden_states_123, key_12, linear_86, view_38, value_states_12, getitem_493, hidden_states_124, value_12, attn_output_48], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf279, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_84, view_36, query_states_12, cos_15, mul_112, x2_24, neg_24, x1_24, cat_49, sin_15, mul_113, q_embed_12, getitem_488, hidden_states_123, key_12, linear_86, view_38, value_states_12, getitem_493, hidden_states_124, value_12, attn_output_48], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf280 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf276, buf277, buf278, reinterpret_tensor(buf279, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf276
            buf281 = buf280[0]
            assert_size_stride(buf281, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf281, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf280
            assert_size_stride(arg182_1, (896, 896), (896, 1))
            buf285 = reinterpret_tensor(buf278, (s50, 896), (896, 1), 0); del buf278  # reuse
            # Topologically Sorted Source Nodes: [transpose_52, reshape_38, attn_output_51], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf281, (s50, 896), (896, 1), 0), reinterpret_tensor(arg182_1, (896, 896), (1, 896), 0), out=buf285)
            del arg182_1
            assert_size_stride(arg184_1, (896, ), (1, ))
            assert_size_stride(arg183_1, (), ())
            buf287 = reinterpret_tensor(buf281, (1, s50, 896), (896*s50, 896, 1), 0); del buf281  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, attn_output_51, hidden_states_125, pow_26, variance_25, rsqrt_25, hidden_states_127, hidden_states_128], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf263, buf269, buf285, arg184_1, arg183_1.item(), buf287, s50, 896, stream=raw_stream0)
            del arg183_1
            del arg184_1
            assert_size_stride(arg185_1, (4864, 896), (896, 1))
            buf288 = reinterpret_tensor(buf268, (s50, 4864), (4864, 1), 0); del buf268  # reuse
            # Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf287, (s50, 896), (896, 1), 0), reinterpret_tensor(arg185_1, (896, 4864), (1, 896), 0), out=buf288)
            del arg185_1
            assert_size_stride(arg186_1, (4864, 896), (896, 1))
            buf289 = buf267; del buf267  # reuse
            # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf287, (s50, 896), (896, 1), 0), reinterpret_tensor(arg186_1, (896, 4864), (1, 896), 0), out=buf289)
            del arg186_1
            buf290 = reinterpret_tensor(buf288, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf288  # reuse
            # Topologically Sorted Source Nodes: [linear_88, silu_12, linear_89, mul_118], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf290, buf289, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg187_1, (896, 4864), (4864, 1))
            buf291 = reinterpret_tensor(buf287, (s50, 896), (896, 1), 0); del buf287  # reuse
            # Topologically Sorted Source Nodes: [linear_88, silu_12, linear_89, mul_118, down_proj_12], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf290, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg187_1, (4864, 896), (1, 4864), 0), out=buf291)
            del arg187_1
            assert_size_stride(arg189_1, (896, ), (1, ))
            assert_size_stride(arg188_1, (), ())
            buf293 = reinterpret_tensor(buf277, (1, s50, 896), (896*s50, 896, 1), 0); del buf277  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, attn_output_51, hidden_states_125, down_proj_12, hidden_states_129, pow_27, variance_26, rsqrt_26, hidden_states_131, hidden_states_132], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf263, buf269, buf285, buf291, arg189_1, arg188_1.item(), buf293, s50, 896, stream=raw_stream0)
            del arg188_1
            del arg189_1
            assert_size_stride(arg191_1, (896, ), (1, ))
            assert_size_stride(arg190_1, (896, 896), (896, 1))
            buf294 = empty_strided_cuda((s50, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_91], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg191_1, reinterpret_tensor(buf293, (s50, 896), (896, 1), 0), reinterpret_tensor(arg190_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf294)
            del arg190_1
            del arg191_1
            assert_size_stride(arg193_1, (128, ), (1, ))
            assert_size_stride(arg192_1, (128, 896), (896, 1))
            buf295 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_92], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg193_1, reinterpret_tensor(buf293, (s50, 896), (896, 1), 0), reinterpret_tensor(arg192_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf295)
            del arg192_1
            del arg193_1
            buf296 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_16, sin_16, linear_92, view_40, key_states_13, mul_123, x2_27, neg_27, x1_27, cat_54, mul_124, k_embed_13], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf295, arg4_1, arg5_1.item(), buf296, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg195_1, (128, ), (1, ))
            assert_size_stride(arg194_1, (128, 896), (896, 1))
            buf297 = buf295; del buf295  # reuse
            # Topologically Sorted Source Nodes: [linear_93], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg195_1, reinterpret_tensor(buf293, (s50, 896), (896, 1), 0), reinterpret_tensor(arg194_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf297)
            del arg194_1
            del arg195_1
            buf298 = reinterpret_tensor(buf293, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf293  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_91, view_39, query_states_13, cos_16, mul_121, x2_26, neg_26, x1_26, cat_53, sin_16, mul_122, q_embed_13, getitem_525, hidden_states_133, key_13, linear_93, view_41, value_states_13, getitem_530, hidden_states_134, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf294, arg4_1, arg5_1.item(), buf298, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf299 = reinterpret_tensor(buf294, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf294  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_91, view_39, query_states_13, cos_16, mul_121, x2_26, neg_26, x1_26, cat_53, sin_16, mul_122, q_embed_13, getitem_525, hidden_states_133, key_13, linear_93, view_41, value_states_13, getitem_530, hidden_states_134, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf296, buf299, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf300 = empty_strided_cuda((1, 14, s50, 64), (896*s50, 64*s50, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_91, view_39, query_states_13, cos_16, mul_121, x2_26, neg_26, x1_26, cat_53, sin_16, mul_122, q_embed_13, getitem_525, hidden_states_133, key_13, linear_93, view_41, value_states_13, getitem_530, hidden_states_134, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf297, buf300, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf301 = buf279; del buf279  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_91, view_39, query_states_13, cos_16, mul_121, x2_26, neg_26, x1_26, cat_53, sin_16, mul_122, q_embed_13, getitem_525, hidden_states_133, key_13, linear_93, view_41, value_states_13, getitem_530, hidden_states_134, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf301, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_91, view_39, query_states_13, cos_16, mul_121, x2_26, neg_26, x1_26, cat_53, sin_16, mul_122, q_embed_13, getitem_525, hidden_states_133, key_13, linear_93, view_41, value_states_13, getitem_530, hidden_states_134, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf302 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf298, buf299, buf300, reinterpret_tensor(buf301, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf298
            del buf299
            buf303 = buf302[0]
            assert_size_stride(buf303, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf303, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf302
            assert_size_stride(arg196_1, (896, 896), (896, 1))
            buf307 = reinterpret_tensor(buf300, (s50, 896), (896, 1), 0); del buf300  # reuse
            # Topologically Sorted Source Nodes: [transpose_56, reshape_41, attn_output_55], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf303, (s50, 896), (896, 1), 0), reinterpret_tensor(arg196_1, (896, 896), (1, 896), 0), out=buf307)
            del arg196_1
            assert_size_stride(arg198_1, (896, ), (1, ))
            assert_size_stride(arg197_1, (), ())
            buf308 = buf263; del buf263  # reuse
            buf310 = reinterpret_tensor(buf303, (1, s50, 896), (896*s50, 896, 1), 0); del buf303  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, attn_output_51, hidden_states_125, down_proj_12, hidden_states_129, attn_output_55, hidden_states_135, pow_28, variance_27, rsqrt_27, hidden_states_137, hidden_states_138], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf308, buf269, buf285, buf291, buf307, arg198_1, arg197_1.item(), buf310, s50, 896, stream=raw_stream0)
            del arg197_1
            del arg198_1
            del buf269
            assert_size_stride(arg199_1, (4864, 896), (896, 1))
            buf311 = reinterpret_tensor(buf290, (s50, 4864), (4864, 1), 0); del buf290  # reuse
            # Topologically Sorted Source Nodes: [pow_28, variance_27, rsqrt_27, hidden_states_137, hidden_states_138, linear_95], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf310, (s50, 896), (896, 1), 0), reinterpret_tensor(arg199_1, (896, 4864), (1, 896), 0), out=buf311)
            del arg199_1
            assert_size_stride(arg200_1, (4864, 896), (896, 1))
            buf312 = buf289; del buf289  # reuse
            # Topologically Sorted Source Nodes: [linear_96], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf310, (s50, 896), (896, 1), 0), reinterpret_tensor(arg200_1, (896, 4864), (1, 896), 0), out=buf312)
            del arg200_1
            buf313 = reinterpret_tensor(buf311, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf311  # reuse
            # Topologically Sorted Source Nodes: [linear_95, silu_13, linear_96, mul_127], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf313, buf312, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg201_1, (896, 4864), (4864, 1))
            buf314 = reinterpret_tensor(buf310, (s50, 896), (896, 1), 0); del buf310  # reuse
            # Topologically Sorted Source Nodes: [linear_95, silu_13, linear_96, mul_127, down_proj_13], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf313, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg201_1, (4864, 896), (1, 4864), 0), out=buf314)
            del arg201_1
            assert_size_stride(arg203_1, (896, ), (1, ))
            assert_size_stride(arg202_1, (), ())
            buf316 = reinterpret_tensor(buf307, (1, s50, 896), (896*s50, 896, 1), 0); del buf307  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, pow_29, variance_28, rsqrt_28, hidden_states_141, hidden_states_142], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf308, buf314, arg203_1, arg202_1.item(), buf316, s50, 896, stream=raw_stream0)
            del arg202_1
            del arg203_1
            assert_size_stride(arg205_1, (896, ), (1, ))
            assert_size_stride(arg204_1, (896, 896), (896, 1))
            buf317 = buf291; del buf291  # reuse
            # Topologically Sorted Source Nodes: [linear_98], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg205_1, reinterpret_tensor(buf316, (s50, 896), (896, 1), 0), reinterpret_tensor(arg204_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf317)
            del arg204_1
            del arg205_1
            assert_size_stride(arg207_1, (128, ), (1, ))
            assert_size_stride(arg206_1, (128, 896), (896, 1))
            buf318 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_99], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg207_1, reinterpret_tensor(buf316, (s50, 896), (896, 1), 0), reinterpret_tensor(arg206_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf318)
            del arg206_1
            del arg207_1
            buf319 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_17, sin_17, linear_99, view_43, key_states_14, mul_132, x2_29, neg_29, x1_29, cat_58, mul_133, k_embed_14], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf318, arg4_1, arg5_1.item(), buf319, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg209_1, (128, ), (1, ))
            assert_size_stride(arg208_1, (128, 896), (896, 1))
            buf320 = buf318; del buf318  # reuse
            # Topologically Sorted Source Nodes: [linear_100], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg209_1, reinterpret_tensor(buf316, (s50, 896), (896, 1), 0), reinterpret_tensor(arg208_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf320)
            del arg208_1
            del arg209_1
            buf321 = reinterpret_tensor(buf316, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf316  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_98, view_42, query_states_14, cos_17, mul_130, x2_28, neg_28, x1_28, cat_57, sin_17, mul_131, q_embed_14, getitem_562, hidden_states_143, key_14, linear_100, view_44, value_states_14, getitem_567, hidden_states_144, value_14, attn_output_56], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf317, arg4_1, arg5_1.item(), buf321, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf322 = reinterpret_tensor(buf317, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf317  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_98, view_42, query_states_14, cos_17, mul_130, x2_28, neg_28, x1_28, cat_57, sin_17, mul_131, q_embed_14, getitem_562, hidden_states_143, key_14, linear_100, view_44, value_states_14, getitem_567, hidden_states_144, value_14, attn_output_56], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf319, buf322, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf323 = reinterpret_tensor(buf285, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf285  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_98, view_42, query_states_14, cos_17, mul_130, x2_28, neg_28, x1_28, cat_57, sin_17, mul_131, q_embed_14, getitem_562, hidden_states_143, key_14, linear_100, view_44, value_states_14, getitem_567, hidden_states_144, value_14, attn_output_56], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf320, buf323, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf324 = buf301; del buf301  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_98, view_42, query_states_14, cos_17, mul_130, x2_28, neg_28, x1_28, cat_57, sin_17, mul_131, q_embed_14, getitem_562, hidden_states_143, key_14, linear_100, view_44, value_states_14, getitem_567, hidden_states_144, value_14, attn_output_56], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf324, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_98, view_42, query_states_14, cos_17, mul_130, x2_28, neg_28, x1_28, cat_57, sin_17, mul_131, q_embed_14, getitem_562, hidden_states_143, key_14, linear_100, view_44, value_states_14, getitem_567, hidden_states_144, value_14, attn_output_56], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf325 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf321, buf322, buf323, reinterpret_tensor(buf324, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf321
            buf326 = buf325[0]
            assert_size_stride(buf326, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf326, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf325
            assert_size_stride(arg210_1, (896, 896), (896, 1))
            buf330 = reinterpret_tensor(buf323, (s50, 896), (896, 1), 0); del buf323  # reuse
            # Topologically Sorted Source Nodes: [transpose_60, reshape_44, attn_output_59], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf326, (s50, 896), (896, 1), 0), reinterpret_tensor(arg210_1, (896, 896), (1, 896), 0), out=buf330)
            del arg210_1
            assert_size_stride(arg212_1, (896, ), (1, ))
            assert_size_stride(arg211_1, (), ())
            buf332 = reinterpret_tensor(buf326, (1, s50, 896), (896*s50, 896, 1), 0); del buf326  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, attn_output_59, hidden_states_145, pow_30, variance_29, rsqrt_29, hidden_states_147, hidden_states_148], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf308, buf314, buf330, arg212_1, arg211_1.item(), buf332, s50, 896, stream=raw_stream0)
            del arg211_1
            del arg212_1
            assert_size_stride(arg213_1, (4864, 896), (896, 1))
            buf333 = reinterpret_tensor(buf313, (s50, 4864), (4864, 1), 0); del buf313  # reuse
            # Topologically Sorted Source Nodes: [linear_102], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf332, (s50, 896), (896, 1), 0), reinterpret_tensor(arg213_1, (896, 4864), (1, 896), 0), out=buf333)
            del arg213_1
            assert_size_stride(arg214_1, (4864, 896), (896, 1))
            buf334 = buf312; del buf312  # reuse
            # Topologically Sorted Source Nodes: [linear_103], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf332, (s50, 896), (896, 1), 0), reinterpret_tensor(arg214_1, (896, 4864), (1, 896), 0), out=buf334)
            del arg214_1
            buf335 = reinterpret_tensor(buf333, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf333  # reuse
            # Topologically Sorted Source Nodes: [linear_102, silu_14, linear_103, mul_136], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf335, buf334, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg215_1, (896, 4864), (4864, 1))
            buf336 = reinterpret_tensor(buf332, (s50, 896), (896, 1), 0); del buf332  # reuse
            # Topologically Sorted Source Nodes: [linear_102, silu_14, linear_103, mul_136, down_proj_14], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf335, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg215_1, (4864, 896), (1, 4864), 0), out=buf336)
            del arg215_1
            assert_size_stride(arg217_1, (896, ), (1, ))
            assert_size_stride(arg216_1, (), ())
            buf338 = reinterpret_tensor(buf322, (1, s50, 896), (896*s50, 896, 1), 0); del buf322  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, attn_output_59, hidden_states_145, down_proj_14, hidden_states_149, pow_31, variance_30, rsqrt_30, hidden_states_151, hidden_states_152], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf308, buf314, buf330, buf336, arg217_1, arg216_1.item(), buf338, s50, 896, stream=raw_stream0)
            del arg216_1
            del arg217_1
            assert_size_stride(arg219_1, (896, ), (1, ))
            assert_size_stride(arg218_1, (896, 896), (896, 1))
            buf339 = empty_strided_cuda((s50, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_105], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg219_1, reinterpret_tensor(buf338, (s50, 896), (896, 1), 0), reinterpret_tensor(arg218_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf339)
            del arg218_1
            del arg219_1
            assert_size_stride(arg221_1, (128, ), (1, ))
            assert_size_stride(arg220_1, (128, 896), (896, 1))
            buf340 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_106], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg221_1, reinterpret_tensor(buf338, (s50, 896), (896, 1), 0), reinterpret_tensor(arg220_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf340)
            del arg220_1
            del arg221_1
            buf341 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_18, sin_18, linear_106, view_46, key_states_15, mul_141, x2_31, neg_31, x1_31, cat_62, mul_142, k_embed_15], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf340, arg4_1, arg5_1.item(), buf341, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg223_1, (128, ), (1, ))
            assert_size_stride(arg222_1, (128, 896), (896, 1))
            buf342 = buf340; del buf340  # reuse
            # Topologically Sorted Source Nodes: [linear_107], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg223_1, reinterpret_tensor(buf338, (s50, 896), (896, 1), 0), reinterpret_tensor(arg222_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf342)
            del arg222_1
            del arg223_1
            buf343 = reinterpret_tensor(buf338, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf338  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_105, view_45, query_states_15, cos_18, mul_139, x2_30, neg_30, x1_30, cat_61, sin_18, mul_140, q_embed_15, getitem_599, hidden_states_153, key_15, linear_107, view_47, value_states_15, getitem_604, hidden_states_154, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf339, arg4_1, arg5_1.item(), buf343, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf344 = reinterpret_tensor(buf339, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf339  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_105, view_45, query_states_15, cos_18, mul_139, x2_30, neg_30, x1_30, cat_61, sin_18, mul_140, q_embed_15, getitem_599, hidden_states_153, key_15, linear_107, view_47, value_states_15, getitem_604, hidden_states_154, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf341, buf344, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf345 = empty_strided_cuda((1, 14, s50, 64), (896*s50, 64*s50, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_105, view_45, query_states_15, cos_18, mul_139, x2_30, neg_30, x1_30, cat_61, sin_18, mul_140, q_embed_15, getitem_599, hidden_states_153, key_15, linear_107, view_47, value_states_15, getitem_604, hidden_states_154, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf342, buf345, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf346 = buf324; del buf324  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_105, view_45, query_states_15, cos_18, mul_139, x2_30, neg_30, x1_30, cat_61, sin_18, mul_140, q_embed_15, getitem_599, hidden_states_153, key_15, linear_107, view_47, value_states_15, getitem_604, hidden_states_154, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf346, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_105, view_45, query_states_15, cos_18, mul_139, x2_30, neg_30, x1_30, cat_61, sin_18, mul_140, q_embed_15, getitem_599, hidden_states_153, key_15, linear_107, view_47, value_states_15, getitem_604, hidden_states_154, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf347 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf343, buf344, buf345, reinterpret_tensor(buf346, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf343
            del buf344
            del buf346
            buf348 = buf347[0]
            assert_size_stride(buf348, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf348, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf347
            assert_size_stride(arg224_1, (896, 896), (896, 1))
            buf352 = reinterpret_tensor(buf345, (s50, 896), (896, 1), 0); del buf345  # reuse
            # Topologically Sorted Source Nodes: [transpose_64, reshape_47, attn_output_63], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf348, (s50, 896), (896, 1), 0), reinterpret_tensor(arg224_1, (896, 896), (1, 896), 0), out=buf352)
            del arg224_1
            assert_size_stride(arg226_1, (896, ), (1, ))
            assert_size_stride(arg225_1, (), ())
            buf353 = buf308; del buf308  # reuse
            buf355 = reinterpret_tensor(buf348, (1, s50, 896), (896*s50, 896, 1), 0); del buf348  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, attn_output_59, hidden_states_145, down_proj_14, hidden_states_149, attn_output_63, hidden_states_155, pow_32, variance_31, rsqrt_31, hidden_states_157, hidden_states_158], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf353, buf314, buf330, buf336, buf352, arg226_1, arg225_1.item(), buf355, s50, 896, stream=raw_stream0)
            del arg225_1
            del arg226_1
            del buf314
            assert_size_stride(arg227_1, (4864, 896), (896, 1))
            buf356 = reinterpret_tensor(buf335, (s50, 4864), (4864, 1), 0); del buf335  # reuse
            # Topologically Sorted Source Nodes: [pow_32, variance_31, rsqrt_31, hidden_states_157, hidden_states_158, linear_109], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf355, (s50, 896), (896, 1), 0), reinterpret_tensor(arg227_1, (896, 4864), (1, 896), 0), out=buf356)
            del arg227_1
            assert_size_stride(arg228_1, (4864, 896), (896, 1))
            buf357 = buf334; del buf334  # reuse
            # Topologically Sorted Source Nodes: [linear_110], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf355, (s50, 896), (896, 1), 0), reinterpret_tensor(arg228_1, (896, 4864), (1, 896), 0), out=buf357)
            del arg228_1
            buf358 = reinterpret_tensor(buf356, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf356  # reuse
            # Topologically Sorted Source Nodes: [linear_109, silu_15, linear_110, mul_145], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf358, buf357, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg229_1, (896, 4864), (4864, 1))
            buf359 = reinterpret_tensor(buf355, (s50, 896), (896, 1), 0); del buf355  # reuse
            # Topologically Sorted Source Nodes: [linear_109, silu_15, linear_110, mul_145, down_proj_15], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf358, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg229_1, (4864, 896), (1, 4864), 0), out=buf359)
            del arg229_1
            assert_size_stride(arg231_1, (896, ), (1, ))
            assert_size_stride(arg230_1, (), ())
            buf361 = reinterpret_tensor(buf352, (1, s50, 896), (896*s50, 896, 1), 0); del buf352  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, pow_33, variance_32, rsqrt_32, hidden_states_161, hidden_states_162], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf353, buf359, arg231_1, arg230_1.item(), buf361, s50, 896, stream=raw_stream0)
            del arg230_1
            del arg231_1
            assert_size_stride(arg233_1, (896, ), (1, ))
            assert_size_stride(arg232_1, (896, 896), (896, 1))
            buf362 = buf336; del buf336  # reuse
            # Topologically Sorted Source Nodes: [linear_112], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg233_1, reinterpret_tensor(buf361, (s50, 896), (896, 1), 0), reinterpret_tensor(arg232_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf362)
            del arg232_1
            del arg233_1
            assert_size_stride(arg235_1, (128, ), (1, ))
            assert_size_stride(arg234_1, (128, 896), (896, 1))
            buf363 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_113], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg235_1, reinterpret_tensor(buf361, (s50, 896), (896, 1), 0), reinterpret_tensor(arg234_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf363)
            del arg234_1
            del arg235_1
            buf364 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_19, sin_19, linear_113, view_49, key_states_16, mul_150, x2_33, neg_33, x1_33, cat_66, mul_151, k_embed_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf363, arg4_1, arg5_1.item(), buf364, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg237_1, (128, ), (1, ))
            assert_size_stride(arg236_1, (128, 896), (896, 1))
            buf365 = buf363; del buf363  # reuse
            # Topologically Sorted Source Nodes: [linear_114], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg237_1, reinterpret_tensor(buf361, (s50, 896), (896, 1), 0), reinterpret_tensor(arg236_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf365)
            del arg236_1
            del arg237_1
            buf366 = reinterpret_tensor(buf361, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf361  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_112, view_48, query_states_16, cos_19, mul_148, x2_32, neg_32, x1_32, cat_65, sin_19, mul_149, q_embed_16, getitem_636, hidden_states_163, key_16, linear_114, view_50, value_states_16, getitem_641, hidden_states_164, value_16, attn_output_64], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf362, arg4_1, arg5_1.item(), buf366, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf367 = reinterpret_tensor(buf362, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf362  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_112, view_48, query_states_16, cos_19, mul_148, x2_32, neg_32, x1_32, cat_65, sin_19, mul_149, q_embed_16, getitem_636, hidden_states_163, key_16, linear_114, view_50, value_states_16, getitem_641, hidden_states_164, value_16, attn_output_64], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf364, buf367, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf368 = reinterpret_tensor(buf330, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf330  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_112, view_48, query_states_16, cos_19, mul_148, x2_32, neg_32, x1_32, cat_65, sin_19, mul_149, q_embed_16, getitem_636, hidden_states_163, key_16, linear_114, view_50, value_states_16, getitem_641, hidden_states_164, value_16, attn_output_64], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf365, buf368, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf369 = empty_strided_cuda((1, 1, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_112, view_48, query_states_16, cos_19, mul_148, x2_32, neg_32, x1_32, cat_65, sin_19, mul_149, q_embed_16, getitem_636, hidden_states_163, key_16, linear_114, view_50, value_states_16, getitem_641, hidden_states_164, value_16, attn_output_64], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf369, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_112, view_48, query_states_16, cos_19, mul_148, x2_32, neg_32, x1_32, cat_65, sin_19, mul_149, q_embed_16, getitem_636, hidden_states_163, key_16, linear_114, view_50, value_states_16, getitem_641, hidden_states_164, value_16, attn_output_64], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf370 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf366, buf367, buf368, reinterpret_tensor(buf369, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf366
            del buf367
            buf371 = buf370[0]
            assert_size_stride(buf371, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf371, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf370
            assert_size_stride(arg238_1, (896, 896), (896, 1))
            buf375 = reinterpret_tensor(buf368, (s50, 896), (896, 1), 0); del buf368  # reuse
            # Topologically Sorted Source Nodes: [transpose_68, reshape_50, attn_output_67], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf371, (s50, 896), (896, 1), 0), reinterpret_tensor(arg238_1, (896, 896), (1, 896), 0), out=buf375)
            del arg238_1
            assert_size_stride(arg240_1, (896, ), (1, ))
            assert_size_stride(arg239_1, (), ())
            buf377 = reinterpret_tensor(buf371, (1, s50, 896), (896*s50, 896, 1), 0); del buf371  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, attn_output_67, hidden_states_165, pow_34, variance_33, rsqrt_33, hidden_states_167, hidden_states_168], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf353, buf359, buf375, arg240_1, arg239_1.item(), buf377, s50, 896, stream=raw_stream0)
            del arg239_1
            del arg240_1
            assert_size_stride(arg241_1, (4864, 896), (896, 1))
            buf378 = reinterpret_tensor(buf358, (s50, 4864), (4864, 1), 0); del buf358  # reuse
            # Topologically Sorted Source Nodes: [linear_116], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf377, (s50, 896), (896, 1), 0), reinterpret_tensor(arg241_1, (896, 4864), (1, 896), 0), out=buf378)
            del arg241_1
            assert_size_stride(arg242_1, (4864, 896), (896, 1))
            buf379 = buf357; del buf357  # reuse
            # Topologically Sorted Source Nodes: [linear_117], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf377, (s50, 896), (896, 1), 0), reinterpret_tensor(arg242_1, (896, 4864), (1, 896), 0), out=buf379)
            del arg242_1
            buf380 = reinterpret_tensor(buf378, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf378  # reuse
            # Topologically Sorted Source Nodes: [linear_116, silu_16, linear_117, mul_154], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf380, buf379, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg243_1, (896, 4864), (4864, 1))
            buf381 = reinterpret_tensor(buf377, (s50, 896), (896, 1), 0); del buf377  # reuse
            # Topologically Sorted Source Nodes: [linear_116, silu_16, linear_117, mul_154, down_proj_16], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf380, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg243_1, (4864, 896), (1, 4864), 0), out=buf381)
            del arg243_1
            assert_size_stride(arg245_1, (896, ), (1, ))
            assert_size_stride(arg244_1, (), ())
            buf383 = empty_strided_cuda((1, s50, 896), (896*s50, 896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, attn_output_67, hidden_states_165, down_proj_16, hidden_states_169, pow_35, variance_34, rsqrt_34, hidden_states_171, hidden_states_172], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf353, buf359, buf375, buf381, arg245_1, arg244_1.item(), buf383, s50, 896, stream=raw_stream0)
            del arg244_1
            del arg245_1
            assert_size_stride(arg247_1, (896, ), (1, ))
            assert_size_stride(arg246_1, (896, 896), (896, 1))
            buf384 = empty_strided_cuda((s50, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_119], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg247_1, reinterpret_tensor(buf383, (s50, 896), (896, 1), 0), reinterpret_tensor(arg246_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf384)
            del arg246_1
            del arg247_1
            assert_size_stride(arg249_1, (128, ), (1, ))
            assert_size_stride(arg248_1, (128, 896), (896, 1))
            buf385 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_120], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg249_1, reinterpret_tensor(buf383, (s50, 896), (896, 1), 0), reinterpret_tensor(arg248_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf385)
            del arg248_1
            del arg249_1
            buf386 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_20, sin_20, linear_120, view_52, key_states_17, mul_159, x2_35, neg_35, x1_35, cat_70, mul_160, k_embed_17], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf385, arg4_1, arg5_1.item(), buf386, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg251_1, (128, ), (1, ))
            assert_size_stride(arg250_1, (128, 896), (896, 1))
            buf387 = buf385; del buf385  # reuse
            # Topologically Sorted Source Nodes: [linear_121], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg251_1, reinterpret_tensor(buf383, (s50, 896), (896, 1), 0), reinterpret_tensor(arg250_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf387)
            del arg250_1
            del arg251_1
            buf388 = reinterpret_tensor(buf383, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf383  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_119, view_51, query_states_17, cos_20, mul_157, x2_34, neg_34, x1_34, cat_69, sin_20, mul_158, q_embed_17, getitem_673, hidden_states_173, key_17, linear_121, view_53, value_states_17, getitem_678, hidden_states_174, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf384, arg4_1, arg5_1.item(), buf388, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf389 = reinterpret_tensor(buf384, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf384  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_119, view_51, query_states_17, cos_20, mul_157, x2_34, neg_34, x1_34, cat_69, sin_20, mul_158, q_embed_17, getitem_673, hidden_states_173, key_17, linear_121, view_53, value_states_17, getitem_678, hidden_states_174, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf386, buf389, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf390 = empty_strided_cuda((1, 14, s50, 64), (896*s50, 64*s50, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_119, view_51, query_states_17, cos_20, mul_157, x2_34, neg_34, x1_34, cat_69, sin_20, mul_158, q_embed_17, getitem_673, hidden_states_173, key_17, linear_121, view_53, value_states_17, getitem_678, hidden_states_174, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf387, buf390, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf391 = buf369; del buf369  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_119, view_51, query_states_17, cos_20, mul_157, x2_34, neg_34, x1_34, cat_69, sin_20, mul_158, q_embed_17, getitem_673, hidden_states_173, key_17, linear_121, view_53, value_states_17, getitem_678, hidden_states_174, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf391, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_119, view_51, query_states_17, cos_20, mul_157, x2_34, neg_34, x1_34, cat_69, sin_20, mul_158, q_embed_17, getitem_673, hidden_states_173, key_17, linear_121, view_53, value_states_17, getitem_678, hidden_states_174, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf392 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf388, buf389, buf390, reinterpret_tensor(buf391, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf388
            del buf389
            buf393 = buf392[0]
            assert_size_stride(buf393, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf393, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf392
            assert_size_stride(arg252_1, (896, 896), (896, 1))
            buf397 = reinterpret_tensor(buf390, (s50, 896), (896, 1), 0); del buf390  # reuse
            # Topologically Sorted Source Nodes: [transpose_72, reshape_53, attn_output_71], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf393, (s50, 896), (896, 1), 0), reinterpret_tensor(arg252_1, (896, 896), (1, 896), 0), out=buf397)
            del arg252_1
            assert_size_stride(arg254_1, (896, ), (1, ))
            assert_size_stride(arg253_1, (), ())
            buf398 = buf353; del buf353  # reuse
            buf400 = reinterpret_tensor(buf393, (1, s50, 896), (896*s50, 896, 1), 0); del buf393  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, attn_output_67, hidden_states_165, down_proj_16, hidden_states_169, attn_output_71, hidden_states_175, pow_36, variance_35, rsqrt_35, hidden_states_177, hidden_states_178], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf398, buf359, buf375, buf381, buf397, arg254_1, arg253_1.item(), buf400, s50, 896, stream=raw_stream0)
            del arg253_1
            del arg254_1
            del buf359
            del buf375
            assert_size_stride(arg255_1, (4864, 896), (896, 1))
            buf401 = reinterpret_tensor(buf380, (s50, 4864), (4864, 1), 0); del buf380  # reuse
            # Topologically Sorted Source Nodes: [pow_36, variance_35, rsqrt_35, hidden_states_177, hidden_states_178, linear_123], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf400, (s50, 896), (896, 1), 0), reinterpret_tensor(arg255_1, (896, 4864), (1, 896), 0), out=buf401)
            del arg255_1
            assert_size_stride(arg256_1, (4864, 896), (896, 1))
            buf402 = buf379; del buf379  # reuse
            # Topologically Sorted Source Nodes: [linear_124], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf400, (s50, 896), (896, 1), 0), reinterpret_tensor(arg256_1, (896, 4864), (1, 896), 0), out=buf402)
            del arg256_1
            buf403 = reinterpret_tensor(buf401, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf401  # reuse
            # Topologically Sorted Source Nodes: [linear_123, silu_17, linear_124, mul_163], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf403, buf402, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg257_1, (896, 4864), (4864, 1))
            buf404 = reinterpret_tensor(buf400, (s50, 896), (896, 1), 0); del buf400  # reuse
            # Topologically Sorted Source Nodes: [linear_123, silu_17, linear_124, mul_163, down_proj_17], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf403, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg257_1, (4864, 896), (1, 4864), 0), out=buf404)
            del arg257_1
            assert_size_stride(arg259_1, (896, ), (1, ))
            assert_size_stride(arg258_1, (), ())
            buf406 = reinterpret_tensor(buf397, (1, s50, 896), (896*s50, 896, 1), 0); del buf397  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, pow_37, variance_36, rsqrt_36, hidden_states_181, hidden_states_182], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf398, buf404, arg259_1, arg258_1.item(), buf406, s50, 896, stream=raw_stream0)
            del arg258_1
            del arg259_1
            assert_size_stride(arg261_1, (896, ), (1, ))
            assert_size_stride(arg260_1, (896, 896), (896, 1))
            buf407 = buf381; del buf381  # reuse
            # Topologically Sorted Source Nodes: [linear_126], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg261_1, reinterpret_tensor(buf406, (s50, 896), (896, 1), 0), reinterpret_tensor(arg260_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf407)
            del arg260_1
            del arg261_1
            assert_size_stride(arg263_1, (128, ), (1, ))
            assert_size_stride(arg262_1, (128, 896), (896, 1))
            buf408 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_127], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg263_1, reinterpret_tensor(buf406, (s50, 896), (896, 1), 0), reinterpret_tensor(arg262_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf408)
            del arg262_1
            del arg263_1
            buf409 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_21, sin_21, linear_127, view_55, key_states_18, mul_168, x2_37, neg_37, x1_37, cat_74, mul_169, k_embed_18], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf408, arg4_1, arg5_1.item(), buf409, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg265_1, (128, ), (1, ))
            assert_size_stride(arg264_1, (128, 896), (896, 1))
            buf410 = buf408; del buf408  # reuse
            # Topologically Sorted Source Nodes: [linear_128], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg265_1, reinterpret_tensor(buf406, (s50, 896), (896, 1), 0), reinterpret_tensor(arg264_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf410)
            del arg264_1
            del arg265_1
            buf411 = reinterpret_tensor(buf406, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf406  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_126, view_54, query_states_18, cos_21, mul_166, x2_36, neg_36, x1_36, cat_73, sin_21, mul_167, q_embed_18, getitem_710, hidden_states_183, key_18, linear_128, view_56, value_states_18, getitem_715, hidden_states_184, value_18, attn_output_72], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf407, arg4_1, arg5_1.item(), buf411, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf412 = reinterpret_tensor(buf407, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf407  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_126, view_54, query_states_18, cos_21, mul_166, x2_36, neg_36, x1_36, cat_73, sin_21, mul_167, q_embed_18, getitem_710, hidden_states_183, key_18, linear_128, view_56, value_states_18, getitem_715, hidden_states_184, value_18, attn_output_72], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf409, buf412, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf413 = empty_strided_cuda((1, 14, s50, 64), (896*s50, 64*s50, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_126, view_54, query_states_18, cos_21, mul_166, x2_36, neg_36, x1_36, cat_73, sin_21, mul_167, q_embed_18, getitem_710, hidden_states_183, key_18, linear_128, view_56, value_states_18, getitem_715, hidden_states_184, value_18, attn_output_72], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf410, buf413, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf414 = buf391; del buf391  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_126, view_54, query_states_18, cos_21, mul_166, x2_36, neg_36, x1_36, cat_73, sin_21, mul_167, q_embed_18, getitem_710, hidden_states_183, key_18, linear_128, view_56, value_states_18, getitem_715, hidden_states_184, value_18, attn_output_72], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf414, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_126, view_54, query_states_18, cos_21, mul_166, x2_36, neg_36, x1_36, cat_73, sin_21, mul_167, q_embed_18, getitem_710, hidden_states_183, key_18, linear_128, view_56, value_states_18, getitem_715, hidden_states_184, value_18, attn_output_72], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf415 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf411, buf412, buf413, reinterpret_tensor(buf414, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf411
            del buf412
            buf416 = buf415[0]
            assert_size_stride(buf416, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf416, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf415
            assert_size_stride(arg266_1, (896, 896), (896, 1))
            buf420 = reinterpret_tensor(buf413, (s50, 896), (896, 1), 0); del buf413  # reuse
            # Topologically Sorted Source Nodes: [transpose_76, reshape_56, attn_output_75], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf416, (s50, 896), (896, 1), 0), reinterpret_tensor(arg266_1, (896, 896), (1, 896), 0), out=buf420)
            del arg266_1
            assert_size_stride(arg268_1, (896, ), (1, ))
            assert_size_stride(arg267_1, (), ())
            buf422 = reinterpret_tensor(buf416, (1, s50, 896), (896*s50, 896, 1), 0); del buf416  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, attn_output_75, hidden_states_185, pow_38, variance_37, rsqrt_37, hidden_states_187, hidden_states_188], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf398, buf404, buf420, arg268_1, arg267_1.item(), buf422, s50, 896, stream=raw_stream0)
            del arg267_1
            del arg268_1
            assert_size_stride(arg269_1, (4864, 896), (896, 1))
            buf423 = reinterpret_tensor(buf403, (s50, 4864), (4864, 1), 0); del buf403  # reuse
            # Topologically Sorted Source Nodes: [linear_130], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf422, (s50, 896), (896, 1), 0), reinterpret_tensor(arg269_1, (896, 4864), (1, 896), 0), out=buf423)
            del arg269_1
            assert_size_stride(arg270_1, (4864, 896), (896, 1))
            buf424 = buf402; del buf402  # reuse
            # Topologically Sorted Source Nodes: [linear_131], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf422, (s50, 896), (896, 1), 0), reinterpret_tensor(arg270_1, (896, 4864), (1, 896), 0), out=buf424)
            del arg270_1
            buf425 = reinterpret_tensor(buf423, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf423  # reuse
            # Topologically Sorted Source Nodes: [linear_130, silu_18, linear_131, mul_172], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf425, buf424, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg271_1, (896, 4864), (4864, 1))
            buf426 = reinterpret_tensor(buf422, (s50, 896), (896, 1), 0); del buf422  # reuse
            # Topologically Sorted Source Nodes: [linear_130, silu_18, linear_131, mul_172, down_proj_18], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf425, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg271_1, (4864, 896), (1, 4864), 0), out=buf426)
            del arg271_1
            assert_size_stride(arg273_1, (896, ), (1, ))
            assert_size_stride(arg272_1, (), ())
            buf428 = empty_strided_cuda((1, s50, 896), (896*s50, 896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, attn_output_75, hidden_states_185, down_proj_18, hidden_states_189, pow_39, variance_38, rsqrt_38, hidden_states_191, hidden_states_192], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf398, buf404, buf420, buf426, arg273_1, arg272_1.item(), buf428, s50, 896, stream=raw_stream0)
            del arg272_1
            del arg273_1
            assert_size_stride(arg275_1, (896, ), (1, ))
            assert_size_stride(arg274_1, (896, 896), (896, 1))
            buf429 = empty_strided_cuda((s50, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_133], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg275_1, reinterpret_tensor(buf428, (s50, 896), (896, 1), 0), reinterpret_tensor(arg274_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf429)
            del arg274_1
            del arg275_1
            assert_size_stride(arg277_1, (128, ), (1, ))
            assert_size_stride(arg276_1, (128, 896), (896, 1))
            buf430 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_134], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg277_1, reinterpret_tensor(buf428, (s50, 896), (896, 1), 0), reinterpret_tensor(arg276_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf430)
            del arg276_1
            del arg277_1
            buf431 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_22, sin_22, linear_134, view_58, key_states_19, mul_177, x2_39, neg_39, x1_39, cat_78, mul_178, k_embed_19], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf430, arg4_1, arg5_1.item(), buf431, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg279_1, (128, ), (1, ))
            assert_size_stride(arg278_1, (128, 896), (896, 1))
            buf432 = buf430; del buf430  # reuse
            # Topologically Sorted Source Nodes: [linear_135], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg279_1, reinterpret_tensor(buf428, (s50, 896), (896, 1), 0), reinterpret_tensor(arg278_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf432)
            del arg278_1
            del arg279_1
            buf433 = reinterpret_tensor(buf428, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf428  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_133, view_57, query_states_19, cos_22, mul_175, x2_38, neg_38, x1_38, cat_77, sin_22, mul_176, q_embed_19, getitem_747, hidden_states_193, key_19, linear_135, view_59, value_states_19, getitem_752, hidden_states_194, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf429, arg4_1, arg5_1.item(), buf433, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf434 = reinterpret_tensor(buf429, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf429  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_133, view_57, query_states_19, cos_22, mul_175, x2_38, neg_38, x1_38, cat_77, sin_22, mul_176, q_embed_19, getitem_747, hidden_states_193, key_19, linear_135, view_59, value_states_19, getitem_752, hidden_states_194, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf431, buf434, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf435 = empty_strided_cuda((1, 14, s50, 64), (896*s50, 64*s50, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_133, view_57, query_states_19, cos_22, mul_175, x2_38, neg_38, x1_38, cat_77, sin_22, mul_176, q_embed_19, getitem_747, hidden_states_193, key_19, linear_135, view_59, value_states_19, getitem_752, hidden_states_194, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf432, buf435, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf436 = buf414; del buf414  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_133, view_57, query_states_19, cos_22, mul_175, x2_38, neg_38, x1_38, cat_77, sin_22, mul_176, q_embed_19, getitem_747, hidden_states_193, key_19, linear_135, view_59, value_states_19, getitem_752, hidden_states_194, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf436, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_133, view_57, query_states_19, cos_22, mul_175, x2_38, neg_38, x1_38, cat_77, sin_22, mul_176, q_embed_19, getitem_747, hidden_states_193, key_19, linear_135, view_59, value_states_19, getitem_752, hidden_states_194, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf437 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf433, buf434, buf435, reinterpret_tensor(buf436, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf433
            del buf434
            buf438 = buf437[0]
            assert_size_stride(buf438, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf438, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf437
            assert_size_stride(arg280_1, (896, 896), (896, 1))
            buf442 = reinterpret_tensor(buf435, (s50, 896), (896, 1), 0); del buf435  # reuse
            # Topologically Sorted Source Nodes: [transpose_80, reshape_59, attn_output_79], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf438, (s50, 896), (896, 1), 0), reinterpret_tensor(arg280_1, (896, 896), (1, 896), 0), out=buf442)
            del arg280_1
            assert_size_stride(arg282_1, (896, ), (1, ))
            assert_size_stride(arg281_1, (), ())
            buf443 = buf398; del buf398  # reuse
            buf445 = reinterpret_tensor(buf438, (1, s50, 896), (896*s50, 896, 1), 0); del buf438  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, attn_output_75, hidden_states_185, down_proj_18, hidden_states_189, attn_output_79, hidden_states_195, pow_40, variance_39, rsqrt_39, hidden_states_197, hidden_states_198], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf443, buf404, buf420, buf426, buf442, arg282_1, arg281_1.item(), buf445, s50, 896, stream=raw_stream0)
            del arg281_1
            del arg282_1
            del buf404
            del buf420
            del buf426
            assert_size_stride(arg283_1, (4864, 896), (896, 1))
            buf446 = reinterpret_tensor(buf425, (s50, 4864), (4864, 1), 0); del buf425  # reuse
            # Topologically Sorted Source Nodes: [pow_40, variance_39, rsqrt_39, hidden_states_197, hidden_states_198, linear_137], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf445, (s50, 896), (896, 1), 0), reinterpret_tensor(arg283_1, (896, 4864), (1, 896), 0), out=buf446)
            del arg283_1
            assert_size_stride(arg284_1, (4864, 896), (896, 1))
            buf447 = buf424; del buf424  # reuse
            # Topologically Sorted Source Nodes: [linear_138], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf445, (s50, 896), (896, 1), 0), reinterpret_tensor(arg284_1, (896, 4864), (1, 896), 0), out=buf447)
            del arg284_1
            buf448 = reinterpret_tensor(buf446, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf446  # reuse
            # Topologically Sorted Source Nodes: [linear_137, silu_19, linear_138, mul_181], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf448, buf447, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg285_1, (896, 4864), (4864, 1))
            buf449 = reinterpret_tensor(buf445, (s50, 896), (896, 1), 0); del buf445  # reuse
            # Topologically Sorted Source Nodes: [linear_137, silu_19, linear_138, mul_181, down_proj_19], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf448, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg285_1, (4864, 896), (1, 4864), 0), out=buf449)
            del arg285_1
            assert_size_stride(arg287_1, (896, ), (1, ))
            assert_size_stride(arg286_1, (), ())
            buf451 = reinterpret_tensor(buf442, (1, s50, 896), (896*s50, 896, 1), 0); del buf442  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, pow_41, variance_40, rsqrt_40, hidden_states_201, hidden_states_202], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf443, buf449, arg287_1, arg286_1.item(), buf451, s50, 896, stream=raw_stream0)
            del arg286_1
            del arg287_1
            assert_size_stride(arg289_1, (896, ), (1, ))
            assert_size_stride(arg288_1, (896, 896), (896, 1))
            buf452 = empty_strided_cuda((s50, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_140], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg289_1, reinterpret_tensor(buf451, (s50, 896), (896, 1), 0), reinterpret_tensor(arg288_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf452)
            del arg288_1
            del arg289_1
            assert_size_stride(arg291_1, (128, ), (1, ))
            assert_size_stride(arg290_1, (128, 896), (896, 1))
            buf453 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_141], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg291_1, reinterpret_tensor(buf451, (s50, 896), (896, 1), 0), reinterpret_tensor(arg290_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf453)
            del arg290_1
            del arg291_1
            buf454 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_23, sin_23, linear_141, view_61, key_states_20, mul_186, x2_41, neg_41, x1_41, cat_82, mul_187, k_embed_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf453, arg4_1, arg5_1.item(), buf454, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg293_1, (128, ), (1, ))
            assert_size_stride(arg292_1, (128, 896), (896, 1))
            buf455 = buf453; del buf453  # reuse
            # Topologically Sorted Source Nodes: [linear_142], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg293_1, reinterpret_tensor(buf451, (s50, 896), (896, 1), 0), reinterpret_tensor(arg292_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf455)
            del arg292_1
            del arg293_1
            buf456 = reinterpret_tensor(buf451, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf451  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_140, view_60, query_states_20, cos_23, mul_184, x2_40, neg_40, x1_40, cat_81, sin_23, mul_185, q_embed_20, getitem_784, hidden_states_203, key_20, linear_142, view_62, value_states_20, getitem_789, hidden_states_204, value_20, attn_output_80], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf452, arg4_1, arg5_1.item(), buf456, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf457 = reinterpret_tensor(buf452, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf452  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_140, view_60, query_states_20, cos_23, mul_184, x2_40, neg_40, x1_40, cat_81, sin_23, mul_185, q_embed_20, getitem_784, hidden_states_203, key_20, linear_142, view_62, value_states_20, getitem_789, hidden_states_204, value_20, attn_output_80], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf454, buf457, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf458 = empty_strided_cuda((1, 14, s50, 64), (896*s50, 64*s50, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_140, view_60, query_states_20, cos_23, mul_184, x2_40, neg_40, x1_40, cat_81, sin_23, mul_185, q_embed_20, getitem_784, hidden_states_203, key_20, linear_142, view_62, value_states_20, getitem_789, hidden_states_204, value_20, attn_output_80], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf455, buf458, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf459 = buf436; del buf436  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_140, view_60, query_states_20, cos_23, mul_184, x2_40, neg_40, x1_40, cat_81, sin_23, mul_185, q_embed_20, getitem_784, hidden_states_203, key_20, linear_142, view_62, value_states_20, getitem_789, hidden_states_204, value_20, attn_output_80], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf459, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_140, view_60, query_states_20, cos_23, mul_184, x2_40, neg_40, x1_40, cat_81, sin_23, mul_185, q_embed_20, getitem_784, hidden_states_203, key_20, linear_142, view_62, value_states_20, getitem_789, hidden_states_204, value_20, attn_output_80], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf460 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf456, buf457, buf458, reinterpret_tensor(buf459, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf456
            del buf457
            buf461 = buf460[0]
            assert_size_stride(buf461, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf461, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf460
            assert_size_stride(arg294_1, (896, 896), (896, 1))
            buf465 = reinterpret_tensor(buf458, (s50, 896), (896, 1), 0); del buf458  # reuse
            # Topologically Sorted Source Nodes: [transpose_84, reshape_62, attn_output_83], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf461, (s50, 896), (896, 1), 0), reinterpret_tensor(arg294_1, (896, 896), (1, 896), 0), out=buf465)
            del arg294_1
            assert_size_stride(arg296_1, (896, ), (1, ))
            assert_size_stride(arg295_1, (), ())
            buf467 = reinterpret_tensor(buf461, (1, s50, 896), (896*s50, 896, 1), 0); del buf461  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, attn_output_83, hidden_states_205, pow_42, variance_41, rsqrt_41, hidden_states_207, hidden_states_208], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf443, buf449, buf465, arg296_1, arg295_1.item(), buf467, s50, 896, stream=raw_stream0)
            del arg295_1
            del arg296_1
            assert_size_stride(arg297_1, (4864, 896), (896, 1))
            buf468 = reinterpret_tensor(buf448, (s50, 4864), (4864, 1), 0); del buf448  # reuse
            # Topologically Sorted Source Nodes: [linear_144], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf467, (s50, 896), (896, 1), 0), reinterpret_tensor(arg297_1, (896, 4864), (1, 896), 0), out=buf468)
            del arg297_1
            assert_size_stride(arg298_1, (4864, 896), (896, 1))
            buf469 = buf447; del buf447  # reuse
            # Topologically Sorted Source Nodes: [linear_145], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf467, (s50, 896), (896, 1), 0), reinterpret_tensor(arg298_1, (896, 4864), (1, 896), 0), out=buf469)
            del arg298_1
            del buf467
            buf470 = reinterpret_tensor(buf468, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf468  # reuse
            # Topologically Sorted Source Nodes: [linear_144, silu_20, linear_145, mul_190], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf470, buf469, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg299_1, (896, 4864), (4864, 1))
            buf471 = empty_strided_cuda((s50, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_144, silu_20, linear_145, mul_190, down_proj_20], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf470, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg299_1, (4864, 896), (1, 4864), 0), out=buf471)
            del arg299_1
            assert_size_stride(arg301_1, (896, ), (1, ))
            assert_size_stride(arg300_1, (), ())
            buf473 = empty_strided_cuda((1, s50, 896), (896*s50, 896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, attn_output_83, hidden_states_205, down_proj_20, hidden_states_209, pow_43, variance_42, rsqrt_42, hidden_states_211, hidden_states_212], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf443, buf449, buf465, buf471, arg301_1, arg300_1.item(), buf473, s50, 896, stream=raw_stream0)
            del arg300_1
            del arg301_1
            assert_size_stride(arg303_1, (896, ), (1, ))
            assert_size_stride(arg302_1, (896, 896), (896, 1))
            buf474 = empty_strided_cuda((s50, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_147], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg303_1, reinterpret_tensor(buf473, (s50, 896), (896, 1), 0), reinterpret_tensor(arg302_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf474)
            del arg302_1
            del arg303_1
            assert_size_stride(arg305_1, (128, ), (1, ))
            assert_size_stride(arg304_1, (128, 896), (896, 1))
            buf475 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_148], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg305_1, reinterpret_tensor(buf473, (s50, 896), (896, 1), 0), reinterpret_tensor(arg304_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf475)
            del arg304_1
            del arg305_1
            buf476 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_24, sin_24, linear_148, view_64, key_states_21, mul_195, x2_43, neg_43, x1_43, cat_86, mul_196, k_embed_21], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf475, arg4_1, arg5_1.item(), buf476, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg307_1, (128, ), (1, ))
            assert_size_stride(arg306_1, (128, 896), (896, 1))
            buf477 = buf475; del buf475  # reuse
            # Topologically Sorted Source Nodes: [linear_149], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg307_1, reinterpret_tensor(buf473, (s50, 896), (896, 1), 0), reinterpret_tensor(arg306_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf477)
            del arg306_1
            del arg307_1
            buf478 = reinterpret_tensor(buf473, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf473  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_147, view_63, query_states_21, cos_24, mul_193, x2_42, neg_42, x1_42, cat_85, sin_24, mul_194, q_embed_21, getitem_821, hidden_states_213, key_21, linear_149, view_65, value_states_21, getitem_826, hidden_states_214, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf474, arg4_1, arg5_1.item(), buf478, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf479 = reinterpret_tensor(buf474, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf474  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_147, view_63, query_states_21, cos_24, mul_193, x2_42, neg_42, x1_42, cat_85, sin_24, mul_194, q_embed_21, getitem_821, hidden_states_213, key_21, linear_149, view_65, value_states_21, getitem_826, hidden_states_214, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf476, buf479, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf480 = empty_strided_cuda((1, 14, s50, 64), (896*s50, 64*s50, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_147, view_63, query_states_21, cos_24, mul_193, x2_42, neg_42, x1_42, cat_85, sin_24, mul_194, q_embed_21, getitem_821, hidden_states_213, key_21, linear_149, view_65, value_states_21, getitem_826, hidden_states_214, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf477, buf480, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf481 = buf459; del buf459  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_147, view_63, query_states_21, cos_24, mul_193, x2_42, neg_42, x1_42, cat_85, sin_24, mul_194, q_embed_21, getitem_821, hidden_states_213, key_21, linear_149, view_65, value_states_21, getitem_826, hidden_states_214, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf481, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_147, view_63, query_states_21, cos_24, mul_193, x2_42, neg_42, x1_42, cat_85, sin_24, mul_194, q_embed_21, getitem_821, hidden_states_213, key_21, linear_149, view_65, value_states_21, getitem_826, hidden_states_214, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf482 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf478, buf479, buf480, reinterpret_tensor(buf481, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf478
            del buf479
            buf483 = buf482[0]
            assert_size_stride(buf483, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf483, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf482
            assert_size_stride(arg308_1, (896, 896), (896, 1))
            buf487 = reinterpret_tensor(buf480, (s50, 896), (896, 1), 0); del buf480  # reuse
            # Topologically Sorted Source Nodes: [transpose_88, reshape_65, attn_output_87], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf483, (s50, 896), (896, 1), 0), reinterpret_tensor(arg308_1, (896, 896), (1, 896), 0), out=buf487)
            del arg308_1
            assert_size_stride(arg310_1, (896, ), (1, ))
            assert_size_stride(arg309_1, (), ())
            buf488 = buf443; del buf443  # reuse
            buf490 = reinterpret_tensor(buf483, (1, s50, 896), (896*s50, 896, 1), 0); del buf483  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, attn_output_83, hidden_states_205, down_proj_20, hidden_states_209, attn_output_87, hidden_states_215, pow_44, variance_43, rsqrt_43, hidden_states_217, hidden_states_218], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf488, buf449, buf465, buf471, buf487, arg310_1, arg309_1.item(), buf490, s50, 896, stream=raw_stream0)
            del arg309_1
            del arg310_1
            del buf449
            del buf465
            del buf471
            assert_size_stride(arg311_1, (4864, 896), (896, 1))
            buf491 = reinterpret_tensor(buf470, (s50, 4864), (4864, 1), 0); del buf470  # reuse
            # Topologically Sorted Source Nodes: [pow_44, variance_43, rsqrt_43, hidden_states_217, hidden_states_218, linear_151], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf490, (s50, 896), (896, 1), 0), reinterpret_tensor(arg311_1, (896, 4864), (1, 896), 0), out=buf491)
            del arg311_1
            assert_size_stride(arg312_1, (4864, 896), (896, 1))
            buf492 = buf469; del buf469  # reuse
            # Topologically Sorted Source Nodes: [linear_152], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf490, (s50, 896), (896, 1), 0), reinterpret_tensor(arg312_1, (896, 4864), (1, 896), 0), out=buf492)
            del arg312_1
            buf493 = reinterpret_tensor(buf491, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf491  # reuse
            # Topologically Sorted Source Nodes: [linear_151, silu_21, linear_152, mul_199], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf493, buf492, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg313_1, (896, 4864), (4864, 1))
            buf494 = reinterpret_tensor(buf490, (s50, 896), (896, 1), 0); del buf490  # reuse
            # Topologically Sorted Source Nodes: [linear_151, silu_21, linear_152, mul_199, down_proj_21], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf493, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg313_1, (4864, 896), (1, 4864), 0), out=buf494)
            del arg313_1
            assert_size_stride(arg315_1, (896, ), (1, ))
            assert_size_stride(arg314_1, (), ())
            buf496 = reinterpret_tensor(buf487, (1, s50, 896), (896*s50, 896, 1), 0); del buf487  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, pow_45, variance_44, rsqrt_44, hidden_states_221, hidden_states_222], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf488, buf494, arg315_1, arg314_1.item(), buf496, s50, 896, stream=raw_stream0)
            del arg314_1
            del arg315_1
            assert_size_stride(arg317_1, (896, ), (1, ))
            assert_size_stride(arg316_1, (896, 896), (896, 1))
            buf497 = empty_strided_cuda((s50, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_154], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg317_1, reinterpret_tensor(buf496, (s50, 896), (896, 1), 0), reinterpret_tensor(arg316_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf497)
            del arg316_1
            del arg317_1
            assert_size_stride(arg319_1, (128, ), (1, ))
            assert_size_stride(arg318_1, (128, 896), (896, 1))
            buf498 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_155], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg319_1, reinterpret_tensor(buf496, (s50, 896), (896, 1), 0), reinterpret_tensor(arg318_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf498)
            del arg318_1
            del arg319_1
            buf499 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_25, sin_25, linear_155, view_67, key_states_22, mul_204, x2_45, neg_45, x1_45, cat_90, mul_205, k_embed_22], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf498, arg4_1, arg5_1.item(), buf499, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg321_1, (128, ), (1, ))
            assert_size_stride(arg320_1, (128, 896), (896, 1))
            buf500 = buf498; del buf498  # reuse
            # Topologically Sorted Source Nodes: [linear_156], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg321_1, reinterpret_tensor(buf496, (s50, 896), (896, 1), 0), reinterpret_tensor(arg320_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf500)
            del arg320_1
            del arg321_1
            buf501 = reinterpret_tensor(buf496, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf496  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_154, view_66, query_states_22, cos_25, mul_202, x2_44, neg_44, x1_44, cat_89, sin_25, mul_203, q_embed_22, getitem_858, hidden_states_223, key_22, linear_156, view_68, value_states_22, getitem_863, hidden_states_224, value_22, attn_output_88], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf497, arg4_1, arg5_1.item(), buf501, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            buf502 = reinterpret_tensor(buf497, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf497  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_154, view_66, query_states_22, cos_25, mul_202, x2_44, neg_44, x1_44, cat_89, sin_25, mul_203, q_embed_22, getitem_858, hidden_states_223, key_22, linear_156, view_68, value_states_22, getitem_863, hidden_states_224, value_22, attn_output_88], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf499, buf502, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf503 = empty_strided_cuda((1, 14, s50, 64), (896*s50, 64*s50, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_154, view_66, query_states_22, cos_25, mul_202, x2_44, neg_44, x1_44, cat_89, sin_25, mul_203, q_embed_22, getitem_858, hidden_states_223, key_22, linear_156, view_68, value_states_22, getitem_863, hidden_states_224, value_22, attn_output_88], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf500, buf503, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf504 = buf481; del buf481  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_154, view_66, query_states_22, cos_25, mul_202, x2_44, neg_44, x1_44, cat_89, sin_25, mul_203, q_embed_22, getitem_858, hidden_states_223, key_22, linear_156, view_68, value_states_22, getitem_863, hidden_states_224, value_22, attn_output_88], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf504, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_154, view_66, query_states_22, cos_25, mul_202, x2_44, neg_44, x1_44, cat_89, sin_25, mul_203, q_embed_22, getitem_858, hidden_states_223, key_22, linear_156, view_68, value_states_22, getitem_863, hidden_states_224, value_22, attn_output_88], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf505 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf501, buf502, buf503, reinterpret_tensor(buf504, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf501
            del buf502
            del buf504
            buf506 = buf505[0]
            assert_size_stride(buf506, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf506, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf505
            assert_size_stride(arg322_1, (896, 896), (896, 1))
            buf510 = reinterpret_tensor(buf503, (s50, 896), (896, 1), 0); del buf503  # reuse
            # Topologically Sorted Source Nodes: [transpose_92, reshape_68, attn_output_91], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf506, (s50, 896), (896, 1), 0), reinterpret_tensor(arg322_1, (896, 896), (1, 896), 0), out=buf510)
            del arg322_1
            assert_size_stride(arg324_1, (896, ), (1, ))
            assert_size_stride(arg323_1, (), ())
            buf512 = reinterpret_tensor(buf506, (1, s50, 896), (896*s50, 896, 1), 0); del buf506  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, attn_output_91, hidden_states_225, pow_46, variance_45, rsqrt_45, hidden_states_227, hidden_states_228], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_10.run(buf488, buf494, buf510, arg324_1, arg323_1.item(), buf512, s50, 896, stream=raw_stream0)
            del arg323_1
            del arg324_1
            assert_size_stride(arg325_1, (4864, 896), (896, 1))
            buf513 = reinterpret_tensor(buf493, (s50, 4864), (4864, 1), 0); del buf493  # reuse
            # Topologically Sorted Source Nodes: [linear_158], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf512, (s50, 896), (896, 1), 0), reinterpret_tensor(arg325_1, (896, 4864), (1, 896), 0), out=buf513)
            del arg325_1
            assert_size_stride(arg326_1, (4864, 896), (896, 1))
            buf514 = buf492; del buf492  # reuse
            # Topologically Sorted Source Nodes: [linear_159], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf512, (s50, 896), (896, 1), 0), reinterpret_tensor(arg326_1, (896, 4864), (1, 896), 0), out=buf514)
            del arg326_1
            del buf512
            buf515 = reinterpret_tensor(buf513, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf513  # reuse
            # Topologically Sorted Source Nodes: [linear_158, silu_22, linear_159, mul_208], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf515, buf514, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            assert_size_stride(arg327_1, (896, 4864), (4864, 1))
            buf516 = empty_strided_cuda((s50, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_158, silu_22, linear_159, mul_208, down_proj_22], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf515, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg327_1, (4864, 896), (1, 4864), 0), out=buf516)
            del arg327_1
            assert_size_stride(arg329_1, (896, ), (1, ))
            assert_size_stride(arg328_1, (), ())
            buf518 = empty_strided_cuda((1, s50, 896), (896*s50, 896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, attn_output_91, hidden_states_225, down_proj_22, hidden_states_229, pow_47, variance_46, rsqrt_46, hidden_states_231, hidden_states_232], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf488, buf494, buf510, buf516, arg329_1, arg328_1.item(), buf518, s50, 896, stream=raw_stream0)
            del arg328_1
            del arg329_1
            assert_size_stride(arg331_1, (896, ), (1, ))
            assert_size_stride(arg330_1, (896, 896), (896, 1))
            buf519 = empty_strided_cuda((s50, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_161], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg331_1, reinterpret_tensor(buf518, (s50, 896), (896, 1), 0), reinterpret_tensor(arg330_1, (896, 896), (1, 896), 0), alpha=1, beta=1, out=buf519)
            del arg330_1
            del arg331_1
            assert_size_stride(arg333_1, (128, ), (1, ))
            assert_size_stride(arg332_1, (128, 896), (896, 1))
            buf520 = empty_strided_cuda((s50, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_162], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg333_1, reinterpret_tensor(buf518, (s50, 896), (896, 1), 0), reinterpret_tensor(arg332_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf520)
            del arg332_1
            del arg333_1
            buf521 = empty_strided_cuda((1, 2, s50, 64), (128*s50, 64, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, cos_26, sin_26, linear_162, view_70, key_states_23, mul_213, x2_47, neg_47, x1_47, cat_94, mul_214, k_embed_23], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.view, aten.mul, aten.slice, aten.neg]
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel = 128*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1.run(buf520, arg4_1, arg5_1.item(), buf521, triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_1_xnumel, stream=raw_stream0)
            assert_size_stride(arg335_1, (128, ), (1, ))
            assert_size_stride(arg334_1, (128, 896), (896, 1))
            buf522 = buf520; del buf520  # reuse
            # Topologically Sorted Source Nodes: [linear_163], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg335_1, reinterpret_tensor(buf518, (s50, 896), (896, 1), 0), reinterpret_tensor(arg334_1, (896, 128), (1, 896), 0), alpha=1, beta=1, out=buf522)
            del arg334_1
            del arg335_1
            buf523 = reinterpret_tensor(buf518, (1, 14, s50, 64), (896*s50, 64, 896, 1), 0); del buf518  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_161, view_69, query_states_23, cos_26, mul_211, x2_46, neg_46, x1_46, cat_93, sin_26, mul_212, q_embed_23, getitem_895, hidden_states_233, key_23, linear_163, view_71, value_states_23, getitem_900, hidden_states_234, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2.run(buf519, arg4_1, arg5_1.item(), buf523, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_2_xnumel, stream=raw_stream0)
            del arg4_1
            del arg5_1
            buf524 = reinterpret_tensor(buf519, (1, 14, s50, 64), (896*s50, 64*s50, 64, 1), 0); del buf519  # reuse
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_161, view_69, query_states_23, cos_26, mul_211, x2_46, neg_46, x1_46, cat_93, sin_26, mul_212, q_embed_23, getitem_895, hidden_states_233, key_23, linear_163, view_71, value_states_23, getitem_900, hidden_states_234, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf521, buf524, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf525 = empty_strided_cuda((1, 14, s50, 64), (896*s50, 64*s50, 64, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_161, view_69, query_states_23, cos_26, mul_211, x2_46, neg_46, x1_46, cat_93, sin_26, mul_212, q_embed_23, getitem_895, hidden_states_233, key_23, linear_163, view_71, value_states_23, getitem_900, hidden_states_234, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel = 896*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3.run(buf522, buf525, s50, ps0, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_3_xnumel, stream=raw_stream0)
            buf526 = empty_strided_cuda((1, 1, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_161, view_69, query_states_23, cos_26, mul_211, x2_46, neg_46, x1_46, cat_93, sin_26, mul_212, q_embed_23, getitem_895, hidden_states_233, key_23, linear_163, view_71, value_states_23, getitem_900, hidden_states_234, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel = s50*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4.run(buf526, s50, triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bmm_cat_clone_constant_pad_nd_cos_expand_le_mul_neg_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_4_xnumel, stream=raw_stream0)
            # Topologically Sorted Source Nodes: [getitem_13, expand_1, arange, position_ids, position_ids_1, getitem_16, position_ids_expanded, matmul, freqs, emb, cos, sin, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, attention_mask, attention_mask_1, linear_161, view_69, query_states_23, cos_26, mul_211, x2_46, neg_46, x1_46, cat_93, sin_26, mul_212, q_embed_23, getitem_895, hidden_states_233, key_23, linear_163, view_71, value_states_23, getitem_900, hidden_states_234, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.sin, aten.le, aten.view, aten.mul, aten.slice, aten.neg, aten.clone, aten._unsafe_view, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            buf527 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf523, buf524, buf525, reinterpret_tensor(buf526, (1, 14, s50, s50), (8*s50*((7 + s50) // 8), 0, 8*((7 + s50) // 8), 1), 0), False, scale=0.125)
            del buf523
            del buf524
            del buf526
            buf528 = buf527[0]
            assert_size_stride(buf528, (1, 14, s50, 64), (896*s50, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf528, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf527
            assert_size_stride(arg336_1, (896, 896), (896, 1))
            buf532 = reinterpret_tensor(buf525, (s50, 896), (896, 1), 0); del buf525  # reuse
            # Topologically Sorted Source Nodes: [transpose_96, reshape_71, attn_output_95], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf528, (s50, 896), (896, 1), 0), reinterpret_tensor(arg336_1, (896, 896), (1, 896), 0), out=buf532)
            del arg336_1
            assert_size_stride(arg338_1, (896, ), (1, ))
            assert_size_stride(arg337_1, (), ())
            buf533 = buf488; del buf488  # reuse
            buf535 = reinterpret_tensor(buf528, (1, s50, 896), (896*s50, 896, 1), 0); del buf528  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, attn_output_91, hidden_states_225, down_proj_22, hidden_states_229, attn_output_95, hidden_states_235, pow_48, variance_47, rsqrt_47, hidden_states_237, hidden_states_238], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf533, buf494, buf510, buf516, buf532, arg338_1, arg337_1.item(), buf535, s50, 896, stream=raw_stream0)
            del arg337_1
            del arg338_1
            del buf494
            del buf510
            del buf516
            del buf532
            assert_size_stride(arg339_1, (4864, 896), (896, 1))
            buf536 = reinterpret_tensor(buf515, (s50, 4864), (4864, 1), 0); del buf515  # reuse
            # Topologically Sorted Source Nodes: [pow_48, variance_47, rsqrt_47, hidden_states_237, hidden_states_238, linear_165], Original ATen: [aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf535, (s50, 896), (896, 1), 0), reinterpret_tensor(arg339_1, (896, 4864), (1, 896), 0), out=buf536)
            del arg339_1
            assert_size_stride(arg340_1, (4864, 896), (896, 1))
            buf537 = buf514; del buf514  # reuse
            # Topologically Sorted Source Nodes: [linear_166], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf535, (s50, 896), (896, 1), 0), reinterpret_tensor(arg340_1, (896, 4864), (1, 896), 0), out=buf537)
            del arg340_1
            buf538 = reinterpret_tensor(buf536, (1, s50, 4864), (4864*s50, 4864, 1), 0); del buf536  # reuse
            # Topologically Sorted Source Nodes: [linear_165, silu_23, linear_166, mul_217], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            triton_poi_fused__unsafe_view_mul_silu_6_xnumel = 4864*s50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_6.run(buf538, buf537, triton_poi_fused__unsafe_view_mul_silu_6_xnumel, stream=raw_stream0)
            del buf537
            assert_size_stride(arg341_1, (896, 4864), (4864, 1))
            buf539 = reinterpret_tensor(buf535, (s50, 896), (896, 1), 0); del buf535  # reuse
            # Topologically Sorted Source Nodes: [linear_165, silu_23, linear_166, mul_217, down_proj_23], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf538, (s50, 4864), (4864, 1), 0), reinterpret_tensor(arg341_1, (4864, 896), (1, 4864), 0), out=buf539)
            del arg341_1
            del buf538
            assert_size_stride(arg343_1, (896, ), (1, ))
            assert_size_stride(arg342_1, (), ())
            buf541 = buf533; del buf533  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, pow_49, variance_48, rsqrt_48, hidden_states_241, hidden_states_242], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf541, buf539, arg343_1, arg342_1.item(), s50, 896, stream=raw_stream0)
            del arg342_1
            del arg343_1
            del buf539
            buf542 = empty_strided_cuda((1, 151936), (151936, 1), torch.float32)
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, pow_49, variance_48, rsqrt_48, hidden_states_241, hidden_states_242, getitem_905, logits], Original ATen: [aten._unsafe_view, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.slice, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf541, (1, 896), (0, 1), (-896) + 896*s50), reinterpret_tensor(arg3_1, (896, 151936), (1, 896), 0), out=buf542)
            del arg3_1
            del buf541
        return (reinterpret_tensor(buf5, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf4, reinterpret_tensor(buf27, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf26, reinterpret_tensor(buf50, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf49, reinterpret_tensor(buf72, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf71, reinterpret_tensor(buf95, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf94, reinterpret_tensor(buf117, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf116, reinterpret_tensor(buf140, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf139, reinterpret_tensor(buf162, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf161, reinterpret_tensor(buf185, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf184, reinterpret_tensor(buf207, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf206, reinterpret_tensor(buf230, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf229, reinterpret_tensor(buf252, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf251, reinterpret_tensor(buf275, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf274, reinterpret_tensor(buf297, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf296, reinterpret_tensor(buf320, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf319, reinterpret_tensor(buf342, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf341, reinterpret_tensor(buf365, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf364, reinterpret_tensor(buf387, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf386, reinterpret_tensor(buf410, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf409, reinterpret_tensor(buf432, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf431, reinterpret_tensor(buf455, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf454, reinterpret_tensor(buf477, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf476, reinterpret_tensor(buf500, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf499, reinterpret_tensor(buf522, (1, 2, s50, 64), (128*s50, 64, 128, 1), 0), buf521, reinterpret_tensor(buf542, (1, 1, 151936), (151936, 151936, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 32
    arg1_1 = 44
    arg2_1 = rand_strided((1, 32), (44, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((151936, 896), (896, 1), device='cuda:0', dtype=torch.float32)
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
    arg14_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg16_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg21_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg30_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg35_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg44_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg49_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg58_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg63_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg72_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg77_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg98_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg100_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg105_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg114_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg119_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg128_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg133_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg142_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg147_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg156_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg161_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
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
    arg182_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg184_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg189_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg198_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg203_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg212_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg217_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg226_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg231_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg240_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg245_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg254_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg259_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg268_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg273_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg282_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg287_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg296_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg301_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg310_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg315_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg324_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg329_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg338_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((), (), device='cpu', dtype=torch.float64)
    arg343_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = 1
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
