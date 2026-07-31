# AOT ID: ['4_inference']
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


# kernel path: <inductor-cache>/mo/cmoj2fbzo4rownmjgzyopvf7ckhtzhcvt3q37huopylk2fj5onle.py
# Topologically Sorted Source Nodes: [linear, silu, linear_1, gated], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   gated => mul
#   linear => view_1
#   linear_1 => view_3
#   silu => add, convert_element_type_2, convert_element_type_3, div, exp, neg
# Graph fragment:
#   %mm : Tensor "f16[512, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm]
#   %mm_1 : Tensor "f16[512, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %view_1 : Tensor "f16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [4, 128, 2048]), kwargs = {})
#   %convert_element_type_2 : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_2,), kwargs = {})
#   %exp : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_2, %add), kwargs = {})
#   %convert_element_type_3 : Tensor "f16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
#   %view_3 : Tensor "f16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [4, 128, 2048]), kwargs = {})
#   %mul : Tensor "f16[4, 128, 2048][262144, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, %view_3), kwargs = {})
#   return %mul
triton_poi_fused__unsafe_view_mul_silu_0 = async_compile.triton('triton_poi_fused__unsafe_view_mul_silu_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_mul_silu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'EB4FEFA44B03C7DE305914399B14B7941D86D9066490932B6E56BBB370DA33AB', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8388608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_silu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(in_out_ptr0 + (x0), tmp9, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        assert_size_stride(arg1_1, (4, 128, 768), (98304, 768, 1))
        assert_size_stride(arg0_1, (2048, 768), (768, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg1_1 = copy_misaligned(arg1_1)
            buf0 = empty_strided_cuda((512, 2048), (2048, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(arg1_1, (512, 768), (768, 1), 0), reinterpret_tensor(arg0_1, (768, 2048), (1, 768), 0), out=buf0)
            del arg0_1
            assert_size_stride(arg2_1, (2048, 768), (768, 1))
            buf1 = empty_strided_cuda((512, 2048), (2048, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(arg1_1, (512, 768), (768, 1), 0), reinterpret_tensor(arg2_1, (768, 2048), (1, 768), 0), out=buf1)
            del arg1_1
            del arg2_1
            buf2 = reinterpret_tensor(buf0, (4, 128, 2048), (262144, 2048, 1), 0); del buf0  # reuse
            # Topologically Sorted Source Nodes: [linear, silu, linear_1, gated], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_0.run(buf2, buf1, 1048576, stream=raw_stream0)
            del buf1
            assert_size_stride(arg3_1, (768, 2048), (2048, 1))
            buf3 = empty_strided_cuda((512, 768), (768, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear, silu, linear_1, gated, linear_2], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf2, (512, 2048), (2048, 1), 0), reinterpret_tensor(arg3_1, (2048, 768), (1, 2048), 0), out=buf3)
            del arg3_1
            del buf2
        return (reinterpret_tensor(buf3, (4, 128, 768), (98304, 768, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((4, 128, 768), (98304, 768, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((2048, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((768, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
