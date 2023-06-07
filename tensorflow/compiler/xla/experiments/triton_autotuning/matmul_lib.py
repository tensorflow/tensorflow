# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Library for running matmuls."""
import itertools
import logging
import math
import typing

import torch
import tqdm
import triton
import triton.language as tl

LOG = logging.getLogger(__name__)

logging.basicConfig(
    format=(
        '%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d]'
        ' %(threadName)15s: %(message)s'
    ),
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
)


class MatmulTiling(typing.NamedTuple):
  """Tiling parameterization of a matmul."""

  BLOCK_M: int
  BLOCK_N: int
  BLOCK_K: int
  SPLIT_K: int
  num_stages: int
  num_warps: int


class MatmulSize(typing.NamedTuple):
  """[M, K] @ [K, N]."""

  M: int
  N: int
  K: int
  quantized_lhs: int


class MatmulTiming(typing.NamedTuple):
  """Timing result of a configuration."""

  dims: MatmulSize
  tiling: MatmulTiling
  min_time_ms: float


def parse_int_list(v: str) -> typing.List[int]:
  """Converts a string of comma-separated ints into a list of strings."""
  return list(map(int, v.split(',')))


def generate_tiling_configs(
    tilings_m: typing.List[int],
    tilings_n: typing.List[int],
    tilings_k: typing.List[int],
    split_ks: typing.List[int],
    num_stages: typing.List[int],
    num_warps: typing.List[int],
) -> typing.Iterator[MatmulTiling]:
  """Generate a list of matmul configs to evaluate."""
  product = itertools.product(
      tilings_m,
      tilings_n,
      tilings_k,
      split_ks,
      num_stages,
      num_warps,
  )
  return [MatmulTiling(*p) for p in product]


@triton.jit
def _matmul_kernel(
    lhs,
    rhs,
    out,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    split_k: tl.constexpr,
    acc_ty: tl.constexpr,
    # Workaround for a bug in Triton cache:
    # force recompilation on different num_warps/num_stages.
    force_num_warps: tl.constexpr,  # pylint: disable=unused-argument
    force_num_stages: tl.constexpr,  # pylint: disable=unused-argument
):
  """Computes a block-level matmul."""
  even_k = k % (block_k * split_k) == 0
  pid0 = tl.program_id(0)
  pid1 = tl.program_id(1)
  pid2 = tl.program_id(2)
  grid_m = (m + block_m - 1) // block_m
  grid_n = (n + block_n - 1) // block_n
  # re-order program ID for better L2 performance
  width = group_m * grid_n
  group_id = pid0 // width
  group_size = min(grid_m - group_id * group_m, group_m)
  pid_m = group_id * group_m + pid0 % group_size
  pid_n = (pid0 % width) // group_size
  rm = pid_m * block_m + tl.arange(0, block_m)
  rn = pid_n * block_n + tl.arange(0, block_n)
  ram = tl.max_contiguous(tl.multiple_of(rm % m, block_m), block_m)
  rbn = tl.max_contiguous(tl.multiple_of(rn % n, block_n), block_n)
  rk = pid1 * block_k + tl.arange(0, block_k)
  lhs += ram[:, None] * stride_am + rk[None, :] * stride_ak + pid2 * m * k
  rhs += rk[:, None] * stride_bk + rbn[None, :] * stride_bn
  acc = tl.zeros((block_m, block_n), dtype=acc_ty)
  # for ki in range(0, k, block_k * split_k):  # pytype: disable=wrong-arg-types
  for ki in range(k, 0, -block_k * split_k):  # pytype: disable=wrong-arg-types
    if even_k:
      a = tl.load(lhs)
      b = tl.load(rhs)
    else:
      a = tl.load(lhs, mask=rk[None, :] < ki, other=0)
      b = tl.load(rhs, mask=rk[:, None] < ki, other=0)
    casted_a = a.to(out.dtype.element_ty)
    casted_b = b.to(out.dtype.element_ty)
    acc += tl.dot(casted_a, casted_b, allow_tf32=True)
    lhs += block_k * split_k * stride_ak
    rhs += block_k * split_k * stride_bk
  acc = acc.to(out.dtype.element_ty)
  # rematerialize rm and rn to save registers
  rm = pid_m * block_m + tl.arange(0, block_m)
  rn = pid_n * block_n + tl.arange(0, block_n)
  out += rm[:, None] * stride_cm + rn[None, :] * stride_cn + pid2 * m * n
  out += m * n * pid1
  mask = (rm < m)[:, None] & (rn < n)[None, :]
  tl.store(out, acc, mask=mask)


@triton.jit
def _reduce_kernel(
    src,
    dest,
    row_size: tl.constexpr,
    col_size: tl.constexpr,
    row_block_size: tl.constexpr,
):
  """Computes a column reduction."""
  pid0 = tl.program_id(0)
  idx = pid0 * row_block_size + tl.arange(0, row_block_size)
  src += idx
  acc = tl.zeros((row_block_size,), dtype=dest.dtype.element_ty)
  for _ in range(col_size):
    acc += tl.load(src, mask=idx < row_size, other=0)
    src += row_size
  tl.store(dest + idx, acc, mask=idx < row_size)


def benchmark_matmul_tiling(
    dims: MatmulSize,
    tiling: MatmulTiling,
    s: torch.cuda.Stream,
    shared_stream: torch.cuda.Stream,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scratchpad: torch.Tensor,  # Largest size: c * SPLIT_K
    repetitions_ms: int,
    debug=False,
) -> typing.Optional[MatmulTiming]:
  """Benchmarks a single matmul tiling."""
  grid = lambda META: (  # pylint: disable=g-long-lambda
      triton.cdiv(dims.M, tiling.BLOCK_M) * triton.cdiv(dims.N, tiling.BLOCK_N),
      tiling.SPLIT_K,
      1,  # batch
  )

  def run_matmul():
    used_output = c if tiling.SPLIT_K == 1 else scratchpad
    _matmul_kernel[grid](
        a,
        b,
        used_output,
        m=int(dims.M),
        n=int(dims.N),
        k=int(dims.K),
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        block_m=int(tiling.BLOCK_M),
        block_n=int(tiling.BLOCK_N),
        block_k=int(tiling.BLOCK_K),
        group_m=8,
        split_k=tiling.SPLIT_K,
        num_warps=tiling.num_warps,
        num_stages=tiling.num_stages,
        force_num_warps=tiling.num_warps,
        force_num_stages=tiling.num_stages,
        acc_ty=tl.float32,
    )
    if tiling.SPLIT_K != 1:
      # Run reduction kernel.
      _reduce_kernel[(triton.cdiv(dims.M * dims.N, 1024),)](
          scratchpad,
          c,
          row_size=int(dims.M),
          col_size=tiling.SPLIT_K,
          num_stages=1,
          num_warps=1024 // 32,
          row_block_size=1024,
      )

  for dim in ['M', 'N', 'K']:
    next_pow2 = lambda v: 2 ** int(math.ceil(math.log2(v)))
    dim_size: int = getattr(dims, dim)
    if dim == 'K':
      dim_size = math.ceil(dim_size / tiling.SPLIT_K)
    tile_size = getattr(tiling, f'BLOCK_{dim}')
    if next_pow2(dim_size) < tile_size:
      if debug:
        LOG.error(
            'Tile %s larger than the dimension %s (%s)',
            tile_size,
            dim,
            dim_size,
        )
      return None

  if tiling.BLOCK_M * tiling.BLOCK_N > 131072:
    if debug:
      LOG.error('Overly large tile')
    return None

  # TODO(cheshire): Compilation time is huge for such tiles.
  if tiling.BLOCK_M > 512 or tiling.BLOCK_N > 512:
    if debug:
      LOG.error('Overly large tile')
    return None

  max_shared_memory = triton.runtime.driver.utils.get_device_properties(
      torch.cuda.current_device()
  )['max_shared_mem']

  required_shared_memory = (
      (tiling.BLOCK_M + tiling.BLOCK_N)
      * tiling.BLOCK_K
      * tiling.num_stages
      * b.element_size()
  )
  if required_shared_memory > max_shared_memory:
    if debug:
      LOG.error('Skipping %s due to exceeding shmem bound', tiling)
    return None
  with torch.cuda.stream(s):
    try:
      run_matmul()  # Warmup on our own stream.
    except Exception as exc:
      LOG.error('%s for %s generated %s', tiling, dims, exc, exc_info=True)
      raise

  # Use shared stream to take actual measurements.
  with torch.cuda.stream(shared_stream):
    try:
      percentiles = triton.testing.do_bench(
          run_matmul,
          warmup=0,
          rep=repetitions_ms,
          quantiles=(0.001, 0.1, 0.5, 0.9),
      )
      min_ms = percentiles[0]
    except Exception as exc:
      LOG.error('%s for %s generated %s', tiling, dims, exc, exc_info=True)
      raise
    return MatmulTiming(dims, tiling, min_ms)


def benchmark_cublas(dims: MatmulSize) -> MatmulTiming:
  """Measure cublas performance."""
  a = torch.randn(dims.M, dims.K, device='cuda', dtype=torch.bfloat16)
  b = torch.randn(dims.K, dims.N, device='cuda', dtype=torch.bfloat16)
  run_matmul = lambda: torch.matmul(a, b)
  percentiles = triton.testing.do_bench(
      run_matmul, warmup=0, rep=300, quantiles=(0.001, 0.1, 0.5, 0.9)
  )
  min_ms = percentiles[0]
  return min_ms


def benchmark_matmul(
    dims: MatmulSize,
    pbar: tqdm.std.tqdm,
    shared_stream: torch.cuda.Stream,
    tilings: typing.List[MatmulTiling],
    repetitions_ms: int,
    debug=False,
) -> typing.Sequence[MatmulTiming]:
  """For a given matmul configuration, benchmark it.

  Args:
    dims: the dimensions of the matmul
    pbar: a progress bar
    shared_stream: stream to execute benchmarks on
    tilings: list of tilings to benchmark
    repetitions_ms: how many milliseconds to spend running each configuration
    debug: whether to print debug output

  Returns:
    A sequence of matmul timings.
  """
  out: list[MatmulTiming] = []
  largest_splitk = max(tilings, key=lambda t: t.SPLIT_K).SPLIT_K

  s = torch.cuda.Stream()

  # Use our own stream for compilation.
  with torch.cuda.stream(s):
    if dims.quantized_lhs:
      a = torch.randint(
          0, 128, (dims.M, dims.K), device='cuda', dtype=torch.int8
      )
    else:
      a = torch.randn(dims.M, dims.K, device='cuda', dtype=torch.bfloat16)

    b = torch.randn(dims.K, dims.N, device='cuda', dtype=torch.bfloat16)
    assert a.shape[1] == b.shape[0], 'incompatible dimensions'
    assert a.is_contiguous(), 'matrix A must be contiguous'
    assert b.is_contiguous(), 'matrix B must be contiguous'
    c = torch.empty((dims.M, dims.N), device=a.device, dtype=torch.bfloat16)
    scratchpad = torch.empty(
        (largest_splitk, dims.M, dims.N), device=a.device, dtype=torch.bfloat16
    )

  LOG.info('Autotuning for %s', dims)

  for tiling in tilings:
    pbar.update(1)

    timing = benchmark_matmul_tiling(
        dims,
        tiling,
        s,
        shared_stream,
        a,
        b,
        c,
        scratchpad,
        repetitions_ms=repetitions_ms,
        debug=debug,
    )
    if not timing:
      continue

    out.append(timing)
  return out


def print_roofline_performance(dims: MatmulSize, time_ms: float):
  """Print theoretical roofline model performance."""
  gbps: float = triton.testing.get_dram_gbps()
  tflops: float = triton.testing.get_max_tensorcore_tflops(torch.bfloat16)
  lhs_size_bytes = dims.M * dims.K
  rhs_size_bytes = dims.K * dims.N * 2
  out_size_bytes = dims.M * dims.N * 2

  size_gb = (lhs_size_bytes + rhs_size_bytes + out_size_bytes) / 1e9
  roofline_time_ms_bw = (size_gb / gbps) * 1e3
  roofline_time_ms_flops = 2 * (dims.M * dims.N * dims.K) / (tflops * 1e9)

  best_time_ms = max(roofline_time_ms_bw, roofline_time_ms_flops)
  bound = (
      'bandwidth' if roofline_time_ms_bw > roofline_time_ms_flops else 'flops'
  )

  print(
      f'Percentage of roofline: {(best_time_ms * 100 / time_ms):0.4f}%'
      f' ({bound} bound)'
  )

  print(f'Roofline time if bandwidth bound: {roofline_time_ms_bw:0.4f}ms')
  print(f'Roofline time if flops bound: {roofline_time_ms_flops:0.4f}ms')
