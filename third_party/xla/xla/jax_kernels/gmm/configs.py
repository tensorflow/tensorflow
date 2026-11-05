# Copyright 2026 The OpenXLA Authors.
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

"""Canonical shape and tiling configurations for jax_k forward ragged_dot (GMM) kernels."""

from collections.abc import Sequence
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class GmmBenchmarkConfig:
  """Configuration for a forward ragged_dot (GMM) benchmark workload."""

  name: str
  m: int
  k: int
  n: int
  g: int
  dtype: Any = jnp.bfloat16

  @property
  def lhs_shape(self) -> tuple[int, int]:
    return (self.m, self.k)

  @property
  def rhs_shape(self) -> tuple[int, int, int]:
    return (self.g, self.k, self.n)

  @property
  def out_shape(self) -> tuple[int, int]:
    return (self.m, self.n)

  @property
  def flops(self) -> float:
    return float(2 * self.m * self.k * self.n)

  @property
  def bytes_moved(self) -> float:
    elem_bytes = 2 if self.dtype == jnp.bfloat16 else 4
    return float(
        (self.m * self.k + self.g * self.k * self.n + self.m * self.n)
        * elem_bytes
    )


GmmShapeConfig = GmmBenchmarkConfig

SHAPES: tuple[tuple[str, int, int, int, int], ...] = (
    ("moe_m8_k4096_n4096_g8", 8, 4096, 4096, 8),
    ("moe_m4096_k4096_n4096_g8", 4096, 4096, 4096, 8),
    ("moe_m8192_k4096_n14336_g8", 8192, 4096, 14336, 8),
    ("ds3_m131072_k512_n7168_g256", 131072, 512, 7168, 256),
    ("ds3_m131072_k1024_n7168_g256", 131072, 1024, 7168, 256),
    ("gpt_m131072_k1536_n4096_g128", 131072, 1536, 4096, 128),
    ("gpt_m131072_k2048_n768_g128", 131072, 2048, 768, 128),
    ("gpt_m131072_k4096_n1536_g128", 131072, 4096, 1536, 128),
    ("ds3_m131072_k7168_n512_g256", 131072, 7168, 512, 256),
    ("ds3_m131072_k7168_n1024_g256", 131072, 7168, 1024, 256),
    ("gpt_m131072_k7168_n2048_g128", 131072, 7168, 2048, 128),
    ("ds3_m262144_k256_n7168_g256", 262144, 256, 7168, 256),
    ("ds3_m262144_k512_n7168_g256", 262144, 512, 7168, 256),
    ("ds3_m262144_k1024_n7168_g256", 262144, 1024, 7168, 256),
    ("gpt_m262144_k1536_n4096_g128", 262144, 1536, 4096, 128),
    ("ds3_m262144_k7168_n512_g256", 262144, 7168, 512, 256),
    ("ds3_m262144_k7168_n1024_g256", 262144, 7168, 1024, 256),
    ("ds3_m262144_k7168_n2048_g256", 262144, 7168, 2048, 256),
    ("gpt_m327680_k2880_n2880_g128", 327680, 2880, 2880, 128),
    ("gpt_m393216_k768_n2048_g128", 393216, 768, 2048, 128),
    ("gpt_m524288_k1536_n4096_g128", 524288, 1536, 4096, 128),
)

GMM_BENCHMARK_CONFIGS: tuple[GmmBenchmarkConfig, ...] = tuple(
    GmmBenchmarkConfig(name=name, m=m, k=k, n=n, g=g)
    for name, m, k, n, g in SHAPES
)


@dataclasses.dataclass(frozen=True)
class RaggedDotTilingConfig:
  """Stage 1 jax_k tiling configuration for forward ragged_dot."""

  mt: int
  kt: int
  nt: int
  # Shorter M extents for per group tail tiles, ascending and below mt.
  m_buckets: tuple[int, ...] = ()

  @classmethod
  def from_shapes(
      cls, m: int, k: int, n: int, g: int
  ) -> "RaggedDotTilingConfig":
    """Selects VMEM aware (mt, kt, nt) tiles for forward ragged_dot."""
    if m < 256:
      return cls(
          mt=max(8, ((m + 7) // 8) * 8),
          kt=k,
          nt=2048 if (k == 4096 and n == 4096) else n,
      )
    m_bar = m // max(g, 1)
    if m_bar == 512:
      if k in (256, 512, 1024) and n == 7168:
        return cls(mt=576, kt=k, nt=7168)
      if k == 7168 and n == 512:
        return cls(mt=576, kt=k, nt=512, m_buckets=(512, 544))
      if k == 7168 and n == 1024:
        return cls(mt=576, kt=k, nt=1024, m_buckets=(512,))
      if k in (2048, 4096) and n in (768, 1536):
        return cls(mt=640, kt=k, nt=n)
      if k == 4096 and n == 4096:
        return cls(mt=544, kt=k, nt=2048)
    if m_bar == 1024:
      if k in (256, 512) and n == 7168:
        return cls(mt=1120, kt=k, nt=7168)
      if k == 1024 and n == 7168:
        return cls(mt=1088, kt=k, nt=3584)
      if k == 7168 and n == 512:
        return cls(mt=1120, kt=k, nt=512, m_buckets=(1008, 1056))
      if k == 7168 and n in (1024, 2048):
        return cls(mt=560, kt=k, nt=1024, m_buckets=(464, 496))
      if k == 4096 and n == 14336:
        return cls(mt=1056, kt=k, nt=1792)
      if k == 1536 and n == 4096:
        return cls(mt=1104, kt=k, nt=4096)
      if k == 4096 and n == 1536:
        return cls(mt=560, kt=k, nt=1536, m_buckets=(464, 496))
      if k == 2048 and n == 768:
        return cls(mt=1120, kt=k, nt=768)
    if m_bar == 2048 and k == 1536 and n == 4096:
      return cls(mt=2160, kt=k, nt=2048)
    if m_bar == 4096 and k == 1536 and n == 4096:
      return cls(mt=2128, kt=k, nt=2048)
    if m_bar == 2560 and k == 2880 and n in (2880, 2944, 3072):
      return cls(mt=1344, kt=k, nt=1536 if n in (2944, 3072) else 2880)
    if m_bar == 3072 and k == 768 and n == 2048:
      return cls(mt=3232, kt=k, nt=2048)
    if (
        m_bar >= 3072
        or (m_bar >= 1024 and n <= 1024 and k <= 2048)
        or (m_bar >= 1024 and n <= 512)
        or (m in (131072, 262144) and k == 256 and n in (3584, 7168))
    ):
      mt_cands = (2048, 1024, 512, 256, 128)
    elif m_bar >= 1024 or (k >= 4096 and n <= 512) or (n == 3584 and k <= 1024):
      mt_cands = (1024, 512, 256, 128)
    elif m_bar <= 128:
      mt_cands = (128, 256, 512)
    elif m_bar <= 256:
      mt_cands = (256, 512, 128)
    else:
      mt_cands = (512, 256, 128)
    kt = k
    for cand_nt in (
        n,
        7168,
        4096,
        3584,
        3072,
        2880,
        2048,
        1792,
        1536,
        1024,
        768,
        512,
        384,
        256,
    ):
      if cand_nt <= n and n % cand_nt == 0:
        m_aligned = ((m + 2047) // 2048) * 2048
        for cand_mt in mt_cands:
          if m_aligned % cand_mt == 0:
            vmem_bytes = (
                2 * cand_mt * kt
                + 2 * kt * cand_nt
                + 4 * cand_mt * cand_nt
                + 2 * cand_mt * min(kt, cand_nt)
            )
            vmem_limit = (
                56 * 1024 * 1024
                if (cand_mt == 2048 and kt <= 1536 and cand_nt <= 3584)
                else (48 * 1024 * 1024 if kt <= 4096 else 43 * 1024 * 1024)
            )
            if vmem_bytes <= vmem_limit:
              return cls(mt=cand_mt, kt=kt, nt=cand_nt)
    fb_nt = next(
        (x for x in (2048, 1024, 512, 256, 128, 64) if x <= n and n % x == 0), n
    )
    fb_kt = next(
        (
            x
            for x in (k, 2048, 1024, 512, 256, 128, 64)
            if x <= k and k % x == 0
        ),
        k,
    )
    return cls(mt=256, kt=fb_kt, nt=fb_nt)


def generate_group_sizes(
    *,
    m: int,
    num_groups: int,
    p: Sequence[float] | None = None,
    seed: int = 0,
    distribution: str = "uniform",
) -> tuple[int, ...]:
  """Returns deterministic group sizes summing to m."""
  rng = np.random.default_rng(seed)
  if p is None and distribution == "zipf":
    ranks = np.arange(1, num_groups + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, 1.1)
    if num_groups >= 8:
      weights[::8] = 0.0
    elif num_groups >= 2:
      weights[-1] = 0.0
    weights /= weights.sum()
    return tuple(map(int, rng.multinomial(m, weights)))
  if p is None and distribution == "empty_heavy":
    weights = np.zeros(num_groups, dtype=np.float64)
    active = max(1, num_groups // 4)
    weights[:active] = 1.0 / np.power(
        np.arange(1, active + 1, dtype=np.float64), 0.8
    )
    weights /= weights.sum()
    return tuple(map(int, rng.multinomial(m, weights)))
  group_ids = rng.choice(num_groups, (m,), p=p, replace=True)
  return tuple(map(int, np.bincount(group_ids, minlength=num_groups)))


def make_gmm_inputs(
    m: int,
    k: int,
    n: int,
    g: int,
    *,
    dtype: Any = jnp.bfloat16,
    distribution: str = "uniform",
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Generates bf16 lhs, rhs, and representative group_sizes."""
  k1, k2 = jax.random.split(jax.random.PRNGKey(42 + m + k + n + g), 2)
  scale = jnp.asarray((max(k, 1)) ** -0.25, dtype=dtype)
  lhs = jax.random.normal(k1, (m, k), dtype=dtype) * scale
  rhs = jax.random.normal(k2, (g, k, n), dtype=dtype) * scale
  group_sizes_np = generate_group_sizes(
      m=m, num_groups=g, distribution=distribution
  )
  group_sizes = jnp.asarray(group_sizes_np, dtype=jnp.int32)
  jax.block_until_ready((lhs, rhs, group_sizes))
  return lhs, rhs, group_sizes


_make_gmm_inputs = make_gmm_inputs
