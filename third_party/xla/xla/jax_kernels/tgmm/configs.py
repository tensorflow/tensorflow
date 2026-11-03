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

"""Canonical shape and tiling configurations for jax_k backward ragged_dot (tgmm) kernels."""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _is_tpu7_device() -> bool:
  try:
    dev = jax.devices()[0]
    device_kind = str(dev.device_kind).lower()
    return (
        dev.platform == "tpu"
        and "lite" not in device_kind
        and not any(t in device_kind for t in ("v2", "v3", "v4", "v5", "v6"))
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def _estimate_tgmm_slots(m: int, g: int, tm: int) -> float:
  if m <= tm:
    return float(max(g, 1))
  g_eff = max(g, 1)
  mu = m / g_eff + 7.5
  sigma = math.sqrt(max((m / g_eff) * (1.0 - 1.0 / g_eff), 1.0) + 21.25)
  inv_s = 1.0 / (sigma * math.sqrt(2.0))
  exp_blks = sum(
      0.5 * (1.0 - math.erf((j * tm - mu) * inv_s)) for j in range(16)
  )
  return max(
      float(g_eff),
      (g_eff - 1) * exp_blks + math.ceil((2.0 * (m / g_eff) + 7.5) / tm),
  )


@dataclasses.dataclass(frozen=True)
class TgmmBenchmarkConfig:
  """Configuration for a backward ragged_dot (tgmm) benchmark workload."""

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
  def rhs_shape(self) -> tuple[int, int]:
    return (self.m, self.n)

  @property
  def out_shape(self) -> tuple[int, int, int]:
    return (self.g, self.k, self.n)

  @property
  def flops(self) -> float:
    return float(2 * self.m * self.k * self.n)

  @property
  def bytes_moved(self) -> float:
    elem_bytes = 2 if self.dtype == jnp.bfloat16 else 4
    return float(
        (self.m * self.k + self.m * self.n + self.g * self.k * self.n)
        * elem_bytes
    )


TgmmShapeConfig = TgmmBenchmarkConfig

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

TGMM_BENCHMARK_CONFIGS: tuple[TgmmBenchmarkConfig, ...] = tuple(
    TgmmBenchmarkConfig(name=name, m=m, k=k, n=n, g=g)
    for name, m, k, n, g in SHAPES
)


@dataclasses.dataclass(frozen=True)
class TgmmTilingConfig:
  """Tiling configuration (mt, kt, nt, nk, nn) for Stage 1 jax_k tgmm."""

  mt: int
  kt: int
  nt: int
  nk: int
  nn: int

  @classmethod
  def from_shapes(cls, m: int, k: int, n: int, g: int) -> "TgmmTilingConfig":
    """Selects VMEM aware (mt, kt, nt, nk, nn) tiles for tgmm."""
    is_tpu7 = _is_tpu7_device()
    if is_tpu7 and m <= 16 and k == 4096 and n == 4096:
      return cls(mt=16, kt=2048, nt=4096, nk=2, nn=1)
    if is_tpu7 and m == 4096 and k == 4096 and n == 4096:
      return cls(mt=576, kt=1024, nt=2048, nk=4, nn=2)
    if is_tpu7 and k == 2880 and n in (2944, 3072) and m >= 2048:
      return cls(mt=1408, kt=2880, nt=1024, nk=1, nn=3)
    if is_tpu7 and m == 131072 and k == 512 and n == 7168 and g == 256:
      return cls(mt=592, kt=512, nt=7168, nk=1, nn=1)
    if is_tpu7 and m == 131072 and k == 1024 and n == 7168 and g == 256:
      return cls(mt=592, kt=1024, nt=3584, nk=1, nn=2)
    if is_tpu7 and m == 262144 and k == 512 and n == 7168 and g == 256:
      return cls(mt=1152, kt=512, nt=3584, nk=1, nn=2)
    if is_tpu7 and m == 262144 and k == 1024 and n == 7168 and g == 256:
      return cls(mt=1152, kt=1024, nt=3584, nk=1, nn=2)
    if (
        is_tpu7
        and g == 256
        and m in (131072, 262144)
        and k == 7168
        and n == 1024
    ):
      return cls(mt=592 if m == 131072 else 1152, kt=1792, nt=1024, nk=4, nn=1)
    if is_tpu7 and g == 256 and m == 262144 and k == 256 and n == 7168:
      return cls(mt=1152, kt=256, nt=7168, nk=1, nn=1)
    vmem_limit = 44 * 1024 * 1024
    best = None
    tm_cands = (
        (1664, 1536, 1408, 1280, 1152, 1024, 896, 768, 640, 512, 384, 256, 128)
        if is_tpu7
        else (1024, 512, 256)
    )
    for nk in range(1, 16):
      if k % nk == 0 and ((k // nk) % 128 == 0 or nk == 1):
        tk = k // nk
      else:
        continue
      for nn in range(1, 16):
        if n % nn == 0 and ((n // nn) % 128 == 0 or nn == 1):
          tn = n // nn
        else:
          continue
        if is_tpu7:
          if tk * tn > 2048 * 1024:
            continue
        else:
          if tk > 512 or tn > 512:
            continue
        for tm in tm_cands:
          if tm > max(m, 128 if is_tpu7 else 256):
            continue
          vmem = 12 * tk * tn + 4 * tm * (tk + tn)
          if vmem <= vmem_limit:
            slots = _estimate_tgmm_slots(m, g, tm)
            eff_m = slots * tm
            hbm_read = eff_m * (nn * k + nk * n) * 2.0
            hbm_write = slots * (nk * nn) * (tk * tn * 2.0)
            comp_ns = (2.0 * eff_m * k * n) / 900e3
            mem_ns = (hbm_read + hbm_write) / 3200.0
            step_ns = slots * (nk * nn) * 350.0
            cost = max(comp_ns, mem_ns) + 0.35 * min(comp_ns, mem_ns) + step_ns
            score = (cost, nk * nn, -tm, -tn)
            if best is None or score < best[0]:
              best = (score, tm, tk, tn, nk, nn)
            if not is_tpu7:
              break
    if best is not None:
      _, tm, tk, tn, nk, nn = best
      return cls(mt=tm, kt=tk, nt=tn, nk=nk, nn=nn)
    fb_tk = next(
        (x for x in (2048, 1024, 512, 256, 128) if x <= k and k % x == 0), k
    )
    fb_tn = next(
        (x for x in (2048, 1024, 512, 256, 128) if x <= n and n % x == 0), n
    )
    fb_tm = 256
    return cls(mt=fb_tm, kt=fb_tk, nt=fb_tn, nk=k // fb_tk, nn=n // fb_tn)


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


def make_tgmm_inputs(
    m: int,
    k: int,
    n: int,
    g: int,
    *,
    dtype: Any = jnp.bfloat16,
    distribution: str = "uniform",
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Generates bf16 lhs (M, K), rhs (M, N), and Tokamax distributed group_sizes."""
  k1, k2 = jax.random.split(jax.random.PRNGKey(42 + m + k + n + g), 2)
  scale = jnp.asarray(1.0 / math.sqrt(max(m // max(g, 1), 16)), dtype=dtype)
  lhs = jax.random.normal(k1, (m, k), dtype=dtype) * scale
  rhs = jax.random.normal(k2, (m, n), dtype=dtype) * scale
  group_sizes_np = np.asarray(
      generate_group_sizes(m=m, num_groups=g, distribution=distribution),
      dtype=np.int32,
  ).copy()
  if g >= 2 and np.all(group_sizes_np > 0):
    group_sizes_np[0] += group_sizes_np[1]
    group_sizes_np[1] = 0
  group_sizes = jnp.asarray(group_sizes_np, dtype=jnp.int32)
  jax.block_until_ready((lhs, rhs, group_sizes))
  return lhs, rhs, group_sizes


_make_tgmm_inputs = make_tgmm_inputs
is_tpu7_device = _is_tpu7_device
