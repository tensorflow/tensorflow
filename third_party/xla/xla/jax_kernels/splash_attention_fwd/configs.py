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

"""Configuration definitions and window tuning tables for jax_k Splash Attention forward hill climbing."""

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class SplashClimbConfig:
  """Specification of a Splash Attention forward benchmark shape and tuning parameters."""

  case_name: str
  batch_size: int
  q_seq_len: int
  kv_seq_len: int
  num_q_heads: int
  num_kv_heads: int
  dim_qk: int
  dim_v: int
  pallas_block_q: int
  pallas_block_kv: int
  pallas_block_kv_compute: int
  q_layout: int = 1
  k_layout: int = 2
  v_layout: int = 1
  use_experimental_scheduler: bool = True
  use_base2_exp: bool = True
  jax_block_q: int = 1024
  jax_block_kv: int = 1024
  causal: bool = True
  ref_pallas_us: float = 0.0
  ref_jax_untuned_us: float = 0.0
  ref_jax_tuned_us: float = 0.0

  @property
  def is_gqa(self) -> bool:
    return self.num_q_heads != self.num_kv_heads

  @property
  def q_groups(self) -> int:
    return self.num_q_heads // self.num_kv_heads

  @property
  def is_long_seq(self) -> bool:
    return self.q_seq_len > 4096

  @property
  def config_key(self) -> tuple[int, ...]:
    return (
        self.batch_size,
        self.num_q_heads,
        self.num_kv_heads,
        self.q_seq_len,
        self.kv_seq_len,
        self.dim_qk,
        self.dim_qk,
        self.dim_v,
        self.jax_block_q,
        self.jax_block_kv,
    )

  @property
  def flops(self) -> float:
    """Returns active attention FLOPs for this configuration."""
    active_fraction = 0.5 if self.causal else 1.0
    qk_flops = (
        2.0
        * self.batch_size
        * self.num_q_heads
        * self.q_seq_len
        * self.kv_seq_len
        * self.dim_qk
    )
    pv_flops = (
        2.0
        * self.batch_size
        * self.num_q_heads
        * self.q_seq_len
        * self.kv_seq_len
        * self.dim_v
    )
    return active_fraction * (qk_flops + pv_flops)


IOTA_DIAG_MASK_SPECS: frozenset[tuple[int, ...]] = frozenset({
    (32, 32, 8, 4096, 4096, 128, 128, 128, 1024, 1024),
    (4, 8, 8, 4096, 4096, 256, 256, 256, 2048, 2048),
    (2, 8, 8, 8192, 8192, 256, 256, 256, 2048, 2048),
    (1, 8, 8, 16384, 16384, 256, 256, 256, 2048, 2048),
})

SPLASH_FUSION_SPECS: dict[str, dict[tuple[int, ...], dict[str, Any]]] = {
    "qk": {
        (1, 16, 16, 16384, 16384, 128, 128, 128, 2048, 2048): {
            "output_window_bounds": (8, 256, 4),
            "kernel_window_bounds": (8, 64, 1),
            "buffering_level": 3,
            "scoped_vmem_limit_kib": 32768,
        },
        (2, 16, 16, 8192, 8192, 128, 128, 128, 2048, 2048): {
            "output_window_bounds": (1, 4, 256, 2),
            "kernel_window_bounds": (1, 4, 256, 1),
            "buffering_level": 4,
            "scoped_vmem_limit_kib": 32768,
        },
        (4, 16, 16, 4096, 4096, 128, 128, 128, 1024, 1024): {
            "output_window_bounds": (1, 16, 64, 8),
            "kernel_window_bounds": (1, 16, 64, 1),
            "buffering_level": 4,
        },
        (8, 16, 16, 2048, 2048, 128, 128, 128, 1024, 1024): {
            "output_window_bounds": (1, 16, 128, 4),
            "kernel_window_bounds": (1, 16, 128, 1),
            "buffering_level": 4,
        },
        (16, 16, 16, 1024, 1024, 128, 128, 128, 1024, 1024): {
            "output_window_bounds": (2, 4, 128, 8),
            "kernel_window_bounds": (2, 4, 128, 1),
            "buffering_level": 2,
        },
        (32, 16, 16, 512, 512, 128, 128, 128, 512, 512): {
            "output_window_bounds": (2, 8, 64, 8),
            "kernel_window_bounds": (2, 8, 64, 2),
            "buffering_level": 2,
        },
        (512, 16, 16, 1024, 1024, 192, 192, 192, 1024, 1024): {
            "output_window_bounds": (1, 16, 32, 8),
            "kernel_window_bounds": (1, 16, 24, 2),
            "buffering_level": 3,
        },
        (32, 32, 8, 4096, 4096, 128, 128, 128, 1024, 1024): {
            "output_window_bounds": (1, 8, 4, 128, 4),
            "kernel_window_bounds": (1, 8, 1, 128, 1),
            "buffering_level": 2,
        },
        (4, 8, 8, 4096, 4096, 256, 256, 256, 2048, 2048): {
            "output_window_bounds": (4, 1, 86, 16),
            "kernel_window_bounds": (4, 1, 86, 2),
            "buffering_level": 2,
            "scoped_vmem_limit_kib": 32768,
        },
        (4, 8, 8, 4096, 4096, 256, 256, 256, 1024, 1024): {
            "output_window_bounds": (1, 4, 64, 8),
            "kernel_window_bounds": (1, 4, 32, 4),
            "buffering_level": 2,
            "scoped_vmem_limit_kib": 32768,
        },
        (8, 8, 8, 2048, 2048, 256, 256, 256, 1024, 1024): {
            "output_window_bounds": (1, 8, 64, 8),
            "kernel_window_bounds": (1, 8, 64, 2),
            "buffering_level": 2,
            "scoped_vmem_limit_kib": 32768,
        },
        (2, 8, 8, 8192, 8192, 256, 256, 256, 2048, 2048): {
            "output_window_bounds": (2, 1, 86, 16),
            "kernel_window_bounds": (2, 1, 86, 2),
            "buffering_level": 2,
            "scoped_vmem_limit_kib": 32768,
        },
        (1, 8, 8, 16384, 16384, 256, 256, 256, 2048, 2048): {
            "output_window_bounds": (4, 128, 16),
            "kernel_window_bounds": (4, 128, 2),
            "buffering_level": 2,
            "scoped_vmem_limit_kib": 32768,
        },
        (2, 8, 8, 8192, 8192, 256, 256, 256, 1024, 1024): {
            "output_window_bounds": (1, 4, 64, 8),
            "kernel_window_bounds": (1, 4, 32, 4),
            "buffering_level": 2,
            "scoped_vmem_limit_kib": 32768,
        },
        (1, 8, 8, 16384, 16384, 256, 256, 256, 1024, 1024): {
            "output_window_bounds": (4, 64, 8),
            "kernel_window_bounds": (4, 32, 4),
            "buffering_level": 2,
            "scoped_vmem_limit_kib": 32768,
        },
    },
    "pv": {
        (32, 32, 32, 512, 512, 64, 64, 64, 512, 512): {
            "output_window_bounds": (2, 8, 9, 2),
            "kernel_window_bounds": (2, 8, 9, 4),
            "buffering_level": 3,
        },
        (16, 32, 32, 1024, 1024, 64, 64, 64, 1024, 1024): {
            "output_window_bounds": (2, 8, 9, 2),
            "kernel_window_bounds": (2, 8, 9, 8),
            "buffering_level": 3,
        },
        (8, 32, 32, 2048, 2048, 64, 64, 64, 1024, 1024): {
            "output_window_bounds": (2, 8, 9, 2),
            "kernel_window_bounds": (2, 8, 9, 8),
            "buffering_level": 3,
        },
        (4, 32, 32, 4096, 4096, 64, 64, 64, 1024, 1024): {
            "output_window_bounds": (2, 8, 9, 2),
            "kernel_window_bounds": (2, 8, 9, 8),
            "buffering_level": 3,
        },
        (2, 32, 32, 8192, 8192, 64, 64, 64, 1024, 1024): {
            "output_window_bounds": (2, 4, 9, 2),
            "kernel_window_bounds": (2, 4, 9, 8),
            "buffering_level": 3,
            "scoped_vmem_limit_kib": 32768,
        },
        (2, 32, 32, 8192, 8192, 64, 64, 64, 2048, 2048): {
            "output_window_bounds": (2, 4, 9, 2),
            "kernel_window_bounds": (2, 4, 9, 8),
            "buffering_level": 3,
            "scoped_vmem_limit_kib": 32768,
        },
        (1, 32, 32, 16384, 16384, 64, 64, 64, 1024, 1024): {
            "output_window_bounds": (4, 9, 2),
            "kernel_window_bounds": (4, 9, 8),
            "buffering_level": 3,
            "scoped_vmem_limit_kib": 20480,
        },
        (1, 32, 32, 16384, 16384, 64, 64, 64, 2048, 2048): {
            "output_window_bounds": (4, 9, 4),
            "kernel_window_bounds": (4, 9, 16),
            "buffering_level": 3,
            "scoped_vmem_limit_kib": 32768,
        },
        (1, 16, 16, 16384, 16384, 128, 128, 128, 2048, 2048): {
            "output_window_bounds": (4, 17, 4),
            "kernel_window_bounds": (4, 17, 16),
            "buffering_level": 3,
            "scoped_vmem_limit_kib": 32768,
        },
        (1, 16, 16, 16384, 16384, 128, 128, 128, 1024, 1024): {
            "output_window_bounds": (2, 17, 8),
            "kernel_window_bounds": (2, 17, 8),
            "buffering_level": 3,
            "scoped_vmem_limit_kib": 20480,
        },
        (2, 16, 16, 8192, 8192, 128, 128, 128, 2048, 2048): {
            "output_window_bounds": (1, 2, 17, 8),
            "kernel_window_bounds": (1, 2, 17, 16),
            "buffering_level": 3,
            "scoped_vmem_limit_kib": 32768,
        },
        (4, 16, 16, 4096, 4096, 128, 128, 128, 1024, 1024): {
            "output_window_bounds": (1, 2, 17, 8),
            "kernel_window_bounds": (1, 2, 17, 8),
            "buffering_level": 3,
        },
        (32, 32, 8, 4096, 4096, 128, 128, 128, 1024, 1024): {
            "output_window_bounds": (2, 2, 4, 17, 2),
            "kernel_window_bounds": (2, 2, 4, 17, 8),
            "buffering_level": 3,
        },
        (8, 16, 16, 2048, 2048, 128, 128, 128, 1024, 1024): {
            "output_window_bounds": (1, 2, 17, 8),
            "kernel_window_bounds": (1, 2, 17, 8),
            "buffering_level": 3,
        },
        (16, 16, 16, 1024, 1024, 128, 128, 128, 1024, 1024): {
            "output_window_bounds": (2, 4, 17, 2),
            "kernel_window_bounds": (2, 4, 17, 8),
            "buffering_level": 3,
        },
        (32, 16, 16, 512, 512, 128, 128, 128, 512, 512): {
            "output_window_bounds": (2, 4, 17, 2),
            "kernel_window_bounds": (2, 4, 17, 4),
            "buffering_level": 3,
        },
        (512, 16, 16, 1024, 1024, 192, 192, 192, 1024, 1024): {
            "output_window_bounds": (2, 4, 25, 2),
            "kernel_window_bounds": (2, 4, 25, 8),
            "buffering_level": 3,
        },
        (32, 8, 8, 512, 512, 256, 256, 256, 512, 512): {
            "output_window_bounds": (2, 2, 33, 2),
            "kernel_window_bounds": (2, 2, 33, 4),
            "buffering_level": 3,
        },
        (16, 8, 8, 1024, 1024, 256, 256, 256, 1024, 1024): {
            "output_window_bounds": (2, 2, 33, 2),
            "kernel_window_bounds": (2, 2, 33, 8),
            "buffering_level": 3,
        },
        (8, 8, 8, 2048, 2048, 256, 256, 256, 1024, 1024): {
            "output_window_bounds": (2, 2, 33, 2),
            "kernel_window_bounds": (2, 2, 33, 8),
            "buffering_level": 3,
        },
        (4, 8, 8, 4096, 4096, 256, 256, 256, 2048, 2048): {
            "output_window_bounds": (1, 2, 33, 4),
            "kernel_window_bounds": (1, 2, 33, 16),
            "buffering_level": 3,
            "scoped_vmem_limit_kib": 32768,
        },
        (4, 8, 8, 4096, 4096, 256, 256, 256, 1024, 1024): {
            "output_window_bounds": (2, 2, 33, 2),
            "kernel_window_bounds": (2, 2, 33, 8),
            "buffering_level": 3,
        },
        (2, 8, 8, 8192, 8192, 256, 256, 256, 1024, 1024): {
            "output_window_bounds": (1, 2, 33, 2),
            "kernel_window_bounds": (1, 2, 33, 8),
            "buffering_level": 3,
            "scoped_vmem_limit_kib": 32768,
        },
        (1, 8, 8, 16384, 16384, 256, 256, 256, 1024, 1024): {
            "output_window_bounds": (2, 33, 8),
            "kernel_window_bounds": (2, 33, 8),
            "buffering_level": 3,
            "scoped_vmem_limit_kib": 20480,
        },
        (2, 8, 8, 8192, 8192, 256, 256, 256, 2048, 2048): {
            "output_window_bounds": (1, 2, 33, 4),
            "kernel_window_bounds": (1, 2, 33, 16),
            "buffering_level": 3,
            "scoped_vmem_limit_kib": 32768,
        },
        (1, 8, 8, 16384, 16384, 256, 256, 256, 2048, 2048): {
            "output_window_bounds": (2, 33, 4),
            "kernel_window_bounds": (2, 33, 16),
            "buffering_level": 3,
            "scoped_vmem_limit_kib": 32768,
        },
    },
}

SPLASH_20SHAPE_FUSION_SPECS = SPLASH_FUSION_SPECS


@dataclasses.dataclass(frozen=True)
class SplashShapeSpec:
  """Canonical Tokamax Splash Attention specification and reference latencies."""

  case_name: str
  batch_size: int
  q_seq_len: int
  kv_seq_len: int
  num_q_heads: int
  num_kv_heads: int
  head_dim: int
  pallas_block_q: int
  pallas_block_kv: int
  pallas_block_kv_compute: int
  q_layout: int = 1
  k_layout: int = 2
  v_layout: int = 1
  use_experimental_scheduler: bool = True
  use_base2_exp: bool = True
  default_jax_k_block_q: int = 1024
  default_jax_k_block_kv: int = 1024
  jax_k_cl_us: float = 0.0
  unopt_jax_k_us: float = 0.0
  pallas_us: float = 0.0
  native_jax_us: float = 0.0
  xprof_session_id: str = ""
  v_head_dim: int | None = None
  is_causal: bool = True
  save_residuals: bool = True

  @property
  def dim_qk(self) -> int:
    return self.head_dim

  @property
  def dim_v(self) -> int:
    return self.v_head_dim if self.v_head_dim is not None else self.head_dim

  @property
  def jax_block_q(self) -> int:
    return self.default_jax_k_block_q

  @property
  def jax_block_kv(self) -> int:
    return self.default_jax_k_block_kv

  @property
  def causal(self) -> bool:
    return self.is_causal

  @property
  def is_gqa(self) -> bool:
    return self.num_q_heads != self.num_kv_heads

  @property
  def q_groups(self) -> int:
    return self.num_q_heads // self.num_kv_heads

  @property
  def config_key(self) -> tuple[int, ...]:
    return (
        self.batch_size,
        self.num_q_heads,
        self.num_kv_heads,
        self.q_seq_len,
        self.kv_seq_len,
        self.head_dim,
        self.head_dim,
        self.dim_v,
        self.default_jax_k_block_q,
        self.default_jax_k_block_kv,
    )


def _spec(
    case_name: str,
    b: int,
    sq: int,
    hq: int,
    hkv: int,
    d: int,
    pbq: int,
    pbkv: int,
    pbkv_c: int,
    ql: int,
    kl: int,
    vl: int,
    exp_sched: bool,
    nbq: int,
    jax_k_us: float,
    pallas_us: float,
    xprof_id: str,
    is_causal: bool,
    save_res: bool,
    unopt_us: float = 0.0,
    jax_us: float = 0.0,
) -> SplashShapeSpec:
  """Constructs a SplashShapeSpec from compact table row fields."""
  return SplashShapeSpec(
      case_name=case_name,
      batch_size=b,
      q_seq_len=sq,
      kv_seq_len=sq,
      num_q_heads=hq,
      num_kv_heads=hkv,
      head_dim=d,
      pallas_block_q=pbq,
      pallas_block_kv=pbkv,
      pallas_block_kv_compute=pbkv_c,
      q_layout=ql,
      k_layout=kl,
      v_layout=vl,
      use_experimental_scheduler=exp_sched,
      use_base2_exp=True,
      default_jax_k_block_q=nbq,
      default_jax_k_block_kv=nbq,
      jax_k_cl_us=jax_k_us,
      unopt_jax_k_us=unopt_us,
      pallas_us=pallas_us,
      native_jax_us=jax_us,
      xprof_session_id=xprof_id,
      v_head_dim=128 if d == 192 else None,
      is_causal=is_causal,
      save_residuals=save_res,
  )


_TOKAMAX_20_BASELINES: dict[str, tuple[float, float]] = {
    "tokamax_b32_q512_h32_d64": (3848.29, 651.77),
    "tokamax_b16_q1024_h32_d64": (8596.82, 1157.09),
    "tokamax_b8_q2048_h32_d64": (9950.55, 1884.20),
    "tokamax_b4_q4096_h32_d64": (11712.59, 3414.18),
    "tokamax_b2_q8192_h32_d64": (28983.40, 6351.00),
    "tokamax_b1_q16384_h32_d64": (36769.49, 11888.63),
    "tokamax_b32_q512_h16_d128": (1925.07, 352.29),
    "tokamax_b16_q1024_h16_d128": (1800.16, 758.71),
    "tokamax_b8_q2048_h16_d128": (4319.35, 1147.25),
    "tokamax_b4_q4096_h16_d128": (5345.28, 2178.98),
    "tokamax_b2_q8192_h16_d128": (15354.42, 3455.40),
    "tokamax_b1_q16384_h16_d128": (29825.32, 6523.49),
    "tokamax_b32_q4096_h32_kv8_d128": (3759627.05, 25873.48),
    "tokamax_b512_q1024_h16_d192": (67958.64, 24773.73),
    "tokamax_b32_q512_h8_d256": (1251.99, 251.94),
    "tokamax_b16_q1024_h8_d256": (1167.89, 369.12),
    "tokamax_b8_q2048_h8_d256": (2412.05, 1001.18),
    "tokamax_b4_q4096_h8_d256": (2942.17, 1462.84),
    "tokamax_b2_q8192_h8_d256": (7882.15, 2746.51),
    "tokamax_b1_q16384_h8_d256": (15297.84, 4012.79),
}

# pyformat: disable
_TOKAMAX_76_ROWS: tuple[tuple[Any, ...], ...] = (
    ("tokamax_b32_q512_h32_d64_causal_res", 32, 512, 32, 32, 64, 512, 512, 512, 2, 2, 2, False, 512, 371.66, 1393.52, "forge-00-38026-3802621153365201214", True, True),
    ("tokamax_b32_q512_h32_d64_causal_nores", 32, 512, 32, 32, 64, 512, 512, 512, 2, 2, 2, True, 512, 367.84, 1047.25, "forge-00-14061-14061346463071233035", True, False),
    ("tokamax_b32_q512_h32_d64_full_res", 32, 512, 32, 32, 64, 512, 512, 512, 2, 2, 2, False, 512, 400.33, 1279.01, "forge-00-12829-12829462525085470178", False, True),
    ("tokamax_b32_q512_h32_d64_full_nores", 32, 512, 32, 32, 64, 512, 512, 512, 2, 2, 2, False, 512, 393.55, 977.08, "forge-00-15430-15430943932128772699", False, False),
    ("tokamax_b16_q1024_h32_d64_causal_res", 16, 1024, 32, 32, 64, 1024, 1024, 1024, 2, 2, 2, False, 1024, 595.39, 1553.17, "forge-00-18168-1816865520998693670", True, True),
    ("tokamax_b16_q1024_h32_d64_causal_nores", 16, 1024, 32, 32, 64, 1024, 1024, 512, 2, 2, 2, True, 1024, 592.73, 1232.72, "forge-00-53183-5318350585943812001", True, False),
    ("tokamax_b16_q1024_h32_d64_full_res", 16, 1024, 32, 32, 64, 1024, 1024, 512, 2, 2, 2, True, 1024, 864.96, 1224.15, "forge-00-31577-3157720504101079634", False, True),
    ("tokamax_b16_q1024_h32_d64_full_nores", 16, 1024, 32, 32, 64, 1024, 1024, 512, 2, 2, 2, True, 1024, 1565.53, 974.88, "forge-00-15711-15711372683292275690", False, False),
    ("tokamax_b8_q2048_h32_d64_causal_res", 8, 2048, 32, 32, 64, 1024, 1024, 512, 2, 2, 2, False, 1024, 997.12, 2035.04, "forge-00-51167-5116779023024353704", True, True),
    ("tokamax_b8_q2048_h32_d64_causal_nores", 8, 2048, 32, 32, 64, 1024, 1024, 512, 2, 2, 2, True, 1024, 992.87, 1682.36, "forge-00-79049-790498603830020632", True, False),
    ("tokamax_b8_q2048_h32_d64_full_res", 8, 2048, 32, 32, 64, 2048, 2048, 512, 2, 2, 2, True, 1024, 1501.74, 1826.88, "forge-00-18057-18057116689204329795", False, True),
    ("tokamax_b8_q2048_h32_d64_full_nores", 8, 2048, 32, 32, 64, 2048, 2048, 1024, 2, 2, 2, False, 1024, 2851.02, 1640.36, "forge-00-10419-10419601301024327617", False, False),
    ("tokamax_b4_q4096_h32_d64_causal_res", 4, 4096, 32, 32, 64, 1024, 1024, 512, 2, 2, 2, True, 1024, 1871.00, 2970.17, "forge-00-37204-3720477141967002632", True, True),
    ("tokamax_b4_q4096_h32_d64_causal_nores", 4, 4096, 32, 32, 64, 1024, 1024, 512, 2, 2, 2, True, 1024, 1842.53, 2551.98, "forge-00-37137-371376563018361797", True, False),
    ("tokamax_b4_q4096_h32_d64_full_res", 4, 4096, 32, 32, 64, 2048, 4096, 512, 2, 2, 2, True, 1024, 3296.12, 3159.54, "forge-00-48791-4879173300275711086", False, True),
    ("tokamax_b4_q4096_h32_d64_full_nores", 4, 4096, 32, 32, 64, 2048, 4096, 1024, 2, 2, 2, False, 1024, 5488.44, 2961.85, "forge-00-81639-8163942807506628861", False, False),
    ("tokamax_b2_q8192_h32_d64_causal_res", 2, 8192, 32, 32, 64, 2048, 2048, 1024, 2, 2, 2, False, 1024, 3282.81, 4573.23, "forge-00-28616-2861673074969266572", True, True),
    ("tokamax_b2_q8192_h32_d64_causal_nores", 2, 8192, 32, 32, 64, 2048, 2048, 512, 2, 2, 2, True, 1024, 3257.46, 4136.29, "forge-00-14868-1486865037123703861", True, False),
    ("tokamax_b2_q8192_h32_d64_full_res", 2, 8192, 32, 32, 64, 4096, 4096, 256, 1, 2, 1, True, 1024, 5050.48, 6202.23, "forge-00-15541-15541477720900092251", False, True),
    ("tokamax_b2_q8192_h32_d64_full_nores", 2, 8192, 32, 32, 64, 4096, 4096, 256, 1, 2, 1, True, 1024, 5003.91, 5883.65, "forge-00-73392-7339289774379198922", False, False),
    ("tokamax_b1_q16384_h32_d64_causal_res", 1, 16384, 32, 32, 64, 2048, 2048, 512, 1, 2, 2, True, 2048, 5696.52, 7286.80, "forge-00-12209-122098391737869947", True, True),
    ("tokamax_b1_q16384_h32_d64_causal_nores", 1, 16384, 32, 32, 64, 2048, 2048, 512, 1, 2, 2, True, 2048, 5672.50, 6909.65, "forge-00-14825-14825428148385135236", True, False),
    ("tokamax_b1_q16384_h32_d64_full_res", 1, 16384, 32, 32, 64, 4096, 4096, 256, 1, 2, 1, True, 2048, 9723.58, 11329.45, "forge-00-89569-8956937522699159001", False, True),
    ("tokamax_b1_q16384_h32_d64_full_nores", 1, 16384, 32, 32, 64, 4096, 4096, 256, 1, 2, 1, True, 2048, 9716.91, 10998.10, "forge-00-13698-1369815540345986782", False, False),
    ("tokamax_b32_q512_h16_d128_causal_res", 32, 512, 16, 16, 128, 512, 512, 512, 1, 2, 1, False, 512, 362.55, 704.30, "forge-00-19052-1905295627870002434", True, True),
    ("tokamax_b32_q512_h16_d128_causal_nores", 32, 512, 16, 16, 128, 512, 512, 512, 1, 2, 2, True, 512, 298.26, 585.20, "forge-00-12732-12732124493071573526", True, False),
    ("tokamax_b32_q512_h16_d128_full_res", 32, 512, 16, 16, 128, 512, 512, 512, 1, 2, 1, False, 512, 271.90, 644.50, "forge-00-16588-16588733135945007803", False, True),
    ("tokamax_b32_q512_h16_d128_full_nores", 32, 512, 16, 16, 128, 512, 512, 512, 1, 2, 2, False, 512, 268.49, 534.33, "forge-00-39474-3947439340624349408", False, False),
    ("tokamax_b16_q1024_h16_d128_causal_res", 16, 1024, 16, 16, 128, 1024, 1024, 512, 1, 2, 1, False, 1024, 661.23, 749.07, "forge-00-67319-6731995764529988856", True, True),
    ("tokamax_b16_q1024_h16_d128_causal_nores", 16, 1024, 16, 16, 128, 1024, 1024, 512, 1, 2, 1, True, 1024, 658.58, 586.04, "forge-00-18331-18331386923103659421", True, False),
    ("tokamax_b16_q1024_h16_d128_full_res", 16, 1024, 16, 16, 128, 1024, 1024, 512, 1, 2, 1, True, 1024, 432.74, 578.42, "forge-00-31112-3111218955311857380", False, True),
    ("tokamax_b16_q1024_h16_d128_full_nores", 16, 1024, 16, 16, 128, 1024, 1024, 512, 1, 2, 1, True, 1024, 432.17, 456.39, "forge-00-34819-3481926039542286969", False, False),
    ("tokamax_b8_q2048_h16_d128_causal_res", 8, 2048, 16, 16, 128, 1024, 1024, 512, 1, 2, 2, False, 1024, 805.25, 1042.41, "forge-00-48935-4893542732428212260", True, True),
    ("tokamax_b8_q2048_h16_d128_causal_nores", 8, 2048, 16, 16, 128, 1024, 1024, 512, 1, 2, 1, True, 1024, 803.88, 806.44, "forge-00-53467-5346745369988272557", True, False),
    ("tokamax_b8_q2048_h16_d128_full_res", 8, 2048, 16, 16, 128, 2048, 2048, 512, 1, 2, 1, True, 1024, 1517.64, 875.38, "forge-00-16027-160277057000019044", False, True),
    ("tokamax_b8_q2048_h16_d128_full_nores", 8, 2048, 16, 16, 128, 2048, 2048, 1024, 1, 2, 1, False, 1024, 815.51, 792.28, "forge-00-14712-14712035304140404375", False, False),
    ("tokamax_b4_q4096_h16_d128_causal_res", 4, 4096, 16, 16, 128, 1024, 1024, 512, 1, 2, 1, True, 1024, 1528.56, 1441.77, "forge-00-81660-8166098916308311085", True, True),
    ("tokamax_b4_q4096_h16_d128_causal_nores", 4, 4096, 16, 16, 128, 1024, 1024, 512, 1, 2, 1, True, 1024, 1291.63, 1236.93, "forge-00-15008-15008308678857751340", True, False),
    ("tokamax_b4_q4096_h16_d128_full_res", 4, 4096, 16, 16, 128, 2048, 4096, 512, 1, 2, 2, True, 1024, 1436.43, 1554.08, "forge-00-38026-3802621153365201836", False, True),
    ("tokamax_b4_q4096_h16_d128_full_nores", 4, 4096, 16, 16, 128, 1024, 4096, 512, 1, 1, 1, True, 1024, 1452.73, 1377.34, "forge-00-14061-14061346463071234149", False, False),
    ("tokamax_b2_q8192_h16_d128_causal_res", 2, 8192, 16, 16, 128, 2048, 2048, 512, 1, 2, 2, True, 2048, 2014.95, 2228.31, "forge-00-12829-12829462525085468372", True, True),
    ("tokamax_b2_q8192_h16_d128_causal_nores", 2, 8192, 16, 16, 128, 2048, 2048, 512, 1, 1, 1, True, 2048, 1991.77, 1967.56, "forge-00-15430-15430943932128774309", True, False),
    ("tokamax_b2_q8192_h16_d128_full_res", 2, 8192, 16, 16, 128, 4096, 4096, 256, 1, 1, 1, True, 2048, 2759.09, 2895.50, "forge-00-18168-1816865520998691060", False, True),
    ("tokamax_b2_q8192_h16_d128_full_nores", 2, 8192, 16, 16, 128, 4096, 4096, 256, 1, 1, 1, True, 2048, 2770.19, 2764.22, "forge-00-53183-5318350585943812335", False, False),
    ("tokamax_b1_q16384_h16_d128_causal_res", 1, 16384, 16, 16, 128, 2048, 2048, 512, 1, 2, 1, True, 2048, 3243.96, 3670.60, "forge-00-31577-3157720504101078920", True, True),
    ("tokamax_b1_q16384_h16_d128_causal_nores", 1, 16384, 16, 16, 128, 2048, 2048, 512, 1, 2, 1, True, 2048, 3251.06, 3468.41, "forge-00-15711-15711372683292276324", True, False),
    ("tokamax_b1_q16384_h16_d128_full_res", 1, 16384, 16, 16, 128, 4096, 4096, 256, 1, 1, 1, True, 2048, 5303.32, 5591.80, "forge-00-51167-5116779023024355634", False, True),
    ("tokamax_b1_q16384_h16_d128_full_nores", 1, 16384, 16, 16, 128, 4096, 4096, 256, 1, 1, 1, True, 2048, 5312.85, 5439.82, "forge-00-79049-790498603830021626", False, False),
    ("tokamax_b32_q4096_h32_kv8_d128_causal_res", 32, 4096, 32, 8, 128, 1024, 1024, 512, 1, 2, 2, True, 1024, 17963.41, 23012.04, "forge-00-18057-18057116689204328665", True, True),
    ("tokamax_b32_q4096_h32_kv8_d128_causal_nores", 32, 4096, 32, 8, 128, 1024, 1024, 512, 1, 2, 1, True, 1024, 17868.73, 19108.86, "forge-00-10419-10419601301024325767", True, False),
    ("tokamax_b512_q1024_h16_d192_causal_res", 512, 1024, 16, 16, 192, 1024, 1024, 1024, 2, 2, 2, False, 1024, 19620.05, 24878.15, "forge-00-37204-3720477141967003022", True, True),
    ("tokamax_b512_q1024_h16_d192_causal_nores", 512, 1024, 16, 16, 192, 1024, 1024, 512, 2, 2, 2, True, 1024, 19400.74, 19250.17, "forge-00-37137-371376563018362011", True, False),
    ("tokamax_b32_q512_h8_d256_causal_res", 32, 512, 8, 8, 256, 512, 512, 512, 1, 2, 2, False, 512, 220.56, 460.86, "forge-00-48791-4879173300275710184", True, True),
    ("tokamax_b32_q512_h8_d256_causal_nores", 32, 512, 8, 8, 256, 512, 512, 512, 1, 2, 2, False, 512, 218.91, 377.54, "forge-00-81639-8163942807506629887", True, False),
    ("tokamax_b32_q512_h8_d256_full_res", 32, 512, 8, 8, 256, 512, 512, 512, 1, 2, 2, True, 512, 209.95, 439.34, "forge-00-28616-2861673074969267094", False, True),
    ("tokamax_b32_q512_h8_d256_full_nores", 32, 512, 8, 8, 256, 512, 512, 512, 1, 2, 2, False, 512, 207.30, 350.87, "forge-00-14868-1486865037123705863", False, False),
    ("tokamax_b16_q1024_h8_d256_causal_res", 16, 1024, 8, 8, 256, 1024, 1024, 512, 1, 2, 2, False, 1024, 331.46, 502.18, "forge-00-15541-15541477720900094217", True, True),
    ("tokamax_b16_q1024_h8_d256_causal_nores", 16, 1024, 8, 8, 256, 1024, 1024, 512, 1, 2, 2, False, 1024, 287.26, 435.44, "forge-00-73392-7339289774379198208", True, False),
    ("tokamax_b16_q1024_h8_d256_full_res", 16, 1024, 8, 8, 256, 1024, 1024, 1024, 1, 2, 2, False, 1024, 326.39, 424.56, "forge-00-12209-122098391737868481", False, True),
    ("tokamax_b16_q1024_h8_d256_full_nores", 16, 1024, 8, 8, 256, 1024, 1024, 256, 1, 2, 1, True, 1024, 325.61, 297.71, "forge-00-14825-14825428148385135026", False, False),
    ("tokamax_b8_q2048_h8_d256_causal_res", 8, 2048, 8, 8, 256, 1024, 1024, 512, 1, 2, 2, False, 1024, 623.42, 653.01, "forge-00-89569-8956937522699161075", True, True),
    ("tokamax_b8_q2048_h8_d256_causal_nores", 8, 2048, 8, 8, 256, 1024, 1024, 256, 1, 2, 2, True, 1024, 551.47, 561.94, "forge-00-13698-1369815540345984408", True, False),
    ("tokamax_b8_q2048_h8_d256_full_res", 8, 2048, 8, 8, 256, 2048, 2048, 512, 1, 2, 1, True, 1024, 498.69, 504.84, "forge-00-19052-1905295627870002796", False, True),
    ("tokamax_b8_q2048_h8_d256_full_nores", 8, 2048, 8, 8, 256, 2048, 2048, 512, 1, 1, 1, True, 1024, 498.28, 411.52, "forge-00-12732-12732124493071576044", False, False),
    ("tokamax_b4_q4096_h8_d256_causal_res", 4, 4096, 8, 8, 256, 2048, 2048, 512, 1, 2, 1, True, 2048, 675.96, 840.60, "forge-00-16588-16588733135945009849", True, True),
    ("tokamax_b4_q4096_h8_d256_causal_nores", 4, 4096, 8, 8, 256, 1024, 1024, 256, 1, 2, 1, True, 2048, 671.25, 755.55, "forge-00-39474-3947439340624350314", True, False),
    ("tokamax_b4_q4096_h8_d256_full_res", 4, 4096, 8, 8, 256, 1024, 4096, 512, 1, 2, 1, True, 2048, 1158.10, 898.96, "forge-00-67319-6731995764529990614", False, True),
    ("tokamax_b4_q4096_h8_d256_full_nores", 4, 4096, 8, 8, 256, 2048, 4096, 512, 1, 1, 1, True, 2048, 1178.74, 756.38, "forge-00-18331-18331386923103660227", False, False),
    ("tokamax_b2_q8192_h8_d256_causal_res", 2, 8192, 8, 8, 256, 2048, 2048, 512, 1, 2, 1, True, 2048, 1212.10, 1238.85, "forge-00-31112-3111218955311857110", True, True),
    ("tokamax_b2_q8192_h8_d256_causal_nores", 2, 8192, 8, 8, 256, 2048, 2048, 512, 1, 2, 1, True, 2048, 1212.16, 1155.17, "forge-00-34819-3481926039542286147", True, False),
    ("tokamax_b2_q8192_h8_d256_full_res", 2, 8192, 8, 8, 256, 2048, 4096, 256, 1, 2, 1, True, 2048, 2113.26, 1684.36, "forge-00-48935-4893542732428210398", False, True),
    ("tokamax_b2_q8192_h8_d256_full_nores", 2, 8192, 8, 8, 256, 4096, 4096, 256, 1, 1, 1, True, 2048, 2148.35, 1518.88, "forge-00-53467-5346745369988273799", False, False),
    ("tokamax_b1_q16384_h8_d256_causal_res", 1, 16384, 8, 8, 256, 2048, 2048, 512, 1, 2, 1, True, 2048, 2318.70, 2042.54, "forge-00-16027-160277057000019062", True, True),
    ("tokamax_b1_q16384_h8_d256_causal_nores", 1, 16384, 8, 8, 256, 2048, 2048, 512, 1, 2, 1, True, 2048, 2323.67, 1943.64, "forge-00-14712-14712035304140404497", True, False),
    ("tokamax_b1_q16384_h8_d256_full_res", 1, 16384, 8, 8, 256, 2048, 4096, 256, 1, 2, 1, True, 2048, 3444.58, 3199.23, "forge-00-81660-8166098916308308427", False, True),
    ("tokamax_b1_q16384_h8_d256_full_nores", 1, 16384, 8, 8, 256, 4096, 4096, 256, 1, 2, 1, True, 2048, 3473.96, 3003.91, "forge-00-15008-15008308678857751950", False, False),
)
# pyformat: enable

TOKAMAX_76_CONFIGS: dict[str, SplashShapeSpec] = {
    row[0]: _spec(*row) for row in _TOKAMAX_76_ROWS
}

TOKAMAX_CONFIGS = TOKAMAX_76_CONFIGS

TOKAMAX_20_SHAPES: dict[str, SplashShapeSpec] = {
    name: dataclasses.replace(
        TOKAMAX_76_CONFIGS[f"{name}_causal_res"],
        case_name=name,
        unopt_jax_k_us=unopt_us,
        native_jax_us=jax_us,
    )
    for name, (unopt_us, jax_us) in _TOKAMAX_20_BASELINES.items()
}
