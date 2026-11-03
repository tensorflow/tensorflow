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

"""Canonical shape and benchmark configurations for Fused Conv1D GDN kernels."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class GdnBenchmarkConfig:
  """Configuration for a Fused Causal Conv1D + Gated Delta Rule workload."""

  name: str
  lengths: tuple[int, ...]
  n_kq: int = 2
  n_v: int = 8
  d_k: int = 128
  d_v: int = 128
  kernel_size: int = 4
  context_lens: tuple[int, ...] | int = 0
  max_reqs: int | None = None
  num_decode_seqs: int | None = None
  distribution: tuple[int, int, int] | None = None
  dtype: Any = jnp.bfloat16
  state_dtype: Any = jnp.bfloat16
  num_spec_tokens: int = 0
  read_offsets: tuple[int, ...] | None = None

  @property
  def num_seqs(self) -> int:
    return len(self.lengths)

  @property
  def effective_max_reqs(self) -> int:
    return self.max_reqs if self.max_reqs is not None else self.num_seqs

  @property
  def num_tokens(self) -> int:
    return sum(self.lengths)

  @property
  def dim_size(self) -> int:
    return 2 * self.n_kq * self.d_k + self.n_v * self.d_v

  @property
  def num_blocks(self) -> int:
    window = self.num_spec_tokens + 1
    return 1 + self.effective_max_reqs * window


GdnShapeConfig = GdnBenchmarkConfig


GDN_BENCHMARK_CONFIGS: tuple[GdnBenchmarkConfig, ...] = (
    # Qwen3.5 397B TP8
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_prefill_1024",
        lengths=(1024,),
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_prefill_2048",
        lengths=(2048,),
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_prefill_4096",
        lengths=(4096,),
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_prefill_8192",
        lengths=(8192,),
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_decode_64",
        lengths=(1,) * 64,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_decode_128",
        lengths=(1,) * 128,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_decode_256",
        lengths=(1,) * 256,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_decode_512",
        lengths=(1,) * 512,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_decode_cached_4_ctx8192",
        lengths=(1,) * 4,
        context_lens=8192,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_decode_cached_8_ctx8192",
        lengths=(1,) * 8,
        context_lens=8192,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_decode_cached_16_ctx8192",
        lengths=(1,) * 16,
        context_lens=8192,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_decode_cached_32_ctx8192",
        lengths=(1,) * 32,
        context_lens=8192,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_decode_cached_64_ctx1024",
        lengths=(1,) * 64,
        context_lens=1024,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_decode_cached_64_ctx8192",
        lengths=(1,) * 64,
        context_lens=8192,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_decode_cached_128_ctx8192",
        lengths=(1,) * 128,
        context_lens=8192,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_mixed_64_t8192",
        lengths=(1,) * 63 + (8129,),
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_mixed_3x512",
        lengths=(256, 128, 128),
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_mixed_prefill_decode_11",
        lengths=(1,) * 8 + (128, 128, 256),
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_padded_mixed_16",
        lengths=(128, 64, 32, 16, 8),
        max_reqs=16,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_padded_decode_64_max512",
        lengths=(1,) * 64,
        max_reqs=512,
        n_kq=2,
        n_v=8,
    ),
    # Qwen3.5 397B ATTN_DP4 EP8
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp4_ep8_prefill_1024",
        lengths=(1024,),
        n_kq=8,
        n_v=32,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp4_ep8_prefill_2048",
        lengths=(2048,),
        n_kq=8,
        n_v=32,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp4_ep8_decode_16_ctx1024",
        lengths=(1,) * 16,
        context_lens=1024,
        n_kq=8,
        n_v=32,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp4_ep8_decode_32_ctx1024",
        lengths=(1,) * 32,
        context_lens=1024,
        n_kq=8,
        n_v=32,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp4_ep8_decode_64_ctx8192",
        lengths=(1,) * 64,
        context_lens=8192,
        n_kq=8,
        n_v=32,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp4_ep8_decode_128_ctx1024",
        lengths=(1,) * 128,
        context_lens=1024,
        n_kq=8,
        n_v=32,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp4_ep8_mixed_16_t1024",
        lengths=(1,) * 15 + (1009,),
        context_lens=(1024,) * 15 + (0,),
        n_kq=8,
        n_v=32,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp4_ep8_mixed_32_t2048",
        lengths=(1,) * 31 + (2017,),
        context_lens=(8192,) * 31 + (0,),
        n_kq=8,
        n_v=32,
    ),
    # Qwen3.5 397B ATTN_DP8 EP8
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp8_ep8_decode_8",
        lengths=(1,) * 8,
        n_kq=16,
        n_v=64,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp8_ep8_decode_64",
        lengths=(1,) * 64,
        n_kq=16,
        n_v=64,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp8_ep8_mixed_8_t1024",
        lengths=(1,) * 7 + (1017,),
        n_kq=16,
        n_v=64,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp8_ep8_mixed_64_t1024",
        lengths=(1,) * 63 + (961,),
        n_kq=16,
        n_v=64,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp8_ep8_prefill_512",
        lengths=(512,),
        n_kq=16,
        n_v=64,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp8_ep8_prefill_1024",
        lengths=(1024,),
        n_kq=16,
        n_v=64,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp8_ep8_prefill_2048",
        lengths=(2048,),
        n_kq=16,
        n_v=64,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp8_ep8_prefill_4096",
        lengths=(4096,),
        n_kq=16,
        n_v=64,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp8_ep8_decode_16_ctx8192",
        lengths=(1,) * 16,
        context_lens=8192,
        n_kq=16,
        n_v=64,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp8_ep8_decode_32_ctx8192",
        lengths=(1,) * 32,
        context_lens=8192,
        n_kq=16,
        n_v=64,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp8_ep8_decode_128_ctx1024",
        lengths=(1,) * 128,
        context_lens=1024,
        n_kq=16,
        n_v=64,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp8_ep8_mixed_32_t512",
        lengths=(1,) * 31 + (481,),
        context_lens=(8192,) * 31 + (0,),
        n_kq=16,
        n_v=64,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp8_ep8_mixed_32_t2048",
        lengths=(1,) * 31 + (2017,),
        context_lens=(8192,) * 31 + (0,),
        n_kq=16,
        n_v=64,
    ),
    # Long context continuation and multi sequence decode
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_delta_prefill_1536_ctx65536",
        lengths=(1536,),
        context_lens=65536,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_delta_prefill_2048_ctx131072",
        lengths=(2048,),
        context_lens=131072,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_dp8_delta_prefill_2048_ctx65536",
        lengths=(2048,),
        context_lens=65536,
        n_kq=16,
        n_v=64,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_multi_seq_decode_8_ctx32768",
        lengths=(1,) * 8,
        context_lens=32768,
        n_kq=2,
        n_v=8,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_397b_tp8_mixed_decode_8_prefill_2048",
        lengths=(1,) * 7 + (2048,),
        context_lens=(32768,) * 7 + (65536,),
        n_kq=2,
        n_v=8,
    ),
    # Qwen3.5 35B and Qwen3.8 Max
    GdnBenchmarkConfig(
        name="qwen3_5_35b_tp8_prefill_2048",
        lengths=(2048,),
        n_kq=2,
        n_v=4,
    ),
    GdnBenchmarkConfig(
        name="qwen3_5_35b_tp8_decode_64_ctx1024",
        lengths=(1,) * 64,
        context_lens=1024,
        n_kq=2,
        n_v=4,
    ),
    GdnBenchmarkConfig(
        name="qwen3_8_max_tp8_prefill_2048",
        lengths=(2048,),
        n_kq=2,
        n_v=16,
    ),
    GdnBenchmarkConfig(
        name="qwen3_8_max_tp8_decode_64_ctx1024",
        lengths=(1,) * 64,
        context_lens=1024,
        n_kq=2,
        n_v=16,
    ),
    GdnBenchmarkConfig(
        name="qwen3_8_max_serving_prefill_2x64",
        lengths=(64, 64),
        n_kq=4,
        n_v=8,
        dtype=jnp.bfloat16,
        state_dtype=jnp.float32,
    ),
)


def make_gdn_inputs(
    key: jax.Array,
    cfg: GdnBenchmarkConfig,
    *,
    has_bias: bool = True,
) -> dict[str, Any]:
  """Generates well conditioned inputs for a GdnBenchmarkConfig."""
  max_reqs = cfg.effective_max_reqs
  num_seqs = cfg.num_seqs
  num_tokens = cfg.num_tokens
  dim_size = cfg.dim_size
  window = cfg.num_spec_tokens + 1
  num_blocks = cfg.num_blocks

  q_loc_np = np.cumsum([0] + list(cfg.lengths), dtype=np.int32)
  if len(q_loc_np) < max_reqs + 1:
    q_loc_np = np.pad(
        q_loc_np,
        (0, max_reqs + 1 - len(q_loc_np)),
        constant_values=q_loc_np[-1],
    )

  if isinstance(cfg.context_lens, int):
    ctx_list = [cfg.context_lens] * num_seqs
  else:
    ctx_list = list(cfg.context_lens)

  s_lens_np = np.array(
      [l + c for l, c in zip(cfg.lengths, ctx_list)], dtype=np.int32
  )
  if len(s_lens_np) < max_reqs:
    s_lens_np = np.pad(
        s_lens_np, (0, max_reqs - len(s_lens_np)), constant_values=0
    )

  if cfg.distribution is not None:
    dist_np = np.array(cfg.distribution, dtype=np.int32)
  else:
    num_decode = cfg.num_decode_seqs
    if num_decode is None:
      num_decode = 0
      for length in cfg.lengths:
        if length <= window and (window > 1 or length == 1):
          num_decode += 1
        else:
          break
    dist_np = np.array([num_decode, num_seqs, num_seqs], dtype=np.int32)

  state_indices_np = np.array(
      [1 + i * window for i in range(max_reqs)], dtype=np.int32
  )

  if cfg.read_offsets is not None:
    ro_list = list(cfg.read_offsets)
    if len(ro_list) < max_reqs:
      ro_list = ro_list + [0] * (max_reqs - len(ro_list))
    read_offsets_arr = jnp.asarray(ro_list, dtype=jnp.int32)
  else:
    read_offsets_arr = jnp.zeros((max_reqs,), dtype=jnp.int32)

  keys = jax.random.split(key, 10)
  qkv = (
      jax.random.normal(keys[0], (num_tokens, dim_size), dtype=jnp.float32)
      * 0.2
  ).astype(cfg.dtype)
  b = (
      jax.random.normal(keys[1], (num_tokens, cfg.n_v), dtype=jnp.float32) * 0.5
  ).astype(cfg.dtype)
  a = (
      jax.random.normal(keys[2], (num_tokens, cfg.n_v), dtype=jnp.float32) * 0.5
  ).astype(cfg.dtype)
  conv_state = (
      jax.random.normal(
          keys[3],
          (num_blocks, cfg.kernel_size - 1, dim_size),
          dtype=jnp.float32,
      )
      * 0.1
  ).astype(cfg.dtype)
  recurrent_state = (
      jax.random.normal(
          keys[4],
          (num_blocks, cfg.n_v, cfg.d_k, cfg.d_v),
          dtype=jnp.float32,
      )
      * 0.05
  ).astype(cfg.state_dtype)
  conv_weight = (
      jax.random.normal(
          keys[5], (dim_size, 1, cfg.kernel_size), dtype=jnp.float32
      )
      * 0.2
  ).astype(cfg.dtype)
  conv_bias = None
  if has_bias:
    conv_bias = (
        jax.random.normal(keys[6], (dim_size,), dtype=jnp.float32) * 0.1
    ).astype(cfg.dtype)
  a_log = jax.random.uniform(
      keys[7], (cfg.n_v,), minval=-2.0, maxval=0.5, dtype=jnp.float32
  )
  dt_bias = (
      jax.random.normal(keys[8], (cfg.n_v,), dtype=jnp.float32) * 0.2
  ).astype(cfg.dtype)

  return dict(
      qkv=qkv,
      b=b,
      a=a,
      conv_state=conv_state,
      recurrent_state=recurrent_state,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      a_log=a_log,
      dt_bias=dt_bias,
      query_start_loc=jnp.asarray(q_loc_np, dtype=jnp.int32),
      state_indices=jnp.asarray(state_indices_np, dtype=jnp.int32),
      distribution=jnp.asarray(dist_np, dtype=jnp.int32),
      seq_lens=jnp.asarray(s_lens_np, dtype=jnp.int32),
      read_state_indices=jnp.asarray(state_indices_np, dtype=jnp.int32),
      read_offsets=read_offsets_arr,
      n_kq=cfg.n_kq,
      n_v=cfg.n_v,
      d_k=cfg.d_k,
      d_v=cfg.d_v,
      kernel_size=cfg.kernel_size,
      num_spec_tokens=cfg.num_spec_tokens,
  )


_make_gdn_inputs = make_gdn_inputs
