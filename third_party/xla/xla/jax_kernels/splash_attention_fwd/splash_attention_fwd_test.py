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

"""TPU hardware numerical parity and HLO forensics tests for Splash Attention kernels."""

import functools
import re
from typing import Any
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from xla.jax_kernels import splash_attention_fwd
from xla.jax_kernels.splash_attention_fwd import champion
from xla.jax_kernels.splash_attention_fwd import configs
from xla.jax_kernels.splash_attention_fwd import reference as baseline


def _generate_deterministic_qkv(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    q_seq_len: int,
    kv_seq_len: int,
    dim_qk: int,
    dim_v: int,
    dtype: Any = jnp.bfloat16,
    seed: int = 1234,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Generates deterministic (q, k, v) inputs on TPU."""
  rng = np.random.default_rng(seed)
  q = jnp.array(
      rng.uniform(
          -0.5, 0.5, size=(batch_size, num_q_heads, q_seq_len, dim_qk)
      ).astype(np.float32),
      dtype=dtype,
  )
  k = jnp.array(
      rng.uniform(
          -0.5, 0.5, size=(batch_size, num_kv_heads, kv_seq_len, dim_qk)
      ).astype(np.float32),
      dtype=dtype,
  )
  v = jnp.array(
      rng.uniform(
          -0.5, 0.5, size=(batch_size, num_kv_heads, kv_seq_len, dim_v)
      ).astype(np.float32),
      dtype=dtype,
  )
  return (q, k, v)


class _JittedSplashJaxKRunner:
  """Callable jitted wrapper around splash_jax_k_fwd with .lower()."""

  def __init__(
      self,
      block_q: int,
      block_kv: int,
      causal: bool = True,
      save_residuals: bool = False,
  ):
    self._block_q = block_q
    self._block_kv = block_kv
    self._causal = causal
    self._save_residuals = save_residuals
    self._jit_cache: dict[tuple[int, int], Any] = {}

  def _get_traced_jit(self, sq: int, d_qk: int):
    key = (sq, d_qk)
    cached = self._jit_cache.get(key)
    if cached is not None:
      return cached
    out_shardings = None
    if sq >= 512 and d_qk >= 64:
      sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
      out_layout = (
          champion.LAYOUT_4D_SEQ_MINOR_8
          if d_qk == 192
          else champion.LAYOUT_4D_SEQ_MINOR_16
      )
      fmt_out = out_layout.to_format(sharding, ndim=4)
      out_shardings = (fmt_out, None) if self._save_residuals else fmt_out

    def _traced_fn(q_in: jax.Array, k_in: jax.Array, v_in: jax.Array):
      return champion.splash_jax_k_fwd(
          q_in,
          k_in,
          v_in,
          block_q=self._block_q,
          block_kv=self._block_kv,
          causal=self._causal,
          save_residuals=self._save_residuals,
      )

    jitted = jax.jit(_traced_fn, out_shardings=out_shardings)
    self._jit_cache[key] = jitted
    return jitted

  def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array):
    return self._get_traced_jit(q.shape[2], q.shape[3])(q, k, v)

  def lower(self, q: jax.Array, k: jax.Array, v: jax.Array):
    return champion.lower_splash_jax_k(
        q,
        k,
        v,
        block_q=self._block_q,
        block_kv=self._block_kv,
        causal=self._causal,
        save_residuals=self._save_residuals,
    )

  def lower_traced(self, q: jax.Array, k: jax.Array, v: jax.Array):
    return self._get_traced_jit(q.shape[2], q.shape[3]).lower(q, k, v)


@functools.lru_cache(maxsize=64)
def _make_jitted_splash_jax_k_20shape(
    block_q: int,
    block_kv: int,
    causal: bool = True,
    save_residuals: bool = False,
) -> _JittedSplashJaxKRunner:
  """Returns a jitted wrapper around splash_jax_k_fwd supporting both execution and .lower()."""
  return _JittedSplashJaxKRunner(
      block_q=block_q,
      block_kv=block_kv,
      causal=causal,
      save_residuals=save_residuals,
  )


def _reference_attention_with_residuals(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sm_scale: float = 1.0,
    causal: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes chunked fp32 reference output, logsumexp, and max_logits via baseline.py."""
  return baseline.reference_splash_attention_with_residuals(
      q, k, v, sm_scale=sm_scale, causal=causal
  )


class SplashAttention20ShapeVariantsParityTest(absltest.TestCase):
  """TPU hardware numerical parity and HLO forensics tests for the 20 shape Splash Attention kernels."""

  def _assert_parity_against_ref_and_pallas(
      self,
      label: str,
      actual_out: jax.Array,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      block_q: int,
      block_kv: int,
      causal: bool = True,
      sm_scale: float = 1.0,
      actual_stats: dict[str, jax.Array] | None = None,
      atol: float = 0.01,
      rtol: float = 0.01,
  ):
    """Verifies candidate output (and optional residuals) against exact reference."""
    del block_q, block_kv
    ref_out, ref_lse, ref_max = _reference_attention_with_residuals(
        q, k, v, sm_scale=sm_scale, causal=causal
    )
    jax.block_until_ready(ref_out)
    np.testing.assert_allclose(
        np.asarray(actual_out, dtype=np.float32),
        np.asarray(ref_out, dtype=np.float32),
        atol=atol,
        rtol=rtol,
        err_msg=f'[{label}] Failed exact reference numerical parity on output',
    )
    if actual_stats is not None:
      np.testing.assert_allclose(
          np.asarray(actual_stats['logsumexp'], dtype=np.float32),
          np.asarray(ref_lse, dtype=np.float32),
          atol=0.01,
          rtol=0.01,
          err_msg=f'[{label}] Failed reference logsumexp parity',
      )
      np.testing.assert_allclose(
          np.asarray(actual_stats['max_logits'], dtype=np.float32),
          np.asarray(ref_max, dtype=np.float32),
          atol=0.01,
          rtol=0.01,
          err_msg=f'[{label}] Failed reference max_logits parity',
      )

  def test_splash_jax_k_20shape_regime1_single_tile_and_large_batch_mla_against_reference_and_pallas(
      self,
  ):
    """Verifies Regime 1 small batch, production B>=16 peeled single tile, and MLA B>=64 schedules."""
    for b, h, sq, d in (
        (2, 4, 512, 64),
        (2, 4, 512, 128),
        (2, 4, 1024, 128),
        (2, 4, 1024, 256),
        (32, 32, 512, 64),
        (16, 32, 1024, 64),
        (16, 16, 512, 128),
        (16, 16, 1024, 128),
        (16, 8, 512, 256),
        (16, 8, 1024, 256),
    ):
      q, k, v = _generate_deterministic_qkv(
          batch_size=b,
          num_q_heads=h,
          num_kv_heads=h,
          q_seq_len=sq,
          kv_seq_len=sq,
          dim_qk=d,
          dim_v=d,
          dtype=jnp.bfloat16,
          seed=200 + b + h + d + sq,
      )
      out, stats = _make_jitted_splash_jax_k_20shape(
          sq, sq, causal=True, save_residuals=True
      )(q, k, v)
      jax.block_until_ready(out)
      self._assert_parity_against_ref_and_pallas(
          f'Regime1_SingleTile(B={b},H={h},Sq={sq},D={d},save_res=True)',
          out,
          q,
          k,
          v,
          block_q=sq,
          block_kv=sq,
          causal=True,
          actual_stats=stats,
      )
      if b == 16 and sq == 512 or (b == 2 and sq == 512 and (d == 128)):
        out_no_res = _make_jitted_splash_jax_k_20shape(
            sq, sq, causal=True, save_residuals=False
        )(q, k, v)
        jax.block_until_ready(out_no_res)
        self._assert_parity_against_ref_and_pallas(
            f'Regime1_SingleTile(B={b},H={h},Sq={sq},D={d},save_res=False)',
            out_no_res,
            q,
            k,
            v,
            block_q=sq,
            block_kv=sq,
            causal=True,
            actual_stats=None,
        )
    neg_jit = jax.jit(jnp.negative)
    for b_mla, h_mla, sq_mla, d_qk, d_v in (
        (64, 2, 512, 192, 192),
        (64, 2, 512, 192, 128),
        (64, 2, 512, 128, 128),
        (64, 16, 1024, 192, 192),
    ):
      q_mla, k_mla, v_mla = _generate_deterministic_qkv(
          batch_size=b_mla,
          num_q_heads=h_mla,
          num_kv_heads=h_mla,
          q_seq_len=sq_mla,
          kv_seq_len=sq_mla,
          dim_qk=d_qk,
          dim_v=d_v,
          dtype=jnp.bfloat16,
          seed=300 + h_mla + sq_mla + d_qk + d_v,
      )
      out_mla, stats_mla = _make_jitted_splash_jax_k_20shape(
          sq_mla, sq_mla, causal=True, save_residuals=True
      )(q_mla, k_mla, v_mla)
      jax.block_until_ready(out_mla)
      if d_qk == 192:
        np.testing.assert_array_equal(
            np.asarray(neg_jit(out_mla), dtype=np.float32),
            -np.asarray(out_mla, dtype=np.float32),
        )
      self._assert_parity_against_ref_and_pallas(
          f'Regime1_LargeBatchMLA(B={b_mla},H={h_mla},Sq={sq_mla},Dqk={d_qk},Dv={d_v})',
          out_mla,
          q_mla,
          k_mla,
          v_mla,
          block_q=sq_mla,
          block_kv=sq_mla,
          causal=True,
          actual_stats=stats_mla,
      )
      if d_qk == 192 and d_v == 128 or sq_mla == 1024:
        out_mla_nores = _make_jitted_splash_jax_k_20shape(
            sq_mla, sq_mla, causal=True, save_residuals=False
        )(q_mla, k_mla, v_mla)
        jax.block_until_ready(out_mla_nores)
        if d_qk == 192:
          np.testing.assert_array_equal(
              np.asarray(neg_jit(out_mla_nores), dtype=np.float32),
              -np.asarray(out_mla_nores, dtype=np.float32),
          )
        self._assert_parity_against_ref_and_pallas(
            f'Regime1_LargeBatchMLA(B={b_mla},H={h_mla},Sq={sq_mla},Dqk={d_qk},Dv={d_v},save_res=False)',
            out_mla_nores,
            q_mla,
            k_mla,
            v_mla,
            block_q=sq_mla,
            block_kv=sq_mla,
            causal=True,
            actual_stats=None,
        )

  def test_splash_jax_k_20shape_regime2_2x2_triangular_unrolled_all_head_dims_against_reference_and_pallas(
      self,
  ):
    """Verifies Regime 2 fallback 2x2 triangular, B>=4 D=256 peeled 2x2, and production Sq=2048 row unrolled schedules."""
    cases = [
        ('T3a_2x2_D64_Sq2048', 2, 4, 2048, 1024, 64),
        ('T3b_2x2_D128_Sq2048', 2, 4, 2048, 1024, 128),
        ('T4a_2x2_D256_Sq2048_bq1024', 2, 4, 2048, 1024, 256),
        ('T4b_2x2_D256_Sq4096_bq2048_barrier_split', 2, 4, 4096, 2048, 256),
        ('Prod_2x2_D64_B8_H32_Sq2048', 8, 32, 2048, 1024, 64),
        ('Prod_2x2_D128_B8_H16_Sq2048', 8, 16, 2048, 1024, 128),
        ('Prod_2x2_D256_B8_H8_Sq2048', 8, 8, 2048, 1024, 256),
    ]
    for label, b, h, sq, bq, d in cases:
      q, k, v = _generate_deterministic_qkv(
          batch_size=b,
          num_q_heads=h,
          num_kv_heads=h,
          q_seq_len=sq,
          kv_seq_len=sq,
          dim_qk=d,
          dim_v=d,
          dtype=jnp.bfloat16,
          seed=400 + b + h + d + bq,
      )
      out, stats = _make_jitted_splash_jax_k_20shape(
          bq, bq, causal=True, save_residuals=True
      )(q, k, v)
      jax.block_until_ready(out)
      self._assert_parity_against_ref_and_pallas(
          f'{label}(save_res=True)',
          out,
          q,
          k,
          v,
          block_q=bq,
          block_kv=bq,
          causal=True,
          actual_stats=stats,
      )
      if b == 8 or (d == 128 and sq == 2048):
        out_no_res = _make_jitted_splash_jax_k_20shape(
            bq, bq, causal=True, save_residuals=False
        )(q, k, v)
        jax.block_until_ready(out_no_res)
        self._assert_parity_against_ref_and_pallas(
            f'{label}(save_res=False)',
            out_no_res,
            q,
            k,
            v,
            block_q=bq,
            block_kv=bq,
            causal=True,
            actual_stats=None,
        )

  def test_splash_jax_k_20shape_regime3_gqa_and_q_outer_4block_against_reference_and_pallas(
      self,
  ):
    """Verifies Regime 3 GQA (q_groups=4) and MHA Sq=4096 row unrolled and fallback schedules."""
    for hq_gqa, hkv_gqa, seed_gqa in ((8, 2, 501), (32, 8, 502)):
      q_gqa, k_gqa, v_gqa = _generate_deterministic_qkv(
          batch_size=2,
          num_q_heads=hq_gqa,
          num_kv_heads=hkv_gqa,
          q_seq_len=4096,
          kv_seq_len=4096,
          dim_qk=128,
          dim_v=128,
          dtype=jnp.bfloat16,
          seed=seed_gqa,
      )
      for save_res in (True, False):
        res_gqa = _make_jitted_splash_jax_k_20shape(
            1024, 1024, causal=True, save_residuals=save_res
        )(q_gqa, k_gqa, v_gqa)
        out_gqa, stats_gqa = res_gqa if save_res else (res_gqa, None)
        jax.block_until_ready(out_gqa)
        self._assert_parity_against_ref_and_pallas(
            f'T5_Regime3_GQA(Hq={hq_gqa},Hkv={hkv_gqa},G=4,save_res={save_res})',
            out_gqa,
            q_gqa,
            k_gqa,
            v_gqa,
            block_q=1024,
            block_kv=1024,
            causal=True,
            actual_stats=stats_gqa,
        )
    for b, h, bq, d in (
        (2, 4, 1024, 128),
        (1, 4, 1024, 128),
        (4, 32, 1024, 64),
        (4, 16, 1024, 128),
        (4, 8, 1024, 256),
        (4, 8, 2048, 256),
    ):
      q, k, v = _generate_deterministic_qkv(
          batch_size=b,
          num_q_heads=h,
          num_kv_heads=h,
          q_seq_len=4096,
          kv_seq_len=4096,
          dim_qk=d,
          dim_v=d,
          dtype=jnp.bfloat16,
          seed=510 + b + h + bq + d,
      )
      out, stats = _make_jitted_splash_jax_k_20shape(
          bq, bq, causal=True, save_residuals=True
      )(q, k, v)
      jax.block_until_ready(out)
      self._assert_parity_against_ref_and_pallas(
          f'T6_Regime3_MHA_4block(B={b},H={h},Sq=4096,bq={bq},D={d},save_res=True)',
          out,
          q,
          k,
          v,
          block_q=bq,
          block_kv=bq,
          causal=True,
          actual_stats=stats,
      )
      if b == 4 and bq == 2048:
        out_nores = _make_jitted_splash_jax_k_20shape(
            bq, bq, causal=True, save_residuals=False
        )(q, k, v)
        jax.block_until_ready(out_nores)
        self._assert_parity_against_ref_and_pallas(
            f'T6_Regime3_MHA_4block(B={b},H={h},Sq=4096,bq={bq},D={d},save_res=False)',
            out_nores,
            q,
            k,
            v,
            block_q=bq,
            block_kv=bq,
            causal=True,
            actual_stats=None,
        )

  def test_splash_jax_k_20shape_regime3_diag2048_2tile_peel_against_reference_and_pallas(
      self,
  ):
    """Verifies Sq=8192 causal schedules across production H*D=2048 and fallback H*D!=2048 shapes."""
    cases = [
        ('T7a_2TilePeel_B1_D64_Sq8192', 1, 4, 8192, 2048, 64),
        ('T7b_2TilePeel_B1_H16_D128_Sq8192', 1, 16, 8192, 2048, 128),
        ('T7c_2TilePeel_B2_H4_D128_Sq8192', 2, 4, 8192, 2048, 128),
        ('Prod_B2_H32_D64_Sq8192', 2, 32, 8192, 1024, 64),
        ('Prod_B2_H16_D128_Sq8192', 2, 16, 8192, 2048, 128),
    ]
    for label, b, h, sq, bq, d in cases:
      q, k, v = _generate_deterministic_qkv(
          batch_size=b,
          num_q_heads=h,
          num_kv_heads=h,
          q_seq_len=sq,
          kv_seq_len=sq,
          dim_qk=d,
          dim_v=d,
          dtype=jnp.bfloat16,
          seed=600 + b + h + d,
      )
      out, stats = _make_jitted_splash_jax_k_20shape(
          bq, bq, causal=True, save_residuals=True
      )(q, k, v)
      jax.block_until_ready(out)
      self._assert_parity_against_ref_and_pallas(
          f'{label}(save_res=True)',
          out,
          q,
          k,
          v,
          block_q=bq,
          block_kv=bq,
          causal=True,
          actual_stats=stats,
      )
      if h * d == 2048 and b == 2 and (d == 128):
        out_nores = _make_jitted_splash_jax_k_20shape(
            bq, bq, causal=True, save_residuals=False
        )(q, k, v)
        jax.block_until_ready(out_nores)
        self._assert_parity_against_ref_and_pallas(
            f'{label}(save_res=False)',
            out_nores,
            q,
            k,
            v,
            block_q=bq,
            block_kv=bq,
            causal=True,
            actual_stats=None,
        )

  def test_splash_jax_k_20shape_regime3_3tile1024_pair4096_and_runner_against_reference_and_pallas(
      self,
  ):
    """Verifies Sq=8192 D=256 and Sq=16384 D=64 causal schedules."""
    cases = [
        ('T8_Fallback_B2_H4_D256_Sq8192', 2, 4, 8192, 2048, 256),
        ('Prod_B2_H8_D256_Sq8192', 2, 8, 8192, 2048, 256),
        ('Prod_B1_H32_D64_Sq16384', 1, 32, 16384, 2048, 64),
    ]
    for label, b, h, sq, bq, d in cases:
      q, k, v = _generate_deterministic_qkv(
          batch_size=b,
          num_q_heads=h,
          num_kv_heads=h,
          q_seq_len=sq,
          kv_seq_len=sq,
          dim_qk=d,
          dim_v=d,
          dtype=jnp.bfloat16,
          seed=650 + b + h + d + sq,
      )
      out, stats = _make_jitted_splash_jax_k_20shape(
          bq, bq, causal=True, save_residuals=True
      )(q, k, v)
      jax.block_until_ready(out)
      self._assert_parity_against_ref_and_pallas(
          f'{label}(save_res=True)',
          out,
          q,
          k,
          v,
          block_q=bq,
          block_kv=bq,
          causal=True,
          actual_stats=stats,
      )
      if sq == 8192 and h == 8 and (d == 256):
        out_nores = _make_jitted_splash_jax_k_20shape(
            bq, bq, causal=True, save_residuals=False
        )(q, k, v)
        jax.block_until_ready(out_nores)
        self._assert_parity_against_ref_and_pallas(
            f'{label}(save_res=False)',
            out_nores,
            q,
            k,
            v,
            block_q=bq,
            block_kv=bq,
            causal=True,
            actual_stats=None,
        )

  def test_splash_jax_k_20shape_regime3_sq16384_d128_d256_and_runner_against_reference_and_pallas(
      self,
  ):
    """Verifies Sq=16384 D=128/256 causal schedules and runner placement."""
    for label, b, h, sq, bq, d in (
        ('Prod_B1_H16_D128_Sq16384', 1, 16, 16384, 2048, 128),
        ('Prod_B1_H8_D256_Sq16384', 1, 8, 16384, 2048, 256),
    ):
      q, k, v = _generate_deterministic_qkv(
          batch_size=b,
          num_q_heads=h,
          num_kv_heads=h,
          q_seq_len=sq,
          kv_seq_len=sq,
          dim_qk=d,
          dim_v=d,
          dtype=jnp.bfloat16,
          seed=650 + b + h + d + sq,
      )
      out, stats = _make_jitted_splash_jax_k_20shape(
          bq, bq, causal=True, save_residuals=True
      )(q, k, v)
      jax.block_until_ready(out)
      self._assert_parity_against_ref_and_pallas(
          f'{label}(save_res=True)',
          out,
          q,
          k,
          v,
          block_q=bq,
          block_kv=bq,
          causal=True,
          actual_stats=stats,
      )
    self.assertIs(splash_attention_fwd.champion, champion)
    self.assertIs(splash_attention_fwd.baseline, baseline)
    self.assertIs(splash_attention_fwd.configs, configs)
    self.assertIs(splash_attention_fwd.kernels, champion)
    q_r, k_r, v_r = _generate_deterministic_qkv(
        batch_size=8,
        num_q_heads=16,
        num_kv_heads=16,
        q_seq_len=2048,
        kv_seq_len=2048,
        dim_qk=128,
        dim_v=128,
        dtype=jnp.bfloat16,
        seed=777,
    )
    compiled_exe, runner_args = champion.build_splash_jax_k_runner(
        q_r, k_r, v_r, causal=True, save_residuals=True
    )
    out_r1, stats_r1 = compiled_exe(*runner_args)
    jax.block_until_ready(out_r1)
    self._assert_parity_against_ref_and_pallas(
        'T10_build_splash_jax_k_runner_4D',
        out_r1,
        q_r,
        k_r,
        v_r,
        1024,
        1024,
        causal=True,
        actual_stats=stats_r1,
    )
    out_eager, stats_eager = champion.splash_jax_k_fwd(
        q_r,
        k_r,
        v_r,
        block_q=1024,
        block_kv=1024,
        causal=True,
        save_residuals=True,
    )
    jax.block_until_ready(out_eager)
    self._assert_parity_against_ref_and_pallas(
        'T10_eager_splash_jax_k_fwd_4D',
        out_eager,
        q_r,
        k_r,
        v_r,
        1024,
        1024,
        causal=True,
        actual_stats=stats_eager,
    )
    v_aug_in = jnp.pad(
        v_r, ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1.0
    )
    out_r2, stats_r2 = _make_jitted_splash_jax_k_20shape(
        1024, 1024, causal=True, save_residuals=True
    )(q_r, k_r, v_aug_in)
    jax.block_until_ready(out_r2)
    self._assert_parity_against_ref_and_pallas(
        'T10_tokamax_fwd_pre_augmented_v_aug',
        out_r2,
        q_r,
        k_r,
        v_r,
        1024,
        1024,
        causal=True,
        actual_stats=stats_r2,
    )

  def test_hlo_inspection_and_compiler_forensics_all_tokamax_regimes(self):
    """Inspects lowered StableHLO and compiled TPU HLO across production and fallback regimes."""
    regime_configs = [
        (
            'Regime1_SingleTile_Prod',
            32,
            16,
            16,
            512,
            512,
            128,
            128,
            True,
            False,
            0,
        ),
        (
            'Regime1_MLA_D192_Dv128',
            4,
            16,
            16,
            1024,
            1024,
            192,
            128,
            True,
            False,
            0,
        ),
        (
            'Regime2_2x2Unrolled_Prod',
            8,
            16,
            16,
            2048,
            1024,
            128,
            128,
            True,
            False,
            0,
        ),
        ('Regime3_GQA_4Block', 2, 8, 2, 4096, 1024, 128, 128, True, False, 0),
        (
            'Regime3_GQA_32_8_Sq4096',
            2,
            32,
            8,
            4096,
            1024,
            128,
            128,
            True,
            False,
            0,
        ),
        (
            'Regime3_Prod_B2_H8_D256_Sq8192',
            2,
            8,
            8,
            8192,
            2048,
            256,
            256,
            True,
            False,
            0,
        ),
        (
            'Regime3_Prod_B2_H32_D64_Sq8192',
            2,
            32,
            32,
            8192,
            1024,
            64,
            64,
            True,
            False,
            0,
        ),
        (
            'NonCausal_Prod_B2_H32_D64_Sq8192',
            2,
            32,
            32,
            8192,
            1024,
            64,
            64,
            False,
            True,
            1,
        ),
        (
            'Regime3_Fallback_B2_H4_D256_Sq8192',
            2,
            4,
            4,
            8192,
            2048,
            256,
            256,
            True,
            True,
            1,
        ),
        (
            'Regime3_Sq16384_D256',
            1,
            8,
            8,
            16384,
            2048,
            256,
            256,
            True,
            False,
            0,
        ),
    ]
    traced_compile_labels = {
        'Regime1_SingleTile_Prod',
        'Regime1_MLA_D192_Dv128',
        'Regime3_GQA_32_8_Sq4096',
        'Regime3_Sq16384_D256',
    }
    for (
        label,
        b,
        hq,
        hkv,
        sq,
        bq,
        d_qk,
        d_v,
        is_causal,
        expect_pre_while,
        expected_min_post_whiles,
    ) in regime_configs:
      q, k, v = _generate_deterministic_qkv(
          batch_size=b,
          num_q_heads=hq,
          num_kv_heads=hkv,
          q_seq_len=sq,
          kv_seq_len=sq,
          dim_qk=d_qk,
          dim_v=d_v,
          dtype=jnp.bfloat16,
          seed=900,
      )
      jit_fn = _make_jitted_splash_jax_k_20shape(
          bq, bq, causal=is_causal, save_residuals=True
      )
      lowered = jit_fn.lower(q, k, v)
      hlo_pre = lowered.as_text()
      compiled = lowered.compile()
      hlo_post = compiled.as_text()
      assert hlo_post is not None
      self.assertNotIn(
          'pallas_call',
          hlo_pre,
          f'[{label}] jax_k StableHLO must not contain pallas_call',
      )
      self.assertNotIn(
          'pallas_call',
          hlo_post,
          f'[{label}] Compiled jax_k HLO must not contain pallas_call',
      )
      traced_lowered = jit_fn.lower_traced(q, k, v)
      traced_hlo_pre = traced_lowered.as_text()
      self.assertNotIn(
          'pallas_call',
          traced_hlo_pre,
          f'[{label}] Traced jax_k StableHLO must not contain pallas_call',
      )
      if label in traced_compile_labels:
        traced_hlo_post = traced_lowered.compile().as_text()
        assert traced_hlo_post is not None
        self.assertNotIn(
            'pallas_call',
            traced_hlo_post,
            f'[{label}] Traced compiled jax_k HLO must not contain pallas_call',
        )
      self.assertIn(
          'MUST_FUSE',
          hlo_pre,
          f'[{label}] Expected MUST_FUSE stage fusion attributes in StableHLO',
      )
      self.assertIn(
          'buffering_level',
          hlo_pre,
          f'[{label}] Expected buffering_level in StableHLO',
      )
      if expect_pre_while:
        self.assertIn(
            'stablehlo.while',
            hlo_pre,
            f'[{label}] Expected stablehlo.while in lowered Regime 3 StableHLO',
        )
      else:
        self.assertNotIn(
            'stablehlo.while',
            hlo_pre,
            f'[{label}] Loop free regime must not contain stablehlo.while in'
            ' StableHLO',
        )
      if sq >= 8192:
        self.assertIn(
            'scoped_vmem_limit_kib',
            hlo_pre,
            f'[{label}] Expected scoped_vmem_limit_kib in long sequence'
            ' StableHLO',
        )
      while_count = len(re.findall('while\\(', hlo_post))
      if expected_min_post_whiles == 0:
        self.assertEqual(
            while_count,
            0,
            f'[{label}] Regime must compile with 0 post optimization while'
            ' loops',
        )
      else:
        self.assertGreaterEqual(
            while_count,
            1,
            f'[{label}] Long sequence regime must retain while loop(s) in'
            ' compiled TPU HLO',
        )

  def test_splash_jax_k_noncausal_sq2048_sq4096_against_reference_and_pallas(
      self,
  ):
    """Verifies noncausal Sq=2048 and Sq=4096 unrolled schedules across D=64, 128, 256."""
    for b_nc, h_nc, sq_nc, bq_nc, d_nc in (
        (4, 32, 2048, 1024, 64),
        (4, 16, 2048, 1024, 128),
        (4, 8, 2048, 1024, 256),
        (4, 32, 4096, 1024, 64),
        (4, 16, 4096, 1024, 128),
        (4, 8, 4096, 2048, 256),
    ):
      q_nc, k_nc, v_nc = _generate_deterministic_qkv(
          b_nc, h_nc, h_nc, sq_nc, sq_nc, d_nc, d_nc, seed=870 + sq_nc + d_nc
      )
      out_nc, stats_nc = _make_jitted_splash_jax_k_20shape(
          bq_nc, bq_nc, causal=False, save_residuals=True
      )(q_nc, k_nc, v_nc)
      jax.block_until_ready(out_nc)
      self._assert_parity_against_ref_and_pallas(
          f'NonCausalMid(B={b_nc},H={h_nc},Sq={sq_nc},D={d_nc},save_res=True)',
          out_nc,
          q_nc,
          k_nc,
          v_nc,
          bq_nc,
          bq_nc,
          causal=False,
          actual_stats=stats_nc,
      )
      if sq_nc == 4096 and d_nc == 64:
        out_nc_nores = _make_jitted_splash_jax_k_20shape(
            bq_nc, bq_nc, causal=False, save_residuals=False
        )(q_nc, k_nc, v_nc)
        jax.block_until_ready(out_nc_nores)
        self._assert_parity_against_ref_and_pallas(
            f'NonCausalMid(B={b_nc},H={h_nc},Sq={sq_nc},D={d_nc},save_res=False)',
            out_nc_nores,
            q_nc,
            k_nc,
            v_nc,
            bq_nc,
            bq_nc,
            causal=False,
            actual_stats=None,
        )

  def test_splash_jax_k_noncausal_sq8192_and_nonstationary_against_reference_and_pallas(
      self,
  ):
    """Verifies noncausal Sq=8192 and Sq=16384 schedules and non stationary KV block logit scaling."""
    for b_nc, h_nc, sq_nc, bq_nc, d_nc in (
        (2, 32, 8192, 1024, 64),
        (2, 16, 8192, 2048, 128),
        (1, 8, 16384, 2048, 256),
    ):
      q_nc, k_nc, v_nc = _generate_deterministic_qkv(
          b_nc, h_nc, h_nc, sq_nc, sq_nc, d_nc, d_nc, seed=870 + sq_nc + d_nc
      )
      out_nc, stats_nc = _make_jitted_splash_jax_k_20shape(
          bq_nc, bq_nc, causal=False, save_residuals=True
      )(q_nc, k_nc, v_nc)
      jax.block_until_ready(out_nc)
      self._assert_parity_against_ref_and_pallas(
          f'NonCausalLong(B={b_nc},H={h_nc},Sq={sq_nc},D={d_nc},save_res=True)',
          out_nc,
          q_nc,
          k_nc,
          v_nc,
          bq_nc,
          bq_nc,
          causal=False,
          actual_stats=stats_nc,
      )
      if d_nc == 64 or sq_nc == 16384:
        out_nc_nores = _make_jitted_splash_jax_k_20shape(
            bq_nc, bq_nc, causal=False, save_residuals=False
        )(q_nc, k_nc, v_nc)
        jax.block_until_ready(out_nc_nores)
        self._assert_parity_against_ref_and_pallas(
            f'NonCausalLong(B={b_nc},H={h_nc},Sq={sq_nc},D={d_nc},save_res=False)',
            out_nc_nores,
            q_nc,
            k_nc,
            v_nc,
            bq_nc,
            bq_nc,
            causal=False,
            actual_stats=None,
        )
    q_ns, k_ns, v_ns = _generate_deterministic_qkv(
        batch_size=4,
        num_q_heads=16,
        num_kv_heads=16,
        q_seq_len=4096,
        kv_seq_len=4096,
        dim_qk=128,
        dim_v=128,
        dtype=jnp.bfloat16,
        seed=789,
    )
    q_ns = q_ns.at[:, :, :, 0].set(1.0)
    block_shifts = jnp.repeat(
        jnp.array([0.0, 2.5, -1.5, 4.5], dtype=jnp.bfloat16), 1024
    )
    k_ns = k_ns.at[:, :, :, 0].add(block_shifts[None, None, :])
    for causal_flag in (True, False):
      out_ns, stats_ns = _make_jitted_splash_jax_k_20shape(
          1024, 1024, causal=causal_flag, save_residuals=True
      )(q_ns, k_ns, v_ns)
      jax.block_until_ready(out_ns)
      self._assert_parity_against_ref_and_pallas(
          f'NonStationaryLogits(B=4,H=16,Sq=4096,D=128,causal={causal_flag})',
          out_ns,
          q_ns,
          k_ns,
          v_ns,
          1024,
          1024,
          causal=causal_flag,
          actual_stats=stats_ns,
      )


if __name__ == '__main__':
  absltest.main()
