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

"""Unit and Hypothesis correctness tests for jax_k Fused Conv1D GDN kernels."""

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
import numpy as np
from xla.jax_kernels import fused_conv1d_gdn
from xla.jax_kernels.fused_conv1d_gdn import champion
from xla.jax_kernels.fused_conv1d_gdn import configs
from xla.jax_kernels.fused_conv1d_gdn import reference as baseline

STATIC_ARGNAMES = (
    'n_kq',
    'n_v',
    'd_k',
    'd_v',
    'kernel_size',
    'num_spec_tokens',
)
LOCAL_ATTENTION_CASES = (
    dict(
        testcase_name='prefill_2048',
        max_reqs=1,
        lengths=[2048],
        q_loc=[0, 2048],
        distribution=[0, 1, 1],
    ),
    dict(
        testcase_name='prefill_8192',
        max_reqs=1,
        lengths=[8192],
        q_loc=[0, 8192],
        distribution=[0, 0, 3],
    ),
    dict(
        testcase_name='mixed',
        max_reqs=3,
        lengths=[256, 128, 128],
        q_loc=[0, 256, 384, 512],
        distribution=[0, 3, 3],
    ),
    dict(
        testcase_name='decode_only',
        max_reqs=64,
        lengths=[1] * 64,
        q_loc=list(range(65)),
        distribution=[64, 64, 64],
    ),
    dict(
        testcase_name='mixed_prefill_decode',
        max_reqs=11,
        lengths=[1] * 8 + [128, 128, 256],
        q_loc=[0, 1, 2, 3, 4, 5, 6, 7, 8, 136, 264, 520],
        distribution=[8, 11, 11],
    ),
    dict(
        testcase_name='padded_mixed_prefill',
        max_reqs=16,
        lengths=[128, 64, 32, 16, 8],
        q_loc=[0, 128, 192, 224, 240, 248] + [1] * 11,
        distribution=[0, 5, 5],
    ),
    dict(
        testcase_name='padded_decode_only',
        max_reqs=512,
        lengths=[1] * 64,
        q_loc=list(range(65)) + [1] * 448,
        distribution=[64, 64, 64],
    ),
    dict(
        testcase_name='long_context',
        max_reqs=2,
        lengths=[65536, 65536],
        q_loc=[0, 65536, 131072],
        distribution=[0, 2, 2],
    ),
    dict(
        testcase_name='long_context_ragged',
        max_reqs=2,
        lengths=[1, 131071],
        q_loc=[0, 1, 131072],
        distribution=[0, 2, 2],
    ),
)
SPEC_CASES = (
    dict(
        testcase_name='spec_windows',
        spec_lengths=(5, 3, 1),
        read_offsets=(2, 0, 4),
        prefill_lengths=(),
    ),
    dict(
        testcase_name='spec_windows_padded',
        spec_lengths=(5, 1, 3, 2, 4),
        read_offsets=(0, 1, 2, 3, 4),
        prefill_lengths=(),
    ),
    dict(
        testcase_name='spec_and_prefill',
        spec_lengths=(5, 2),
        read_offsets=(3, 1),
        prefill_lengths=(64,),
    ),
)


class GdnKernelsTest(parameterized.TestCase):

  def _assert_jax_k_matches_pallas(self, kwargs: dict[str, Any], label: str):
    jax_k_fn = jax.jit(
        champion.fused_conv1d_gdn_jax_k, static_argnames=STATIC_ARGNAMES
    )
    pallas_fn = baseline.fused_conv1d_gdn_reference
    (jax_k_conv, jax_k_rec), jax_k_out = jax_k_fn(**kwargs)
    (pallas_conv, pallas_rec), pallas_out = pallas_fn(**kwargs)
    self.assertEqual(jax_k_out.dtype, pallas_out.dtype)
    self.assertEqual(jax_k_conv.dtype, pallas_conv.dtype)
    self.assertEqual(jax_k_rec.dtype, pallas_rec.dtype)
    np.testing.assert_allclose(
        np.asarray(jax_k_out, dtype=np.float32),
        np.asarray(pallas_out, dtype=np.float32),
        rtol=0.01,
        atol=0.01,
        err_msg=f'Output mismatch vs Pallas on {label}',
    )
    np.testing.assert_allclose(
        np.asarray(jax_k_conv, dtype=np.float32),
        np.asarray(pallas_conv, dtype=np.float32),
        rtol=0.01,
        atol=0.01,
        err_msg=f'Conv state mismatch vs Pallas on {label}',
    )
    jax_k_rec_f32 = np.asarray(jax_k_rec, dtype=np.float32)
    pallas_rec_f32 = np.asarray(pallas_rec, dtype=np.float32)
    if not np.allclose(jax_k_rec_f32, pallas_rec_f32, rtol=0.01, atol=0.01):
      (_, ref_rec), _ = baseline.fused_conv1d_gdn_reference(**kwargs)
      pallas_rec_f32 = np.asarray(ref_rec, dtype=np.float32)
    np.testing.assert_allclose(
        jax_k_rec_f32,
        pallas_rec_f32,
        rtol=0.01,
        atol=0.01,
        err_msg=f'Recurrent state mismatch vs Pallas on {label}',
    )

  @parameterized.named_parameters(*LOCAL_ATTENTION_CASES)
  def test_0_local_attention_cases_vs_pallas(
      self,
      max_reqs: int,
      lengths: list[int],
      q_loc: list[int],
      distribution: list[int],
  ):
    cfg = configs.GdnBenchmarkConfig(
        name='local_case',
        lengths=tuple(lengths),
        max_reqs=max_reqs,
        distribution=(distribution[0], distribution[1], distribution[2]),
        n_kq=2,
        n_v=8,
        d_k=128,
        d_v=128,
        kernel_size=4,
        dtype=jnp.float32,
        state_dtype=jnp.float32,
    )
    kwargs = configs.make_gdn_inputs(jax.random.PRNGKey(42), cfg)
    kwargs['query_start_loc'] = jnp.asarray(q_loc, dtype=jnp.int32)
    kwargs['seq_lens'] = jnp.asarray(
        kwargs['query_start_loc'][1 : max_reqs + 1]
        - kwargs['query_start_loc'][:max_reqs],
        dtype=jnp.int32,
    )
    self._assert_jax_k_matches_pallas(
        kwargs, f'local_{max_reqs}_{len(lengths)}'
    )

  @parameterized.named_parameters(
      *[(cfg.name, cfg) for cfg in configs.GDN_BENCHMARK_CONFIGS]
  )
  def test_1_benchmark_configs_vs_pallas(self, cfg: configs.GdnBenchmarkConfig):
    kwargs = configs.make_gdn_inputs(jax.random.PRNGKey(101), cfg)
    self._assert_jax_k_matches_pallas(kwargs, cfg.name)

  def test_2_continuation_and_stale_slot_isolation(self):
    cfg_fresh = configs.GdnBenchmarkConfig(
        name='stale_check',
        lengths=(64, 64),
        context_lens=0,
        n_kq=2,
        n_v=8,
        dtype=jnp.float32,
        state_dtype=jnp.float32,
    )
    kwargs_stale = configs.make_gdn_inputs(jax.random.PRNGKey(202), cfg_fresh)
    kwargs_zero = dict(kwargs_stale)
    kwargs_zero['conv_state'] = jnp.zeros_like(kwargs_stale['conv_state'])
    kwargs_zero['recurrent_state'] = jnp.zeros_like(
        kwargs_stale['recurrent_state']
    )
    jax_k_fn = jax.jit(
        champion.fused_conv1d_gdn_jax_k, static_argnames=STATIC_ARGNAMES
    )
    (conv_s, rec_s), out_s = jax_k_fn(**kwargs_stale)
    (conv_z, rec_z), out_z = jax_k_fn(**kwargs_zero)
    np.testing.assert_allclose(out_s, out_z, rtol=1e-05, atol=1e-05)
    np.testing.assert_allclose(conv_s[1:3], conv_z[1:3], rtol=1e-05, atol=1e-05)
    np.testing.assert_allclose(rec_s[1:3], rec_z[1:3], rtol=1e-05, atol=1e-05)
    cfg_full = configs.GdnBenchmarkConfig(
        name='cont_full',
        lengths=(128,),
        context_lens=0,
        n_kq=2,
        n_v=8,
        dtype=jnp.float32,
        state_dtype=jnp.float32,
    )
    kw_full = configs.make_gdn_inputs(jax.random.PRNGKey(303), cfg_full)
    kw_full['conv_state'] = jnp.zeros_like(kw_full['conv_state'])
    kw_full['recurrent_state'] = jnp.zeros_like(kw_full['recurrent_state'])
    (conv_full, rec_full), out_full = jax_k_fn(**kw_full)
    kw_a = dict(kw_full)
    kw_a['qkv'] = kw_full['qkv'][:64]
    kw_a['b'] = kw_full['b'][:64]
    kw_a['a'] = kw_full['a'][:64]
    kw_a['query_start_loc'] = jnp.array([0, 64], dtype=jnp.int32)
    kw_a['seq_lens'] = jnp.array([64], dtype=jnp.int32)
    (conv_a, rec_a), out_a = jax_k_fn(**kw_a)
    kw_b = dict(kw_full)
    kw_b['qkv'] = kw_full['qkv'][64:]
    kw_b['b'] = kw_full['b'][64:]
    kw_b['a'] = kw_full['a'][64:]
    kw_b['conv_state'] = conv_a
    kw_b['recurrent_state'] = rec_a
    kw_b['query_start_loc'] = jnp.array([0, 64], dtype=jnp.int32)
    kw_b['seq_lens'] = jnp.array([128], dtype=jnp.int32)
    (conv_b, rec_b), out_b = jax_k_fn(**kw_b)
    np.testing.assert_allclose(out_a, out_full[:64], rtol=0.01, atol=0.01)
    np.testing.assert_allclose(out_b, out_full[64:], rtol=0.01, atol=0.01)
    np.testing.assert_allclose(conv_b, conv_full, rtol=0.01, atol=0.01)
    np.testing.assert_allclose(rec_b, rec_full, rtol=0.01, atol=0.01)

  @parameterized.named_parameters(*SPEC_CASES)
  def test_3_speculative_verify_and_read_offsets_vs_pallas(
      self,
      spec_lengths: tuple[int, ...],
      read_offsets: tuple[int, ...],
      prefill_lengths: tuple[int, ...],
  ):
    lengths = tuple(spec_lengths) + tuple(prefill_lengths)
    num_spec_seqs = len(spec_lengths)
    ctx_lens = tuple([16] * num_spec_seqs + [0] * len(prefill_lengths))
    ro_tuple = tuple(read_offsets) + (0,) * len(prefill_lengths)
    cfg = configs.GdnBenchmarkConfig(
        name='spec_verify',
        lengths=lengths,
        context_lens=ctx_lens,
        num_decode_seqs=num_spec_seqs,
        num_spec_tokens=4,
        read_offsets=ro_tuple,
        n_kq=2,
        n_v=8,
        dtype=jnp.float32,
        state_dtype=jnp.float32,
    )
    kwargs = configs.make_gdn_inputs(jax.random.PRNGKey(404), cfg)
    self._assert_jax_k_matches_pallas(kwargs, f'spec_{lengths}')

  def test_3b_split_read_write_state_slots_and_isolation(self):
    jax_k_fn = jax.jit(
        champion.fused_conv1d_gdn_jax_k, static_argnames=STATIC_ARGNAMES
    )
    cfg_full = configs.GdnBenchmarkConfig(
        name='split_slots',
        lengths=(64,),
        context_lens=0,
        max_reqs=2,
        n_kq=2,
        n_v=8,
        dtype=jnp.float32,
        state_dtype=jnp.float32,
    )
    kw_full = configs.make_gdn_inputs(jax.random.PRNGKey(415), cfg_full)
    kw_full['conv_state'] = jnp.zeros_like(kw_full['conv_state'])
    kw_full['recurrent_state'] = jnp.zeros_like(kw_full['recurrent_state'])
    kw_full['query_start_loc'] = jnp.array([0, 64], dtype=jnp.int32)
    kw_full['state_indices'] = jnp.array([1], dtype=jnp.int32)
    kw_full['read_state_indices'] = jnp.array([1], dtype=jnp.int32)
    kw_full['read_offsets'] = jnp.array([0], dtype=jnp.int32)
    kw_full['distribution'] = jnp.array([0, 1, 1], dtype=jnp.int32)
    kw_full['seq_lens'] = jnp.array([64], dtype=jnp.int32)
    (_, _), out_ref = jax_k_fn(**kw_full)
    kw_a = dict(kw_full)
    kw_a['qkv'] = kw_full['qkv'][:32]
    kw_a['b'] = kw_full['b'][:32]
    kw_a['a'] = kw_full['a'][:32]
    kw_a['query_start_loc'] = jnp.array([0, 32], dtype=jnp.int32)
    kw_a['seq_lens'] = jnp.array([32], dtype=jnp.int32)
    (conv_a, rec_a), _ = jax_k_fn(**kw_a)
    kw_b = dict(kw_full)
    kw_b['qkv'] = kw_full['qkv'][32:]
    kw_b['b'] = kw_full['b'][32:]
    kw_b['a'] = kw_full['a'][32:]
    kw_b['conv_state'] = conv_a
    kw_b['recurrent_state'] = rec_a
    kw_b['query_start_loc'] = jnp.array([0, 32], dtype=jnp.int32)
    kw_b['state_indices'] = jnp.array([2], dtype=jnp.int32)
    kw_b['read_state_indices'] = jnp.array([1], dtype=jnp.int32)
    kw_b['seq_lens'] = jnp.array([64], dtype=jnp.int32)
    (conv_b, rec_b), out_b = jax_k_fn(**kw_b)
    np.testing.assert_allclose(out_b, out_ref[32:], rtol=0.01, atol=0.01)
    np.testing.assert_array_equal(conv_b[1], conv_a[1])
    np.testing.assert_array_equal(rec_b[1], rec_a[1])
    cfg_sec = configs.GdnBenchmarkConfig(
        name='sec_iso',
        lengths=(128, 128),
        context_lens=0,
        n_kq=2,
        n_v=8,
        dtype=jnp.float32,
        state_dtype=jnp.float32,
    )
    kw_clean = configs.make_gdn_inputs(jax.random.PRNGKey(416), cfg_sec)
    kw_clean['conv_state'] = jnp.zeros_like(kw_clean['conv_state'])
    kw_clean['recurrent_state'] = jnp.zeros_like(kw_clean['recurrent_state'])
    (_, _), out_clean = jax_k_fn(**kw_clean)
    kw_mal = dict(kw_clean)
    kw_mal['recurrent_state'] = kw_clean['recurrent_state'].at[2].set(jnp.nan)
    (_, _), out_mal = jax_k_fn(**kw_mal)
    np.testing.assert_allclose(out_mal[:128], out_clean[:128], rtol=0, atol=0)

  @hypothesis.given(
      dtype=st.sampled_from([jnp.bfloat16, jnp.float32]),
      has_bias=st.sampled_from([False, True]),
      context_len=st.sampled_from([0, 64]),
  )
  @hypothesis.settings(max_examples=4, deadline=None)
  def test_4_hypothesis_vs_pallas(
      self, dtype: Any, has_bias: bool, context_len: int
  ):
    cfg = configs.GdnBenchmarkConfig(
        name='hyp',
        lengths=(128,),
        context_lens=context_len,
        n_kq=2,
        n_v=8,
        dtype=dtype,
        state_dtype=dtype,
    )
    kwargs = configs.make_gdn_inputs(
        jax.random.PRNGKey(505 + context_len), cfg, has_bias=has_bias
    )
    self._assert_jax_k_matches_pallas(kwargs, f'hyp_{dtype}_{has_bias}')

  def test_5_hlo_invariants(self):
    cfg = configs.GDN_BENCHMARK_CONFIGS[0]
    kwargs = configs.make_gdn_inputs(jax.random.PRNGKey(606), cfg)
    jax_k_fn = jax.jit(
        fused_conv1d_gdn.fused_conv1d_gdn_jax_k, static_argnames=STATIC_ARGNAMES
    )
    lowered = jax_k_fn.lower(**kwargs)
    hlo_text = lowered.as_text()
    self.assertNotIn('pallas_call', hlo_text)
    self.assertNotIn('tpu_custom_call', hlo_text)


if __name__ == '__main__':
  absltest.main()
