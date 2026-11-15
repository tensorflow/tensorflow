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

"""Unit and Hypothesis correctness tests for jax_k forward ragged_dot (GMM) kernels."""

from absl.testing import absltest
import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
import numpy as np
from xla.jax_kernels import gmm
from xla.jax_kernels.gmm import champion
from xla.jax_kernels.gmm import configs
from xla.jax_kernels.gmm import reference as baseline


class GmmKernelsTest(absltest.TestCase):

  @hypothesis.given(
      expert_counts=st.lists(
          st.sampled_from([0, 15, 32, 49, 64]), min_size=2, max_size=5
      ),
      k_dim=st.sampled_from([40, 64]),
      n_dim=st.sampled_from([40, 64]),
  )
  @hypothesis.settings(max_examples=12, deadline=None)
  def test_0_gmm_active_tile_pipeline_hypothesis(
      self, expert_counts: list[int], k_dim: int, n_dim: int
  ):
    num_experts = len(expert_counts)
    m_total = sum(expert_counts)
    if m_total == 0:
      expert_counts[0] = 24
      m_total = 24
    key = jax.random.PRNGKey(2026)
    kx, kw = jax.random.split(key)
    x = jax.random.normal(kx, (m_total, k_dim), dtype=jnp.float32)
    w = jax.random.normal(kw, (num_experts, k_dim, n_dim), dtype=jnp.float32)
    group_starts = jnp.array(
        [0] + list(np.cumsum(expert_counts)), dtype=jnp.int32
    )
    out = jnp.zeros((m_total, n_dim), dtype=jnp.float32)
    actual = jax.jit(
        champion.gmm_active_tile_pipeline,
        static_argnames=('tile_m', 'tile_k', 'tile_n'),
    )(
        x, w, group_starts, out, tile_m=32, tile_k=32, tile_n=32
    ).block_until_ready()
    expected_chunks = []
    for e, count in enumerate(expert_counts):
      start = int(group_starts[e])
      end = int(group_starts[e + 1])
      if count > 0:
        expected_chunks.append(jnp.matmul(x[start:end], w[e]))
    expected = (
        jnp.concatenate(expected_chunks, axis=0)
        if expected_chunks
        else jnp.zeros_like(out)
    )
    np.testing.assert_allclose(actual, expected, rtol=0.001, atol=0.001)

  def test_1_ragged_dot_forward_zeroes_rows_past_group_sizes(self):
    m, k, n, g = (4096, 256, 256, 8)
    kx, kw = jax.random.split(jax.random.PRNGKey(7))
    scale = jnp.asarray(max(k, 1) ** (-0.25), dtype=jnp.bfloat16)
    lhs = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16) * scale
    rhs = jax.random.normal(kw, (g, k, n), dtype=jnp.bfloat16) * scale
    spec = gmm.bulk_slot_meta_spec(m, k, n, g)
    assert spec is not None
    tile_m, split_m, max_slots = spec

    def fwd_impl(a, b, gs):
      ends = jnp.cumsum(jnp.maximum(gs, 0), dtype=jnp.int32)
      starts = ends - jnp.maximum(gs, 0)
      slot_meta = gmm.build_slot_meta(
          jnp.clip(starts, 0, split_m),
          jnp.clip(ends, 0, split_m),
          tile_m,
          max_slots,
          m // 8,
      )
      return gmm.ragged_dot_forward_kernel(a, b, gs, slot_meta)

    fwd = jax.jit(fwd_impl)
    full = [m // g] * g
    partial = [m // g - 37 * (i % 3) for i in range(g)]
    partial[5] = 0
    for counts in (partial, [0] * g, full):
      group_sizes = jnp.array(counts, dtype=jnp.int32)
      actual = np.asarray(fwd(lhs, rhs, group_sizes), np.float32)
      expected = np.asarray(
          jax.lax.ragged_dot(
              lhs, rhs, group_sizes, preferred_element_type=jnp.float32
          ).astype(jnp.bfloat16),
          np.float32,
      )
      covered = sum(counts)
      np.testing.assert_allclose(
          actual[:covered], expected[:covered], rtol=0.01, atol=0.01
      )
      np.testing.assert_array_equal(actual[covered:], 0.0)

  def test_2_ragged_dot_jax_k_vs_tokamax_and_ref(self):
    for m, k, n, g in ((4096, 256, 256, 8), (4096, 512, 512, 8)):
      for dist in ('uniform', 'zipf', 'empty_heavy'):
        lhs, rhs, group_sizes = configs.make_gmm_inputs(
            m, k, n, g, distribution=dist
        )
        for return_residuals in (False, True):
          jax_k_res = jax.jit(
              champion.ragged_dot_jax_k, static_argnames=('return_residuals',)
          )(lhs, rhs, group_sizes, return_residuals=return_residuals)
          ref_res = jax.jit(
              baseline.gmm_reference, static_argnames=('return_residuals',)
          )(lhs, rhs, group_sizes, return_residuals=return_residuals)
          jax_k_out = jax_k_res[0] if return_residuals else jax_k_res
          ref_out = ref_res[0] if return_residuals else ref_res
          np.testing.assert_allclose(
              np.asarray(jax_k_out, np.float32),
              np.asarray(ref_out, np.float32),
              rtol=0.01,
              atol=0.01,
          )
          if return_residuals:
            np.testing.assert_array_equal(
                np.asarray(jax_k_res[1], np.float32),
                np.asarray(jax_k_out, np.float32),
            )

  def test_4_dispatch_branches_m8_async_m_buckets_and_packed_wb(self):
    m8, k8, n8, g8 = (8, 4096, 4096, 8)
    kx8, kw8 = jax.random.split(jax.random.PRNGKey(801))
    scale8 = jnp.asarray(k8 ** (-0.25), dtype=jnp.bfloat16)
    lhs8 = jax.random.normal(kx8, (m8, k8), dtype=jnp.bfloat16) * scale8
    rhs8 = jax.random.normal(kw8, (g8, k8, n8), dtype=jnp.bfloat16) * scale8
    for counts8 in ([1] * g8, [3, 0, 2, 0, 1, 0, 2, 0]):
      gs8 = jnp.array(counts8, dtype=jnp.int32)
      for return_residuals in (False, True):
        jax_k_res8 = jax.jit(
            champion.ragged_dot_jax_k, static_argnames=('return_residuals',)
        )(lhs8, rhs8, gs8, return_residuals=return_residuals)
        ref_res8 = jax.jit(
            baseline.gmm_reference, static_argnames=('return_residuals',)
        )(lhs8, rhs8, gs8, return_residuals=return_residuals)
        jax_k_out8 = jax_k_res8[0] if return_residuals else jax_k_res8
        ref_out8 = ref_res8[0] if return_residuals else ref_res8
        np.testing.assert_allclose(
            np.asarray(jax_k_out8, np.float32),
            np.asarray(ref_out8, np.float32),
            rtol=0.01,
            atol=0.01,
        )
    bucket_cfg = configs.RaggedDotTilingConfig(
        mt=512, kt=512, nt=512, m_buckets=(256, 384)
    )
    m_b, k_b, n_b, g_b = (4096, 512, 512, 8)
    kxb, kw_b = jax.random.split(jax.random.PRNGKey(802))
    scale_b = jnp.asarray(k_b ** (-0.25), dtype=jnp.bfloat16)
    lhs_b = jax.random.normal(kxb, (m_b, k_b), dtype=jnp.bfloat16) * scale_b
    rhs_b = (
        jax.random.normal(kw_b, (g_b, k_b, n_b), dtype=jnp.bfloat16) * scale_b
    )
    gs_b = jnp.array([200, 320, 480, 0, 768, 900, 628, 800], dtype=jnp.int32)
    out_b = jax.jit(
        lambda a, b, gs: champion.ragged_dot_jax_k(a, b, gs, cfg=bucket_cfg)
    )(lhs_b, rhs_b, gs_b)
    ref_b = jax.jit(baseline.gmm_reference)(lhs_b, rhs_b, gs_b)
    np.testing.assert_allclose(
        np.asarray(out_b, np.float32),
        np.asarray(ref_b, np.float32),
        rtol=0.01,
        atol=0.01,
    )
    for m, k, n, g, return_residuals in (
        (4096, 4096, 4096, 8, True),
        (131072, 4096, 1536, 128, False),
        (131072, 7168, 512, 256, False),
    ):
      k1, k2 = jax.random.split(jax.random.PRNGKey(803 + m + k + n + g))
      scale = jnp.asarray(k ** (-0.25), dtype=jnp.bfloat16)
      lhs = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16) * scale
      rhs = jax.random.normal(k2, (g, k, n), dtype=jnp.bfloat16) * scale
      jit_jax_k = jax.jit(
          champion.ragged_dot_jax_k, static_argnames=('return_residuals',)
      )
      jit_ref = jax.jit(
          baseline.gmm_reference, static_argnames=('return_residuals',)
      )
      for dist in ('uniform', 'zipf'):
        gs = jnp.asarray(
            configs.generate_group_sizes(m=m, num_groups=g, distribution=dist),
            dtype=jnp.int32,
        )
        jax_k_res = jit_jax_k(lhs, rhs, gs, return_residuals=return_residuals)
        ref_res = jit_ref(lhs, rhs, gs, return_residuals=return_residuals)
        jax_k_out = jax_k_res[0] if return_residuals else jax_k_res
        ref_out = ref_res[0] if return_residuals else ref_res
        np.testing.assert_allclose(
            np.asarray(jax_k_out, np.float32),
            np.asarray(ref_out, np.float32),
            rtol=0.01,
            atol=0.01,
        )
        if return_residuals:
          np.testing.assert_array_equal(
              np.asarray(jax_k_res[1], np.float32),
              np.asarray(jax_k_out, np.float32),
          )

  def test_3_hlo_invariants(self):
    m, k, n, g = (4096, 512, 512, 8)
    lhs = jnp.ones((m, k), dtype=jnp.bfloat16)
    rhs = jnp.ones((g, k, n), dtype=jnp.bfloat16)
    group_sizes = jnp.array([m // g] * g, dtype=jnp.int32)
    jit_fn = jax.jit(
        champion.ragged_dot_jax_k, static_argnames=('return_residuals',)
    )
    lowered = jit_fn.lower(lhs, rhs, group_sizes, return_residuals=False)
    hlo_pre = lowered.as_text()
    self.assertNotIn('ragged-dot', hlo_pre)
    self.assertNotIn('pallas_call', hlo_pre)
    self.assertIn('stablehlo.while', hlo_pre)
    self.assertIn('xla_disable_while_loop_copies', hlo_pre)
    compiled = lowered.compile()
    hlo_opt = compiled.as_text()
    self.assertIsNotNone(hlo_opt)
    self.assertTrue('while' in hlo_opt or 'unrolled' in hlo_opt)

  def test_5_medium_m_bar_transpose_rhs(self):
    for m, k, n, g in ((8192, 1536, 6144, 128), (8192, 6144, 1536, 128)):
      cfg = configs.RaggedDotTilingConfig.from_shapes(m, k, n, g)
      self.assertEqual(cfg, configs.RaggedDotTilingConfig(mt=128, kt=k, nt=n))
      lhs, rhs, group_sizes = configs.make_gmm_inputs(
          m, k, n, g, distribution='uniform'
      )
      gs_zipf = jnp.asarray(
          configs.generate_group_sizes(m=m, num_groups=g, distribution='zipf'),
          dtype=jnp.int32,
      )
      rhs_t = jnp.copy(jnp.swapaxes(rhs, 1, 2))
      jax.block_until_ready((rhs_t, gs_zipf))
      jit_ref = jax.jit(baseline.gmm_reference)
      ref_uniform = np.asarray(jit_ref(lhs, rhs, group_sizes), np.float32)
      ref_zipf = np.asarray(jit_ref(lhs, rhs, gs_zipf), np.float32)
      for transpose_rhs in (False, True):
        rhs_in = rhs_t if transpose_rhs else rhs
        for return_residuals in (False, True):
          jit_jax_k = jax.jit(
              champion.ragged_dot_jax_k,
              static_argnames=('return_residuals', 'transpose_rhs'),
          )
          for gs_in, expected_np in (
              (group_sizes, ref_uniform),
              (gs_zipf, ref_zipf),
          ):
            jax_k_res = jit_jax_k(
                lhs,
                rhs_in,
                gs_in,
                return_residuals=return_residuals,
                transpose_rhs=transpose_rhs,
            )
            jax.block_until_ready(jax_k_res)
            jax_k_out = jax_k_res[0] if return_residuals else jax_k_res
            np.testing.assert_allclose(
                np.asarray(jax_k_out, np.float32),
                expected_np,
                rtol=0.01,
                atol=0.01,
            )
            if return_residuals:
              np.testing.assert_array_equal(
                  np.asarray(jax_k_res[1], np.float32),
                  np.asarray(jax_k_out, np.float32),
              )

  def _assert_matches_ragged_dot(self, actual, lhs, rhs, group_sizes):
    expected = jax.lax.ragged_dot(
        lhs, rhs, group_sizes, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        rtol=0.02,
        atol=0.02,
    )

  def test_6_more_than_256_groups(self):
    rng = np.random.default_rng(0)
    for m, k, n, g in ((8192, 1024, 1024, 512), (4096, 512, 1024, 512)):
      lhs, rhs, uniform_sizes = configs.make_gmm_inputs(m, k, n, g)
      live = rng.choice(g, size=g // 3, replace=False)
      live[0] = g - 1
      probs = rng.zipf(1.5, size=live.size).astype(np.float64)
      probs /= probs.sum()
      picks = rng.choice(live, size=m, p=probs)
      skewed_sizes = jnp.asarray(np.bincount(picks, minlength=g), jnp.int32)
      self.assertGreater(int(jnp.sum(skewed_sizes[g // 2 :])), 0)
      for group_sizes in (uniform_sizes, skewed_sizes):
        actual = jax.jit(champion.ragged_dot_jax_k)(lhs, rhs, group_sizes)
        self._assert_matches_ragged_dot(actual, lhs, rhs, group_sizes)


if __name__ == '__main__':
  absltest.main()
