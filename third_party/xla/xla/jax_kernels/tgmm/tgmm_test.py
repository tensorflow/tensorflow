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

"""Unit tests for Stage 1 jax_k backward ragged_dot (tgmm)."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from xla.jax_kernels import tgmm
from xla.jax_kernels.tgmm import champion
from xla.jax_kernels.tgmm import reference as baseline


@jax.jit
def _max_allclose_excess(actual: jax.Array, expected: jax.Array) -> jax.Array:
  a = actual.astype(jnp.float32)
  b = expected.astype(jnp.float32)
  return jnp.max(jnp.abs(a - b) - (0.01 + 0.01 * jnp.abs(b)))


class TgmmKernelsTest(absltest.TestCase):

  def test_0_ragged_dot_tgmm_correctness_and_empty_groups(self):
    for m, k, n, g in (
        (8, 512, 512, 8),
        (512, 512, 512, 8),
        (4096, 512, 512, 8),
        (4096, 2048, 2048, 8),
    ):
      kx, ky = jax.random.split(jax.random.PRNGKey(13 + m))
      scale = jnp.asarray(max(m // max(g, 1), 1) ** (-0.25), dtype=jnp.bfloat16)
      lhs = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16) * scale
      rhs = jax.random.normal(ky, (m, n), dtype=jnp.bfloat16) * scale
      spec = tgmm.tgmm_slot_meta_spec(m, k, n, g)
      assert spec is not None
      tile_m, m_pad, max_slots = spec

      def tgmm_impl(a, b, gs, tm=tile_m, mp=m_pad, ms=max_slots):
        ends = jnp.cumsum(jnp.maximum(gs, 0), dtype=jnp.int32)
        starts = ends - jnp.maximum(gs, 0)
        slot_meta = tgmm.build_tgmm_slot_meta(starts, ends, tm, ms, mp // 16)
        return tgmm.ragged_dot_tgmm_kernel(a, b, gs, slot_meta)

      tgmm_fn = jax.jit(tgmm_impl)
      full = [m // g] * g
      if m >= 4096:
        partial = [1300, 0, 19, 805, 0, 1100, 500, 100]
      elif m >= 512:
        partial = [140, 0, 19, 85, 0, 110, 50, 10]
      else:
        partial = [3, 0, 2, 0, 1, 0, 2, 0]
      for counts in (partial, [0] * g, full):
        group_sizes = jnp.array(counts, dtype=jnp.int32)
        out_dev = tgmm_fn(lhs, rhs, group_sizes)
        actual = np.asarray(out_dev, np.float32)
        expected = np.asarray(
            baseline.tgmm_reference(lhs, rhs, group_sizes), np.float32
        )
        np.testing.assert_allclose(actual, expected, rtol=0.01, atol=0.01)
        for e, cnt in enumerate(counts):
          if cnt == 0:
            np.testing.assert_array_equal(actual[e], 0.0)

  def test_2_scratch_overflow_multi_tile_and_non_g8_shapes(self):
    jit_ref = jax.jit(baseline.tgmm_reference)
    jit_tgmm = jax.jit(champion.tgmm_jax_k)
    m4, k4, n4, g4 = (4096, 4096, 4096, 4)
    kx4, ky4 = jax.random.split(jax.random.PRNGKey(41))
    scale4 = jnp.asarray((m4 // g4) ** (-0.25), dtype=jnp.bfloat16)
    lhs4 = jax.random.normal(kx4, (m4, k4), dtype=jnp.bfloat16) * scale4
    rhs4 = jax.random.normal(ky4, (m4, n4), dtype=jnp.bfloat16) * scale4
    gs4 = jnp.array([1500, 0, 1800, 796], dtype=jnp.int32)
    out4 = np.asarray(jit_tgmm(lhs4, rhs4, gs4), np.float32)
    ref4 = np.asarray(jit_ref(lhs4, rhs4, gs4), np.float32)
    np.testing.assert_allclose(out4, ref4, rtol=0.01, atol=0.01)
    np.testing.assert_array_equal(out4[1], 0.0)
    m8, k8, n8, g8 = (8, 4096, 4096, 8)
    kx8, ky8 = jax.random.split(jax.random.PRNGKey(42))
    lhs8 = jax.random.normal(kx8, (m8, k8), dtype=jnp.bfloat16)
    rhs8 = jax.random.normal(ky8, (m8, n8), dtype=jnp.bfloat16)
    for counts8 in ([1] * g8, [3, 0, 2, 0, 1, 0, 2, 0]):
      gs8 = jnp.array(counts8, dtype=jnp.int32)
      out8 = np.asarray(jit_tgmm(lhs8, rhs8, gs8), np.float32)
      ref8 = np.asarray(jit_ref(lhs8, rhs8, gs8), np.float32)
      np.testing.assert_allclose(out8, ref8, rtol=0.01, atol=0.01)
    mg, kg, ng, gg = (4096, 4096, 4096, 8)
    kxg, kyg = jax.random.split(jax.random.PRNGKey(43))
    scaleg = jnp.asarray((mg // gg) ** (-0.25), dtype=jnp.bfloat16)
    lhsg = jax.random.normal(kxg, (mg, kg), dtype=jnp.bfloat16) * scaleg
    rhsg = jax.random.normal(kyg, (mg, ng), dtype=jnp.bfloat16) * scaleg
    for countsg in (
        [512] * gg,
        [1300, 0, 19, 805, 0, 1100, 500, 100],
        [4096, 0, 0, 0, 0, 0, 0, 0],
    ):
      gsg = jnp.array(countsg, dtype=jnp.int32)
      outg = np.asarray(jit_tgmm(lhsg, rhsg, gsg), np.float32)
      refg = np.asarray(jit_ref(lhsg, rhsg, gsg), np.float32)
      np.testing.assert_allclose(outg, refg, rtol=0.01, atol=0.01)
    cfg_tm1024 = champion.TgmmTilingConfig(
        mt=1024, kt=1024, nt=2048, nk=4, nn=2
    )
    gs_span5119 = jnp.array([4096, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    out_tm1024 = np.asarray(
        jax.jit(lambda a, b, gs: champion.tgmm_jax_k(a, b, gs, cfg=cfg_tm1024))(
            lhsg, rhsg, gs_span5119
        ),
        np.float32,
    )
    ref_tm1024 = np.asarray(jit_ref(lhsg, rhsg, gs_span5119), np.float32)
    np.testing.assert_allclose(out_tm1024, ref_tm1024, rtol=0.01, atol=0.01)
    m128, k128, n128, g128 = (131072, 2048, 768, 128)
    kx128, ky128 = jax.random.split(jax.random.PRNGKey(44))
    scale128 = jnp.asarray((m128 // g128) ** (-0.25), dtype=jnp.bfloat16)
    lhs128 = (
        jax.random.normal(kx128, (m128, k128), dtype=jnp.bfloat16) * scale128
    )
    rhs128 = (
        jax.random.normal(ky128, (m128, n128), dtype=jnp.bfloat16) * scale128
    )
    for dist128 in ('uniform', 'zipf'):
      gs128 = jnp.asarray(
          champion.configs.generate_group_sizes(
              m=m128, num_groups=g128, seed=1234, distribution=dist128
          ),
          dtype=jnp.int32,
      )
      out128_dev = jit_tgmm(lhs128, rhs128, gs128)
      ref128_dev = jit_ref(lhs128, rhs128, gs128)
      self.assertLessEqual(
          float(_max_allclose_excess(out128_dev, ref128_dev)), 0.0
      )
      np.testing.assert_allclose(
          np.asarray(out128_dev[:8], np.float32),
          np.asarray(ref128_dev[:8], np.float32),
          rtol=0.01,
          atol=0.01,
      )
    m, k, n, g = (131072, 512, 7168, 256)
    kx, ky = jax.random.split(jax.random.PRNGKey(99))
    scale = jnp.asarray(1.0 / np.sqrt(m // g), dtype=jnp.bfloat16)
    lhs = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16) * scale
    rhs = jax.random.normal(ky, (m, n), dtype=jnp.bfloat16) * scale
    spec = tgmm.tgmm_slot_meta_spec(m, k, n, g)
    self.assertIsNotNone(spec)
    tile_m, m_pad, max_slots = spec

    def tgmm_ds3_fn(a, b, gs):
      ends = jnp.cumsum(jnp.maximum(gs, 0), dtype=jnp.int32)
      starts = ends - jnp.maximum(gs, 0)
      slot_meta = tgmm.build_tgmm_slot_meta(
          starts, ends, tile_m, max_slots, m_pad // 16
      )
      return tgmm.ragged_dot_tgmm_kernel(a, b, gs, slot_meta)

    jit_ds3 = jax.jit(tgmm_ds3_fn)
    two_ov = [1024, 1024, 0, 0] + [512] * 252
    three_tile = [1536, 0, 0] + [512] * 253
    one_hot = [m] + [0] * (g - 1)
    zipf_counts = list(
        champion.configs.generate_group_sizes(
            m=m, num_groups=g, seed=1234, distribution='zipf'
        )
    )
    empty_heavy_counts = list(
        champion.configs.generate_group_sizes(
            m=m, num_groups=g, seed=1234, distribution='empty_heavy'
        )
    )
    for counts in (
        two_ov,
        three_tile,
        one_hot,
        zipf_counts,
        empty_heavy_counts,
    ):
      gs = jnp.array(counts, dtype=jnp.int32)
      actual_dev = jit_ds3(lhs, rhs, gs)
      expected_dev = jit_ref(lhs, rhs, gs)
      self.assertLessEqual(
          float(_max_allclose_excess(actual_dev, expected_dev)), 0.0
      )
      np.testing.assert_allclose(
          np.asarray(actual_dev[:8], np.float32),
          np.asarray(expected_dev[:8], np.float32),
          rtol=0.01,
          atol=0.01,
      )

  def test_1_hlo_invariants(self):
    m, k, n, g = (4096, 512, 512, 8)
    lhs = jnp.ones((m, k), dtype=jnp.bfloat16)
    rhs = jnp.ones((m, n), dtype=jnp.bfloat16)
    group_sizes = jnp.array([m // g] * g, dtype=jnp.int32)
    jit_fn = jax.jit(champion.tgmm_jax_k)
    lowered = jit_fn.lower(lhs, rhs, group_sizes)
    hlo_pre = lowered.as_text()
    self.assertNotIn('ragged-dot', hlo_pre)
    self.assertNotIn('pallas_call', hlo_pre)
    self.assertIn('stablehlo.while', hlo_pre)
    self.assertIn('xla_disable_while_loop_copies', hlo_pre)
    compiled = lowered.compile()
    hlo_opt = compiled.as_text()
    self.assertIsNotNone(hlo_opt)
    self.assertTrue('while' in hlo_opt or 'unrolled' in hlo_opt)

  def test_3_large_kn_128_groups(self):
    jit_ref = jax.jit(baseline.tgmm_reference)
    jit_tgmm = jax.jit(champion.tgmm_jax_k)
    for m, k, n, g in ((8192, 1536, 6144, 128), (8192, 6144, 1536, 128)):
      kx, ky = jax.random.split(jax.random.PRNGKey(77 + k))
      scale = jnp.asarray((m // g) ** (-0.25), dtype=jnp.bfloat16)
      lhs = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16) * scale
      rhs = jax.random.normal(ky, (m, n), dtype=jnp.bfloat16) * scale
      for dist in ('uniform', 'zipf'):
        gs = jnp.asarray(
            champion.configs.generate_group_sizes(
                m=m, num_groups=g, seed=1234, distribution=dist
            ),
            dtype=jnp.int32,
        )
        out_dev = jit_tgmm(lhs, rhs, gs)
        ref_dev = jit_ref(lhs, rhs, gs)
        self.assertEqual(out_dev.shape, (g, k, n))
        self.assertEqual(out_dev.dtype, jnp.bfloat16)
        self.assertLessEqual(float(_max_allclose_excess(out_dev, ref_dev)), 0.0)
        np.testing.assert_allclose(
            np.asarray(out_dev[:8], np.float32),
            np.asarray(ref_dev[:8], np.float32),
            rtol=0.01,
            atol=0.01,
        )
        if dist == 'zipf':
          np.testing.assert_array_equal(np.asarray(out_dev[0], np.float32), 0.0)


if __name__ == '__main__':
  absltest.main()
