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
# ==============================================================================
"""Unit tests for CostModel and DenseCostModel."""

from collections.abc import Sequence
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from xla.benchmarks.core import platform_info
from xla.benchmarks.pallas_microbenchmarks import cost_model


def _get_dtype_name(dtype: Any) -> str:
  return getattr(dtype, "__name__", str(dtype))


class _CustomCostModel(cost_model.CostModel):
  """Test subclass exercising additional operands, input, and output fusions."""

  def __init__(
      self,
      *args,
      extra_k_bytes: int = 0,
      input_fusion_vmem_bytes: int = 0,
      output_fusion_vmem_bytes: int = 0,
      input_fusion_stage_cost: Any = None,
      output_fusion_stage_cost: Any = None,
      **kwargs,
  ):
    self._extra_k_bytes = extra_k_bytes
    self._input_fusion_vmem_bytes = input_fusion_vmem_bytes
    self._output_fusion_vmem_bytes = output_fusion_vmem_bytes
    if input_fusion_stage_cost is None:
      input_fusion_stage_cost = cost_model.PipelineStageCost()
    if output_fusion_stage_cost is None:
      output_fusion_stage_cost = cost_model.PipelineStageCost()
    self._input_fusion_stage_cost = input_fusion_stage_cost
    self._output_fusion_stage_cost = output_fusion_stage_cost
    super().__init__(*args, **kwargs)

  def _get_additional_operands(self) -> Sequence[Any]:
    if self._extra_k_bytes > 0:
      return (
          cost_model.Operand(
              size_bytes=np.full(
                  self._grid_shape, self._extra_k_bytes, dtype=np.int64
              ),
              axes=(cost_model.Axis.M, cost_model.Axis.K),
              is_output=False,
          ),
      )
    return ()

  def _get_input_fusion_vmem_usage_bytes(self) -> int | np.ndarray:
    return self._input_fusion_vmem_bytes

  def _get_output_fusion_vmem_usage_bytes(self) -> int | np.ndarray:
    return self._output_fusion_vmem_bytes

  def _get_input_fusion_stage_cost(self) -> cost_model.PipelineStageCost:
    return self._input_fusion_stage_cost

  def _get_output_fusion_stage_cost(self) -> cost_model.PipelineStageCost:
    return self._output_fusion_stage_cost


_COMMON_KWARGS = dict(
    m=4096,
    k=4096,
    n=4096,
    lhs_mem=pltpu.HBM,
    rhs_mem=pltpu.HBM,
    out_mem=pltpu.HBM,
    lhs_dtype=jnp.bfloat16,
    rhs_dtype=jnp.bfloat16,
    out_dtype=jnp.bfloat16,
    acc_dtype=jnp.float32,
    chip_version=pltpu.ChipVersion.TPU_7X,
)
_P_STATE = 1


def _clock_ghz() -> float:
  pinfo = platform_info.get_platform_info(pltpu.ChipVersion.TPU_7X)
  return pinfo.clock_speed_ghz_by_p_state[_P_STATE]


class CostModelTest(parameterized.TestCase):

  def test_dense_cost_model_select_window(self):
    model = cost_model.DenseCostModel(
        m=16384,
        k=16384,
        n=16384,
        lhs_mem=pltpu.HBM,
        rhs_mem=pltpu.HBM,
        out_mem=pltpu.HBM,
        lhs_dtype=jnp.bfloat16,
        rhs_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        subblock_m=1024,
        chip_version=pltpu.ChipVersion.TPU_7X,
    )
    block_m, block_k, block_n = model.select_window(
        vmem_limit_bytes=32 * 1024 * 1024,
        p_state=1,
    )
    self.assertGreater(int(block_m), 0)
    self.assertGreater(int(block_k), 0)
    self.assertGreater(int(block_n), 0)
    self.assertEqual(int(block_m) % 128, 0)
    self.assertEqual(int(block_k) % 256, 0)
    self.assertEqual(int(block_n) % 256, 0)

  @parameterized.parameters(
      pltpu.ChipVersion.TPU_V5E,
      pltpu.ChipVersion.TPU_V5P,
      pltpu.ChipVersion.TPU_V6E,
      pltpu.ChipVersion.TPU_7X,
      pltpu.ChipVersion.TPU_8I,
      pltpu.ChipVersion.TPU_8T,
  )
  def test_dtype_objects_match_scalar_types(self, chip: pltpu.ChipVersion):
    pinfo = platform_info.get_platform_info(chip)
    scalar_types = (
        jnp.float32,
        jnp.bfloat16,
        jnp.float8_e4m3fn,
        jnp.float8_e5m2,
        jnp.float8_e4m3b11fnuz,
        jnp.int8,
        jnp.uint8,
        jnp.int4,
        jnp.uint4,
        jnp.float4_e2m1fn,
    )
    mappings = (
        pinfo.matmul_cadence_cycles_by_dtype,
        pinfo.matmul_latency_cycles_by_dtype,
        pinfo.latch_cadence_cycles_by_dtype,
    )
    for scalar_type in scalar_types:
      dtype_object = jnp.dtype(scalar_type)
      with self.subTest(dtype=dtype_object.name):
        for mapping in mappings:
          expected = pinfo.get_emulated_dtype(scalar_type, mapping)
          actual = pinfo.get_emulated_dtype(dtype_object, mapping)
          self.assertEqual(actual, expected)
          self.assertIn(actual, mapping)
        self.assertEqual(
            pinfo.get_matmul_cadence_cycles(dtype_object),
            pinfo.get_matmul_cadence_cycles(scalar_type),
        )
        self.assertEqual(
            pinfo.get_matmul_latency_cycles(dtype_object),
            pinfo.get_matmul_latency_cycles(scalar_type),
        )
        self.assertEqual(
            pinfo.get_latch_cadence_cycles(dtype_object),
            pinfo.get_latch_cadence_cycles(scalar_type),
        )
        self.assertEqual(
            pinfo.mxu_size_by_dtype(dtype_object),
            pinfo.mxu_size_by_dtype(scalar_type),
        )

  def test_gf_dtype_objects_are_native(self):
    pinfo = platform_info.get_platform_info(pltpu.ChipVersion.TPU_7X)
    cadence_by_dtype = pinfo.matmul_cadence_cycles_by_dtype
    self.assertEqual(
        pinfo.get_emulated_dtype(jnp.dtype(jnp.bfloat16), cadence_by_dtype),
        jnp.bfloat16,
    )
    self.assertEqual(
        pinfo.get_emulated_dtype(
            jnp.dtype(jnp.float8_e4m3fn), cadence_by_dtype
        ),
        jnp.float8_e4m3fn,
    )
    self.assertEqual(
        pinfo.get_matmul_cadence_cycles(jnp.dtype(jnp.bfloat16)), 8
    )
    self.assertEqual(
        pinfo.get_matmul_latency_cycles(jnp.dtype(jnp.float8_e4m3fn)), 204
    )
    self.assertEqual(
        pinfo.get_emulated_dtype(jnp.dtype(jnp.int4), cadence_by_dtype),
        jnp.float8_e4m3fn,
    )

  def test_native_sub_byte_mxu_size_for_dtype_objects(self):
    pinfo = platform_info.get_platform_info(pltpu.ChipVersion.TPU_8T)
    mxu_k, mxu_n = pinfo.mxu_size
    self.assertEqual(
        pinfo.mxu_size_by_dtype(jnp.dtype(jnp.int4)), (2 * mxu_k, mxu_n)
    )
    self.assertEqual(
        pinfo.mxu_size_by_dtype(jnp.dtype(jnp.float4_e2m1fn)),
        (2 * mxu_k, mxu_n),
    )

  @parameterized.parameters(
      (jnp.bfloat16, jnp.bfloat16, jnp.bfloat16, jnp.float32),
      (jnp.float8_e4m3fn, jnp.float8_e4m3fn, jnp.bfloat16, jnp.float32),
      (jnp.bfloat16, jnp.float8_e4m3fn, jnp.bfloat16, jnp.float32),
      (jnp.int4, jnp.int4, jnp.bfloat16, jnp.int32),
  )
  def test_gf_dense_windows_match_for_dtype_objects(
      self, lhs_dtype: Any, rhs_dtype: Any, out_dtype: Any, acc_dtype: Any
  ):
    pinfo = platform_info.get_platform_info(pltpu.ChipVersion.TPU_7X)

    def _make_model(dtype_fn):
      return cost_model.DenseCostModel(
          m=4096,
          k=4096,
          n=4096,
          lhs_mem=pltpu.HBM,
          rhs_mem=pltpu.HBM,
          out_mem=pltpu.HBM,
          lhs_dtype=dtype_fn(lhs_dtype),
          rhs_dtype=dtype_fn(rhs_dtype),
          out_dtype=dtype_fn(out_dtype),
          acc_dtype=dtype_fn(acc_dtype),
          chip_version=pltpu.ChipVersion.TPU_7X,
      )

    scalar_model = _make_model(lambda d: d)
    object_model = _make_model(jnp.dtype)
    self.assertEqual(
        object_model.matmul_lhs_dtype, scalar_model.matmul_lhs_dtype
    )
    self.assertEqual(
        object_model.matmul_rhs_dtype, scalar_model.matmul_rhs_dtype
    )
    select_kwargs = dict(
        vmem_limit_bytes=pinfo.default_vmem_limit_kib * 1024,
        p_state=pinfo.default_p_state,
    )
    np.testing.assert_array_equal(
        np.asarray(object_model.select_window(**select_kwargs)),
        np.asarray(scalar_model.select_window(**select_kwargs)),
    )

  def test_matmul_stage_cost_split_and_unnormalized(self):
    model = _CustomCostModel(**_COMMON_KWARGS)
    cost = model._get_matmul_stage_cost()
    self.assertFalse(hasattr(cost, "mxu_cycles"))
    self.assertTrue(hasattr(cost, "matmul_cycles"))
    self.assertTrue(hasattr(cost, "latch_cycles"))

    mxu_k, mxu_n = model._get_mxu_size()
    cadence = model._get_matmul_cadence_cycles()
    k_blocks = model._block_k // mxu_k
    n_blocks = model._block_n // mxu_n
    num_latches = k_blocks * n_blocks
    vmatmuls_per_latch = model._get_vmatmuls_per_latch()
    hazard_cycles = np.where(vmatmuls_per_latch == 1, 4, 0)
    expected_matmul_cycles = (
        cadence * vmatmuls_per_latch * num_latches
        + hazard_cycles * (k_blocks - 1) * n_blocks
    )
    latch_latency_cycles = model._get_latch_latency_cycles()
    expected_latch_cycles = latch_latency_cycles * num_latches
    np.testing.assert_allclose(cost.matmul_cycles, expected_matmul_cycles)
    np.testing.assert_allclose(cost.latch_cycles, expected_latch_cycles)

  def test_pipeline_mxu_sums_across_stages(self):
    # Matmul and latch cycles reported by a fusion stage add to their respective
    # units, and the overall MXU throughput is the max over both units.
    base_model = _CustomCostModel(**_COMMON_KWARGS)
    num_mxus = base_model._platform_info.num_mxus
    base_cost = base_model._get_matmul_stage_cost()
    base_throughput = (
        np.maximum(base_cost.matmul_cycles, base_cost.latch_cycles) / num_mxus
    )
    cases = (
        ("matmul_only", 500.0, 0.0),
        ("latch_only", 0.0, 100.0),
        ("latch_dominates", 0.0, 1e7),
        ("both", 500.0, 100.0),
    )
    for name, extra_matmul_cycles, extra_latch_cycles in cases:
      with self.subTest(msg=name):
        model = _CustomCostModel(
            **_COMMON_KWARGS,
            output_fusion_stage_cost=cost_model.PipelineStageCost(
                matmul_cycles=extra_matmul_cycles,
                latch_cycles=extra_latch_cycles,
            ),
        )
        model_throughput = (
            np.maximum(
                base_cost.matmul_cycles + extra_matmul_cycles,
                base_cost.latch_cycles + extra_latch_cycles,
            )
            / num_mxus
        )
        expected_extra_cycles = model_throughput - base_throughput
        compute_diff = model._compute_latency_ns_per_iteration(
            _P_STATE
        ) - base_model._compute_latency_ns_per_iteration(_P_STATE)
        np.testing.assert_allclose(
            compute_diff,
            np.broadcast_to(
                expected_extra_cycles / _clock_ghz(),
                model._grid_shape,
            ),
        )

  @parameterized.product(
      chip=[
          pltpu.ChipVersion.TPU_V5E,
          pltpu.ChipVersion.TPU_V5P,
          pltpu.ChipVersion.TPU_V6E,
          pltpu.ChipVersion.TPU_7X,
          pltpu.ChipVersion.TPU_8T,
      ],
      dtypes=[
          (jnp.bfloat16, jnp.bfloat16, jnp.bfloat16, jnp.float32),
          (jnp.float8_e4m3fn, jnp.float8_e4m3fn, jnp.bfloat16, jnp.float32),
          (jnp.int4, jnp.int4, jnp.bfloat16, jnp.int32),
      ],
  )
  def test_dense_compute_matches_closed_form(
      self, chip: pltpu.ChipVersion, dtypes: tuple[Any, Any, Any, Any]
  ):
    """Guards against regressions in the dense (fusion-free) compute model."""
    lhs_dtype, rhs_dtype, out_dtype, acc_dtype = dtypes
    model = cost_model.DenseCostModel(
        m=4096,
        k=4096,
        n=4096,
        lhs_mem=pltpu.HBM,
        rhs_mem=pltpu.HBM,
        out_mem=pltpu.HBM,
        lhs_dtype=lhs_dtype,
        rhs_dtype=rhs_dtype,
        out_dtype=out_dtype,
        acc_dtype=acc_dtype,
        chip_version=chip,
    )
    pinfo = platform_info.get_platform_info(chip)
    p_state = pinfo.default_p_state
    clock_speed_ghz = pinfo.clock_speed_ghz_by_p_state[p_state]
    num_mxus = pinfo.num_mxus
    mxu_k, mxu_n = model._get_mxu_size()
    cadence = model._get_matmul_cadence_cycles()
    k_blocks = model._block_k // mxu_k
    n_blocks = model._block_n // mxu_n
    num_latches = k_blocks * n_blocks
    vmatmuls_per_latch = model._get_vmatmuls_per_latch()
    hazard_cycles = np.where(vmatmuls_per_latch == 1, 4, 0)
    total_matmul_cycles = (
        cadence * vmatmuls_per_latch * num_latches
        + hazard_cycles * (k_blocks - 1) * n_blocks
    )
    latch_latency_cycles = model._get_latch_latency_cycles()
    total_latch_cycles = latch_latency_cycles * num_latches
    expected_ns = (
        np.maximum(total_matmul_cycles, total_latch_cycles) / num_mxus
        + latch_latency_cycles
        + pinfo.get_matmul_latency_cycles(model.matmul_lhs_dtype)
    ) / clock_speed_ghz
    np.testing.assert_allclose(
        model._compute_latency_ns_per_iteration(p_state),
        np.broadcast_to(expected_ns, model._grid_shape),
        rtol=1e-12,
    )

  def test_subclass_hooks_affect_vmem_and_latency(self):
    clock_speed_ghz = _clock_ghz()
    base_model = _CustomCostModel(**_COMMON_KWARGS)
    custom_model = _CustomCostModel(
        **_COMMON_KWARGS,
        extra_k_bytes=1024,
        input_fusion_vmem_bytes=2048,
        output_fusion_vmem_bytes=4096,
        input_fusion_stage_cost=cost_model.PipelineStageCost(
            latency_cycles=100.0 * clock_speed_ghz
        ),
        output_fusion_stage_cost=cost_model.PipelineStageCost(
            latency_cycles=200.0 * clock_speed_ghz
        ),
    )

    vmem_diff = (
        custom_model._compute_vmem_usage_bytes()
        - base_model._compute_vmem_usage_bytes()
    )
    self.assertEqual(int(vmem_diff[0, 0, 0]), 2 * 1024 + 2048 + 4096)

    compute_diff = custom_model._compute_latency_ns_per_iteration(
        _P_STATE
    ) - base_model._compute_latency_ns_per_iteration(_P_STATE)
    self.assertAlmostEqual(float(compute_diff[0, 0, 0]), 300.0)

    self.assertGreater(
        float(custom_model._get_initial_operand_transfer_latency()[0, 0, 0]),
        float(base_model._get_initial_operand_transfer_latency()[0, 0, 0]),
    )


if __name__ == "__main__":
  absltest.main()
