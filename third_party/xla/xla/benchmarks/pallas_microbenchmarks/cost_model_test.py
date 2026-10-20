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
import jax
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

  @parameterized.parameters(False, True)
  def test_subchannel_cost_model_select_window(self, pre_quantize_lhs: bool):
    model = cost_model.SubchannelCostModel(
        m=16384,
        k=16384,
        n=16384,
        lhs_mem=pltpu.HBM,
        rhs_mem=pltpu.HBM,
        out_mem=pltpu.HBM,
        lhs_dtype=jnp.bfloat16,
        rhs_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        outer_acc_dtype=jnp.float32,
        lhs_quantized_dtype=jnp.float8_e4m3fn,
        rhs_quantized_dtype=jnp.float8_e4m3fn,
        inner_acc_dtype=jnp.float32,
        subchannel_size=512,
        pre_quantize_lhs=pre_quantize_lhs,
        chip_version=pltpu.ChipVersion.TPU_7X,
    )
    block_m, block_k, block_n = model.select_window(
        vmem_limit_bytes=32 * 1024 * 1024,
        p_state=1,
    )
    self.assertGreater(int(block_m), 0)
    self.assertGreater(int(block_k), 0)
    self.assertGreater(int(block_n), 0)
    self.assertEqual(int(block_k) % 512, 0)

    # Verify subchannel additional operands are included.
    expected_num_extra_operands = 2 if pre_quantize_lhs else 1
    extra_operands = model._get_additional_operands()
    self.assertLen(extra_operands, expected_num_extra_operands)

    # Verify input fusion cost is positive when pre_quantize_lhs is False and
    # zero when pre_quantize_lhs is True (fp8 is native on TPU_7X, so there is
    # no unpacking). The input fusion never uses the MXU.
    in_cost = model._get_input_fusion_stage_cost()
    self.assertEqual(float(np.asarray(in_cost.matmul_cycles).max()), 0.0)
    self.assertEqual(float(np.asarray(in_cost.latch_cycles).max()), 0.0)
    if pre_quantize_lhs:
      self.assertEqual(float(np.asarray(in_cost.alu_cycles).max()), 0.0)
      self.assertEqual(float(np.asarray(in_cost.xlu_cycles).max()), 0.0)
      self.assertEqual(float(np.asarray(in_cost.latency_cycles).max()), 0.0)
    else:
      self.assertGreater(float(np.asarray(in_cost.alu_cycles).min()), 0.0)
      self.assertGreater(float(np.asarray(in_cost.xlu_cycles).min()), 0.0)
      self.assertGreater(float(np.asarray(in_cost.latency_cycles).min()), 0.0)

    # Verify output fusion cost (VPU only) and VMEM usage are positive.
    out_cost = model._get_output_fusion_stage_cost()
    self.assertGreater(float(np.asarray(out_cost.alu_cycles).min()), 0.0)
    self.assertEqual(float(np.asarray(out_cost.matmul_cycles).max()), 0.0)
    self.assertEqual(float(np.asarray(out_cost.latch_cycles).max()), 0.0)
    self.assertEqual(float(np.asarray(out_cost.xlu_cycles).max()), 0.0)
    self.assertGreater(float(np.asarray(out_cost.latency_cycles).min()), 0.0)
    out_fusion_vmem = model._get_output_fusion_vmem_usage_bytes()
    self.assertGreater(float(np.asarray(out_fusion_vmem).min()), 0.0)

  @parameterized.parameters(
      (128, 4096, 4096, False),
      (128, 8192, 8192, False),
      (128, 16384, 16384, False),
      (1024, 4096, 4096, False),
      (128, 4096, 4096, True),
      (1024, 4096, 4096, True),
  )
  def test_subchannel_cost_model_hardware_shapes(
      self, m: int, k: int, n: int, pre_quantize_lhs: bool
  ):
    model = cost_model.SubchannelCostModel(
        m=m,
        k=k,
        n=n,
        lhs_mem=pltpu.HBM,
        rhs_mem=pltpu.HBM,
        out_mem=pltpu.HBM,
        lhs_dtype=jnp.bfloat16,
        rhs_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        outer_acc_dtype=jnp.bfloat16,
        lhs_quantized_dtype=jnp.float8_e4m3fn,
        rhs_quantized_dtype=jnp.int4,
        inner_acc_dtype=jnp.float32,
        subchannel_size=1024,
        pre_quantize_lhs=pre_quantize_lhs,
        chip_version=pltpu.ChipVersion.TPU_7X,
    )
    window = model.select_window(
        vmem_limit_bytes=32 * 1024 * 1024,
        p_state=3,
    )
    window_tuple = (int(window[0]), int(window[1]), int(window[2]))
    self.assertEqual(window_tuple[0] % 128, 0)
    self.assertEqual(window_tuple[1] % 1024, 0)
    self.assertEqual(window_tuple[2] % 256, 0)

  def test_xlu_cycles(self):
    model = _CustomCostModel(**_COMMON_KWARGS)
    xlane_max = platform_info.XluOp.XLANE_REDUCE
    # On TPU_7X: num_xlus=2, and xlane reduce max has a cadence of 8 cycles for
    # bf16 and 4 cycles for f32. The XLU pipeline latency is not included.
    with self.subTest(msg="bf16"):
      self.assertAlmostEqual(
          float(model._xlu_cycles(8, xlane_max, jnp.bfloat16)), 8 * 8 / 2
      )
    with self.subTest(msg="f32"):
      self.assertAlmostEqual(
          float(model._xlu_cycles(8, xlane_max, jnp.float32)), 8 * 4 / 2
      )
    with self.subTest(msg="zero_ops"):
      self.assertAlmostEqual(
          float(model._xlu_cycles(0, xlane_max, jnp.bfloat16)), 0.0
      )

  def test_xlu_timings_by_op_and_dtype(self):
    pinfo = platform_info.get_platform_info(pltpu.ChipVersion.TPU_7X)
    xlane_max = platform_info.XluOp.XLANE_REDUCE
    self.assertEqual(pinfo.get_xlu_cadence_cycles(xlane_max, jnp.float32), 4)
    self.assertEqual(pinfo.get_xlu_cadence_cycles(xlane_max, jnp.bfloat16), 8)
    self.assertEqual(pinfo.get_xlu_latency_cycles(xlane_max, jnp.float32), 141)
    self.assertEqual(pinfo.get_xlu_latency_cycles(xlane_max, jnp.bfloat16), 145)
    # dtype objects (rather than scalar types) map to the same entries.
    self.assertEqual(
        pinfo.get_xlu_latency_cycles(xlane_max, jnp.dtype(jnp.bfloat16)), 145
    )
    # 8-bit dtypes without native XLU support fall back to bf16.
    self.assertEqual(
        pinfo.get_xlu_dtype(xlane_max, jnp.float8_e4m3fn), jnp.bfloat16
    )
    self.assertEqual(
        pinfo.get_xlu_cadence_cycles(xlane_max, jnp.float8_e4m3fn), 8
    )
    self.assertEqual(
        pinfo.get_xlu_latency_cycles(xlane_max, jnp.float8_e4m3fn), 145
    )

  @parameterized.parameters(
      pltpu.ChipVersion.TPU_V5E,
      pltpu.ChipVersion.TPU_V5P,
      pltpu.ChipVersion.TPU_V6E,
      pltpu.ChipVersion.TPU_7X,
      pltpu.ChipVersion.TPU_8I,
      pltpu.ChipVersion.TPU_8T,
  )
  def test_all_platforms_have_xlu_timings(self, chip: pltpu.ChipVersion):
    pinfo = platform_info.get_platform_info(chip)
    for op in platform_info.XluOp:
      for dtype in (jnp.bfloat16, jnp.float32):
        self.assertGreater(pinfo.get_xlu_cadence_cycles(op, dtype), 0)
        self.assertGreater(pinfo.get_xlu_latency_cycles(op, dtype), 0)

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

  @parameterized.parameters(
      (jnp.bfloat16, 8, 145, 16),
      (jnp.float32, 4, 141, 8),
  )
  def test_subchannel_xlu_uses_xlane_reduce_max(
      self,
      lhs_dtype: Any,
      expected_cadence: int,
      expected_latency: int,
      expected_rows_per_xlu_op: int,
  ):
    model = cost_model.SubchannelCostModel(
        m=1024,
        k=8192,
        n=8192,
        lhs_mem=pltpu.HBM,
        rhs_mem=pltpu.HBM,
        out_mem=pltpu.HBM,
        lhs_dtype=lhs_dtype,
        rhs_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        outer_acc_dtype=jnp.float32,
        lhs_quantized_dtype=jnp.float8_e4m3fn,
        rhs_quantized_dtype=jnp.float8_e4m3fn,
        subchannel_size=1024,
        pre_quantize_lhs=False,
        chip_version=pltpu.ChipVersion.TPU_7X,
    )
    pinfo = platform_info.get_platform_info(pltpu.ChipVersion.TPU_7X)
    num_xlus = pinfo.num_xlus
    block_m, block_k = model._block_m, model._block_k
    subblocks_per_tile = block_k // model.subchannel_size
    in_cost = model._get_input_fusion_stage_cost()

    # XLU throughput: one xlane reduce max per vreg of each subchannel.
    expected_xlu_cycles = (
        subblocks_per_tile
        * np.ceil(block_m / expected_rows_per_xlu_op)
        * expected_cadence
    )
    np.testing.assert_allclose(
        np.broadcast_to(in_cost.xlu_cycles, model._grid_shape),
        np.broadcast_to(expected_xlu_cycles, model._grid_shape),
    )

    # Latency: quantization of the first MXU op's subchannel, plus issuing its
    # xlane reduce max ops, plus the xlane reduce max pipeline latency.
    first_op_rows = np.minimum(block_m, model._get_rows_per_vmatmul())
    expected_latency_cycles = (
        model._alu_latency_cycles(
            first_op_rows,
            model.subchannel_size,
            model._NUM_QUANTIZE_ALU_OPS,
            model._get_input_fusion_dtype(),
        )
        + np.ceil(first_op_rows / expected_rows_per_xlu_op)
        * expected_cadence
        / num_xlus
        + expected_latency
    )
    np.testing.assert_allclose(
        np.broadcast_to(in_cost.latency_cycles, model._grid_shape),
        np.broadcast_to(expected_latency_cycles, model._grid_shape),
    )

  def test_pipeline_matmul_bound(self):
    # When fusions have negligible throughput cost, only their latencies add to
    # the per-iteration compute latency.
    base_model = _CustomCostModel(**_COMMON_KWARGS)
    model = _CustomCostModel(
        **_COMMON_KWARGS,
        input_fusion_stage_cost=cost_model.PipelineStageCost(
            latency_cycles=100.0
        ),
        output_fusion_stage_cost=cost_model.PipelineStageCost(
            latency_cycles=50.0
        ),
    )
    compute_diff = model._compute_latency_ns_per_iteration(
        _P_STATE
    ) - base_model._compute_latency_ns_per_iteration(_P_STATE)
    np.testing.assert_allclose(compute_diff, 150.0 / _clock_ghz())

  @parameterized.named_parameters(
      ("input_vpu_bound", 1e6, 0.0, 0.0, 1e6),
      ("output_vpu_bound", 0.0, 0.0, 1e6, 1e6),
      ("vpu_sums_across_stages", 1e6, 0.0, 2e6, 3e6),
      ("max_over_units", 0.0, 1e6, 2e6, 2e6),
      ("xlu_bound", 0.0, 3e6, 1e6, 3e6),
      ("vpu_and_xlu_overlap", 1e6, 2.5e6, 1e6, 2.5e6),
  )
  def test_pipeline_fusion_bound(
      self,
      in_alu_cycles: float,
      in_xlu_cycles: float,
      out_alu_cycles: float,
      expected: float,
  ):
    # When a fusion unit is the throughput bottleneck, it replaces the matmul
    # throughput term, and all three stage latencies are added. VPU work from
    # both fusions shares the VALUs and adds up, while the XLU is balanced
    # against the VPU as a separate unit.
    in_latency_cycles, out_latency_cycles = 30.0, 20.0
    pinfo = platform_info.get_platform_info(_COMMON_KWARGS["chip_version"])
    model = _CustomCostModel(
        **_COMMON_KWARGS,
        input_fusion_stage_cost=cost_model.PipelineStageCost(
            alu_cycles=in_alu_cycles * pinfo.num_alus,
            xlu_cycles=in_xlu_cycles * pinfo.num_xlus,
            latency_cycles=in_latency_cycles,
        ),
        output_fusion_stage_cost=cost_model.PipelineStageCost(
            alu_cycles=out_alu_cycles * pinfo.num_alus,
            latency_cycles=out_latency_cycles,
        ),
    )
    matmul_cost = model._get_matmul_stage_cost()
    # Sanity check that a fusion unit is actually the bottleneck.
    matmul_throughput = (
        np.maximum(
            matmul_cost.matmul_cycles,
            matmul_cost.latch_cycles,
        )
        / model._platform_info.num_mxus
    )
    self.assertTrue(np.all(matmul_throughput < expected))
    expected_ns = (
        expected
        + in_latency_cycles
        + matmul_cost.latency_cycles
        + out_latency_cycles
    ) / _clock_ghz()
    np.testing.assert_allclose(
        model._compute_latency_ns_per_iteration(_P_STATE),
        np.broadcast_to(expected_ns, model._grid_shape),
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

  def test_subchannel_vpu_sums_input_and_output_fusions(self):
    # For on-the-fly LHS quantization with an emulated s4 RHS, the VPU load is
    # the sum of quantization, RHS unpacking, and dequantization/accumulation.
    model = cost_model.SubchannelCostModel(
        m=1024,
        k=8192,
        n=8192,
        lhs_mem=pltpu.HBM,
        rhs_mem=pltpu.HBM,
        out_mem=pltpu.HBM,
        lhs_dtype=jnp.bfloat16,
        rhs_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        outer_acc_dtype=jnp.bfloat16,
        lhs_quantized_dtype=jnp.float8_e4m3fn,
        rhs_quantized_dtype=jnp.int4,
        subchannel_size=1024,
        pre_quantize_lhs=False,
        chip_version=pltpu.ChipVersion.TPU_7X,
    )
    block_m, block_k, block_n = model._block_m, model._block_k, model._block_n
    subblocks_per_tile = block_k // model.subchannel_size
    in_dtype = model._get_input_fusion_dtype()
    out_dtype = model._get_output_fusion_dtype()
    expected_in_vpu = model._alu_cycles(
        block_m, block_k, model._NUM_QUANTIZE_ALU_OPS, in_dtype
    )
    if model.lhs_quantized_dtype != model.matmul_lhs_dtype:
      expected_in_vpu = expected_in_vpu + model._alu_cycles(
          block_m, block_k, model._NUM_UNPACK_ALU_OPS, in_dtype
      )
    if model.rhs_quantized_dtype != model.matmul_rhs_dtype:
      expected_in_vpu = expected_in_vpu + model._alu_cycles(
          block_k, block_n, model._NUM_UNPACK_ALU_OPS, in_dtype
      )
    expected_out_vpu = subblocks_per_tile * model._alu_cycles(
        block_m, block_n, model._NUM_DEQUANT_ACCUM_ALU_OPS, out_dtype
    )
    in_cost = model._get_input_fusion_stage_cost()
    out_cost = model._get_output_fusion_stage_cost()
    np.testing.assert_allclose(
        np.broadcast_to(in_cost.alu_cycles, model._grid_shape),
        np.broadcast_to(expected_in_vpu, model._grid_shape),
    )
    np.testing.assert_allclose(
        np.broadcast_to(out_cost.alu_cycles, model._grid_shape),
        np.broadcast_to(expected_out_vpu, model._grid_shape),
    )

    # The per-iteration compute latency is driven by the busiest unit, where
    # the VPU load is the sum of both fusions.
    matmul_cost = model._get_matmul_stage_cost()
    pinfo = platform_info.get_platform_info(pltpu.ChipVersion.TPU_7X)
    matmul_throughput = (
        np.maximum(
            matmul_cost.matmul_cycles,
            matmul_cost.latch_cycles,
        )
        / pinfo.num_mxus
    )
    throughput_cycles = np.maximum(
        np.maximum(
            matmul_throughput,
            (expected_in_vpu + expected_out_vpu) / pinfo.num_alus,
        ),
        in_cost.xlu_cycles / pinfo.num_xlus,
    )
    p_state = pinfo.default_p_state
    expected_ns = (
        throughput_cycles
        + in_cost.latency_cycles
        + matmul_cost.latency_cycles
        + out_cost.latency_cycles
    ) / pinfo.clock_speed_ghz_by_p_state[p_state]
    np.testing.assert_allclose(
        model._compute_latency_ns_per_iteration(p_state),
        np.broadcast_to(expected_ns, model._grid_shape),
    )

  @parameterized.product(
      chip=[
          pltpu.ChipVersion.TPU_V5E,
          pltpu.ChipVersion.TPU_V5P,
          pltpu.ChipVersion.TPU_V6E,
          pltpu.ChipVersion.TPU_7X,
      ],
      lhs_dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_subchannel_vpu_bf16_to_f32_conversion(
      self,
      chip: pltpu.ChipVersion,
      lhs_dtype: Any,
  ):
    model = cost_model.SubchannelCostModel(
        m=1024,
        k=4096,
        n=4096,
        lhs_mem=pltpu.HBM,
        rhs_mem=pltpu.HBM,
        out_mem=pltpu.HBM,
        lhs_dtype=lhs_dtype,
        rhs_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        outer_acc_dtype=jnp.bfloat16,
        lhs_quantized_dtype=jnp.float8_e5m2,
        rhs_quantized_dtype=jnp.float8_e5m2,
        subchannel_size=1024,
        pre_quantize_lhs=False,
        chip_version=chip,
    )
    in_dtype = model._get_input_fusion_dtype()
    in_cost = model._get_input_fusion_stage_cost()
    base_alu = model._alu_cycles(
        model._block_m, model._block_k, model._NUM_QUANTIZE_ALU_OPS, in_dtype
    )
    if chip in (
        pltpu.ChipVersion.TPU_V5E,
        pltpu.ChipVersion.TPU_V5P,
    ) and jnp.dtype(lhs_dtype) == jnp.dtype(jnp.bfloat16):
      conversion_alu = model._alu_cycles(
          model._block_m,
          model._block_k,
          model._NUM_CONVERT_BF16_TO_F32_ALU_OPS,
          in_dtype,
      )
      expected_alu = base_alu + conversion_alu
    else:
      expected_alu = base_alu
    np.testing.assert_allclose(
        np.broadcast_to(in_cost.alu_cycles, model._grid_shape),
        np.broadcast_to(expected_alu, model._grid_shape),
    )

  @parameterized.parameters(
      (pltpu.ChipVersion.TPU_7X, jnp.dtype(jnp.float8_e4m3fn)),
      (pltpu.ChipVersion.TPU_V6E, jnp.dtype(jnp.bfloat16)),
      (pltpu.ChipVersion.TPU_V5P, jnp.dtype(jnp.bfloat16)),
      (pltpu.ChipVersion.TPU_V5E, jnp.dtype(jnp.bfloat16)),
  )
  def test_subchannel_s4_rhs_unpack_across_platforms(
      self,
      chip: pltpu.ChipVersion,
      expected_matmul_rhs_dtype: jnp.dtype,
  ):
    model = cost_model.SubchannelCostModel(
        m=1024,
        k=4096,
        n=4096,
        lhs_mem=pltpu.HBM,
        rhs_mem=pltpu.HBM,
        out_mem=pltpu.HBM,
        lhs_dtype=jnp.bfloat16,
        rhs_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        outer_acc_dtype=jnp.bfloat16,
        lhs_quantized_dtype=jnp.float8_e5m2,
        rhs_quantized_dtype=jnp.int4,
        subchannel_size=1024,
        pre_quantize_lhs=False,
        chip_version=chip,
    )
    self.assertEqual(model.matmul_rhs_dtype, expected_matmul_rhs_dtype)
    self.assertNotEqual(model.rhs_quantized_dtype, model.matmul_rhs_dtype)
    in_cost = model._get_input_fusion_stage_cost()
    in_dtype = model._get_input_fusion_dtype()
    unpack_alu = model._alu_cycles(
        model._block_k, model._block_n, model._NUM_UNPACK_ALU_OPS, in_dtype
    )
    self.assertTrue(np.all(in_cost.alu_cycles >= unpack_alu))

  @parameterized.product(
      chip=[
          pltpu.ChipVersion.TPU_V5E,
          pltpu.ChipVersion.TPU_V5P,
          pltpu.ChipVersion.TPU_V6E,
          pltpu.ChipVersion.TPU_7X,
          pltpu.ChipVersion.TPU_8I,
          pltpu.ChipVersion.TPU_8T,
      ],
      lhs_dtype=[jnp.bfloat16, jnp.float32, jnp.dtype(jnp.bfloat16)],
      outer_acc_dtype=[jnp.bfloat16, jnp.float32, jnp.dtype(jnp.float32)],
  )
  def test_subchannel_fusion_dtypes_across_platforms(
      self,
      chip: pltpu.ChipVersion,
      lhs_dtype: Any,
      outer_acc_dtype: Any,
  ):
    model = cost_model.SubchannelCostModel(
        m=1024,
        k=4096,
        n=4096,
        lhs_mem=pltpu.HBM,
        rhs_mem=pltpu.HBM,
        out_mem=pltpu.HBM,
        lhs_dtype=lhs_dtype,
        rhs_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        outer_acc_dtype=outer_acc_dtype,
        lhs_quantized_dtype=jnp.float8_e4m3fn,
        rhs_quantized_dtype=jnp.float8_e4m3fn,
        subchannel_size=1024,
        pre_quantize_lhs=False,
        chip_version=chip,
    )
    if chip in (pltpu.ChipVersion.TPU_V5E, pltpu.ChipVersion.TPU_V5P):
      self.assertEqual(model._get_input_fusion_dtype(), jnp.float32)
      self.assertEqual(model._get_output_fusion_dtype(), jnp.float32)
    else:
      expected_in = (
          jnp.float32
          if jnp.dtype(lhs_dtype) == jnp.dtype(jnp.float32)
          else jnp.bfloat16
      )
      expected_out = (
          jnp.float32
          if jnp.dtype(outer_acc_dtype) == jnp.dtype(jnp.float32)
          else jnp.bfloat16
      )
      self.assertEqual(model._get_input_fusion_dtype(), expected_in)
      self.assertEqual(model._get_output_fusion_dtype(), expected_out)

  @parameterized.product(
      chip=[
          pltpu.ChipVersion.TPU_V5E,
          pltpu.ChipVersion.TPU_7X,
          pltpu.ChipVersion.TPU_8T,
      ],
      lhs_dtype=[jnp.bfloat16, jnp.float32],
      outer_acc_dtype=[jnp.bfloat16, jnp.float32],
  )
  def test_subchannel_derived_throughput_and_latency(
      self,
      chip: pltpu.ChipVersion,
      lhs_dtype: Any,
      outer_acc_dtype: Any,
  ):
    model = cost_model.SubchannelCostModel(
        m=1024,
        k=4096,
        n=4096,
        lhs_mem=pltpu.HBM,
        rhs_mem=pltpu.HBM,
        out_mem=pltpu.HBM,
        lhs_dtype=lhs_dtype,
        rhs_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        outer_acc_dtype=outer_acc_dtype,
        lhs_quantized_dtype=jnp.float8_e4m3fn,
        rhs_quantized_dtype=jnp.float8_e4m3fn,
        subchannel_size=1024,
        pre_quantize_lhs=False,
        chip_version=chip,
    )
    pinfo = platform_info.get_platform_info(chip)
    in_dtype = model._get_input_fusion_dtype()
    out_dtype = model._get_output_fusion_dtype()
    xlu_op = platform_info.XluOp.XLANE_REDUCE
    xlu_cadence = pinfo.get_xlu_cadence_cycles(xlu_op, in_dtype)
    xlu_latency = pinfo.get_xlu_latency_cycles(xlu_op, in_dtype)

    sublane_count, lane_count = pinfo.vreg_size
    in_bits = jax.dtypes.itemsize_bits(in_dtype)
    in_rows_per_vreg = sublane_count * (32 // in_bits)
    out_bits = jax.dtypes.itemsize_bits(out_dtype)
    out_rows_per_vreg = sublane_count * (32 // out_bits)

    block_m, block_k, block_n = model._block_m, model._block_k, model._block_n
    subblocks = block_k // model.subchannel_size
    first_op_rows = np.minimum(block_m, model._get_rows_per_vmatmul())
    _, mxu_non_contracting_dim = model._get_mxu_size()
    last_op_rows = np.minimum(block_m, model._get_rows_per_vmatmul())

    in_vregs = np.ceil(block_m / in_rows_per_vreg) * np.ceil(
        block_k / lane_count
    )
    expected_in_alu = in_vregs * model._NUM_QUANTIZE_ALU_OPS * 2
    if pinfo.chip_version in (
        pltpu.ChipVersion.TPU_V5E,
        pltpu.ChipVersion.TPU_V5P,
    ) and jnp.dtype(lhs_dtype) == jnp.dtype(jnp.bfloat16):
      expected_in_alu = (
          expected_in_alu
          + in_vregs * model._NUM_CONVERT_BF16_TO_F32_ALU_OPS * 2
      )
    if model.lhs_quantized_dtype != model.matmul_lhs_dtype:
      expected_in_alu = expected_in_alu + model._alu_cycles(
          block_m, block_k, model._NUM_UNPACK_ALU_OPS, in_dtype
      )
    if model.rhs_quantized_dtype != model.matmul_rhs_dtype:
      expected_in_alu = expected_in_alu + model._alu_cycles(
          block_k, block_n, model._NUM_UNPACK_ALU_OPS, in_dtype
      )
    xlu_ops = subblocks * np.ceil(block_m / in_rows_per_vreg)
    expected_in_xlu = xlu_ops * xlu_cadence
    first_op_vregs = np.ceil(first_op_rows / in_rows_per_vreg) * np.ceil(
        model.subchannel_size / lane_count
    )
    first_op_xlu_ops = np.ceil(first_op_rows / in_rows_per_vreg)
    expected_in_latency = (
        (first_op_vregs * model._NUM_QUANTIZE_ALU_OPS * 2) / pinfo.num_alus
        + (first_op_xlu_ops * xlu_cadence) / pinfo.num_xlus
        + xlu_latency
    )
    if pinfo.chip_version in (
        pltpu.ChipVersion.TPU_V5E,
        pltpu.ChipVersion.TPU_V5P,
    ) and jnp.dtype(lhs_dtype) == jnp.dtype(jnp.bfloat16):
      expected_in_latency = (
          expected_in_latency
          + (first_op_vregs * model._NUM_CONVERT_BF16_TO_F32_ALU_OPS * 2)
          / pinfo.num_alus
      )
    if model.lhs_quantized_dtype != model.matmul_lhs_dtype:
      expected_in_latency = expected_in_latency + model._alu_latency_cycles(
          first_op_rows,
          model._get_mxu_size()[0],
          model._NUM_UNPACK_ALU_OPS,
          in_dtype,
      )
    if model.rhs_quantized_dtype != model.matmul_rhs_dtype:
      expected_in_latency = expected_in_latency + model._alu_latency_cycles(
          model._get_mxu_size()[0],
          mxu_non_contracting_dim,
          model._NUM_UNPACK_ALU_OPS,
          in_dtype,
      )

    in_cost = model._get_input_fusion_stage_cost()
    np.testing.assert_allclose(
        np.broadcast_to(in_cost.alu_cycles, model._grid_shape),
        np.broadcast_to(expected_in_alu, model._grid_shape),
    )
    np.testing.assert_allclose(
        np.broadcast_to(in_cost.xlu_cycles, model._grid_shape),
        np.broadcast_to(expected_in_xlu, model._grid_shape),
    )
    np.testing.assert_allclose(
        np.broadcast_to(in_cost.latency_cycles, model._grid_shape),
        np.broadcast_to(expected_in_latency, model._grid_shape),
    )

    out_vregs = np.ceil(block_m / out_rows_per_vreg) * np.ceil(
        block_n / lane_count
    )
    expected_out_alu = subblocks * out_vregs * 4 * 2
    last_op_vregs = np.ceil(last_op_rows / out_rows_per_vreg) * np.ceil(
        mxu_non_contracting_dim / lane_count
    )
    expected_out_latency = (last_op_vregs * 4 * 2) / pinfo.num_alus

    out_cost = model._get_output_fusion_stage_cost()
    np.testing.assert_allclose(
        np.broadcast_to(out_cost.alu_cycles, model._grid_shape),
        np.broadcast_to(expected_out_alu, model._grid_shape),
    )
    np.testing.assert_allclose(
        np.broadcast_to(out_cost.latency_cycles, model._grid_shape),
        np.broadcast_to(expected_out_latency, model._grid_shape),
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
