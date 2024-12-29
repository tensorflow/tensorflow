# Copyright 2024 The OpenXLA Authors.
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

from collections.abc import Callable, Sequence
import dataclasses
import itertools
import math

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from xla.backends.cpu import testlib as testlib_cpu
from xla.codegen import testlib as testlib_base
from xla.codegen.testlib import utilities as testlib_utilities

HloOpcode = testlib_base.HloOpcode
create_literal = testlib_base.utilities.create_literal_from_np
HloInstruction = testlib_base.HloInstruction
ComparisonDirection = testlib_base.ComparisonDirection
_inf = float("inf")


def create_input(
    value_range: tuple[float, float],
    shape: Sequence[int],
    dtype: np.dtype,
    shuffle: bool = False,
) -> np.ndarray:
  size = np.prod(shape)
  result = np.linspace(
      value_range[0], value_range[1], size, dtype=dtype
  ).reshape(shape)

  if shuffle:
    np.random.shuffle(result)

  return result


def np_erf(x):
  return np.vectorize(math.erf, otypes=[x.dtype])(x)


@dataclasses.dataclass(frozen=True)
class ElementalHloOpcodeDef:
  op: HloOpcode
  np_op: Callable[[np.ndarray, ...], np.ndarray]
  input_ranges: tuple[float, float] = (-1.0, 1.0)
  decimal_precision: int = 6

  # For simple unpacking
  def __iter__(self):
    return iter(
        (self.op, self.np_op, self.input_ranges, self.decimal_precision)
    )

  def __repr__(self):
    return f"{self.op.name}({self.input_ranges})"


@parameterized.product(
    op_def=[
        ElementalHloOpcodeDef(HloOpcode.sine, np.sin),
        ElementalHloOpcodeDef(HloOpcode.cosine, np.cos),
        ElementalHloOpcodeDef(HloOpcode.tan, np.tan),
        ElementalHloOpcodeDef(HloOpcode.exponential, np.exp),
        ElementalHloOpcodeDef(HloOpcode.log, np.log, (0.01, 10.0)),
        ElementalHloOpcodeDef(HloOpcode.log_plus_one, np.log1p),
        ElementalHloOpcodeDef(HloOpcode.sqrt, np.sqrt),
        ElementalHloOpcodeDef(
            HloOpcode.rsqrt, lambda x: np.reciprocal(np.sqrt(x))
        ),
        ElementalHloOpcodeDef(HloOpcode.cbrt, np.cbrt),
        ElementalHloOpcodeDef(HloOpcode.power, np.pow),
        ElementalHloOpcodeDef(HloOpcode.add, np.add),
        ElementalHloOpcodeDef(HloOpcode.subtract, np.subtract),
        ElementalHloOpcodeDef(HloOpcode.multiply, np.multiply),
        ElementalHloOpcodeDef(HloOpcode.divide, np.divide),
        ElementalHloOpcodeDef(HloOpcode.maximum, np.maximum),
        ElementalHloOpcodeDef(HloOpcode.minimum, np.minimum),
        ElementalHloOpcodeDef(HloOpcode.sign, np.sign),
        ElementalHloOpcodeDef(HloOpcode.negate, np.negative),
        ElementalHloOpcodeDef(HloOpcode.is_finite, np.isfinite, (-_inf, _inf)),
        ElementalHloOpcodeDef(HloOpcode.ceil, np.ceil, (-10.0, 10.0)),
        ElementalHloOpcodeDef(HloOpcode.floor, np.floor, (-5.0, 5.0)),
        ElementalHloOpcodeDef(HloOpcode.tanh, np.tanh),
        ElementalHloOpcodeDef(HloOpcode.atan2, np.arctan2),
        ElementalHloOpcodeDef(HloOpcode.erf, np_erf),
        ElementalHloOpcodeDef(HloOpcode.exponential_minus_one, np.expm1),
        # TODO(willfroom): Update to use better inputs for the following.
        ElementalHloOpcodeDef(HloOpcode.clamp, np.clip),
        # TODO(willfroom): Add complex ops.
        # ElementalHloOpcodeDef(HloOpcode.complex, np.complex),
        # ElementalHloOpcodeDef(HloOpcode.real, np.real),
        # ElementalHloOpcodeDef(HloOpcode.imag, np.imag),
        # TODO(willfroom): go through ElementalIrEmitter interface and ensure
        # that all ops are implemented.
        # ...
    ],
    shape=[(4,), (4, 3), (4, 3, 10)],
    dtype=[np.dtype(np.float32), np.dtype(np.float64)],
)
class ElementalKernelRunnerTest(absltest.TestCase):

  def id(self):
    return self._test_params_reprs.get(self._testMethodName, "")

  def test_elemental_kernel_emitter(
      self,
      op_def: ElementalHloOpcodeDef,
      shape: tuple[int, ...],
      dtype: np.dtype,
  ):

    [op, np_op, input_ranges, decimal_precision] = op_def

    num_inputs = testlib_utilities.opcode_arity(op)
    self.assertIsNotNone(num_inputs)

    np_inputs = [
        create_input(input_ranges, shape, dtype) for _ in range(num_inputs)
    ]
    input_literals = [create_literal(input_array) for input_array in np_inputs]

    expected_output = np_op(*np_inputs)
    output_literal = create_literal(
        np.ndarray(shape, dtype=expected_output.dtype)
    )

    hlo_parameters = [
        HloInstruction.create_parameter(idx, literal.shape(), f"input_{idx}")
        for [idx, literal] in enumerate(input_literals)
    ]

    hlo_op = HloInstruction.create_variadic(
        output_literal.shape(), op, hlo_parameters
    )

    emitter = testlib_cpu.ElementalKernelEmitter(hlo_op)
    kernel_spec = emitter.emit_kernel_spec()
    self.assertIsNotNone(kernel_spec)

    # kernel_spec is consumed by the runner, so we need to save the IR string
    # before passing it to the runner.
    ir_string = str(kernel_spec.kernel_source())

    runner = testlib_cpu.KernelRunner.create(kernel_spec)

    runner.call(list(itertools.chain(input_literals, [output_literal])))
    np.testing.assert_array_almost_equal(
        np.asarray(output_literal),
        expected_output,
        decimal=decimal_precision,
        err_msg=ir_string,
    )


@parameterized.product(
    op_def=[
        (ComparisonDirection.kEq, np.equal),
        (ComparisonDirection.kNe, np.not_equal),
        (ComparisonDirection.kGe, np.greater_equal),
        (ComparisonDirection.kGt, np.greater),
        (ComparisonDirection.kLe, np.less_equal),
        (ComparisonDirection.kLt, np.less),
    ],
    shape=[(4,), (4, 3), (4, 3, 10)],
    dtype=[
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
        np.dtype(np.uint64),
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.int32),
        np.dtype(np.int64),
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    ],
)
class ElementalComparisonKernelRunnerTest(absltest.TestCase):

  def test_elemental_comparision_kernel_emitter(self, op_def, shape, dtype):
    [direction, np_op] = op_def

    is_unsigned = np.issubdtype(dtype, np.unsignedinteger)
    value_range = (0.0, 20.0) if is_unsigned else (-10.0, 10.0)
    lhs_np = create_input(value_range, shape, dtype, shuffle=True)
    rhs_np = create_input(value_range, shape, dtype, shuffle=True)

    lhs_literal = create_literal(lhs_np)
    rhs_literal = create_literal(rhs_np)

    output_literal = create_literal(np.ndarray(shape, dtype=np.bool))

    lhs_param = HloInstruction.create_parameter(0, lhs_literal.shape(), "lhs")
    rhs_param = HloInstruction.create_parameter(1, rhs_literal.shape(), "rhs")

    hlo_op = HloInstruction.create_compare(
        output_literal.shape(), lhs_param, rhs_param, direction
    )

    emitter = testlib_cpu.ElementalKernelEmitter(hlo_op)

    runner = testlib_cpu.KernelRunner.create(emitter.emit_kernel_spec())

    runner.call([lhs_literal, rhs_literal, output_literal])
    np.testing.assert_equal(
        np.asarray(output_literal),
        np_op(lhs_np, rhs_np),
    )


if __name__ == "__main__":
  absltest.main()
