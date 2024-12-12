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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from xla.backends.cpu.testlib import kernel_runner
from xla.codegen.testlib import kernel_runner as kernel_runner_base
from xla.python import xla_extension

HloOpcode = kernel_runner_base.HloOpcode
create_literal = kernel_runner_base.create_literal_from_np
_inf = float("inf")


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
        # TODO(willfroom): Update to use better inputs for the following.
        ElementalHloOpcodeDef(HloOpcode.clamp, np.clip),
        # TODO(willfroom): Enable the following once real ir emitter is
        # implemented.
        # ElementalHloOpcodeDef(HloOpcode.tanh, np.tanh),
        # ElementalHloOpcodeDef(HloOpcode.atan2, np.arctan2),
        # ElementalHloOpcodeDef(HloOpcode.erf, np.erf),
        # ElementalHloOpcodeDef(HloOpcode.exponential_minus_one, np.expm1),
        # TODO(willfroom): Add comparision ops once they are implemented.
        # ...
        # TODO(willfroom): Add complex ops once they are implemented.
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

  def create_input(
      self,
      value_range: tuple[float, float],
      shape: Sequence[int],
      dtype: np.dtype,
  ) -> np.ndarray:
    size = np.prod(shape)
    return np.linspace(
        value_range[0], value_range[1], size, dtype=dtype
    ).reshape(shape)

  def test_elemental_kernel_emitter(
      self,
      op_def: ElementalHloOpcodeDef,
      shape: tuple[int, ...],
      dtype: np.dtype,
  ):

    if (op_def.op == HloOpcode.log) and (dtype == np.float64):
      self.skipTest("TODO(willfroom): Look into why this fails.")

    [op, np_op, input_ranges, decimal_precision] = op_def

    num_inputs = kernel_runner_base.opcode_arity(op)
    self.assertIsNotNone(num_inputs)

    np_inputs = [self.create_input(input_ranges, shape, dtype)] * num_inputs
    input_literals = [create_literal(input_array) for input_array in np_inputs]

    expected_output = np_op(*np_inputs)
    output_literal = create_literal(
        np.ndarray(shape, dtype=expected_output.dtype)
    )

    # TODO(willfroom): Add support to get the shape directly from the Literal.
    input_shape = xla_extension.Shape.array_shape(dtype, shape)
    output_shape = xla_extension.Shape.array_shape(expected_output.dtype, shape)
    emitter = kernel_runner.ElementalKernelEmitter(
        op.name, op, [input_shape] * num_inputs, output_shape
    )

    runner = kernel_runner.KernelRunner.create(emitter.emit_kernel_spec())

    runner.call(list(itertools.chain(input_literals, [output_literal])))
    np.testing.assert_array_almost_equal(
        np.asarray(output_literal),
        expected_output,
        decimal=decimal_precision,
    )


if __name__ == "__main__":
  absltest.main()
