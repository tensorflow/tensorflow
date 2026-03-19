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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from xla.backends.cpu import testlib as cpu_testlib
from xla.backends.cpu.testlib import utilities
from xla.codegen import testlib as base_testlib
from xla.codegen.testlib import utilities as base_utilities

create_literal = base_utilities.create_literal_from_np
HloInstruction = base_testlib.HloInstruction


class ConcatenateKernelRunnerTest(parameterized.TestCase):

  @parameterized.product(
      cycle_layout=[True, False],
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
      concat_dimension=[0, 1, 2],
  )
  def test_concatenate(self, cycle_layout, dtype, concat_dimension):
    num_inputs = 5
    shape = (4, 4, 4)
    np_inputs = [
        (np.random.rand(*shape) * 10).astype(dtype) for _ in range(num_inputs)
    ]
    if cycle_layout:
      # Give the inputs different layouts to test the slow path.
      default_layout = [0, 1, 2]
      input_literals = [
          create_literal(input_array, np.roll(default_layout, idx))
          for idx, input_array in enumerate(np_inputs)
      ]
    else:
      input_literals = [
          create_literal(input_array) for input_array in np_inputs
      ]

    expected_output = np.concatenate(np_inputs, axis=concat_dimension)
    output_literal = create_literal(np.zeros_like(expected_output))

    hlo_parameters = [
        HloInstruction.create_parameter(idx, literal.shape(), f"input_{idx}")
        for [idx, literal] in enumerate(input_literals)
    ]

    hlo_op = HloInstruction.create_concatenate(
        output_literal.shape(), hlo_parameters, concat_dimension
    )

    hlo_module, buffer_assignment = utilities.build_hlo_module(
        hlo_op, *hlo_parameters
    )
    jit_compiler = cpu_testlib.JitCompiler(hlo_module.get_config())

    emitter = cpu_testlib.ConcatenateKernelEmitter(
        hlo_module.get_root_instruction(),
        buffer_assignment,
        jit_compiler.get_target_machine(),
    )

    kernel_definition = emitter.emit_kernel_definition()
    self.assertIsNotNone(kernel_definition)

    runner = cpu_testlib.KernelRunner.create(kernel_definition, jit_compiler)
    runner.call(list(itertools.chain(input_literals, [output_literal])))

    np.testing.assert_array_equal(np.asarray(output_literal), expected_output)


if __name__ == "__main__":
  absltest.main()
