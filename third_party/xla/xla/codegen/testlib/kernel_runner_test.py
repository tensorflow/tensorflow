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

from absl.testing import absltest
import numpy as np
from xla.codegen.testlib import _extension
from xla.codegen.testlib import utilities as testlib_utilities
from xla.python import xla_extension


create_literal = testlib_utilities.create_literal_from_np


class HloModuleParse(absltest.TestCase):

  def test_from_instruction(self):
    shape = xla_extension.Shape.array_shape(np.dtype(np.int32), (4,))
    hlo_parameter = _extension.HloInstruction.create_parameter(
        0, shape, "input"
    )
    hlo_op = _extension.HloInstruction.create_variadic(
        shape, _extension.HloOpcode.sine, [hlo_parameter]
    )
    hlo_module = _extension.HloModule(hlo_op.name() + "_module")
    hlo_module.add_entry_computation(
        _extension.build_hlo_computation(hlo_op, hlo_parameter)
    )
    expected_parts = [
        "HloModule sine_module,",
        "{",
        "%input = s32[4]{0} parameter(0)",
        "ROOT %sine = s32[4]{0} sine(%input)",
        "}",
    ]
    self.assertContainsInOrder(
        expected_parts,
        str(hlo_module),
    )


class LiteralFromNpTest(absltest.TestCase):

  def test_output_same_as_input(self):
    array = np.array([1, 2, 3, 4], dtype=np.int32)
    got = create_literal(array)
    np.testing.assert_array_equal(np.asarray(got), array)


class DummyKernelRunnerTest(absltest.TestCase):

  def test_dummy_kernel(self):
    runner = _extension.DummyAddKernelRunner()
    in_arg1 = create_literal(np.array([1, 2, 3, 4], dtype=np.int32))
    in_arg2 = create_literal(np.array([5, 6, 7, 8], dtype=np.int32))
    out_arg = create_literal(np.array([0, 0, 0, 0], dtype=np.int32))
    runner.call([in_arg1, in_arg2, out_arg])
    np.testing.assert_array_equal(
        np.asarray(out_arg), np.asarray(in_arg1) + np.asarray(in_arg2)
    )


if __name__ == "__main__":
  absltest.main()
