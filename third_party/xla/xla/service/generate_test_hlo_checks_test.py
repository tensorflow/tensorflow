# Copyright 2021 The OpenXLA Authors.
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
"""Tests for generate_test_hlo_checks."""

from absl.testing import absltest

from xla.service import generate_test_hlo_checks


class GenerateTestHloChecksTest(absltest.TestCase):

  def test_replacement(self):
    input_hlo = """
%param.0 # Do not replace if it's not CHECK'd.
// CHECK: %computation { # Do not replace computations
// CHECK: %param.0 = parameter(0) # Replace
// CHECK: %param_1 = parameter(1)
// CHECK-NEXT: %add.1 = add(%param.0, %param_1) # Replace for any CHECK-directive
// CHECK-NEXT: ROOT %reduce = reduce(%add.1)
// CHECK-NEXT: }
// CHECK: %computation.2 { # New computation resets the counter.
// CHECK-NEXT: %parameter.0 = parameter(0)
// CHECK-NEXT: %get-tuple-element.1 = get-tuple-element(%parameter.0)
// CHECK-NEXT: ROOT %bitcast-convert = bitcast-convert(%get-tuple-element.1)
"""
    self.assertEqual(
        generate_test_hlo_checks.replace_instruction_names(input_hlo), """
%param.0 # Do not replace if it's not CHECK'd.
// CHECK: %computation { # Do not replace computations
// CHECK: [[param_0_0:%[^ ]+]] = parameter(0) # Replace
// CHECK: [[param_1_1:%[^ ]+]] = parameter(1)
// CHECK-NEXT: [[add_1_2:%[^ ]+]] = add([[param_0_0]], [[param_1_1]]) # Replace for any CHECK-directive
// CHECK-NEXT: ROOT [[reduce_3:%[^ ]+]] = reduce([[add_1_2]])
// CHECK-NEXT: }
// CHECK: %computation.2 { # New computation resets the counter.
// CHECK-NEXT: [[parameter_0_0:%[^ ]+]] = parameter(0)
// CHECK-NEXT: [[get_tuple_element_1_1:%[^ ]+]] = get-tuple-element([[parameter_0_0]])
// CHECK-NEXT: ROOT [[bitcast_convert_2:%[^ ]+]] = bitcast-convert([[get_tuple_element_1_1]])
""")


if __name__ == '__main__':
  absltest.main()
