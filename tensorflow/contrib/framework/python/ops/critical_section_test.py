# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""critical section tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import critical_section_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


class CriticalSectionTest(test.TestCase):

  def testCreateCriticalSectionRaw(self):
    handle = critical_section_ops.critical_section_op("cs")
    v = resource_variable_ops.ResourceVariable(0.0, name="v")

    @function.Defun(dtypes.float32, dtypes.float32)
    def fn(a, b):
      c = v.read_value()
      with ops.control_dependencies([c]):
        nv = v.assign_add(a * b)
        with ops.control_dependencies([nv]):
          return array_ops.identity(c)

    def execute(fn, *args):
      output_args = fn.definition.signature.output_arg
      return resource_variable_ops.execute_in_critical_section(
          critical_section=handle,
          arguments=list(args) + fn.captured_inputs,
          f=fn,
          output_types=[out.type for out in output_args],
          output_shapes=[tensor_shape.TensorShape(None) for _ in output_args])

    num_concurrent = 1000
    r = [execute(fn, 1.0, 2.0)[0] for _ in range(num_concurrent)]
    self.evaluate(v.initializer)
    r_value = self.evaluate(r)
    self.assertAllClose([2.0 * i for i in range(num_concurrent)],
                        sorted(r_value))


if __name__ == "__main__":
  test.main()
