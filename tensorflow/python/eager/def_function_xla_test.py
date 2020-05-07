# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class DefFunctionTests(xla_test.XLATestCase):

  def testVarInitializedInFunction(self):
    with self.test_scope():
      v_holder = []

      @def_function.function
      def add_var(x):
        if not v_holder:
          v = variables.Variable([1., 2.])
          v_holder.append(v)
          already_initialized = variables.Variable(3.)
          with ops.init_scope():
            already_initialized.assign(10.)
          v_holder.append(already_initialized)
        return v_holder[0] + v_holder[1] + x

      self.assertAllClose([13., 14.], add_var(constant_op.constant(2.)))


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
