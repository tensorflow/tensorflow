# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for XLA call module op wrapper with disabled platform check.

This test runs with --tf_xla_call_module_disabled_checks=platform
"""
from typing import Tuple

import numpy as np

from tensorflow.compiler.mlir.stablehlo import stablehlo
from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


def serialize(module_str: str) -> Tuple[str, int]:
  target = stablehlo.get_minimum_version()
  byte_str = stablehlo.serialize_portable_artifact(module_str, target)
  return byte_str, xla.call_module_maximum_supported_version()


class XlaCallModuleOpTest(xla_test.XLATestCase):

  def _assertOpOutputMatchesExpected(self,
                                     op,
                                     args,
                                     expected,
                                     equality_fn=None):
    """Asserts op(*args) == expected."""
    with self.session() as session:
      with self.test_scope():
        placeholders = [
            array_ops.placeholder(dtypes.as_dtype(arg.dtype), arg.shape)
            for arg in args
        ]
        feeds = {placeholders[i]: args[i] for i in range(0, len(args))}
        output = op(*placeholders)
      result = session.run(output, feeds)
      if not equality_fn:
        equality_fn = self.assertAllClose
      equality_fn(result, expected, rtol=1e-3)

  def test_platforms_errors(self):
    """Error reporting for the platforms attribute."""
    x = np.float32(0.)

    module_str = """
module @jit_f.0 {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    return %arg0 : tensor<f32>
  }
}
"""
    module, version = serialize(module_str)
    def f(x):
      return xla.call_module(
          [x], version=version,
          module=module,
          Tout=[np.float32],
          Sout=[()],
          platforms=['RANDOM_PLATFORM'],
          disabled_checks=[])
    # No error even though the `platforms` does not match the testing platform
    self._assertOpOutputMatchesExpected(f, (x,), (x,))


if __name__ == '__main__':
  # This test is using Tensorflow sessions which are not compatible with eager
  # mode.
  ops.disable_eager_execution()
  googletest.main()
