# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Dependency test for rccl to test behavior when RCCL is not installed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import rccl
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


class RcclDependencyTest(test.TestCase):
  """Verifies that importing rccl ops lib does not fail even if RCCL is not
  installed but rccl ops throws an exception on use if RCCL is not installed.
  """

  def test_rccl_ops(self):
    """Tests behavior of rccl ops when RCCL is not installed."""

    public_methods = [
        m[0]
        for m in tf_inspect.getmembers(rccl, tf_inspect.isfunction)
        if not m[0].startswith('_')
    ]
    for method_name in public_methods:
      with ops.device('/device:CPU:0'):
        tensor = constant_op.constant(1)

      if method_name == 'broadcast':
        arg = tensor
      else:
        arg = [tensor]

      rccl_op = getattr(rccl, method_name)
      with ops.device('/device:CPU:0'):
        with self.assertRaisesRegexp(errors_impl.NotFoundError,
                                     r'cannot open shared object file'):
          rccl_op(arg)


if __name__ == '__main__':
  test.main()
