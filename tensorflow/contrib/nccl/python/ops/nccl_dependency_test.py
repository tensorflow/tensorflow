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
"""Dependency test for nccl to test behavior when NCCL is not installed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import nccl
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


class NcclDependencyTest(test.TestCase):
  """Verifies that importing nccl ops lib does not fail even if NCCL is not
  installed but nccl ops throws an exception on use if NCCL is not installed.
  """

  def test_nccl_ops(self):
    """Tests behavior of nccl ops when NCCL is not installed."""

    public_methods = [
        m[0]
        for m in tf_inspect.getmembers(nccl, tf_inspect.isfunction)
        if not m[0].startswith('_')
    ]
    for method_name in public_methods:
      with ops.device('/device:CPU:0'):
        tensor = constant_op.constant(1)

      if method_name == 'broadcast':
        arg = tensor
      else:
        arg = [tensor]

      nccl_op = getattr(nccl, method_name)
      with ops.device('/device:CPU:0'):
        with self.assertRaisesRegexp(errors_impl.NotFoundError,
                                     r'cannot open shared object file'):
          nccl_op(arg)


if __name__ == '__main__':
  test.main()
