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
"""Tests for the XRT client."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.xla.python import xla_client
from tensorflow.compiler.xla.python import xrt
from tensorflow.python.platform import test


def BuildAddAndScaleComputation(shape1, shape2):
  """Builds the computation (a + b) * 3."""
  b = xla_client.ComputationBuilder("add-and-scale")
  x = b.ParameterWithShape(shape1)
  y = b.ParameterWithShape(shape2)
  dtype = shape1.numpy_dtype().type
  b.Mul(b.Add(x, y), b.Constant(dtype(3)))
  return b.Build()


# TODO(phawkins): add more tests, beyond a simple "hello world" example.
class XrtBackendTest(test.TestCase):

  def testBasics(self):
    (worker,), _ = test.create_local_cluster(num_workers=1, num_ps=0)
    self.assertTrue(worker.target.startswith("grpc://"))
    tf_context = xrt.get_tf_context(worker.target[len("grpc://"):], "worker")
    backend = xrt.XrtBackend(tf_context, "XLA_CPU")

    a = np.arange(10)
    b = np.arange(10)

    c = BuildAddAndScaleComputation(
        xla_client.Shape.from_pyval(a), xla_client.Shape.from_pyval(b))

    executable = c.Compile(backend=backend)
    output = executable.ExecuteWithPythonValues((a, b))
    self.assertAllEqual(output, (a + b) * 3)


if __name__ == "__main__":
  test.main()
