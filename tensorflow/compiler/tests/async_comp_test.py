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
"""Tests for asynchronous compilation on the CPU and GPU devices."""

import os

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def RunMetadataLabels(run_metadata):
  """Returns all labels in run_metadata."""
  labels = []
  for dev_stats in run_metadata.step_stats.dev_stats:
    for node_stats in dev_stats.node_stats:
      labels.append(node_stats.timeline_label)
  return labels


def InLabels(labels, substr):
  """Returns true iff one of the labels contains substr."""
  return any(substr in x for x in labels)


def MetadataHasXlaRunOp(run_metadata):
  """Returns true if there are XlaRun kernels in run_metadata's timeline."""

  # TODO(phawkins): find a less hacky way to test whether a kernel ran.
  return InLabels(RunMetadataLabels(run_metadata), "_XlaRun")


class AsyncCompilationTest(test.TestCase):

  # Asynchrobnous compilation uses the existing fallback path and existing
  # compiler. This test only tests that asynchronus compilation is performed.
  def testAsyncCompilationJit(self):

    @function.Defun(compiled=True)
    def CompiledFunction(x):
      return math_ops.log(x)

    with session_lib.Session() as sess:
      x = array_ops.placeholder(dtypes.float32)
      y = CompiledFunction(x)

      run_metadata = config_pb2.RunMetadata()
      sess.run(
          y,
          feed_dict={x: [0.] * 60},
          run_metadata=run_metadata,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
      # For The first iteration, the fall back path is chosen.
      hasXlaRunOp = MetadataHasXlaRunOp(run_metadata)
      self.assert_(not hasXlaRunOp)

      # Execute the session until after asynchronous compilation is finished
      # and the compiled cluster has been executed once.
      while (not hasXlaRunOp):
        run_metadata = config_pb2.RunMetadata()
        sess.run(
            y,
            feed_dict={x: [0.] * 60},
            run_metadata=run_metadata,
            options=config_pb2.RunOptions(
                trace_level=config_pb2.RunOptions.FULL_TRACE))
        hasXlaRunOp = MetadataHasXlaRunOp(run_metadata)


if __name__ == "__main__":
  os.environ["TF_XLA_FLAGS"] = ("--tf_xla_async_compilation=true " +
                                "--tf_xla_enable_lazy_compilation=true " +
                                os.environ.get("TF_XLA_FLAGS", ""))
  # This test is using Tensorflow sessions which are not compatible with eager
  # mode.
  ops.disable_eager_execution()
  test.main()
