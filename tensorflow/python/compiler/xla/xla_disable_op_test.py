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
"""Test for checking if tf.where op is excluded from XLA auto-clustering."""

import functools
import numpy as np
import os
import subprocess

from tensorflow.compat.v1 import config
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


cmds_linux = {
    "grep_where": (
        "grep 'Where' /tmp/xla_logs/* && rm -rf /tmp/xla_logs/"),
}

def run_shell_cmd(args):
  """Executes shell commands and returns output.

  Args:
    args: String of shell commands to run.

  Returns:
    Tuple output (stdoutdata, stderrdata) from running the shell commands.
  """
  proc = subprocess.Popen(
      args,
      shell=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT
  )
  return proc.communicate()

class XlaDisableOpTest(test.TestCase):

  @def_function.function()
  def _runEvaluation(self, x, y, predictions):
    dummy_loss = 0.9
    predictions = array_ops.reshape(predictions, [-1])
    display_ids = x
    display_ids = array_ops.reshape(display_ids, [-1])
    labels = array_ops.reshape(y, [-1])
    sorted_ids = sort_ops.argsort(display_ids)
    display_ids = array_ops.gather(display_ids, indices=sorted_ids)
    predictions = array_ops.gather(predictions, indices=sorted_ids)
    labels = array_ops.gather(labels, indices=sorted_ids)
    _, display_ids_idx, display_ids_ads_count = array_ops.unique_with_counts(
        display_ids, out_idx=dtypes.int64)
    pad_length = 30 - math_ops.reduce_max(display_ids_ads_count)
    preds = ragged_tensor.RaggedTensor.from_value_rowids(
        predictions, display_ids_idx).to_tensor()
    labels = ragged_tensor.RaggedTensor.from_value_rowids(
        labels, display_ids_idx).to_tensor()
    labels_mask = math_ops.reduce_max(labels, 1)
    preds_masked = array_ops.boolean_mask(preds, labels_mask)
    labels_masked = array_ops.boolean_mask(labels, labels_mask)
    labels_masked = math_ops.argmax(labels_masked, axis=1, output_type=dtypes.int32)
    labels_masked = array_ops.reshape(labels_masked, [-1, 1])

    preds_masked = array_ops.pad(preds_masked, [(0, 0), (0, pad_length)])
    _, predictions_idx = nn_ops.top_k(preds_masked, 12)
    indices = math_ops.equal(predictions_idx, labels_masked)
    return math_ops.cast(array_ops.shape(indices)[0], dtypes.float64)

  def testRunEval(self):
    dim_prediction = 1024
    config.optimizer.set_jit(True)
    pre = np.random.random((dim_prediction, 1))
    y_tmp = np.zeros((dim_prediction, 1), dtype=float)

    num_ones = np.random.randint(1, dim_prediction+1, 1)
    id_one = np.random.randint(0, dim_prediction, num_ones)
    for i in id_one:
      y_tmp[i][0] = 1.
    x_tmp = np.random.randint(0, dim_prediction,
                              (dim_prediction, 1), dtype=np.int64)
    display_id_counter = self._runEvaluation(x_tmp, y_tmp, pre)

    out,err = run_shell_cmd(cmds_linux['grep_where'])
    self.assertEqual(err, None)
    self.assertEqual(len(out), 0)


if __name__ == '__main__':
  os.environ['XLA_FLAGS'] = "--xla_dump_to='/tmp/xla_logs'"
  os.environ['TF_XLA_FLAGS'] = "--tf_xla_cluster_exclude_ops=Where"
  test.main()
