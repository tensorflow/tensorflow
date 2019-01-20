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
"""Tests for trainer hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from tensorflow.contrib.boosted_trees.estimator_batch import trainer_hooks
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import monitored_session


class FeatureImportanceSummarySaverTest(test_util.TensorFlowTestCase):

  def test_invalid_input(self):
    with self.assertRaises(ValueError):
      trainer_hooks.FeatureImportanceSummarySaver(model_dir=None)

  def test_invalid_graph(self):
    # Create inputs.
    model_dir = tempfile.mkdtemp()
    hook = trainer_hooks.FeatureImportanceSummarySaver(model_dir)
    with ops.Graph().as_default():
      # Begin won't be able to find the required tensors in the graph.
      _ = variables.get_or_create_global_step()
      with self.assertRaises(KeyError):
        hook.begin()

  def test_run(self):
    # Create inputs.
    model_dir = tempfile.mkdtemp()
    hook = trainer_hooks.FeatureImportanceSummarySaver(model_dir)
    with ops.Graph().as_default(), tf_session.Session() as sess:
      global_step = variables.get_or_create_global_step()
      with ops.name_scope("gbdt"):
        constant_op.constant(["featA", "featB"], name="feature_names")
        constant_op.constant([0, 2], name="feature_usage_counts")
        constant_op.constant([0, 0.8], name="feature_gains")
      # Begin finds tensors in the graph.
      hook.begin()
      sess.run(tf_variables.global_variables_initializer())
      # Run hook in a monitored session.
      train_op = state_ops.assign_add(global_step, 1)
      mon_sess = monitored_session._HookedSession(sess, [hook])
      mon_sess.run(train_op)
      hook.end(sess)
      # Ensure output summary dirs are created.
      self.assertTrue(os.path.exists(os.path.join(model_dir, "featA")))
      self.assertTrue(os.path.exists(os.path.join(model_dir, "featB")))


if __name__ == "__main__":
  googletest.main()
