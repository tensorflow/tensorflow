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
"""SavedModel integration tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

import tensorflow.compat.v2 as tf

from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging


class SavedModelTest(tf.test.TestCase):

  def assertCommandSucceeded(self, script_name, **flags):
    """Runs a test script via run_script."""
    run_script = resource_loader.get_path_to_datafile("run_script")
    command_parts = [run_script]
    for flag_key, flag_value in flags.items():
      command_parts.append("--%s=%s" % (flag_key, flag_value))
    env = dict(TF2_BEHAVIOR="enabled", SCRIPT_NAME=script_name)
    logging.info("Running: %s with environment flags %s" % (command_parts, env))
    subprocess.check_call(command_parts, env=dict(os.environ, **env))

  def test_text_rnn(self):
    export_dir = self.get_temp_dir()
    self.assertCommandSucceeded("export_text_rnn_model", export_dir=export_dir)
    self.assertCommandSucceeded("use_text_rnn_model", model_dir=export_dir)

  def test_rnn_cell(self):
    export_dir = self.get_temp_dir()
    self.assertCommandSucceeded("export_rnn_cell", export_dir=export_dir)
    self.assertCommandSucceeded("use_rnn_cell", model_dir=export_dir)

  def test_text_embedding_in_sequential_keras(self):
    export_dir = self.get_temp_dir()
    self.assertCommandSucceeded(
        "export_simple_text_embedding", export_dir=export_dir)
    self.assertCommandSucceeded(
        "use_model_in_sequential_keras", model_dir=export_dir)

  def test_text_embedding_in_dataset(self):
    export_dir = self.get_temp_dir()
    self.assertCommandSucceeded(
        "export_simple_text_embedding", export_dir=export_dir)
    self.assertCommandSucceeded(
        "use_text_embedding_in_dataset", model_dir=export_dir)

  def test_mnist_cnn(self):
    export_dir = self.get_temp_dir()
    self.assertCommandSucceeded(
        "export_mnist_cnn", export_dir=export_dir, fast_test_mode="true")
    self.assertCommandSucceeded(
        "use_mnist_cnn", export_dir=export_dir, fast_test_mode="true")

  def test_mnist_cnn_with_mirrored_strategy(self):
    self.skipTest(
        "b/129134185 - saved model and distribution strategy integration")
    export_dir = self.get_temp_dir()
    self.assertCommandSucceeded(
        "export_mnist_cnn",
        export_dir=export_dir,
        fast_test_mode="true")
    self.assertCommandSucceeded(
        "use_mnist_cnn",
        export_dir=export_dir,
        fast_test_mode="true",
        use_mirrored_strategy=True,
    )

if __name__ == "__main__":
  tf.test.main()
