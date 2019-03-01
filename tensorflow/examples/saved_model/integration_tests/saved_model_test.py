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

from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging


class SavedModelTest(tf.test.TestCase):

  def assertCommandSucceeded(self, binary, **flags):
    command_parts = [binary]
    for flag_key, flag_value in flags.items():
      command_parts.append("--%s=%s" % (flag_key, flag_value))

    logging.info("Running: %s" % command_parts)
    subprocess.check_call(
        command_parts, env=dict(os.environ, TF2_BEHAVIOR="enabled"))

  @test_util.run_v2_only
  def test_text_rnn(self):
    export_dir = self.get_temp_dir()
    export_binary = resource_loader.get_path_to_datafile(
        "export_text_rnn_model")
    self.assertCommandSucceeded(export_binary, export_dir=export_dir)

    use_binary = resource_loader.get_path_to_datafile("use_text_rnn_model")
    self.assertCommandSucceeded(use_binary, model_dir=export_dir)

  @test_util.run_v2_only
  def test_rnn_cell(self):
    export_dir = self.get_temp_dir()
    export_binary = resource_loader.get_path_to_datafile(
        "export_rnn_cell")
    self.assertCommandSucceeded(export_binary, export_dir=export_dir)

    use_binary = resource_loader.get_path_to_datafile("use_rnn_cell")
    self.assertCommandSucceeded(use_binary, model_dir=export_dir)

  @test_util.run_v2_only
  def test_text_embedding_in_sequential_keras(self):
    export_dir = self.get_temp_dir()
    export_binary = resource_loader.get_path_to_datafile(
        "export_simple_text_embedding")
    self.assertCommandSucceeded(export_binary, export_dir=export_dir)

    use_binary = resource_loader.get_path_to_datafile(
        "use_model_in_sequential_keras")
    self.assertCommandSucceeded(use_binary, model_dir=export_dir)

  @test_util.run_v2_only
  def test_mnist_cnn(self):
    export_dir = self.get_temp_dir()
    export_binary = resource_loader.get_path_to_datafile("export_mnist_cnn")
    self.assertCommandSucceeded(export_binary, export_dir=export_dir,
                                fast_test_mode="true")

    use_binary = resource_loader.get_path_to_datafile("use_mnist_cnn")
    self.assertCommandSucceeded(use_binary, export_dir=export_dir,
                                fast_test_mode="true")

if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
