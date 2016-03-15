# Copyright 2016 Google Inc. All Rights Reserved.
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

"""Library for getting system information during TensorFlow tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.tools.test import run_and_gather_logs_lib

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("test_name", "", """Test target to run.""")
tf.app.flags.DEFINE_string(
    "test_args", "", """Test arguments, space separated.""")
tf.app.flags.DEFINE_string(
    "test_log_output", "", """Filename to write logs.""")
tf.app.flags.DEFINE_bool(
    "test_log_output_use_tmpdir", False,
    """Store the log output into tmpdir?.""")


def main(unused_args):
  test_name = FLAGS.test_name
  test_args = FLAGS.test_args
  test_results, _ = run_and_gather_logs_lib.run_and_gather_logs(
      test_name, test_args)
  serialized_test_results = text_format.MessageToString(test_results)

  if not FLAGS.test_log_output:
    print(serialized_test_results)
    return

  if FLAGS.test_log_output_use_tmpdir:
    tmpdir = tf.test.get_temp_dir()
    output_path = os.path.join(tmpdir, FLAGS.test_log_output)
  else:
    output_path = os.path.abspath(FLAGS.test_log_output)
  tf.gfile.GFile(output_path, "w").write(serialized_test_results)
  print("Test results written to: %s" % output_path)


if __name__ == "__main__":
  tf.app.run()
