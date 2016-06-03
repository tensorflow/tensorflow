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

"""Test runner for TensorFlow tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shlex
import sys

import tensorflow as tf

# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
# pylint: disable=unused-import
# Note: cpuinfo and psutil are not installed for you in the TensorFlow
# OSS tree.  They are installable via pip.
try:
  import cpuinfo
  import psutil
except ImportError as e:
  tf.logging.error("\n\n\nERROR: Unable to import necessary library: {}.  "
                   "Issuing a soft exit.\n\n\n".format(e))
  sys.exit(0)
# pylint: enable=g-bad-import-order
# pylint: enable=unused-import

from google.protobuf import text_format
from tensorflow.core.util import test_log_pb2
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
tf.app.flags.DEFINE_string(
    "compilation_mode", "", """Mode used during this build (e.g. opt, dbg).""")
tf.app.flags.DEFINE_string(
    "cc_flags", "", """CC flags used during this build.""")


def gather_build_configuration():
  build_config = test_log_pb2.BuildConfiguration()
  build_config.mode = FLAGS.compilation_mode
  # Include all flags except includes
  cc_flags = [
      flag for flag in shlex.split(FLAGS.cc_flags)
      if not flag.startswith("-i")]
  build_config.cc_flags.extend(cc_flags)
  return build_config


def main(unused_args):
  test_name = FLAGS.test_name
  test_args = FLAGS.test_args
  test_results, _ = run_and_gather_logs_lib.run_and_gather_logs(
      test_name, test_args)

  # Additional bits we receive from bazel
  test_results.build_configuration.CopyFrom(gather_build_configuration())

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
  tf.logging.info("Test results written to: %s" % output_path)


if __name__ == "__main__":
  tf.app.run()
