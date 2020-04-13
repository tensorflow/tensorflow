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
"""Utility to write SavedModel integration tests.

SavedModel testing requires isolation between the process that creates and
consumes it. This file helps doing that by relaunching the same binary that
calls `assertCommandSucceeded` with an environment flag indicating what source
file to execute. That binary must start by calling `MaybeRunScriptInstead`.

This allows to wire this into existing building systems without having to depend
on data dependencies. And as so allow to keep a fixed binary size and allows
interop with GPU tests.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import subprocess
import sys

from absl import app
from absl import flags as absl_flags
import tensorflow.compat.v2 as tf

from tensorflow.python.platform import tf_logging as logging


class TestCase(tf.test.TestCase):
  """Base class to write SavedModel integration tests."""

  def assertCommandSucceeded(self, script_name, **flags):
    """Runs an integration test script with given flags."""
    run_script = sys.argv[0]
    if run_script.endswith(".py"):
      command_parts = [sys.executable, run_script]
    else:
      command_parts = [run_script]
    command_parts.append("--alsologtostderr")  # For visibility in sponge.
    for flag_key, flag_value in flags.items():
      command_parts.append("--%s=%s" % (flag_key, flag_value))

    # TODO(b/143247229): Remove forwarding this flag once the BUILD rule
    # `distribute_py_test()` stops setting it.
    deepsea_flag_name = "register_deepsea_platform"
    deepsea_flag_value = getattr(absl_flags.FLAGS, deepsea_flag_name, None)
    if deepsea_flag_value is not None:
      command_parts.append("--%s=%s" % (deepsea_flag_name,
                                        str(deepsea_flag_value).lower()))

    env = dict(TF2_BEHAVIOR="enabled", SCRIPT_NAME=script_name)
    logging.info("Running %s with added environment variables %s" %
                 (command_parts, env))
    subprocess.check_call(command_parts, env=dict(os.environ, **env))


def MaybeRunScriptInstead():
  if "SCRIPT_NAME" in os.environ:
    # Append current path to import path and execute `SCRIPT_NAME` main.
    sys.path.extend([os.path.dirname(__file__)])
    module_name = os.environ["SCRIPT_NAME"]
    retval = app.run(importlib.import_module(module_name).main)  # pylint: disable=assignment-from-no-return
    sys.exit(retval)
