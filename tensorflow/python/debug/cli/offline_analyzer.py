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
"""Offline dump analyzer of TensorFlow Debugger (tfdbg)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# Google-internal import(s).
from tensorflow.python.debug import debug_data
from tensorflow.python.debug.cli import analyzer_cli
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("dump_dir", "", "tfdbg dump directory to load")
flags.DEFINE_boolean(
    "log_usage", True, "Whether the usage of this tool is to be logged")
flags.DEFINE_boolean(
    "validate_graph", True,
    "Whether the dumped tensors will be validated against the GraphDefs")


def main(_):
  if FLAGS.log_usage:
    pass  # No logging for open-source.

  if not FLAGS.dump_dir:
    print("ERROR: dump_dir flag is empty.", file=sys.stderr)
    sys.exit(1)

  print("tfdbg offline: FLAGS.dump_dir = %s" % FLAGS.dump_dir)

  debug_dump = debug_data.DebugDumpDir(
      FLAGS.dump_dir, validate=FLAGS.validate_graph)
  cli = analyzer_cli.create_analyzer_curses_cli(
      debug_dump,
      tensor_filters={"has_inf_or_nan": debug_data.has_inf_or_nan})

  title = "tfdbg offline @ %s" % FLAGS.dump_dir
  cli.run_ui(title=title, title_color="black_on_white", init_command="lt")


if __name__ == "__main__":
  app.run()
