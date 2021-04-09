# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Notebook front-end to TensorFlow.

When you run this binary, you'll see something like below, which indicates
the serving URL of the notebook:

   The IPython Notebook is running at: http://127.0.0.1:8888/

Press "Shift+Enter" to execute a cell
Press "Enter" on a cell to go into edit mode.
Press "Escape" to go back into command mode and use arrow keys to navigate.
Press "a" in command mode to insert cell above or "b" to insert cell below.

Your root notebooks directory is FLAGS.notebook_dir
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import socket
import sys

from absl import app

# pylint: disable=g-import-not-at-top
# Official recommended way of turning on fast protocol buffers as of 10/21/14
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

FLAGS = None

ORIG_ARGV = sys.argv
# Main notebook process calls itself with argv[1]="kernel" to start kernel
# subprocesses.
IS_KERNEL = len(sys.argv) > 1 and sys.argv[1] == "kernel"


def main(unused_argv):
  sys.argv = ORIG_ARGV

  if not IS_KERNEL:
    # Drop all flags.
    sys.argv = [sys.argv[0]]
    # NOTE(sadovsky): For some reason, putting this import at the top level
    # breaks inline plotting.  It's probably a bug in the stone-age version of
    # matplotlib.
    from IPython.html.notebookapp import NotebookApp  # pylint: disable=g-import-not-at-top
    notebookapp = NotebookApp.instance()
    notebookapp.open_browser = True

    # password functionality adopted from quality/ranklab/main/tools/notebook.py
    # add options to run with "password"
    if FLAGS.password:
      from IPython.lib import passwd  # pylint: disable=g-import-not-at-top
      notebookapp.ip = "0.0.0.0"
      notebookapp.password = passwd(FLAGS.password)
    else:
      print("\nNo password specified; Notebook server will only be available"
            " on the local machine.\n")
    notebookapp.initialize(argv=["--notebook-dir", FLAGS.notebook_dir])

    if notebookapp.ip == "0.0.0.0":
      proto = "https" if notebookapp.certfile else "http"
      url = "%s://%s:%d%s" % (proto, socket.gethostname(), notebookapp.port,
                              notebookapp.base_project_url)
      print("\nNotebook server will be publicly available at: %s\n" % url)

    notebookapp.start()
    return

  # Drop the --flagfile flag so that notebook doesn't complain about an
  # "unrecognized alias" when parsing sys.argv.
  sys.argv = ([sys.argv[0]] +
              [z for z in sys.argv[1:] if not z.startswith("--flagfile")])
  from IPython.kernel.zmq.kernelapp import IPKernelApp  # pylint: disable=g-import-not-at-top
  kernelapp = IPKernelApp.instance()
  kernelapp.initialize()

  # Enable inline plotting. Equivalent to running "%matplotlib inline".
  ipshell = kernelapp.shell
  ipshell.enable_matplotlib("inline")

  kernelapp.start()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--password",
      type=str,
      default=None,
      help="""\
      Password to require. If set, the server will allow public access. Only
      used if notebook config file does not exist.\
      """)
  parser.add_argument(
      "--notebook_dir",
      type=str,
      default="experimental/brain/notebooks",
      help="root location where to store notebooks")

  # When the user starts the main notebook process, we don't touch sys.argv.
  # When the main process launches kernel subprocesses, it writes all flags
  # to a tmpfile and sets --flagfile to that tmpfile, so for kernel
  # subprocesses here we drop all flags *except* --flagfile, then call
  # app.run(), and then (in main) restore all flags before starting the
  # kernel app.
  if IS_KERNEL:
    # Drop everything except --flagfile.
    sys.argv = (
        [sys.argv[0]] + [x for x in sys.argv[1:] if x.startswith("--flagfile")])

  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
