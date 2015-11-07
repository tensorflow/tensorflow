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


import os
import socket
import sys

# pylint: disable=g-import-not-at-top
# Official recommended way of turning on fast protocol buffers as of 10/21/14
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "password", None,
    "Password to require. If set, the server will allow public access."
    " Only used if notebook config file does not exist.")

flags.DEFINE_string("notebook_dir", "experimental/brain/notebooks",
                    "root location where to store notebooks")

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
      print ("\nNo password specified; Notebook server will only be available"
             " on the local machine.\n")
    notebookapp.initialize(argv=["--notebook-dir", FLAGS.notebook_dir])

    if notebookapp.ip == "0.0.0.0":
      proto = "https" if notebookapp.certfile else "http"
      url = "%s://%s:%d%s" % (proto, socket.gethostname(), notebookapp.port,
                              notebookapp.base_project_url)
      print "\nNotebook server will be publicly available at: %s\n" % url

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
  # When the user starts the main notebook process, we don't touch sys.argv.
  # When the main process launches kernel subprocesses, it writes all flags
  # to a tmpfile and sets --flagfile to that tmpfile, so for kernel
  # subprocesses here we drop all flags *except* --flagfile, then call
  # app.run(), and then (in main) restore all flags before starting the
  # kernel app.
  if IS_KERNEL:
    # Drop everything except --flagfile.
    sys.argv = ([sys.argv[0]] +
                [x for x in sys.argv[1:] if x.startswith("--flagfile")])
  app.run()
