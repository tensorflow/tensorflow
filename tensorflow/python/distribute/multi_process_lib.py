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
"""Library for multi-process testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import multiprocessing
import os
import platform
import sys
import unittest
from absl import app
from absl import logging

from tensorflow.python.eager import test


def is_oss():
  """Returns whether the test is run under OSS."""
  return len(sys.argv) >= 1 and 'bazel' in sys.argv[0]


def _is_enabled():
  # Note that flags may not be parsed at this point and simply importing the
  # flags module causes a variety of unusual errors.
  tpu_args = [arg for arg in sys.argv if arg.startswith('--tpu')]
  if is_oss() and tpu_args:
    return False
  if sys.version_info == (3, 8) and platform.system() == 'Linux':
    return False  # TODO(b/171242147)
  return sys.platform != 'win32'


class _AbslProcess:
  """A process that runs using absl.app.run."""

  def __init__(self, *args, **kwargs):
    super(_AbslProcess, self).__init__(*args, **kwargs)
    # Monkey-patch that is carried over into the spawned process by pickle.
    self._run_impl = getattr(self, 'run')
    self.run = self._run_with_absl

  def _run_with_absl(self):
    app.run(lambda _: self._run_impl())


if _is_enabled():

  class AbslForkServerProcess(_AbslProcess,
                              multiprocessing.context.ForkServerProcess):
    """An absl-compatible Forkserver process.

    Note: Forkserver is not available in windows.
    """

  class AbslForkServerContext(multiprocessing.context.ForkServerContext):
    _name = 'absl_forkserver'
    Process = AbslForkServerProcess  # pylint: disable=invalid-name

  multiprocessing = AbslForkServerContext()
  Process = multiprocessing.Process

else:

  class Process(object):
    """A process that skips test (until windows is supported)."""

    def __init__(self, *args, **kwargs):
      del args, kwargs
      raise unittest.SkipTest(
          'TODO(b/150264776): Windows is not supported in MultiProcessRunner.')


_test_main_called = False


def _set_spawn_exe_path():
  """Set the path to the executable for spawned processes.

  This utility searches for the binary the parent process is using, and sets
  the executable of multiprocessing's context accordingly.

  Raises:
    RuntimeError: If the binary path cannot be determined.
  """
  # TODO(b/150264776): This does not work with Windows. Find a solution.
  if sys.argv[0].endswith('.py'):
    def guess_path(package_root):
      # If all we have is a python module path, we'll need to make a guess for
      # the actual executable path.
      if 'bazel-out' in sys.argv[0] and package_root in sys.argv[0]:
        # Guess the binary path under bazel. For target
        # //tensorflow/python/distribute:input_lib_test_multiworker_gpu, the
        # argv[0] is in the form of
        # /.../tensorflow/python/distribute/input_lib_test.py
        # and the binary is
        # /.../tensorflow/python/distribute/input_lib_test_multiworker_gpu
        package_root_base = sys.argv[0][:sys.argv[0].rfind(package_root)]
        binary = os.environ['TEST_TARGET'][2:].replace(':', '/', 1)
        possible_path = os.path.join(package_root_base, package_root,
                                     binary)
        logging.info('Guessed test binary path: %s', possible_path)
        if os.access(possible_path, os.X_OK):
          return possible_path
        return None
    path = guess_path('org_tensorflow')
    if not path:
      path = guess_path('org_keras')
    if path is None:
      logging.error(
          'Cannot determine binary path. sys.argv[0]=%s os.environ=%s',
          sys.argv[0], os.environ)
      raise RuntimeError('Cannot determine binary path')
    sys.argv[0] = path
  # Note that this sets the executable for *all* contexts.
  multiprocessing.get_context().set_executable(sys.argv[0])


def _if_spawn_run_and_exit():
  """If spawned process, run requested spawn task and exit. Else a no-op."""

  # `multiprocessing` module passes a script "from multiprocessing.x import y"
  # to subprocess, followed by a main function call. We use this to tell if
  # the process is spawned. Examples of x are "forkserver" or
  # "semaphore_tracker".
  is_spawned = ('-c' in sys.argv[1:] and
                sys.argv[sys.argv.index('-c') +
                         1].startswith('from multiprocessing.'))

  if not is_spawned:
    return
  cmd = sys.argv[sys.argv.index('-c') + 1]
  # As a subprocess, we disregarding all other interpreter command line
  # arguments.
  sys.argv = sys.argv[0:1]

  # Run the specified command - this is expected to be one of:
  # 1. Spawn the process for semaphore tracker.
  # 2. Spawn the initial process for forkserver.
  # 3. Spawn any process as requested by the "spawn" method.
  exec(cmd)  # pylint: disable=exec-used
  sys.exit(0)  # Semaphore tracker doesn't explicitly sys.exit.


def test_main():
  """Main function to be called within `__main__` of a test file."""
  global _test_main_called
  _test_main_called = True

  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

  if _is_enabled():
    _set_spawn_exe_path()
    _if_spawn_run_and_exit()

  # Only runs test.main() if not spawned process.
  test.main()


def initialized():
  """Returns whether the module is initialized."""
  return _test_main_called
