# Lint as: python3
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
"""Run doctests for tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import pkgutil
import sys

from absl import flags
from absl.testing import absltest
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.tools.docs import tf_doctest_lib

# We put doctest after absltest so that it picks up the unittest monkeypatch.
# Otherwise doctest tests aren't runnable at all.
import doctest  # pylint: disable=g-bad-import-order

tf.compat.v1.enable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_list('module', [], 'A list of specific module to run doctest on.')
flags.DEFINE_list('module_prefix_skip', [],
                  'A list of modules to ignore when resolving modules.')
flags.DEFINE_boolean('list', None,
                     'List all the modules in the core package imported.')
flags.DEFINE_string('file', None, 'A specific file to run doctest on.')
flags.DEFINE_integer('required_gpus', 0,
                     'The number of GPUs required for the tests.')

flags.mark_flags_as_mutual_exclusive(['module', 'file'])
flags.mark_flags_as_mutual_exclusive(['list', 'file'])

# Both --module and --module_prefix_skip are relative to PACKAGE.
PACKAGE = 'tensorflow.python.'


def recursive_import(root):
  """Recursively imports all the sub-modules under a root package.

  Args:
    root: A python package.
  """
  for _, name, _ in pkgutil.walk_packages(
      root.__path__, prefix=root.__name__ + '.'):
    try:
      importlib.import_module(name)
    except (AttributeError, ImportError):
      pass


def find_modules():
  """Finds all the modules in the core package imported.

  Returns:
    A list containing all the modules in tensorflow.python.
  """

  tf_modules = []
  for name, module in sys.modules.items():
    if name.startswith(PACKAGE):
      tf_modules.append(module)

  return tf_modules


def filter_on_submodules(all_modules, submodules):
  """Filters all the modules based on the modules flag.

  The module flag has to be relative to the core package imported.
  For example, if `module=keras.layers` then, this function will return
  all the modules in the submodule.

  Args:
    all_modules: All the modules in the core package.
    submodules: Submodules to filter from all the modules.

  Returns:
    All the modules in the submodule.
  """

  filtered_modules = [
      mod for mod in all_modules
      if any(PACKAGE + submodule in mod.__name__ for submodule in submodules)
  ]
  return filtered_modules


def get_module_and_inject_docstring(file_path):
  """Replaces the docstring of the module with the changed file's content.

  Args:
    file_path: Path to the file

  Returns:
    A list containing the module changed by the file.
  """

  file_path = os.path.abspath(file_path)
  mod_index = file_path.find(PACKAGE.replace('.', os.sep))
  file_mod_name, _ = os.path.splitext(file_path[mod_index:])
  file_module = sys.modules[file_mod_name.replace(os.sep, '.')]

  with open(file_path, 'r') as f:
    content = f.read()

  file_module.__doc__ = content

  return [file_module]


def setup_gpu(required_gpus):
  """Sets up the GPU devices.

  If there're more available GPUs than needed, it hides the additional ones. If
  there're less, it creates logical devices. This is to make sure the tests see
  a fixed number of GPUs regardless of the environment.

  Args:
    required_gpus: an integer. The number of GPUs required.

  Raises:
    ValueError: if num_gpus is larger than zero but no GPU is available.
  """
  if required_gpus == 0:
    return
  available_gpus = tf.config.experimental.list_physical_devices('GPU')
  if not available_gpus:
    raise ValueError('requires at least one physical GPU')
  if len(available_gpus) >= required_gpus:
    tf.config.set_visible_devices(available_gpus[:required_gpus])
  else:
    # Create logical GPUs out of one physical GPU for simplicity. Note that the
    # other physical GPUs are still available and corresponds to one logical GPU
    # each.
    num_logical_gpus = required_gpus - len(available_gpus) + 1
    logical_gpus = [
        tf.config.LogicalDeviceConfiguration(memory_limit=256)
        for _ in range(num_logical_gpus)
    ]
    tf.config.set_logical_device_configuration(available_gpus[0], logical_gpus)


class TfTestCase(tf.test.TestCase):

  def set_up(self, test):
    self.setUp()

  def tear_down(self, test):
    self.tearDown()


def load_tests(unused_loader, tests, unused_ignore):
  """Loads all the tests in the docstrings and runs them."""

  tf_modules = find_modules()

  if FLAGS.module:
    tf_modules = filter_on_submodules(tf_modules, FLAGS.module)

  if FLAGS.list:
    print('**************************************************')
    for mod in tf_modules:
      print(mod.__name__)
    print('**************************************************')
    return tests

  if FLAGS.file:
    tf_modules = get_module_and_inject_docstring(FLAGS.file)

  for module in tf_modules:
    if any(
        module.__name__.startswith(PACKAGE + prefix)
        for prefix in FLAGS.module_prefix_skip):
      continue
    testcase = TfTestCase()
    tests.addTests(
        doctest.DocTestSuite(
            module,
            test_finder=doctest.DocTestFinder(exclude_empty=False),
            extraglobs={
                'tf': tf,
                'np': np,
                'os': os
            },
            setUp=testcase.set_up,
            tearDown=testcase.tear_down,
            checker=tf_doctest_lib.TfDoctestOutputChecker(),
            optionflags=(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
                         | doctest.IGNORE_EXCEPTION_DETAIL
                         | doctest.DONT_ACCEPT_BLANKLINE),
        ))
  return tests


# We can only create logical devices before initializing Tensorflow. This is
# called by unittest framework before running any test.
# https://docs.python.org/3/library/unittest.html#setupmodule-and-teardownmodule
def setUpModule():
  setup_gpu(FLAGS.required_gpus)


if __name__ == '__main__':
  # Use importlib to import python submodule of tensorflow.
  # We delete python submodule in root __init__.py file. This means
  # normal import won't work for some Python versions.
  tf_python_root = importlib.import_module(PACKAGE[:-1])
  recursive_import(tf_python_root)
  absltest.main()
