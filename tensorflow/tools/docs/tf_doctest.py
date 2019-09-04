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

import os
import pkgutil
import numpy as np

from absl import flags
from absl.testing import absltest

import tensorflow.compat.v2 as tf
import tensorflow.python as core
tf.compat.v1.enable_v2_behavior()

# We put doctest after absltest so that it picks up the unittest monkeypatch.
# Otherwise doctest tests aren't runnable at all.
import doctest  # pylint: disable=g-import-not-at-top; import: after=absltest

FLAGS = flags.FLAGS

flags.DEFINE_string('module', '', 'A specific module to run doctest on.')
flags.DEFINE_boolean('list', False,
                     'List all the modules in the core package imported.')


def find_modules():
  """Finds all the modules in the core package imported."""

  tf_modules = []
  for _, name, _ in pkgutil.walk_packages(
      core.__path__, prefix=core.__name__ + '.'):
    try:
      tf_modules.append(__import__(name, fromlist=['']))
    except (ImportError, AttributeError):
      pass

  return tf_modules


def filter_on_submodules(all_modules, submodule):
  """Filters all the modules based on the module flag.

  The module flag has to be relative to the core package imported.
  For example, if `submodule=keras.layers` then, this function will return
  all the modules in the submodule.

  Args:
    all_modules: All the modules in the core package.
    submodule: Submodule to filter from all the modules.

  Returns:
    All the modules in the submodule.
  """

  filtered_modules = [
      mod for mod in all_modules
      if core.__name__ + '.' + submodule in mod.__name__
  ]
  return filtered_modules


def load_tests(loader, tests, ignore):
  """Loads all the tests in the docstrings and runs them."""

  tf_modules = find_modules()

  if FLAGS.module:
    tf_modules = filter_on_submodules(tf_modules, FLAGS.module)

  if FLAGS.list:
    for mod in tf_modules:
      print(mod.__name__)
    return tests

  for module in tf_modules:
    tests.addTests(
        doctest.DocTestSuite(
            module,
            test_finder=doctest.DocTestFinder(exclude_empty=False),
            extraglobs={
                'tf': tf,
                'np': np,
                'os': os
            },
            optionflags=(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)))
  return tests


if __name__ == '__main__':
  absltest.main()
