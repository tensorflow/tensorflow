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

"""Imports absltest as a replacement for testing.pybase.googletest."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import itertools
import os
import sys
import tempfile

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from absl.testing.absltest import *
# pylint: enable=wildcard-import

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import app
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


Benchmark = benchmark.TensorFlowBenchmark  # pylint: disable=invalid-name

absltest_main = main

# We keep a global variable in this module to make sure we create the temporary
# directory only once per test binary invocation.
_googletest_temp_dir = ''


# pylint: disable=invalid-name
# pylint: disable=undefined-variable
def g_main(argv):
  """Delegate to absltest.main after redefining testLoader."""
  if 'TEST_SHARD_STATUS_FILE' in os.environ:
    try:
      f = None
      try:
        f = open(os.environ['TEST_SHARD_STATUS_FILE'], 'w')
        f.write('')
      except IOError:
        sys.stderr.write('Error opening TEST_SHARD_STATUS_FILE (%s). Exiting.'
                         % os.environ['TEST_SHARD_STATUS_FILE'])
        sys.exit(1)
    finally:
      if f is not None: f.close()

  if ('TEST_TOTAL_SHARDS' not in os.environ or
      'TEST_SHARD_INDEX' not in os.environ):
    return absltest_main(argv=argv)

  total_shards = int(os.environ['TEST_TOTAL_SHARDS'])
  shard_index = int(os.environ['TEST_SHARD_INDEX'])
  base_loader = TestLoader()

  delegate_get_names = base_loader.getTestCaseNames
  bucket_iterator = itertools.cycle(range(total_shards))

  def getShardedTestCaseNames(testCaseClass):
    filtered_names = []
    for testcase in sorted(delegate_get_names(testCaseClass)):
      bucket = next(bucket_iterator)
      if bucket == shard_index:
        filtered_names.append(testcase)
    return filtered_names

  # Override getTestCaseNames
  base_loader.getTestCaseNames = getShardedTestCaseNames

  absltest_main(argv=argv, testLoader=base_loader)


# Redefine main to allow running benchmarks
def main(argv=None):  # pylint: disable=function-redefined
  def main_wrapper():
    args = argv
    if args is None:
      args = sys.argv
    return app.run(main=g_main, argv=args)
  benchmark.benchmarks_main(true_main=main_wrapper)


def GetTempDir():
  """Return a temporary directory for tests to use."""
  global _googletest_temp_dir
  if not _googletest_temp_dir:
    if os.environ.get('TEST_TMPDIR'):
      temp_dir = tempfile.mkdtemp(prefix=os.environ['TEST_TMPDIR'])
    else:
      first_frame = tf_inspect.stack()[-1][0]
      temp_dir = os.path.join(tempfile.gettempdir(),
                              os.path.basename(tf_inspect.getfile(first_frame)))
      temp_dir = tempfile.mkdtemp(prefix=temp_dir.rstrip('.py'))

    # Make sure we have the correct path separators.
    temp_dir = temp_dir.replace('/', os.sep)

    def delete_temp_dir(dirname=temp_dir):
      try:
        file_io.delete_recursively(dirname)
      except errors.OpError as e:
        logging.error('Error removing %s: %s', dirname, e)

    atexit.register(delete_temp_dir)

    _googletest_temp_dir = temp_dir

  return _googletest_temp_dir


def test_src_dir_path(relative_path):
  """Creates an absolute test srcdir path given a relative path.

  Args:
    relative_path: a path relative to tensorflow root.
      e.g. "contrib/session_bundle/example".

  Returns:
    An absolute path to the linked in runfiles.
  """
  return os.path.join(os.environ['TEST_SRCDIR'],
                      'org_tensorflow/tensorflow', relative_path)


def StatefulSessionAvailable():
  return False


@tf_export(v1=['test.StubOutForTesting'])
class StubOutForTesting(object):
  """Support class for stubbing methods out for unit testing.

  Sample Usage:

  You want os.path.exists() to always return true during testing.

     stubs = StubOutForTesting()
     stubs.Set(os.path, 'exists', lambda x: 1)
       ...
     stubs.CleanUp()

  The above changes os.path.exists into a lambda that returns 1.  Once
  the ... part of the code finishes, the CleanUp() looks up the old
  value of os.path.exists and restores it.
  """

  def __init__(self):
    self.cache = []
    self.stubs = []

  def __del__(self):
    """Do not rely on the destructor to undo your stubs.

    You cannot guarantee exactly when the destructor will get called without
    relying on implementation details of a Python VM that may change.
    """
    self.CleanUp()

  # __enter__ and __exit__ allow use as a context manager.
  def __enter__(self):
    return self

  def __exit__(self, unused_exc_type, unused_exc_value, unused_tb):
    self.CleanUp()

  def CleanUp(self):
    """Undoes all SmartSet() & Set() calls, restoring original definitions."""
    self.SmartUnsetAll()
    self.UnsetAll()

  def SmartSet(self, obj, attr_name, new_attr):
    """Replace obj.attr_name with new_attr.

    This method is smart and works at the module, class, and instance level
    while preserving proper inheritance. It will not stub out C types however
    unless that has been explicitly allowed by the type.

    This method supports the case where attr_name is a staticmethod or a
    classmethod of obj.

    Notes:
      - If obj is an instance, then it is its class that will actually be
        stubbed. Note that the method Set() does not do that: if obj is
        an instance, it (and not its class) will be stubbed.
      - The stubbing is using the builtin getattr and setattr. So, the __get__
        and __set__ will be called when stubbing (TODO: A better idea would
        probably be to manipulate obj.__dict__ instead of getattr() and
        setattr()).

    Args:
      obj: The object whose attributes we want to modify.
      attr_name: The name of the attribute to modify.
      new_attr: The new value for the attribute.

    Raises:
      AttributeError: If the attribute cannot be found.
    """
    _, obj = tf_decorator.unwrap(obj)
    if (tf_inspect.ismodule(obj) or
        (not tf_inspect.isclass(obj) and attr_name in obj.__dict__)):
      orig_obj = obj
      orig_attr = getattr(obj, attr_name)
    else:
      if not tf_inspect.isclass(obj):
        mro = list(tf_inspect.getmro(obj.__class__))
      else:
        mro = list(tf_inspect.getmro(obj))

      mro.reverse()

      orig_attr = None
      found_attr = False

      for cls in mro:
        try:
          orig_obj = cls
          orig_attr = getattr(obj, attr_name)
          found_attr = True
        except AttributeError:
          continue

      if not found_attr:
        raise AttributeError('Attribute not found.')

    # Calling getattr() on a staticmethod transforms it to a 'normal' function.
    # We need to ensure that we put it back as a staticmethod.
    old_attribute = obj.__dict__.get(attr_name)
    if old_attribute is not None and isinstance(old_attribute, staticmethod):
      orig_attr = staticmethod(orig_attr)

    self.stubs.append((orig_obj, attr_name, orig_attr))
    setattr(orig_obj, attr_name, new_attr)

  def SmartUnsetAll(self):
    """Reverses SmartSet() calls, restoring things to original definitions.

    This method is automatically called when the StubOutForTesting()
    object is deleted; there is no need to call it explicitly.

    It is okay to call SmartUnsetAll() repeatedly, as later calls have
    no effect if no SmartSet() calls have been made.
    """
    for args in reversed(self.stubs):
      setattr(*args)

    self.stubs = []

  def Set(self, parent, child_name, new_child):
    """In parent, replace child_name's old definition with new_child.

    The parent could be a module when the child is a function at
    module scope.  Or the parent could be a class when a class' method
    is being replaced.  The named child is set to new_child, while the
    prior definition is saved away for later, when UnsetAll() is
    called.

    This method supports the case where child_name is a staticmethod or a
    classmethod of parent.

    Args:
      parent: The context in which the attribute child_name is to be changed.
      child_name: The name of the attribute to change.
      new_child: The new value of the attribute.
    """
    old_child = getattr(parent, child_name)

    old_attribute = parent.__dict__.get(child_name)
    if old_attribute is not None and isinstance(old_attribute, staticmethod):
      old_child = staticmethod(old_child)

    self.cache.append((parent, old_child, child_name))
    setattr(parent, child_name, new_child)

  def UnsetAll(self):
    """Reverses Set() calls, restoring things to their original definitions.

    This method is automatically called when the StubOutForTesting()
    object is deleted; there is no need to call it explicitly.

    It is okay to call UnsetAll() repeatedly, as later calls have no
    effect if no Set() calls have been made.
    """
    # Undo calls to Set() in reverse order, in case Set() was called on the
    # same arguments repeatedly (want the original call to be last one undone)
    for (parent, old_child, child_name) in reversed(self.cache):
      setattr(parent, child_name, old_child)
    self.cache = []
