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

"""Check that TensorFlow python files have certain __future__ imports.

This makes it easier to find Python 2.7 / Python 3.x incompatibility bugs.
In particular, this test makes it illegal to write a Python file that
doesn't import division from __future__, which can catch subtle division
bugs in Python 3.

Note: We can't use tf.test in this file because it needs to run in an
environment that doesn't include license-free gen_blah_ops.py files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import os
import re

import six

BASE_DIR = os.path.normpath(os.path.join(__file__, '../../..'))
FUTURES_PATTERN = re.compile(r'^from __future__ import (\w+)\s*$')
FUTURES_PATTERN_2 = re.compile(
    r'^from __future__ import (\w+), (\w+), (\w+)\s*$')
REQUIRED_FUTURES = frozenset(['absolute_import', 'division', 'print_function'])

WHITELIST = [
    'python/platform/control_imports.py',
    'tools/docker/jupyter_notebook_config.py',
    'tools/ci_build/update_version.py',
]

# Tests that must *not* import division
OLD_DIVISION = [
    'python/framework/tensor_shape_div_test.py',
    'python/kernel_tests/division_past_test.py',
]


def check_file(path, old_division):
  futures = set()
  count = 0
  for line in open(path, encoding='utf-8') if six.PY3 else open(path):
    count += 1
    m = FUTURES_PATTERN.match(line)
    if m:
      futures.add(m.group(1))
    else:
      m = FUTURES_PATTERN_2.match(line)
      if m:
        for entry in m.groups():
          futures.add(entry)
  if not count:
    return  # Skip empty files
  if old_division:
    # This file checks correct behavior without importing division
    # from __future__, so make sure it's doing that.
    expected = set(['absolute_import', 'print_function'])
    if futures != expected:
      raise AssertionError(('Incorrect futures for old_division file:\n'
                            '  expected = %s\n  got = %s') %
                           (' '.join(expected), ' '.join(futures)))
  else:
    missing = REQUIRED_FUTURES - futures
    if missing:
      raise AssertionError('Missing futures: %s' % ' '.join(missing))


def main():
  # Make sure BASE_DIR ends with tensorflow.  If it doesn't, we probably
  # computed the wrong directory.
  if os.path.split(BASE_DIR)[-1] != 'tensorflow':
    raise AssertionError("BASE_DIR = '%s' doesn't end with tensorflow" %
                         BASE_DIR)

  # Verify that all files have futures
  whitelist = frozenset(os.path.join(BASE_DIR, w) for w in WHITELIST)
  old_division = frozenset(os.path.join(BASE_DIR, w) for w in OLD_DIVISION)
  for root, _, filenames in os.walk(BASE_DIR):
    for f in fnmatch.filter(filenames, '*.py'):
      path = os.path.join(root, f)
      if path not in whitelist:
        try:
          check_file(path, old_division=path in old_division)
        except AssertionError as e:
          short_path = path[len(BASE_DIR) + 1:]
          raise AssertionError('Error in %s: %s' % (short_path, str(e)))


if __name__ == '__main__':
  main()
