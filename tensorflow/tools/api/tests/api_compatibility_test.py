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
#
# ==============================================================================
"""TensorFlow API compatibility tests.

This test ensures all changes to the public API of TensorFlow are intended.

If this test fails, it means a change has been made to the public API. Backwards
incompatible changes are not allowed. You can run the test with
"--update_goldens" flag set to "True" to update goldens when making changes to
the public TF python API.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import unittest

import tensorflow as tf

from google.protobuf import text_format

from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.tools.api.lib import api_objects_pb2
from tensorflow.tools.api.lib import python_object_to_proto_visitor
from tensorflow.tools.common import public_api
from tensorflow.tools.common import traverse

# FLAGS defined at the bottom:
FLAGS = None
# DEFINE_boolean, update_goldens, default False:
_UPDATE_GOLDENS_HELP = """
     Update stored golden files if API is updated. WARNING: All API changes
     have to be authorized by TensorFlow leads.
"""

# DEFINE_boolean, verbose_diffs, default False:
_VERBOSE_DIFFS_HELP = """
     If set to true, print line by line diffs on all libraries. If set to
     false, only print which libraries have differences.
"""

_API_GOLDEN_FOLDER = 'tensorflow/tools/api/golden'
_TEST_README_FILE = 'tensorflow/tools/api/tests/README.txt'
_UPDATE_WARNING_FILE = 'tensorflow/tools/api/tests/API_UPDATE_WARNING.txt'


def _KeyToFilePath(key):
  """From a given key, construct a filepath."""
  def _ReplaceCapsWithDash(matchobj):
    match = matchobj.group(0)
    return '-%s' % (match.lower())

  case_insensitive_key = re.sub('([A-Z]{1})', _ReplaceCapsWithDash, key)
  return os.path.join(_API_GOLDEN_FOLDER, '%s.pbtxt' % case_insensitive_key)


def _FileNameToKey(filename):
  """From a given filename, construct a key we use for api objects."""
  def _ReplaceDashWithCaps(matchobj):
    match = matchobj.group(0)
    return match[1].upper()

  base_filename = os.path.basename(filename)
  base_filename_without_ext = os.path.splitext(base_filename)[0]
  api_object_key = re.sub(
      '((-[a-z]){1})', _ReplaceDashWithCaps, base_filename_without_ext)
  return api_object_key


class ApiCompatibilityTest(test.TestCase):

  def __init__(self, *args, **kwargs):
    super(ApiCompatibilityTest, self).__init__(*args, **kwargs)

    golden_update_warning_filename = os.path.join(
        resource_loader.get_root_dir_with_all_resources(),
        _UPDATE_WARNING_FILE)
    self._update_golden_warning = file_io.read_file_to_string(
        golden_update_warning_filename)

    test_readme_filename = os.path.join(
        resource_loader.get_root_dir_with_all_resources(),
        _TEST_README_FILE)
    self._test_readme_message = file_io.read_file_to_string(
        test_readme_filename)

  def _AssertProtoDictEquals(self,
                             expected_dict,
                             actual_dict,
                             verbose=False,
                             update_goldens=False):
    """Diff given dicts of protobufs and report differences a readable way.

    Args:
      expected_dict: a dict of TFAPIObject protos constructed from golden
          files.
      actual_dict: a ict of TFAPIObject protos constructed by reading from the
          TF package linked to the test.
      verbose: Whether to log the full diffs, or simply report which files were
          different.
      update_goldens: Whether to update goldens when there are diffs found.
    """
    diffs = []
    verbose_diffs = []

    expected_keys = set(expected_dict.keys())
    actual_keys = set(actual_dict.keys())
    only_in_expected = expected_keys - actual_keys
    only_in_actual = actual_keys - expected_keys
    all_keys = expected_keys | actual_keys

    # This will be populated below.
    updated_keys = []

    for key in all_keys:
      diff_message = ''
      verbose_diff_message = ''
      # First check if the key is not found in one or the other.
      if key in only_in_expected:
        diff_message = 'Object %s expected but not found (removed).' % key
        verbose_diff_message = diff_message
      elif key in only_in_actual:
        diff_message = 'New object %s found (added).' % key
        verbose_diff_message = diff_message
      else:
        # Now we can run an actual proto diff.
        try:
          self.assertProtoEquals(expected_dict[key], actual_dict[key])
        except AssertionError as e:
          updated_keys.append(key)
          diff_message = 'Change detected in python object: %s.' % key
          verbose_diff_message = str(e)

      # All difference cases covered above. If any difference found, add to the
      # list.
      if diff_message:
        diffs.append(diff_message)
        verbose_diffs.append(verbose_diff_message)

    # If diffs are found, handle them based on flags.
    if diffs:
      diff_count = len(diffs)
      logging.error(self._test_readme_message)
      logging.error('%d differences found between API and golden.', diff_count)
      messages = verbose_diffs if verbose else diffs
      for i in range(diff_count):
        logging.error('Issue %d\t: %s', i + 1, messages[i])

      if update_goldens:
        # Write files if requested.
        logging.warning(self._update_golden_warning)

        # If the keys are only in expected, some objects are deleted.
        # Remove files.
        for key in only_in_expected:
          filepath = _KeyToFilePath(key)
          file_io.delete_file(filepath)

        # If the files are only in actual (current library), these are new
        # modules. Write them to files. Also record all updates in files.
        for key in only_in_actual | set(updated_keys):
          filepath = _KeyToFilePath(key)
          file_io.write_string_to_file(
              filepath, text_format.MessageToString(actual_dict[key]))
      else:
        # Fail if we cannot fix the test by updating goldens.
        self.fail('%d differences found between API and golden.' % diff_count)

    else:
      logging.info('No differences found between API and golden.')

  @unittest.skipUnless(
      sys.version_info.major == 2,
      'API compabitility test goldens are generated using python2.')
  def testAPIBackwardsCompatibility(self):
    # Extract all API stuff.
    visitor = python_object_to_proto_visitor.PythonObjectToProtoVisitor()

    public_api_visitor = public_api.PublicAPIVisitor(visitor)
    public_api_visitor.do_not_descend_map['tf'].append('contrib')
    public_api_visitor.do_not_descend_map['tf.GPUOptions'] = ['Experimental']
    traverse.traverse(tf, public_api_visitor)

    proto_dict = visitor.GetProtos()

    # Read all golden files.
    expression = os.path.join(
        resource_loader.get_root_dir_with_all_resources(),
        _KeyToFilePath('*'))
    golden_file_list = file_io.get_matching_files(expression)

    def _ReadFileToProto(filename):
      """Read a filename, create a protobuf from its contents."""
      ret_val = api_objects_pb2.TFAPIObject()
      text_format.Merge(file_io.read_file_to_string(filename), ret_val)
      return ret_val

    golden_proto_dict = {
        _FileNameToKey(filename): _ReadFileToProto(filename)
        for filename in golden_file_list
    }

    # Diff them. Do not fail if called with update.
    # If the test is run to update goldens, only report diffs but do not fail.
    self._AssertProtoDictEquals(
        golden_proto_dict,
        proto_dict,
        verbose=FLAGS.verbose_diffs,
        update_goldens=FLAGS.update_goldens)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--update_goldens', type=bool, default=False, help=_UPDATE_GOLDENS_HELP)
  parser.add_argument(
      '--verbose_diffs', type=bool, default=False, help=_VERBOSE_DIFFS_HELP)
  FLAGS, unparsed = parser.parse_known_args()

  # Now update argv, so that unittest library does not get confused.
  sys.argv = [sys.argv[0]] + unparsed
  test.main()
