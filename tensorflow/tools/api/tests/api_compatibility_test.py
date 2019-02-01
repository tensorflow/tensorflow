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

import tensorflow as tf
from tensorflow._api.v2 import v2 as tf_v2

from google.protobuf import message
from google.protobuf import text_format

from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import test_util
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

# DEFINE_boolean, only_test_core_api, default False:
_ONLY_TEST_CORE_API_HELP = """
    Some TF APIs are being moved outside of the tensorflow/ directory. There is
    no guarantee which versions of these APIs will be present when running this
    test. Therefore, do not error out on API changes in non-core TF code
    if this flag is set.
"""

# DEFINE_boolean, verbose_diffs, default True:
_VERBOSE_DIFFS_HELP = """
     If set to true, print line by line diffs on all libraries. If set to
     false, only print which libraries have differences.
"""

_API_GOLDEN_FOLDER_V1 = 'tensorflow/tools/api/golden/v1'
_API_GOLDEN_FOLDER_V2 = 'tensorflow/tools/api/golden/v2'
_TEST_README_FILE = 'tensorflow/tools/api/tests/README.txt'
_UPDATE_WARNING_FILE = 'tensorflow/tools/api/tests/API_UPDATE_WARNING.txt'

_NON_CORE_PACKAGES = ['estimator']


def _KeyToFilePath(key, api_version):
  """From a given key, construct a filepath.

  Filepath will be inside golden folder for api_version.
  """

  def _ReplaceCapsWithDash(matchobj):
    match = matchobj.group(0)
    return '-%s' % (match.lower())

  case_insensitive_key = re.sub('([A-Z]{1})', _ReplaceCapsWithDash, key)
  api_folder = (
      _API_GOLDEN_FOLDER_V2 if api_version == 2 else _API_GOLDEN_FOLDER_V1)
  return os.path.join(api_folder, '%s.pbtxt' % case_insensitive_key)


def _FileNameToKey(filename):
  """From a given filename, construct a key we use for api objects."""

  def _ReplaceDashWithCaps(matchobj):
    match = matchobj.group(0)
    return match[1].upper()

  base_filename = os.path.basename(filename)
  base_filename_without_ext = os.path.splitext(base_filename)[0]
  api_object_key = re.sub('((-[a-z]){1})', _ReplaceDashWithCaps,
                          base_filename_without_ext)
  return api_object_key


def _VerifyNoSubclassOfMessageVisitor(path, parent, unused_children):
  """A Visitor that crashes on subclasses of generated proto classes."""
  # If the traversed object is a proto Message class
  if not (isinstance(parent, type) and issubclass(parent, message.Message)):
    return
  if parent is message.Message:
    return
  # Check that it is a direct subclass of Message.
  if message.Message not in parent.__bases__:
    raise NotImplementedError(
        'Object tf.%s is a subclass of a generated proto Message. '
        'They are not yet supported by the API tools.' % path)


def _FilterNonCoreGoldenFiles(golden_file_list):
  """Filter out non-core API pbtxt files."""
  filtered_file_list = []
  filtered_package_prefixes = ['tensorflow.%s.' % p for p in _NON_CORE_PACKAGES]
  for f in golden_file_list:
    if any(
        f.rsplit('/')[-1].startswith(pre) for pre in filtered_package_prefixes
    ):
      continue
    filtered_file_list.append(f)
  return filtered_file_list


class ApiCompatibilityTest(test.TestCase):

  def __init__(self, *args, **kwargs):
    super(ApiCompatibilityTest, self).__init__(*args, **kwargs)

    golden_update_warning_filename = os.path.join(
        resource_loader.get_root_dir_with_all_resources(), _UPDATE_WARNING_FILE)
    self._update_golden_warning = file_io.read_file_to_string(
        golden_update_warning_filename)

    test_readme_filename = os.path.join(
        resource_loader.get_root_dir_with_all_resources(), _TEST_README_FILE)
    self._test_readme_message = file_io.read_file_to_string(
        test_readme_filename)

  def _AssertProtoDictEquals(self,
                             expected_dict,
                             actual_dict,
                             verbose=False,
                             update_goldens=False,
                             additional_missing_object_message='',
                             api_version=2):
    """Diff given dicts of protobufs and report differences a readable way.

    Args:
      expected_dict: a dict of TFAPIObject protos constructed from golden files.
      actual_dict: a ict of TFAPIObject protos constructed by reading from the
        TF package linked to the test.
      verbose: Whether to log the full diffs, or simply report which files were
        different.
      update_goldens: Whether to update goldens when there are diffs found.
      additional_missing_object_message: Message to print when a symbol is
        missing.
      api_version: TensorFlow API version to test.
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
        diff_message = 'Object %s expected but not found (removed). %s' % (
            key, additional_missing_object_message)
        verbose_diff_message = diff_message
      elif key in only_in_actual:
        diff_message = 'New object %s found (added).' % key
        verbose_diff_message = diff_message
      else:
        # Do not truncate diff
        self.maxDiff = None  # pylint: disable=invalid-name
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
        print('Issue %d\t: %s' % (i + 1, messages[i]), file=sys.stderr)

      if update_goldens:
        # Write files if requested.
        logging.warning(self._update_golden_warning)

        # If the keys are only in expected, some objects are deleted.
        # Remove files.
        for key in only_in_expected:
          filepath = _KeyToFilePath(key, api_version)
          file_io.delete_file(filepath)

        # If the files are only in actual (current library), these are new
        # modules. Write them to files. Also record all updates in files.
        for key in only_in_actual | set(updated_keys):
          filepath = _KeyToFilePath(key, api_version)
          file_io.write_string_to_file(
              filepath, text_format.MessageToString(actual_dict[key]))
      else:
        # Fail if we cannot fix the test by updating goldens.
        self.fail('%d differences found between API and golden.' % diff_count)

    else:
      logging.info('No differences found between API and golden.')

  def testNoSubclassOfMessage(self):
    visitor = public_api.PublicAPIVisitor(_VerifyNoSubclassOfMessageVisitor)
    visitor.do_not_descend_map['tf'].append('contrib')
    # Skip compat.v1 and compat.v2 since they are validated in separate tests.
    visitor.private_map['tf.compat'] = ['v1', 'v2']
    traverse.traverse(tf, visitor)

  def testNoSubclassOfMessageV1(self):
    if not hasattr(tf.compat, 'v1'):
      return
    visitor = public_api.PublicAPIVisitor(_VerifyNoSubclassOfMessageVisitor)
    visitor.do_not_descend_map['tf'].append('contrib')
    if FLAGS.only_test_core_api:
      visitor.do_not_descend_map['tf'].extend(_NON_CORE_PACKAGES)
    traverse.traverse(tf_v2.compat.v1, visitor)

  def testNoSubclassOfMessageV2(self):
    if not hasattr(tf.compat, 'v2'):
      return
    visitor = public_api.PublicAPIVisitor(_VerifyNoSubclassOfMessageVisitor)
    visitor.do_not_descend_map['tf'].append('contrib')
    if FLAGS.only_test_core_api:
      visitor.do_not_descend_map['tf'].extend(_NON_CORE_PACKAGES)
    traverse.traverse(tf_v2, visitor)

  def _checkBackwardsCompatibility(self,
                                   root,
                                   golden_file_pattern,
                                   api_version,
                                   additional_private_map=None):
    # Extract all API stuff.
    visitor = python_object_to_proto_visitor.PythonObjectToProtoVisitor()

    public_api_visitor = public_api.PublicAPIVisitor(visitor)
    public_api_visitor.private_map['tf'] = ['contrib']
    if api_version == 2:
      public_api_visitor.private_map['tf'].append('enable_v2_behavior')

    public_api_visitor.do_not_descend_map['tf.GPUOptions'] = ['Experimental']
    if FLAGS.only_test_core_api:
      public_api_visitor.do_not_descend_map['tf'].extend(_NON_CORE_PACKAGES)
    if additional_private_map:
      public_api_visitor.private_map.update(additional_private_map)

    traverse.traverse(root, public_api_visitor)
    proto_dict = visitor.GetProtos()

    # Read all golden files.
    golden_file_list = file_io.get_matching_files(golden_file_pattern)
    if FLAGS.only_test_core_api:
      golden_file_list = _FilterNonCoreGoldenFiles(golden_file_list)

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
        update_goldens=FLAGS.update_goldens,
        api_version=api_version)

  @test_util.run_v1_only('b/120545219')
  def testAPIBackwardsCompatibility(self):
    api_version = 1
    golden_file_pattern = os.path.join(
        resource_loader.get_root_dir_with_all_resources(),
        _KeyToFilePath('*', api_version))
    self._checkBackwardsCompatibility(
        tf,
        golden_file_pattern,
        api_version,
        # Skip compat.v1 and compat.v2 since they are validated
        # in separate tests.
        additional_private_map={'tf.compat': ['v1', 'v2']})

    # Also check that V1 API has contrib
    self.assertTrue(
        'tensorflow.python.util.lazy_loader.LazyLoader'
        in str(type(tf.contrib)))

  @test_util.run_v1_only('b/120545219')
  def testAPIBackwardsCompatibilityV1(self):
    api_version = 1
    golden_file_pattern = os.path.join(
        resource_loader.get_root_dir_with_all_resources(),
        _KeyToFilePath('*', api_version))
    self._checkBackwardsCompatibility(tf_v2.compat.v1, golden_file_pattern,
                                      api_version)

  def testAPIBackwardsCompatibilityV2(self):
    api_version = 2
    golden_file_pattern = os.path.join(
        resource_loader.get_root_dir_with_all_resources(),
        _KeyToFilePath('*', api_version))
    self._checkBackwardsCompatibility(
        tf_v2,
        golden_file_pattern,
        api_version,
        additional_private_map={'tf.compat': ['v1']})


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--update_goldens', type=bool, default=False, help=_UPDATE_GOLDENS_HELP)
  # TODO(mikecase): Create Estimator's own API compatibility test or
  # a more general API compatibility test for use for TF components.
  parser.add_argument(
      '--only_test_core_api',
      type=bool,
      default=False,
      help=_ONLY_TEST_CORE_API_HELP)
  parser.add_argument(
      '--verbose_diffs', type=bool, default=True, help=_VERBOSE_DIFFS_HELP)
  FLAGS, unparsed = parser.parse_known_args()

  # Now update argv, so that unittest library does not get confused.
  sys.argv = [sys.argv[0]] + unparsed
  test.main()
