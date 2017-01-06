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

This test ensures all changes to the public API of TensorFlow is checked.

If this test fails, it means there is a change made to the public API.
You can still go forward with your change, however you will need explicit
approval from TensorFlow leads for your API change.

After you receive approval, you can run the test as follows to update test
goldens and package them with your change.

    $ bazel build tensorflow/tools/api/test:api_compatibility_test
    $ bazel-bin/tensorflow/tools/api/test/api_compatibility_test --update_goldens True
"""
import argparse
import os
import sys
import tensorflow as tf

from google.protobuf import text_format

from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.tools.api.lib import python_object_to_proto_visitor
from tensorflow.tools.api.lib import api_objects_pb2
from tensorflow.tools.common import public_api
from tensorflow.tools.common import traverse

# FLAGS defined at the bottom:
FLAGS = None
# DEFINE_boolean, update_goldens, default False:
_UPDATE_GOLDENS_HELP="""
     Update stored golden files if API is updated. WARNING: All API changes
     have to be authorized by TensorFlow leads.
"""

# DEFINE_boolean, verbose_diffs, default False:
_VERBOSE_DIFFS_HELP="""
     If set to true, print line by line diffs on all libraries. If set to
     false, only print which libraries have differences.
"""

_API_GOLDEN_FOLDER = 'tensorflow/tools/api/golden'
_GOLDEN_UPDATE_WARNING = """
Golden file update requested!
All test failures have been skipped, see the logs for detected diffs.
This test is now going to write new golden files.
Make sure to:
  1-Package the updates together with your change.
  2-Get an API update approval from wicke<TODO: add more people here.
"""


class ApiCompatibilityTest(test.TestCase):

  def _AssertProtoDictEquals(
      self, expected_dict, actual_dict, fail_on_diffs=False, verbose=False):
    """Diff given dicts of protobufs and report differences a readable way.

    Args:
      expected_dict: a dict of TFAPIObject protos constructed from golden
          files.
      actual_dict: a dict of TFAPIObject protos extracted from the current API.
      fail_on_diffs: Whether to fail the test if any diffs are found or not.
      verbose: Whether to log the full diffs, or simply report which files were
          different.
    """
    diffs = []
    verbose_diffs = []

    expected_keys = set(expected_dict.keys())
    actual_keys = set(actual_dict.keys())
    only_in_expected = expected_keys - actual_keys
    only_in_actual = actual_keys - expected_keys
    all_keys = expected_keys | actual_keys

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
          # TODO(gunan): Create a structural diff tool.
          self.assertProtoEquals(expected_dict[key], actual_dict[key])
        except AssertionError as e:
          diff_message = 'Change detected in python object: %s.' % key
          verbose_diff_message = str(e)

      # All difference cases covered above. If any difference found, add to the
      # list.
      if diff_message:
        diffs.append(diff_message)
        verbose_diffs.append(verbose_diff_message)

    # If diffs are found, handle them base on flags.
    if diffs:
      diff_count = len(diffs)
      logging.error('%d differences found between API and golden.', diff_count)
      messages = verbose_diffs if verbose else diffs
      for i in range(diff_count):
        logging.error('Issue %d\t: %s', i+1, messages[i])

      if fail_on_diffs:
        self.fail()
    else:
      logging.info('No differences found between API and golden.')


  def testAPIBackwardsCompatibility(self):
    # Extract all API stuff.
    visitor = python_object_to_proto_visitor.PythonObjectToProtoVisitor()
    traverse.traverse(tf, public_api.PublicAPIVisitor(visitor))
    proto_dict = visitor.GetProtos()

    # Read all golden files.
    # TODO(gunan) figure out why resource loader is failing.
    #golden_file_list = file_io.get_matching_files(
    #    '%s/%s/*.pbtxt' % (resource_loader.get_data_files_path(),
    #                       _API_GOLDEN_FOLDER))
    expression = '%s/*.pbtxt' % _API_GOLDEN_FOLDER
    golden_file_list = file_io.get_matching_files(expression)

    def _ReadFileToProto(filename):
      """Read a filename, create a protobuf from its contents."""
      ret_val = api_objects_pb2.TFAPIObject()
      text_format.Merge(file_io.read_file_to_string(filename), ret_val)
      return ret_val

    golden_proto_dict = {
        os.path.basename(filename[:-6]): _ReadFileToProto(filename) for
        filename in golden_file_list}

    # Diff them. Do not fail if called with update.
    # If the test is run to update goldens, only report diffs but do not fail.
    self._AssertProtoDictEquals(golden_proto_dict, proto_dict,
                                fail_on_diffs=(not FLAGS.update_goldens),
                                verbose=FLAGS.verbose_diffs)

    # Write files if requested.
    if FLAGS.update_goldens:
      logging.warning(_GOLDEN_UPDATE_WARNING)
      for key, proto in proto_dict.iteritems():
        filepath = os.path.join(_API_GOLDEN_FOLDER, '%s.pbtxt' % key)
        file_io.write_string_to_file(
            filepath, text_format.MessageToString(proto))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--update_goldens', type=bool, default=False, help=_UPDATE_GOLDENS_HELP)
  parser.add_argument(
      '--verbose_diffs', type=bool, default=False, help=_VERBOSE_DIFFS_HELP)
  FLAGS, unparsed = parser.parse_known_args()

  # Now update argv, so that unittest does not get confused.
  sys.argv = [sys.argv[0]] + unparsed
  test.main()
