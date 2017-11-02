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
from collections import defaultdict
from operator import attrgetter
import os
import re
import subprocess
import sys
import unittest

import tensorflow as tf

from google.protobuf import text_format

from tensorflow.core.framework import api_def_pb2
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

_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
_CONVERT_FROM_MULTILINE_SCRIPT = 'tensorflow/tools/api/tests/convert_from_multiline'
_BASE_API_DIR = 'tensorflow/core/api_def/base_api'
_PYTHON_API_DIR = 'tensorflow/core/api_def/python_api'
_HIDDEN_OPS_FILE = 'tensorflow/python/ops/hidden_ops.txt'


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


def _GetSymbol(symbol_id):
  """Get TensorFlow symbol based on the given identifier.

  Args:
    symbol_id: Symbol identifier in the form module1.module2. ... .sym.

  Returns:
    Symbol corresponding to the given id.
  """
  # Ignore first module which should be tensorflow
  symbol_id_split = symbol_id.split('.')[1:]
  symbol = tf
  for sym in symbol_id_split:
    symbol = getattr(symbol, sym)
  return symbol


def _IsGenModule(module_name):
  if not module_name:
    return False
  module_name_split = module_name.split('.')
  return module_name_split[-1].startswith('gen_')


def _GetHiddenOps():
  hidden_ops_file = file_io.FileIO(_HIDDEN_OPS_FILE, 'r')
  hidden_ops = set()
  for line in hidden_ops_file:
    line = line.strip()
    if not line:
      continue
    if line[0] == '#':  # comment line
      continue
    # If line is of the form "op_name # comment", only keep the op_name.
    line_split = line.split('#')
    hidden_ops.add(line_split[0].strip())
  return hidden_ops


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
      sys.version_info.major == 2 and os.uname()[0] == 'Linux',
      'API compabitility test goldens are generated using python2 on Linux.')
  def testAPIBackwardsCompatibility(self):
    # Extract all API stuff.
    visitor = python_object_to_proto_visitor.PythonObjectToProtoVisitor()

    public_api_visitor = public_api.PublicAPIVisitor(visitor)
    public_api_visitor.do_not_descend_map['tf'].append('contrib')
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


class ApiDefTest(test.TestCase):

  def __init__(self, *args, **kwargs):
    super(ApiDefTest, self).__init__(*args, **kwargs)
    self._first_cap_pattern = re.compile('(.)([A-Z][a-z]+)')
    self._all_cap_pattern = re.compile('([a-z0-9])([A-Z])')

  def _GenerateLowerCaseOpName(self, op_name):
    lower_case_name = self._first_cap_pattern.sub(r'\1_\2', op_name)
    return self._all_cap_pattern.sub(r'\1_\2', lower_case_name).lower()

  def _CreatePythonApiDef(self, base_api_def, endpoint_names):
    """Creates Python ApiDef that overrides base_api_def if needed.

    Args:
      base_api_def: (api_def_pb2.ApiDef) base ApiDef instance.
      endpoint_names: List of Python endpoint names.

    Returns:
      api_def_pb2.ApiDef instance with overrides for base_api_def
      if module.name endpoint is different from any existing
      endpoints in base_api_def. Otherwise, returns None.
    """
    endpoint_names_set = set(endpoint_names)
    base_endpoint_names_set = {
        self._GenerateLowerCaseOpName(endpoint.name)
        for endpoint in base_api_def.endpoint}

    if endpoint_names_set == base_endpoint_names_set:
      return None  # All endpoints are the same

    api_def = api_def_pb2.ApiDef()
    api_def.graph_op_name = base_api_def.graph_op_name

    for endpoint_name in sorted(endpoint_names):
      new_endpoint = api_def.endpoint.add()
      new_endpoint.name = endpoint_name

    return api_def

  def _GetBaseApiMap(self):
    """Get a map from graph op name to its base ApiDef.

    Returns:
      Dictionary mapping graph op name to corresponding ApiDef.
    """
    # Convert base ApiDef in Multiline format to Proto format.
    converted_base_api_dir = os.path.join(
        test.get_temp_dir(), 'temp_base_api_defs')
    subprocess.check_call(
        [os.path.join(resource_loader.get_root_dir_with_all_resources(),
                      _CONVERT_FROM_MULTILINE_SCRIPT),
         _BASE_API_DIR, converted_base_api_dir])

    name_to_base_api_def = {}
    base_api_files = file_io.get_matching_files(
        os.path.join(converted_base_api_dir, 'api_def_*.pbtxt'))
    for base_api_file in base_api_files:
      if file_io.file_exists(base_api_file):
        api_defs = api_def_pb2.ApiDefs()
        text_format.Merge(
            file_io.read_file_to_string(base_api_file), api_defs)
        for api_def in api_defs.op:
          name_to_base_api_def[api_def.graph_op_name] = api_def
    return name_to_base_api_def

  def _AddHiddenOpOverrides(self, name_to_base_api_def, api_def_map):
    """Adds ApiDef overrides to api_def_map for hidden Python ops.

    Args:
      name_to_base_api_def: Map from op name to base api_def_pb2.ApiDef.
      api_def_map: Map from first op name character (in caps) to
        api_def_pb2.ApiDefs for Python API overrides.
    """
    hidden_ops = _GetHiddenOps()
    for hidden_op in hidden_ops:
      if hidden_op not in name_to_base_api_def:
        logging.warning('Unexpected hidden op name: %s' % hidden_op)
        continue

      base_api_def = name_to_base_api_def[hidden_op]
      if base_api_def.visibility != api_def_pb2.ApiDef.HIDDEN:
        api_def = api_def_pb2.ApiDef()
        api_def.graph_op_name = base_api_def.graph_op_name
        api_def.visibility = api_def_pb2.ApiDef.HIDDEN
        api_def_map[api_def.graph_op_name[0].upper()].op.extend([api_def])

  @unittest.skipUnless(
      sys.version_info.major == 2 and os.uname()[0] == 'Linux',
      'API compabitility test goldens are generated using python2 on Linux.')
  def testAPIDefCompatibility(self):
    # Get base ApiDef
    name_to_base_api_def = self._GetBaseApiMap()
    snake_to_camel_graph_op_names = {
        self._GenerateLowerCaseOpName(name): name
        for name in name_to_base_api_def.keys()}
    # Extract Python API
    visitor = python_object_to_proto_visitor.PythonObjectToProtoVisitor()
    public_api_visitor = public_api.PublicAPIVisitor(visitor)
    public_api_visitor.do_not_descend_map['tf'].append('contrib')
    traverse.traverse(tf, public_api_visitor)
    proto_dict = visitor.GetProtos()

    # Map from first character of op name to Python ApiDefs.
    api_def_map = defaultdict(api_def_pb2.ApiDefs)
    # We need to override all endpoints even if 1 endpoint differs from base
    # ApiDef. So, we first create a map from an op to all its endpoints.
    op_to_endpoint_name = defaultdict(list)

    # Generate map from generated python op to endpoint names.
    for public_module, value in proto_dict.items():
      module_obj = _GetSymbol(public_module)
      for sym in value.tf_module.member_method:
        obj = getattr(module_obj, sym.name)

        # Check if object is defined in gen_* module. That is,
        # the object has been generated from OpDef.
        if hasattr(obj, '__module__') and _IsGenModule(obj.__module__):
          if obj.__name__ not in snake_to_camel_graph_op_names:
            # Symbol might be defined only in Python and not generated from
            # C++ api.
            continue
          relative_public_module = public_module[len('tensorflow.'):]
          full_name = (relative_public_module + '.' + sym.name
                       if relative_public_module else sym.name)
          op_to_endpoint_name[obj].append(full_name)

    # Generate Python ApiDef overrides.
    for op, endpoint_names in op_to_endpoint_name.items():
      graph_op_name = snake_to_camel_graph_op_names[op.__name__]
      api_def = self._CreatePythonApiDef(
          name_to_base_api_def[graph_op_name], endpoint_names)
      if api_def:
        api_defs = api_def_map[graph_op_name[0].upper()]
        api_defs.op.extend([api_def])

    self._AddHiddenOpOverrides(name_to_base_api_def, api_def_map)

    for key in _ALPHABET:
      # Get new ApiDef for the given key.
      new_api_defs_str = ''
      if key in api_def_map:
        new_api_defs = api_def_map[key]
        new_api_defs.op.sort(key=attrgetter('graph_op_name'))
        new_api_defs_str = str(new_api_defs)

      # Get current ApiDef for the given key.
      api_defs_file_path = os.path.join(
          _PYTHON_API_DIR, 'api_def_%s.pbtxt' % key)
      old_api_defs_str = ''
      if file_io.file_exists(api_defs_file_path):
        old_api_defs_str = file_io.read_file_to_string(api_defs_file_path)

      if old_api_defs_str == new_api_defs_str:
        continue

      if FLAGS.update_goldens:
        if not new_api_defs_str:
          logging.info('Deleting %s...' % api_defs_file_path)
          file_io.delete_file(api_defs_file_path)
        else:
          logging.info('Updating %s...' % api_defs_file_path)
          file_io.write_string_to_file(api_defs_file_path, new_api_defs_str)
      else:
        self.assertMultiLineEqual(
            old_api_defs_str, new_api_defs_str,
            'To update golden API files, run api_compatibility_test locally '
            'with --update_goldens=True flag.')


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
