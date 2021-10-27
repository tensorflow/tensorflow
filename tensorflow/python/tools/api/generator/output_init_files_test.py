# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Tests for api_init_files.bzl and api_init_files_v1.bzl."""
import sys

# The unused imports are needed so that the python and lite modules are
# available in sys.modules
# pylint: disable=unused-import
from tensorflow import python as _tf_for_api_traversal
from tensorflow.lite.python import lite as _tflite_for_api_traversal
from tensorflow.lite.python.authoring import authoring
from tensorflow.python import modules_with_exports
from tensorflow.python.distribute import merge_call_interim
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.framework import combinations
from tensorflow.python.framework import test_combinations
# pylint: enable=unused-import
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.util import tf_decorator


def _get_module_from_symbol(symbol):
  if '.' not in symbol:
    return ''
  return '.'.join(symbol.split('.')[:-1])


def _get_modules(package, attr_name, constants_attr_name):
  """Get list of TF API modules.

  Args:
    package: We only look at modules that contain package in the name.
    attr_name: Attribute set on TF symbols that contains API names.
    constants_attr_name: Attribute set on TF modules that contains
      API constant names.

  Returns:
    Set of TensorFlow API modules.
  """
  modules = set()
  # TODO(annarev): split up the logic in create_python_api.py so that
  #   it can be reused in this test.
  for module in list(sys.modules.values()):
    if (not module or not hasattr(module, '__name__') or
        package not in module.__name__):
      continue

    for module_contents_name in dir(module):
      attr = getattr(module, module_contents_name)
      _, attr = tf_decorator.unwrap(attr)

      # Add modules to _tf_api_constants attribute.
      if module_contents_name == constants_attr_name:
        for exports, _ in attr:
          modules.update(
              [_get_module_from_symbol(export) for export in exports])
        continue

      # Add modules for _tf_api_names attribute.
      if (hasattr(attr, '__dict__') and attr_name in attr.__dict__):
        modules.update([
            _get_module_from_symbol(export)
            for export in getattr(attr, attr_name)])
  return modules


def _get_files_set(path, start_tag, end_tag):
  """Get set of file paths from the given file.

  Args:
    path: Path to file. File at `path` is expected to contain a list of paths
      where entire list starts with `start_tag` and ends with `end_tag`. List
      must be comma-separated and each path entry must be surrounded by double
      quotes.
    start_tag: String that indicates start of path list.
    end_tag: String that indicates end of path list.

  Returns:
    List of string paths.
  """
  with open(path, 'r') as f:
    contents = f.read()
    start = contents.find(start_tag) + len(start_tag) + 1
    end = contents.find(end_tag)
    contents = contents[start:end]
    file_paths = [
        file_path.strip().strip('"') for file_path in contents.split(',')]
    return set(file_path for file_path in file_paths if file_path)


def _module_to_paths(module):
  """Get all API __init__.py file paths for the given module.

  Args:
    module: Module to get file paths for.

  Returns:
    List of paths for the given module. For e.g. module foo.bar
    requires 'foo/__init__.py' and 'foo/bar/__init__.py'.
  """
  submodules = []
  module_segments = module.split('.')
  for i in range(len(module_segments)):
    submodules.append('.'.join(module_segments[:i+1]))
  paths = []
  for submodule in submodules:
    if not submodule:
      paths.append('__init__.py')
      continue
    paths.append('%s/__init__.py' % (submodule.replace('.', '/')))
  return paths


class OutputInitFilesTest(test.TestCase):
  """Test that verifies files that list paths for TensorFlow API."""

  def _validate_paths_for_modules(
      self, actual_paths, expected_paths, file_to_update_on_error):
    """Validates that actual_paths match expected_paths.

    Args:
      actual_paths: */__init__.py file paths listed in file_to_update_on_error.
      expected_paths: */__init__.py file paths that we need to create for
        TensorFlow API.
      file_to_update_on_error: File that contains list of */__init__.py files.
        We include it in error message printed if the file list needs to be
        updated.
    """
    self.assertTrue(actual_paths)
    self.assertTrue(expected_paths)
    missing_paths = expected_paths - actual_paths
    extra_paths = actual_paths - expected_paths

    # Surround paths with quotes so that they can be copy-pasted
    # from error messages as strings.
    missing_paths = ['\'%s\'' % path for path in missing_paths]
    extra_paths = ['\'%s\'' % path for path in extra_paths]

    self.assertFalse(
        missing_paths,
        'Please add %s to %s.' % (
            ',\n'.join(sorted(missing_paths)), file_to_update_on_error))
    self.assertFalse(
        extra_paths,
        'Redundant paths, please remove %s in %s.' % (
            ',\n'.join(sorted(extra_paths)), file_to_update_on_error))

  def test_V2_init_files(self):
    modules = _get_modules(
        'tensorflow', '_tf_api_names', '_tf_api_constants')
    file_path = resource_loader.get_path_to_datafile(
        'api_init_files.bzl')
    paths = _get_files_set(
        file_path, '# BEGIN GENERATED FILES', '# END GENERATED FILES')
    module_paths = set(
        f for module in modules for f in _module_to_paths(module))
    self._validate_paths_for_modules(
        paths, module_paths, file_to_update_on_error=file_path)

  def test_V1_init_files(self):
    modules = _get_modules(
        'tensorflow', '_tf_api_names_v1', '_tf_api_constants_v1')
    file_path = resource_loader.get_path_to_datafile(
        'api_init_files_v1.bzl')
    paths = _get_files_set(
        file_path, '# BEGIN GENERATED FILES', '# END GENERATED FILES')
    module_paths = set(
        f for module in modules for f in _module_to_paths(module))
    self._validate_paths_for_modules(
        paths, module_paths, file_to_update_on_error=file_path)


if __name__ == '__main__':
  test.main()
