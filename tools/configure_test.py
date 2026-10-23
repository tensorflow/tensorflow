# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests for TensorFlow configure script and helper utilities."""

import argparse
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tools.configure_helper import (
    BazelrcConfig,
    PromptManager,
    apply_cli_overrides,
    safe_decode,
    split_path_list,
)
import configure


class SplitPathListTest(unittest.TestCase):
  """Tests for split_path_list helper."""

  def test_empty_or_none(self):
    self.assertEqual(split_path_list(None), [])
    self.assertEqual(split_path_list(''), [])
    self.assertEqual(split_path_list('   '), [])

  def test_single_path_posix(self):
    self.assertEqual(split_path_list('/usr/lib/python3'), ['/usr/lib/python3'])

  def test_multiple_paths_semicolon(self):
    paths = r'C:\Python314\Lib;D:\workspace\tensorflow;E:\libs'
    expected = [r'C:\Python314\Lib', r'D:\workspace\tensorflow', r'E:\libs']
    self.assertEqual(split_path_list(paths), expected)

  def test_windows_single_path_with_drive_colon(self):
    # On Windows, drive letters like C:\ should NOT be split into 'C' and '\...'
    with patch('tools.configure_helper.is_windows', return_value=True):
      res = split_path_list(r'C:\Python314\Lib')
      self.assertEqual(res, [r'C:\Python314\Lib'])

  def test_posix_colon_separated(self):
    with patch('tools.configure_helper.is_windows', return_value=False):
      with patch('tools.configure_helper.is_cygwin', return_value=False):
        with patch('os.pathsep', ':'):
          res = split_path_list('/usr/lib:/usr/local/lib')
          self.assertEqual(res, ['/usr/lib', '/usr/local/lib'])


class SafeDecodeTest(unittest.TestCase):
  """Tests for safe_decode helper."""

  def test_none_and_str(self):
    self.assertEqual(safe_decode(None), '')
    self.assertEqual(safe_decode('  hello  '), 'hello')

  def test_bytes_utf8(self):
    self.assertEqual(safe_decode(b'hello world\n'), 'hello world')

  def test_bytes_non_utf8(self):
    # Byte sequence with invalid UTF-8 should not crash
    raw_bytes = b'error \xff\xfe'
    decoded = safe_decode(raw_bytes)
    self.assertIn('error', decoded)


class BazelrcConfigTest(unittest.TestCase):
  """Tests for BazelrcConfig."""

  def setUp(self):
    self.test_dir = tempfile.mkdtemp()
    self.bazelrc_path = os.path.join(self.test_dir, '.tf_configure.bazelrc')

  def tearDown(self):
    shutil.rmtree(self.test_dir, ignore_errors=True)

  def test_in_memory_accumulation(self):
    config = BazelrcConfig()
    config.add_line('build --copt="-O3"')
    config.add_action_env('PYTHON_BIN_PATH', '/usr/bin/python3')
    config.add_repo_env('cuda', 'HERMETIC_CUDA_VERSION', '12.3')

    expected = (
        'build --copt="-O3"\n'
        'build --action_env PYTHON_BIN_PATH="/usr/bin/python3"\n'
        'build:cuda --repo_env HERMETIC_CUDA_VERSION="12.3"'
    )
    self.assertEqual(config.to_string(), expected)

  def test_file_writing_and_reset(self):
    config = BazelrcConfig(self.bazelrc_path)
    config.add_action_env('TEST_VAR', '1')

    self.assertTrue(os.path.exists(self.bazelrc_path))
    with open(self.bazelrc_path, 'r', encoding='utf-8') as f:
      content = f.read()
    self.assertIn('build --action_env TEST_VAR="1"', content)

    config.reset()
    self.assertEqual(config.lines, [])
    with open(self.bazelrc_path, 'r', encoding='utf-8') as f:
      self.assertEqual(f.read(), '')


class PromptManagerTest(unittest.TestCase):
  """Tests for PromptManager."""

  def test_interactive_mode(self):
    mock_input = MagicMock(return_value='my_custom_input')
    pm = PromptManager(non_interactive=False, input_fn=mock_input)
    response = pm.get_input('Specify path: ')
    self.assertEqual(response, 'my_custom_input')
    mock_input.assert_called_once_with('Specify path: ')

  def test_non_interactive_mode(self):
    mock_input = MagicMock()
    pm = PromptManager(non_interactive=True, input_fn=mock_input)
    response = pm.get_input('Specify path: ')
    self.assertEqual(response, '')
    mock_input.assert_not_called()

  def test_eof_handling(self):
    mock_input = MagicMock(side_effect=EOFError)
    pm = PromptManager(non_interactive=False, input_fn=mock_input)
    response = pm.get_input('Specify path: ')
    self.assertEqual(response, '')


class ConvertVersionToIntTest(unittest.TestCase):
  """Tests for convert_version_to_int in configure.py."""

  def test_valid_versions(self):
    self.assertEqual(configure.convert_version_to_int('2.16.0'), 2016000)
    self.assertEqual(configure.convert_version_to_int('0.24'), 24000)
    self.assertEqual(configure.convert_version_to_int('7.1.0-homebrew'), 7001000)
    self.assertEqual(configure.convert_version_to_int('18.1.3'), 18001003)

  def test_invalid_versions(self):
    self.assertIsNone(configure.convert_version_to_int(''))
    self.assertIsNone(configure.convert_version_to_int(None))
    self.assertIsNone(configure.convert_version_to_int('git-head'))
    self.assertIsNone(configure.convert_version_to_int('custom.version.beta'))


class GetVarTest(unittest.TestCase):
  """Tests for get_var boolean evaluation."""

  def test_env_truthy_values(self):
    for val in ('1', 't', 'true', 'y', 'yes', 'TRUE', 'Yes'):
      env = {'TF_TEST_VAR': val}
      result = configure.get_var(
          env, 'TF_TEST_VAR', 'Test feature', enabled_by_default=False
      )
      self.assertTrue(result, f'Failed for value {val}')

  def test_env_falsy_values(self):
    for val in ('0', 'f', 'false', 'n', 'no', 'FALSE', 'No'):
      env = {'TF_TEST_VAR': val}
      result = configure.get_var(
          env, 'TF_TEST_VAR', 'Test feature', enabled_by_default=True
      )
      self.assertFalse(result, f'Failed for value {val}')

  def test_invalid_env_value_raises_user_input_error(self):
    env = {'TF_TEST_VAR': 'not_a_boolean'}
    with self.assertRaises(configure.UserInputError):
      configure.get_var(
          env, 'TF_TEST_VAR', 'Test feature', enabled_by_default=False
      )

  def test_non_interactive_fallback(self):
    env = {}
    with patch.object(configure._PROMPT_MANAGER, 'non_interactive', True):
      result_default_true = configure.get_var(
          env, 'TF_UNSET_VAR_1', 'Feature 1', enabled_by_default=True
      )
      self.assertTrue(result_default_true)

      result_default_false = configure.get_var(
          env, 'TF_UNSET_VAR_2', 'Feature 2', enabled_by_default=False
      )
      self.assertFalse(result_default_false)


class CliOverridesTest(unittest.TestCase):
  """Tests for CLI arguments override parsing."""

  def test_apply_cli_overrides(self):
    parser = configure.setup_argument_parser()
    args = parser.parse_args([
        '--python-bin-path', '/custom/bin/python',
        '--cuda',
        '--no-rocm',
        '--clang',
        '--no-xla',
    ])
    env = {}
    apply_cli_overrides(args, env)

    self.assertEqual(env['PYTHON_BIN_PATH'], '/custom/bin/python')
    self.assertEqual(env['TF_NEED_CUDA'], '1')
    self.assertEqual(env['TF_NEED_ROCM'], '0')
    self.assertEqual(env['TF_NEED_CLANG'], '1')
    self.assertEqual(env['TF_ENABLE_XLA'], '0')


class ConfigureMainTest(unittest.TestCase):
  """End-to-end execution test for configure.main."""

  def setUp(self):
    self.temp_workspace = tempfile.mkdtemp()
    self.temp_bazelrc = os.path.join(self.temp_workspace, '.tf_configure.bazelrc')

  def tearDown(self):
    shutil.rmtree(self.temp_workspace, ignore_errors=True)

  @patch('configure.retrieve_bazel_version', return_value='6.5.0')
  def test_main_non_interactive_execution(self, mock_bazel):
    test_args = [
        'configure.py',
        '--workspace',
        self.temp_workspace,
        '--output-bazelrc',
        self.temp_bazelrc,
        '--non-interactive',
        '--python-bin-path',
        sys.executable,
        '--no-cuda',
        '--no-rocm',
        '--no-clang',
    ]
    with patch.object(sys, 'argv', test_args):
      configure.main()

    self.assertTrue(os.path.exists(self.temp_bazelrc))
    with open(self.temp_bazelrc, 'r', encoding='utf-8') as f:
      content = f.read()

    self.assertIn('PYTHON_BIN_PATH', content)
    self.assertIn('PYTHON_LIB_PATH', content)

    # Verify tools/python_bin_path.sh was created
    bin_sh_path = os.path.join(self.temp_workspace, 'tools', 'python_bin_path.sh')
    self.assertTrue(os.path.exists(bin_sh_path))


if __name__ == '__main__':
  unittest.main()

