# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""configure script to get build parameters from user."""

import argparse
import errno
import json
import os
import platform
import re
import shutil
import subprocess
import sys


_DEFAULT_CUDA_COMPUTE_CAPABILITIES = '3.5,7.0'

_SUPPORTED_ANDROID_NDK_VERSIONS = [19, 20, 21, 25]

_DEFAULT_PROMPT_ASK_ATTEMPTS = 10

_TF_BAZELRC_FILENAME = '.tf_configure.bazelrc'
_TF_WORKSPACE_ROOT = ''
_TF_BAZELRC = ''
_TF_CURRENT_BAZEL_VERSION = None

NCCL_LIB_PATHS = [
    'lib64/', 'lib/powerpc64le-linux-gnu/', 'lib/x86_64-linux-gnu/', ''
]

# List of files to configure when building Bazel on Apple platforms.
APPLE_BAZEL_FILES = [
    'tensorflow/lite/ios/BUILD', 'tensorflow/lite/objc/BUILD',
    'tensorflow/lite/swift/BUILD',
    'tensorflow/lite/tools/benchmark/experimental/ios/BUILD'
]

# List of files to move when building for iOS.
IOS_FILES = [
    'tensorflow/lite/objc/TensorFlowLiteObjC.podspec',
    'tensorflow/lite/swift/TensorFlowLiteSwift.podspec',
]


class UserInputError(Exception):
  pass


def is_windows():
  return platform.system() == 'Windows'


def is_linux():
  return platform.system() == 'Linux'


def is_macos():
  return platform.system() == 'Darwin'


def is_ppc64le():
  return platform.machine() == 'ppc64le'


def is_s390x():
  return platform.machine() == 's390x'


def is_cygwin():
  return platform.system().startswith('CYGWIN_NT')


def get_input(question):
  try:
    try:
      answer = raw_input(question)
    except NameError:
      answer = input(question)  # pylint: disable=bad-builtin
  except EOFError:
    answer = ''
  return answer


def symlink_force(target, link_name):
  """Force symlink, equivalent of 'ln -sf'.

  Args:
    target: items to link to.
    link_name: name of the link.
  """
  try:
    os.symlink(target, link_name)
  except OSError as e:
    if e.errno == errno.EEXIST:
      os.remove(link_name)
      os.symlink(target, link_name)
    else:
      raise e


def write_to_bazelrc(line):
  with open(_TF_BAZELRC, 'a') as f:
    f.write(line + '\n')


def write_action_env_to_bazelrc(var_name, var):
  write_to_bazelrc('build --action_env {}="{}"'.format(var_name, str(var)))


def write_repo_env_to_bazelrc(config_name, var_name, var):
  write_to_bazelrc(
      'build:{} --repo_env {}="{}"'.format(config_name, var_name, str(var))
  )


def run_shell(cmd, allow_non_zero=False, stderr=None):
  if stderr is None:
    stderr = sys.stdout
  if allow_non_zero:
    try:
      output = subprocess.check_output(cmd, stderr=stderr)
    except subprocess.CalledProcessError as e:
      output = e.output
  else:
    output = subprocess.check_output(cmd, stderr=stderr)
  return output.decode('UTF-8').strip()


def cygpath(path):
  """Convert path from posix to windows."""
  return os.path.abspath(path).replace('\\', '/')


def get_python_path(environ_cp, python_bin_path):
  """Get the python site package paths."""
  python_paths = []
  if environ_cp.get('PYTHONPATH'):
    python_paths = environ_cp.get('PYTHONPATH').split(':')
  try:
    stderr = open(os.devnull, 'wb')
    library_paths = run_shell([
        python_bin_path, '-c',
        'import site; print("\\n".join(site.getsitepackages()))'
    ],
                              stderr=stderr).split('\n')
  except subprocess.CalledProcessError:
    library_paths = [
        run_shell([
            python_bin_path,
            '-c',
            'import sysconfig;print(sysconfig.get_path("purelib")',
        ])
    ]

  all_paths = set(python_paths + library_paths)
  # Sort set so order is deterministic
  all_paths = sorted(all_paths)

  paths = []
  for path in all_paths:
    if os.path.isdir(path):
      paths.append(path)
  return paths


def get_python_major_version(python_bin_path):
  """Get the python major version."""
  return run_shell([python_bin_path, '-c', 'import sys; print(sys.version[0])'])


def setup_python(environ_cp):
  """Setup python related env variables."""
  # Get PYTHON_BIN_PATH, default is the current running python.
  default_python_bin_path = sys.executable
  ask_python_bin_path = ('Please specify the location of python. [Default is '
                         '{}]: ').format(default_python_bin_path)
  while True:
    python_bin_path = get_from_env_or_user_or_default(environ_cp,
                                                      'PYTHON_BIN_PATH',
                                                      ask_python_bin_path,
                                                      default_python_bin_path)
    # Check if the path is valid
    if os.path.isfile(python_bin_path) and os.access(python_bin_path, os.X_OK):
      break
    elif not os.path.exists(python_bin_path):
      print('Invalid python path: {} cannot be found.'.format(python_bin_path))
    else:
      print('{} is not executable.  Is it the python binary?'.format(
          python_bin_path))
    environ_cp['PYTHON_BIN_PATH'] = ''

  # Convert python path to Windows style before checking lib and version
  if is_windows() or is_cygwin():
    python_bin_path = cygpath(python_bin_path)

  # Get PYTHON_LIB_PATH
  python_lib_path = environ_cp.get('PYTHON_LIB_PATH')
  if not python_lib_path:
    python_lib_paths = get_python_path(environ_cp, python_bin_path)
    if environ_cp.get('USE_DEFAULT_PYTHON_LIB_PATH') == '1':
      python_lib_path = python_lib_paths[0]
    else:
      print('Found possible Python library paths:\n  %s' %
            '\n  '.join(python_lib_paths))
      default_python_lib_path = python_lib_paths[0]
      python_lib_path = get_input(
          'Please input the desired Python library path to use.  '
          'Default is [{}]\n'.format(python_lib_paths[0]))
      if not python_lib_path:
        python_lib_path = default_python_lib_path
    environ_cp['PYTHON_LIB_PATH'] = python_lib_path

  python_major_version = get_python_major_version(python_bin_path)
  if python_major_version == '2':
    write_to_bazelrc('build --host_force_python=PY2')

  # Convert python path to Windows style before writing into bazel.rc
  if is_windows() or is_cygwin():
    python_lib_path = cygpath(python_lib_path)

  # Set-up env variables used by python_configure.bzl
  write_action_env_to_bazelrc('PYTHON_BIN_PATH', python_bin_path)
  write_action_env_to_bazelrc('PYTHON_LIB_PATH', python_lib_path)
  write_to_bazelrc('build --python_path=\"{}"'.format(python_bin_path))
  environ_cp['PYTHON_BIN_PATH'] = python_bin_path

  # If chosen python_lib_path is from a path specified in the PYTHONPATH
  # variable, need to tell bazel to include PYTHONPATH
  if environ_cp.get('PYTHONPATH'):
    python_paths = environ_cp.get('PYTHONPATH').split(':')
    if python_lib_path in python_paths:
      write_action_env_to_bazelrc('PYTHONPATH', environ_cp.get('PYTHONPATH'))

  # Write tools/python_bin_path.sh
  with open(
      os.path.join(_TF_WORKSPACE_ROOT, 'tools', 'python_bin_path.sh'),
      'w') as f:
    f.write('export PYTHON_BIN_PATH="{}"'.format(python_bin_path))


def reset_tf_configure_bazelrc():
  """Reset file that contains customized config settings."""
  open(_TF_BAZELRC, 'w').close()


def cleanup_makefile():
  """Delete any leftover BUILD files from the Makefile build.

  These files could interfere with Bazel parsing.
  """
  makefile_download_dir = os.path.join(_TF_WORKSPACE_ROOT, 'tensorflow',
                                       'contrib', 'makefile', 'downloads')
  if os.path.isdir(makefile_download_dir):
    for root, _, filenames in os.walk(makefile_download_dir):
      for f in filenames:
        if f.endswith('BUILD'):
          os.remove(os.path.join(root, f))


def get_var(environ_cp,
            var_name,
            query_item,
            enabled_by_default,
            question=None,
            yes_reply=None,
            no_reply=None):
  """Get boolean input from user.

  If var_name is not set in env, ask user to enable query_item or not. If the
  response is empty, use the default.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    query_item: string for feature related to the variable, e.g. "CUDA for
      Nvidia GPUs".
    enabled_by_default: boolean for default behavior.
    question: optional string for how to ask for user input.
    yes_reply: optional string for reply when feature is enabled.
    no_reply: optional string for reply when feature is disabled.

  Returns:
    boolean value of the variable.

  Raises:
    UserInputError: if an environment variable is set, but it cannot be
      interpreted as a boolean indicator, assume that the user has made a
      scripting error, and will continue to provide invalid input.
      Raise the error to avoid infinitely looping.
  """
  if not question:
    question = 'Do you wish to build TensorFlow with {} support?'.format(
        query_item)
  if not yes_reply:
    yes_reply = '{} support will be enabled for TensorFlow.'.format(query_item)
  if not no_reply:
    no_reply = 'No {}'.format(yes_reply)

  yes_reply += '\n'
  no_reply += '\n'

  if enabled_by_default:
    question += ' [Y/n]: '
  else:
    question += ' [y/N]: '

  var = environ_cp.get(var_name)
  if var is not None:
    var_content = var.strip().lower()
    true_strings = ('1', 't', 'true', 'y', 'yes')
    false_strings = ('0', 'f', 'false', 'n', 'no')
    if var_content in true_strings:
      var = True
    elif var_content in false_strings:
      var = False
    else:
      raise UserInputError(
          'Environment variable %s must be set as a boolean indicator.\n'
          'The following are accepted as TRUE : %s.\n'
          'The following are accepted as FALSE: %s.\n'
          'Current value is %s.' %
          (var_name, ', '.join(true_strings), ', '.join(false_strings), var))

  while var is None:
    user_input_origin = get_input(question)
    user_input = user_input_origin.strip().lower()
    if user_input == 'y':
      print(yes_reply)
      var = True
    elif user_input == 'n':
      print(no_reply)
      var = False
    elif not user_input:
      if enabled_by_default:
        print(yes_reply)
        var = True
      else:
        print(no_reply)
        var = False
    else:
      print('Invalid selection: {}'.format(user_input_origin))
  return var


def set_action_env_var(environ_cp,
                       var_name,
                       query_item,
                       enabled_by_default,
                       question=None,
                       yes_reply=None,
                       no_reply=None,
                       bazel_config_name=None):
  """Set boolean action_env variable.

  Ask user if query_item will be enabled. Default is used if no input is given.
  Set environment variable and write to .bazelrc.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    query_item: string for feature related to the variable, e.g. "CUDA for
      Nvidia GPUs".
    enabled_by_default: boolean for default behavior.
    question: optional string for how to ask for user input.
    yes_reply: optional string for reply when feature is enabled.
    no_reply: optional string for reply when feature is disabled.
    bazel_config_name: adding config to .bazelrc instead of action_env.
  """
  var = int(
      get_var(environ_cp, var_name, query_item, enabled_by_default, question,
              yes_reply, no_reply))

  if not bazel_config_name:
    write_action_env_to_bazelrc(var_name, var)
  elif var:
    write_to_bazelrc('build --config=%s' % bazel_config_name)
  environ_cp[var_name] = str(var)


def convert_version_to_int(version):
  """Convert a version number to a integer that can be used to compare.

  Version strings of the form X.YZ and X.Y.Z-xxxxx are supported. The
  'xxxxx' part, for instance 'homebrew' on OS/X, is ignored.

  Args:
    version: a version to be converted

  Returns:
    An integer if converted successfully, otherwise return None.
  """
  version = version.split('-')[0]
  version_segments = version.split('.')
  # Treat "0.24" as "0.24.0"
  if len(version_segments) == 2:
    version_segments.append('0')
  for seg in version_segments:
    if not seg.isdigit():
      return None

  version_str = ''.join(['%03d' % int(seg) for seg in version_segments])
  return int(version_str)


def retrieve_bazel_version():
  """Retrieve installed bazel version (or bazelisk).

  Returns:
    The bazel version detected.
  """
  bazel_executable = shutil.which('bazel')
  if bazel_executable is None:
    bazel_executable = shutil.which('bazelisk')
    if bazel_executable is None:
      print('Cannot find bazel. Please install bazel/bazelisk.')
      sys.exit(1)

  stderr = open(os.devnull, 'wb')
  curr_version = run_shell([bazel_executable, '--version'],
                           allow_non_zero=True,
                           stderr=stderr)
  if curr_version.startswith('bazel '):
    curr_version = curr_version.split('bazel ')[1]

  curr_version_int = convert_version_to_int(curr_version)

  # Check if current bazel version can be detected properly.
  if not curr_version_int:
    print('WARNING: current bazel installation is not a release version.')
    return curr_version

  print('You have bazel %s installed.' % curr_version)
  return curr_version


def set_cc_opt_flags(environ_cp):
  """Set up architecture-dependent optimization flags.

  Also append CC optimization flags to bazel.rc..

  Args:
    environ_cp: copy of the os.environ.
  """
  if is_ppc64le():
    # gcc on ppc64le does not support -march, use mcpu instead
    default_cc_opt_flags = '-mcpu=native'
  elif is_windows():
    default_cc_opt_flags = '/arch:AVX'
  else:
    # On all other platforms, no longer use `-march=native` as this can result
    # in instructions that are too modern being generated. Users that want
    # maximum performance should compile TF in their environment and can pass
    # `-march=native` there.
    # See https://github.com/tensorflow/tensorflow/issues/45744 and duplicates
    default_cc_opt_flags = '-Wno-sign-compare'
  question = ('Please specify optimization flags to use during compilation when'
              ' bazel option "--config=opt" is specified [Default is %s]: '
             ) % default_cc_opt_flags
  cc_opt_flags = get_from_env_or_user_or_default(environ_cp, 'CC_OPT_FLAGS',
                                                 question, default_cc_opt_flags)
  for opt in cc_opt_flags.split():
    write_to_bazelrc('build:opt --copt=%s' % opt)
    write_to_bazelrc('build:opt --host_copt=%s' % opt)


def set_tf_cuda_clang(environ_cp):
  """set TF_CUDA_CLANG action_env.

  Args:
    environ_cp: copy of the os.environ.
  """
  question = 'Do you want to use clang as CUDA compiler?'
  yes_reply = 'Clang will be used as CUDA compiler.'
  no_reply = 'nvcc will be used as CUDA compiler.'
  set_action_env_var(
      environ_cp,
      'TF_CUDA_CLANG',
      None,
      True,
      question=question,
      yes_reply=yes_reply,
      no_reply=no_reply,
      bazel_config_name='cuda_clang',
  )


def set_tf_download_clang(environ_cp):
  """Set TF_DOWNLOAD_CLANG action_env."""
  question = 'Do you wish to download a fresh release of clang? (Experimental)'
  yes_reply = 'Clang will be downloaded and used to compile tensorflow.'
  no_reply = 'Clang will not be downloaded.'
  set_action_env_var(
      environ_cp,
      'TF_DOWNLOAD_CLANG',
      None,
      False,
      question=question,
      yes_reply=yes_reply,
      no_reply=no_reply,
      bazel_config_name='download_clang')


def get_from_env_or_user_or_default(environ_cp, var_name, ask_for_var,
                                    var_default):
  """Get var_name either from env, or user or default.

  If var_name has been set as environment variable, use the preset value, else
  ask for user input. If no input is provided, the default is used.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    ask_for_var: string for how to ask for user input.
    var_default: default value string.

  Returns:
    string value for var_name
  """
  var = environ_cp.get(var_name)
  if not var:
    var = get_input(ask_for_var)
    print('\n')
  if not var:
    var = var_default
  return var


def prompt_loop_or_load_from_env(environ_cp,
                                 var_name,
                                 var_default,
                                 ask_for_var,
                                 check_success,
                                 error_msg,
                                 suppress_default_error=False,
                                 resolve_symlinks=False,
                                 n_ask_attempts=_DEFAULT_PROMPT_ASK_ATTEMPTS):
  """Loop over user prompts for an ENV param until receiving a valid response.

  For the env param var_name, read from the environment or verify user input
  until receiving valid input. When done, set var_name in the environ_cp to its
  new value.

  Args:
    environ_cp: (Dict) copy of the os.environ.
    var_name: (String) string for name of environment variable, e.g. "TF_MYVAR".
    var_default: (String) default value string.
    ask_for_var: (String) string for how to ask for user input.
    check_success: (Function) function that takes one argument and returns a
      boolean. Should return True if the value provided is considered valid. May
      contain a complex error message if error_msg does not provide enough
      information. In that case, set suppress_default_error to True.
    error_msg: (String) String with one and only one '%s'. Formatted with each
      invalid response upon check_success(input) failure.
    suppress_default_error: (Bool) Suppress the above error message in favor of
      one from the check_success function.
    resolve_symlinks: (Bool) Translate symbolic links into the real filepath.
    n_ask_attempts: (Integer) Number of times to query for valid input before
      raising an error and quitting.

  Returns:
    [String] The value of var_name after querying for input.

  Raises:
    UserInputError: if a query has been attempted n_ask_attempts times without
      success, assume that the user has made a scripting error, and will
      continue to provide invalid input. Raise the error to avoid infinitely
      looping.
  """
  default = environ_cp.get(var_name) or var_default
  full_query = '%s [Default is %s]: ' % (
      ask_for_var,
      default,
  )

  for _ in range(n_ask_attempts):
    val = get_from_env_or_user_or_default(environ_cp, var_name, full_query,
                                          default)
    if check_success(val):
      break
    if not suppress_default_error:
      print(error_msg % val)
    environ_cp[var_name] = ''
  else:
    raise UserInputError('Invalid %s setting was provided %d times in a row. '
                         'Assuming to be a scripting mistake.' %
                         (var_name, n_ask_attempts))

  if resolve_symlinks:
    val = os.path.realpath(val)
  environ_cp[var_name] = val
  return val


def set_clang_cuda_compiler_path(environ_cp):
  """Set CLANG_CUDA_COMPILER_PATH."""
  # Upon clang 19 drop the check for 16
  default_clang_path = '/usr/lib/llvm-18/bin/clang'
  if not os.path.exists(default_clang_path):
    default_clang_path = '/usr/lib/llvm-17/bin/clang'
    if not os.path.exists(default_clang_path):
      default_clang_path = '/usr/lib/llvm-16/bin/clang'
    if not os.path.exists(default_clang_path):
      default_clang_path = shutil.which('clang') or ''

  clang_cuda_compiler_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='CLANG_CUDA_COMPILER_PATH',
      var_default=default_clang_path,
      ask_for_var='Please specify clang path that to be used as host compiler.',
      check_success=os.path.exists,
      resolve_symlinks=True,
      error_msg='Invalid clang path. %s cannot be found.',
  )

  # Set CLANG_CUDA_COMPILER_PATH
  environ_cp['CLANG_CUDA_COMPILER_PATH'] = clang_cuda_compiler_path
  write_action_env_to_bazelrc('CLANG_CUDA_COMPILER_PATH',
                              clang_cuda_compiler_path)
  return clang_cuda_compiler_path


def create_android_ndk_rule(environ_cp):
  """Set ANDROID_NDK_HOME and write Android NDK WORKSPACE rule."""
  if is_windows() or is_cygwin():
    default_ndk_path = cygpath('%s/Android/Sdk/ndk-bundle' %
                               environ_cp['APPDATA'])
  elif is_macos():
    default_ndk_path = '%s/library/Android/Sdk/ndk-bundle' % environ_cp['HOME']
  else:
    default_ndk_path = '%s/Android/Sdk/ndk-bundle' % environ_cp['HOME']

  def valid_ndk_path(path):
    return (os.path.exists(path) and
            os.path.exists(os.path.join(path, 'source.properties')))

  android_ndk_home_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_NDK_HOME',
      var_default=default_ndk_path,
      ask_for_var='Please specify the home path of the Android NDK to use.',
      check_success=valid_ndk_path,
      error_msg=('The path %s or its child file "source.properties" '
                 'does not exist.'))
  write_action_env_to_bazelrc('ANDROID_NDK_HOME', android_ndk_home_path)
  write_action_env_to_bazelrc(
      'ANDROID_NDK_API_LEVEL',
      get_ndk_api_level(environ_cp, android_ndk_home_path))


def create_android_sdk_rule(environ_cp):
  """Set Android variables and write Android SDK WORKSPACE rule."""
  if is_windows() or is_cygwin():
    default_sdk_path = cygpath('%s/Android/Sdk' % environ_cp['APPDATA'])
  elif is_macos():
    default_sdk_path = '%s/library/Android/Sdk' % environ_cp['HOME']
  else:
    default_sdk_path = '%s/Android/Sdk' % environ_cp['HOME']

  def valid_sdk_path(path):
    return (os.path.exists(path) and
            os.path.exists(os.path.join(path, 'platforms')) and
            os.path.exists(os.path.join(path, 'build-tools')))

  android_sdk_home_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_SDK_HOME',
      var_default=default_sdk_path,
      ask_for_var='Please specify the home path of the Android SDK to use.',
      check_success=valid_sdk_path,
      error_msg=('Either %s does not exist, or it does not contain the '
                 'subdirectories "platforms" and "build-tools".'))

  platforms = os.path.join(android_sdk_home_path, 'platforms')
  api_levels = sorted(os.listdir(platforms))
  api_levels = [x.replace('android-', '') for x in api_levels]

  def valid_api_level(api_level):
    return os.path.exists(
        os.path.join(android_sdk_home_path, 'platforms',
                     'android-' + api_level))

  android_api_level = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_API_LEVEL',
      var_default=api_levels[-1],
      ask_for_var=('Please specify the Android SDK API level to use. '
                   '[Available levels: %s]') % api_levels,
      check_success=valid_api_level,
      error_msg='Android-%s is not present in the SDK path.')

  build_tools = os.path.join(android_sdk_home_path, 'build-tools')
  versions = sorted(os.listdir(build_tools))

  def valid_build_tools(version):
    return os.path.exists(
        os.path.join(android_sdk_home_path, 'build-tools', version))

  android_build_tools_version = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_BUILD_TOOLS_VERSION',
      var_default=versions[-1],
      ask_for_var=('Please specify an Android build tools version to use. '
                   '[Available versions: %s]') % versions,
      check_success=valid_build_tools,
      error_msg=('The selected SDK does not have build-tools version %s '
                 'available.'))

  write_action_env_to_bazelrc('ANDROID_BUILD_TOOLS_VERSION',
                              android_build_tools_version)
  write_action_env_to_bazelrc('ANDROID_SDK_API_LEVEL', android_api_level)
  write_action_env_to_bazelrc('ANDROID_SDK_HOME', android_sdk_home_path)


def get_ndk_api_level(environ_cp, android_ndk_home_path):
  """Gets the appropriate NDK API level to use for the provided Android NDK path.
  """

  # First check to see if we're using a blessed version of the NDK.
  properties_path = '%s/source.properties' % android_ndk_home_path
  if is_windows() or is_cygwin():
    properties_path = cygpath(properties_path)
  with open(properties_path, 'r') as f:
    filedata = f.read()

  revision = re.search(r'Pkg.Revision = (\d+)', filedata)
  if revision:
    ndk_version = revision.group(1)
  else:
    raise Exception('Unable to parse NDK revision.')
  if int(ndk_version) not in _SUPPORTED_ANDROID_NDK_VERSIONS:
    print('WARNING: The NDK version in %s is %s, which is not '
          'supported by Bazel (officially supported versions: %s). Please use '
          'another version. Compiling Android targets may result in confusing '
          'errors.\n' %
          (android_ndk_home_path, ndk_version, _SUPPORTED_ANDROID_NDK_VERSIONS))
  write_action_env_to_bazelrc('ANDROID_NDK_VERSION', ndk_version)

  # Now grab the NDK API level to use. Note that this is different from the
  # SDK API level, as the NDK API level is effectively the *min* target SDK
  # version.
  meta = open(os.path.join(android_ndk_home_path, 'meta/platforms.json'))
  platforms = json.load(meta)
  meta.close()
  aliases = platforms['aliases']
  api_levels = sorted(list(set([aliases[i] for i in aliases])))

  android_ndk_api_level = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_NDK_API_LEVEL',
      var_default='21',  # 21 is required for ARM64 support.
      ask_for_var=(
          'Please specify the (min) Android NDK API level to use. '
          '[Available levels: %s]'
      )
      % api_levels,
      check_success=(lambda *_: True),
      error_msg='Android-%s is not present in the NDK path.',
  )

  return android_ndk_api_level


def set_gcc_host_compiler_path(environ_cp):
  """Set GCC_HOST_COMPILER_PATH."""
  default_gcc_host_compiler_path = shutil.which('gcc') or ''

  gcc_host_compiler_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='GCC_HOST_COMPILER_PATH',
      var_default=default_gcc_host_compiler_path,
      ask_for_var='Please specify which gcc should be used by nvcc as the host '
      'compiler.',
      check_success=os.path.exists,
      resolve_symlinks=True,
      error_msg='Invalid gcc path. %s cannot be found.',
  )

  write_action_env_to_bazelrc('GCC_HOST_COMPILER_PATH', gcc_host_compiler_path)


def choose_compiler(environ_cp):
  question = 'Do you want to use Clang to build TensorFlow?'
  yes_reply = 'Clang will be used to compile TensorFlow.'
  no_reply = 'GCC will be used to compile TensorFlow.'
  var = int(
      get_var(
          environ_cp, 'TF_NEED_CLANG', None, True, question, yes_reply, no_reply
      )
  )
  return var


def choose_compiler_Win(environ_cp):
  question = 'Do you want to use Clang to build TensorFlow?'
  yes_reply = 'Add "--config=win_clang" to compile TensorFlow with CLANG.'
  no_reply = 'MSVC will be used to compile TensorFlow.'
  var = int(
      get_var(
          environ_cp, 'TF_NEED_CLANG', None, True, question, yes_reply, no_reply
      )
  )
  return var


def set_clang_compiler_path(environ_cp):
  """Set CLANG_COMPILER_PATH and environment variables.

  Loop over user prompts for clang path until receiving a valid response.
  Default is used if no input is given. Set CLANG_COMPILER_PATH and write
  environment variables CC and BAZEL_COMPILER to .bazelrc.

  Args:
    environ_cp: (Dict) copy of the os.environ.

  Returns:
    string value for clang_compiler_path.
  """
  # Default path if clang-18 is installed by using apt-get install
  # remove 16 logic upon release of 19
  default_clang_path = '/usr/lib/llvm-18/bin/clang'
  if not os.path.exists(default_clang_path):
    default_clang_path = '/usr/lib/llvm-17/bin/clang'
    if not os.path.exists(default_clang_path):
      default_clang_path = '/usr/lib/llvm-16/bin/clang'
    if not os.path.exists(default_clang_path):
      default_clang_path = shutil.which('clang') or ''

  clang_compiler_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='CLANG_COMPILER_PATH',
      var_default=default_clang_path,
      ask_for_var='Please specify the path to clang executable.',
      check_success=os.path.exists,
      resolve_symlinks=True,
      error_msg=(
          'Invalid clang path. %s cannot be found. Note that TensorFlow now'
          ' requires clang to compile. You may override this behavior by'
          ' setting TF_NEED_CLANG=0'
      ),
  )

  write_action_env_to_bazelrc('CLANG_COMPILER_PATH', clang_compiler_path)
  write_to_bazelrc('build --repo_env=CC=%s' % clang_compiler_path)
  write_to_bazelrc('build --repo_env=BAZEL_COMPILER=%s' % clang_compiler_path)

  return clang_compiler_path


def set_clang_compiler_path_win(environ_cp):
  """Set CLANG_COMPILER_PATH and environment variables.

  Loop over user prompts for clang path until receiving a valid response.
  Default is used if no input is given. Set CLANG_COMPILER_PATH and write
  environment variables CC and BAZEL_COMPILER to .bazelrc.

  Args:
    environ_cp: (Dict) copy of the os.environ.

  Returns:
    string value for clang_compiler_path.
  """
  # Default path if clang-16 is installed by using apt-get install
  default_clang_path = 'C:/Program Files/LLVM/bin/clang.exe'
  if not os.path.exists(default_clang_path):
    default_clang_path = shutil.which('clang') or ''

  clang_compiler_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='CLANG_COMPILER_PATH',
      var_default=default_clang_path,
      ask_for_var='Please specify the path to clang executable.',
      check_success=os.path.exists,
      resolve_symlinks=True,
      error_msg=(
          'Invalid clang path. %s cannot be found. Note that Clang is now'
          'preferred compiler. You may use MSVC by removing --config=win_clang'
      ),
  )

  write_action_env_to_bazelrc('CLANG_COMPILER_PATH', clang_compiler_path)
  write_to_bazelrc(f'build --repo_env=CC="{clang_compiler_path}"')
  write_to_bazelrc(f'build --repo_env=BAZEL_COMPILER="{clang_compiler_path}"')

  return clang_compiler_path


def retrieve_clang_version(clang_executable):
  """Retrieve installed clang version.

  Args:
    clang_executable: (String) path to clang executable

  Returns:
    The clang version detected.
  """
  stderr = open(os.devnull, 'wb')
  curr_version = run_shell([clang_executable, '--version'],
                           allow_non_zero=True,
                           stderr=stderr)

  curr_version_split = curr_version.lower().split('clang version ')
  if len(curr_version_split) > 1:
    curr_version = curr_version_split[1].split()[0].split('git')

  if len(curr_version) > 1:
    print('WARNING: current clang installation is not a release version.\n')

  curr_version = curr_version[0]
  curr_version_int = convert_version_to_int(curr_version)
  # Check if current clang version can be detected properly.
  if not curr_version_int:
    print('WARNING: current clang installation version unknown.\n')
    return None

  print('You have Clang %s installed.\n' % curr_version)
  return curr_version


# Disable clang extension that rejects type definitions within offsetof.
# This was added in clang-16 by https://reviews.llvm.org/D133574.
# Still required for clang-17.
# Can be removed once upb is updated, since a type definition is used within
# offset of in the current version of ubp. See
# https://github.com/protocolbuffers/upb/blob/9effcbcb27f0a665f9f345030188c0b291e32482/upb/upb.c#L183.
def disable_clang_offsetof_extension(clang_version):
  if int(clang_version.split('.')[0]) in (16, 17):
    write_to_bazelrc('build --copt=-Wno-gnu-offsetof-extensions')


def set_hermetic_cuda_version(environ_cp):
  """Set HERMETIC_CUDA_VERSION."""
  ask_cuda_version = (
      'Please specify the hermetic CUDA version you want to use '
      'or leave empty to use the default version. '
  )
  hermetic_cuda_version = get_from_env_or_user_or_default(
      environ_cp, 'HERMETIC_CUDA_VERSION', ask_cuda_version, None
  )
  if hermetic_cuda_version:
    environ_cp['HERMETIC_CUDA_VERSION'] = hermetic_cuda_version
    write_repo_env_to_bazelrc(
        'cuda', 'HERMETIC_CUDA_VERSION', hermetic_cuda_version
    )


def set_hermetic_cudnn_version(environ_cp):
  """Set HERMETIC_CUDNN_VERSION."""
  ask_cudnn_version = (
      'Please specify the hermetic cuDNN version you want to use '
      'or leave empty to use the default version. '
  )
  hermetic_cudnn_version = get_from_env_or_user_or_default(
      environ_cp, 'HERMETIC_CUDNN_VERSION', ask_cudnn_version, None
  )
  if hermetic_cudnn_version:
    environ_cp['HERMETIC_CUDNN_VERSION'] = hermetic_cudnn_version
    write_repo_env_to_bazelrc(
        'cuda', 'HERMETIC_CUDNN_VERSION', hermetic_cudnn_version
    )


def set_hermetic_cuda_compute_capabilities(environ_cp):
  """Set HERMETIC_CUDA_COMPUTE_CAPABILITIES."""
  while True:
    default_cuda_compute_capabilities = _DEFAULT_CUDA_COMPUTE_CAPABILITIES

    ask_cuda_compute_capabilities = (
        'Please specify a list of comma-separated CUDA compute capabilities '
        'you want to build with.\nYou can find the compute capability of your '
        'device at: https://developer.nvidia.com/cuda-gpus. Each capability '
        'can be specified as "x.y" or "compute_xy" to include both virtual and'
        ' binary GPU code, or as "sm_xy" to only include the binary '
        'code.\nPlease note that each additional compute capability '
        'significantly increases your build time and binary size, and that '
        'TensorFlow only supports compute capabilities >= 3.5 [Default is: '
        '%s]: ' % default_cuda_compute_capabilities)
    hermetic_cuda_compute_capabilities = get_from_env_or_user_or_default(
        environ_cp,
        'HERMETIC_CUDA_COMPUTE_CAPABILITIES',
        ask_cuda_compute_capabilities,
        default_cuda_compute_capabilities,
    )
    # Check whether all capabilities from the input is valid
    all_valid = True
    # Remove all whitespace characters before splitting the string
    # that users may insert by accident, as this will result in error
    hermetic_cuda_compute_capabilities = ''.join(
        hermetic_cuda_compute_capabilities.split()
    )
    for compute_capability in hermetic_cuda_compute_capabilities.split(','):
      m = re.match('[0-9]+.[0-9]+', compute_capability)
      if not m:
        # We now support sm_35,sm_50,sm_60,compute_70.
        sm_compute_match = re.match('(sm|compute)_?([0-9]+[0-9]+)',
                                    compute_capability)
        if not sm_compute_match:
          print('Invalid compute capability: %s' % compute_capability)
          all_valid = False
        else:
          ver = int(sm_compute_match.group(2))
          if ver < 30:
            print(
                'ERROR: TensorFlow only supports small CUDA compute'
                ' capabilities of sm_30 and higher. Please re-specify the list'
                ' of compute capabilities excluding version %s.' % ver)
            all_valid = False
          if ver < 35:
            print('WARNING: XLA does not support CUDA compute capabilities '
                  'lower than sm_35. Disable XLA when running on older GPUs.')
      else:
        ver = float(m.group(0))
        if ver < 3.0:
          print('ERROR: TensorFlow only supports CUDA compute capabilities 3.0 '
                'and higher. Please re-specify the list of compute '
                'capabilities excluding version %s.' % ver)
          all_valid = False
        if ver < 3.5:
          print('WARNING: XLA does not support CUDA compute capabilities '
                'lower than 3.5. Disable XLA when running on older GPUs.')

    if all_valid:
      break

    # Reset and Retry
    environ_cp['HERMETIC_CUDA_COMPUTE_CAPABILITIES'] = ''

  # Set HERMETIC_CUDA_COMPUTE_CAPABILITIES
  environ_cp['HERMETIC_CUDA_COMPUTE_CAPABILITIES'] = (
      hermetic_cuda_compute_capabilities
  )
  write_repo_env_to_bazelrc(
      'cuda',
      'HERMETIC_CUDA_COMPUTE_CAPABILITIES',
      hermetic_cuda_compute_capabilities,
  )


def set_cuda_local_path(environ_cp, dist_name, env_var):
  ask_path = (
      'Please specify the local {} path you want to use '
      'or leave empty to use the default version. '
  ).format(dist_name)
  local_path = get_from_env_or_user_or_default(
      environ_cp, env_var, ask_path, None
  )
  if local_path:
    environ_cp[env_var] = local_path
    write_repo_env_to_bazelrc('cuda', env_var, local_path)


def set_other_cuda_vars(environ_cp):
  """Set other CUDA related variables."""
  # If CUDA is enabled, always use GPU during build and test.
  if environ_cp.get('TF_CUDA_CLANG') == '1':
    write_to_bazelrc('build --config=cuda_clang')
  else:
    write_to_bazelrc('build --config=cuda')


def system_specific_test_config(environ_cp):
  """Add default build and test flags required for TF tests to bazelrc."""
  write_to_bazelrc('test --test_size_filters=small,medium')

  # Each instance of --test_tag_filters or --build_tag_filters overrides all
  # previous instances, so we need to build up a complete list and write a
  # single list of filters for the .bazelrc file.

  # Filters to use with both --test_tag_filters and --build_tag_filters
  test_and_build_filters = ['-benchmark-test', '-no_oss', '-oss_excluded']
  # Additional filters for --test_tag_filters beyond those in
  # test_and_build_filters
  test_only_filters = ['-oss_serial']
  if is_windows():
    test_and_build_filters += ['-no_windows', '-windows_excluded']
    if ((environ_cp.get('TF_NEED_CUDA', None) == '1') or
        (environ_cp.get('TF_NEED_ROCM', None) == '1')):
      test_and_build_filters += ['-no_windows_gpu', '-no_gpu']
    else:
      test_and_build_filters.append('-gpu')
  elif is_macos():
    test_and_build_filters += ['-gpu', '-nomac', '-no_mac', '-mac_excluded']
  elif is_linux():
    if ((environ_cp.get('TF_NEED_CUDA', None) == '1') or
        (environ_cp.get('TF_NEED_ROCM', None) == '1')):
      test_and_build_filters.append('-no_gpu')
      write_to_bazelrc('test --test_env=LD_LIBRARY_PATH')
    else:
      test_and_build_filters.append('-gpu')

  # Disable tests with "v1only" tag in "v2" Bazel config, but not in "v1" config
  write_to_bazelrc('test:v1 --test_tag_filters=%s' %
                   ','.join(test_and_build_filters + test_only_filters))
  write_to_bazelrc('test:v1 --build_tag_filters=%s' %
                   ','.join(test_and_build_filters))
  write_to_bazelrc(
      'test:v2 --test_tag_filters=%s' %
      ','.join(test_and_build_filters + test_only_filters + ['-v1only']))
  write_to_bazelrc('test:v2 --build_tag_filters=%s' %
                   ','.join(test_and_build_filters + ['-v1only']))


def set_system_libs_flag(environ_cp):
  """Set system libs flags."""
  syslibs = environ_cp.get('TF_SYSTEM_LIBS', '')

  if is_s390x() and 'boringssl' not in syslibs:
    syslibs = 'boringssl' + (', ' + syslibs if syslibs else '')

  if syslibs:
    if ',' in syslibs:
      syslibs = ','.join(sorted(syslibs.split(',')))
    else:
      syslibs = ','.join(sorted(syslibs.split()))
    write_action_env_to_bazelrc('TF_SYSTEM_LIBS', syslibs)

  for varname in ('PREFIX', 'LIBDIR', 'INCLUDEDIR', 'PROTOBUF_INCLUDE_PATH'):
    if varname in environ_cp:
      write_to_bazelrc('build --define=%s=%s' % (varname, environ_cp[varname]))


def set_windows_build_flags(environ_cp):
  """Set Windows specific build options."""

  # First available in VS 16.4. Speeds up Windows compile times by a lot. See
  # https://groups.google.com/a/tensorflow.org/d/topic/build/SsW98Eo7l3o/discussion
  # pylint: disable=line-too-long
  write_to_bazelrc(
      'build --copt=/d2ReducedOptimizeHugeFunctions --host_copt=/d2ReducedOptimizeHugeFunctions'
  )

  if get_var(
      environ_cp, 'TF_OVERRIDE_EIGEN_STRONG_INLINE', 'Eigen strong inline',
      True, ('Would you like to override eigen strong inline for some C++ '
             'compilation to reduce the compilation time?'),
      'Eigen strong inline overridden.', 'Not overriding eigen strong inline, '
      'some compilations could take more than 20 mins.'):
    # Due to a known MSVC compiler issue
    # https://github.com/tensorflow/tensorflow/issues/10521
    # Overriding eigen strong inline speeds up the compiling of
    # conv_grad_ops_3d.cc and conv_ops_3d.cc by 20 minutes,
    # but this also hurts the performance. Let users decide what they want.
    write_to_bazelrc('build --define=override_eigen_strong_inline=true')


def config_info_line(name, help_text):
  """Helper function to print formatted help text for Bazel config options."""
  print('\t--config=%-12s\t# %s' % (name, help_text))


def configure_ios(environ_cp):
  """Configures TensorFlow for iOS builds."""
  if not is_macos():
    return
  if not get_var(environ_cp, 'TF_CONFIGURE_IOS', 'iOS', False):
    return
  for filepath in APPLE_BAZEL_FILES:
    existing_filepath = os.path.join(_TF_WORKSPACE_ROOT, filepath + '.apple')
    renamed_filepath = os.path.join(_TF_WORKSPACE_ROOT, filepath)
    symlink_force(existing_filepath, renamed_filepath)
  for filepath in IOS_FILES:
    filename = os.path.basename(filepath)
    new_filepath = os.path.join(_TF_WORKSPACE_ROOT, filename)
    symlink_force(filepath, new_filepath)


def get_gcc_compiler(environ_cp):
  gcc_env = environ_cp.get('CXX') or environ_cp.get('CC') or shutil.which('gcc')
  if gcc_env is not None:
    gcc_version = run_shell([gcc_env, '--version']).split()
    if gcc_version[0] in ('gcc', 'g++'):
      return gcc_env
  return None


def main():
  global _TF_WORKSPACE_ROOT
  global _TF_BAZELRC
  global _TF_CURRENT_BAZEL_VERSION

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--workspace',
      type=str,
      default=os.path.abspath(os.path.dirname(__file__)),
      help='The absolute path to your active Bazel workspace.')
  args = parser.parse_args()

  _TF_WORKSPACE_ROOT = args.workspace
  _TF_BAZELRC = os.path.join(_TF_WORKSPACE_ROOT, _TF_BAZELRC_FILENAME)

  # Make a copy of os.environ to be clear when functions and getting and setting
  # environment variables.
  environ_cp = dict(os.environ)

  try:
    current_bazel_version = retrieve_bazel_version()
  except subprocess.CalledProcessError as e:
    print('Error retrieving bazel version: ', e.output.decode('UTF-8').strip())
    raise e

  _TF_CURRENT_BAZEL_VERSION = convert_version_to_int(current_bazel_version)

  reset_tf_configure_bazelrc()

  cleanup_makefile()
  setup_python(environ_cp)

  if is_windows():
    environ_cp['TF_NEED_OPENCL'] = '0'
    environ_cp['TF_CUDA_CLANG'] = '0'
    # TODO(ibiryukov): Investigate using clang as a cpu or cuda compiler on
    # Windows.
    environ_cp['TF_DOWNLOAD_CLANG'] = '0'
    environ_cp['TF_NEED_MPI'] = '0'

  if is_ppc64le():
    # Enable MMA Dynamic Dispatch support if 'gcc' and if linker >= 2.35
    gcc_env = get_gcc_compiler(environ_cp)
    if gcc_env is not None:

      # Use gold linker if 'gcc' and if 'ppc64le'
      write_to_bazelrc('build --linkopt="-fuse-ld=gold"')

      # Get the linker version
      ld_version = run_shell([gcc_env, '-Wl,-version']).split()

      ld_version_int = 0
      for i in range(len(ld_version)):
        ld_version_int = convert_version_to_int(ld_version[i])
        if ld_version_int is not None:
          break

      if ld_version_int is None:
        ld_version_int = 0

      # Enable if 'ld' version >= 2.35
      if ld_version_int >= 2035000:
        write_to_bazelrc(
            'build --copt="-DEIGEN_ALTIVEC_ENABLE_MMA_DYNAMIC_DISPATCH=1"')

  with_xla_support = environ_cp.get('TF_ENABLE_XLA', None)
  if with_xla_support is not None:
    write_to_bazelrc('build --define=with_xla_support=%s' %
                     ('true' if int(with_xla_support) else 'false'))

  set_action_env_var(
      environ_cp, 'TF_NEED_ROCM', 'ROCm', False, bazel_config_name='rocm')
  if (environ_cp.get('TF_NEED_ROCM') == '1' and
      'LD_LIBRARY_PATH' in environ_cp and
      environ_cp.get('LD_LIBRARY_PATH') != '1'):
    write_action_env_to_bazelrc('LD_LIBRARY_PATH',
                                environ_cp.get('LD_LIBRARY_PATH'))

  if (environ_cp.get('TF_NEED_ROCM') == '1' and environ_cp.get('ROCM_PATH')):
    write_action_env_to_bazelrc('ROCM_PATH', environ_cp.get('ROCM_PATH'))

  if (environ_cp.get('TF_NEED_ROCM') == '1' and environ_cp.get('HIP_PLATFORM')):
    write_action_env_to_bazelrc('HIP_PLATFORM', environ_cp.get('HIP_PLATFORM'))

  if is_windows():
    print('\nWARNING: Cannot build with CUDA support on Windows.\n'
          'Starting in TF 2.11, CUDA build is not supported for Windows. '
          'For using TensorFlow GPU on Windows, you will need to build/install '
          'TensorFlow in WSL2.\n')
    environ_cp['TF_NEED_CUDA'] = '0'
  else:
    environ_cp['TF_NEED_CUDA'] = str(
        int(get_var(environ_cp, 'TF_NEED_CUDA', 'CUDA', False)))
  if environ_cp.get('TF_NEED_CUDA') == '1':
    set_hermetic_cuda_version(environ_cp)
    set_hermetic_cudnn_version(environ_cp)
    set_hermetic_cuda_compute_capabilities(environ_cp)
    set_cuda_local_path(environ_cp, 'CUDA', 'LOCAL_CUDA_PATH')
    set_cuda_local_path(environ_cp, 'CUDNN', 'LOCAL_CUDNN_PATH')
    set_cuda_local_path(environ_cp, 'NCCL', 'LOCAL_NCCL_PATH')

    if 'LD_LIBRARY_PATH' in environ_cp and environ_cp.get(
        'LD_LIBRARY_PATH') != '1':
      write_action_env_to_bazelrc('LD_LIBRARY_PATH',
                                  environ_cp.get('LD_LIBRARY_PATH'))

    set_tf_cuda_clang(environ_cp)
    if environ_cp.get('TF_CUDA_CLANG') == '1':
      # Set up which clang we should use as the cuda / host compiler.
      clang_cuda_compiler_path = set_clang_cuda_compiler_path(environ_cp)
      clang_version = retrieve_clang_version(clang_cuda_compiler_path)
      disable_clang_offsetof_extension(clang_version)
    else:
      # Set up which gcc nvcc should use as the host compiler
      # No need to set this on Windows
      if not is_windows():
        set_gcc_host_compiler_path(environ_cp)
    set_other_cuda_vars(environ_cp)
  else:
    # CUDA not required. Ask whether we should use clang for the CPU build.
    if is_linux():
      environ_cp['TF_NEED_CLANG'] = str(choose_compiler(environ_cp))
      if environ_cp.get('TF_NEED_CLANG') == '1':
        clang_compiler_path = set_clang_compiler_path(environ_cp)
        clang_version = retrieve_clang_version(clang_compiler_path)
        disable_clang_offsetof_extension(clang_version)
    if is_windows():
      environ_cp['TF_NEED_CLANG'] = str(choose_compiler_Win(environ_cp))
      if environ_cp.get('TF_NEED_CLANG') == '1':
        clang_compiler_path = set_clang_compiler_path_win(environ_cp)
        clang_version = retrieve_clang_version(clang_compiler_path)
        disable_clang_offsetof_extension(clang_version)

  # ROCm / CUDA are mutually exclusive.
  # At most 1 GPU platform can be configured.
  gpu_platform_count = 0
  if environ_cp.get('TF_NEED_ROCM') == '1':
    gpu_platform_count += 1
  if environ_cp.get('TF_NEED_CUDA') == '1':
    gpu_platform_count += 1
  if gpu_platform_count >= 2:
    raise UserInputError('CUDA / ROCm are mututally exclusive. '
                         'At most 1 GPU platform can be configured.')

  set_cc_opt_flags(environ_cp)
  set_system_libs_flag(environ_cp)
  if is_windows():
    set_windows_build_flags(environ_cp)

  if get_var(environ_cp, 'TF_SET_ANDROID_WORKSPACE', 'android workspace', False,
             ('Would you like to interactively configure ./WORKSPACE for '
              'Android builds?'), 'Searching for NDK and SDK installations.',
             'Not configuring the WORKSPACE for Android builds.'):
    create_android_ndk_rule(environ_cp)
    create_android_sdk_rule(environ_cp)

  system_specific_test_config(environ_cp)

  configure_ios(environ_cp)

  print('Preconfigured Bazel build configs. You can use any of the below by '
        'adding "--config=<>" to your build command. See .bazelrc for more '
        'details.')
  config_info_line('mkl', 'Build with MKL support.')
  config_info_line(
      'mkl_aarch64',
      'Build with oneDNN and Compute Library for the Arm Architecture (ACL).')
  config_info_line('monolithic', 'Config for mostly static monolithic build.')
  config_info_line('numa', 'Build with NUMA support.')
  config_info_line(
      'dynamic_kernels',
      '(Experimental) Build kernels into separate shared objects.')
  config_info_line('v1', 'Build with TensorFlow 1 API instead of TF 2 API.')

  print('Preconfigured Bazel build configs to DISABLE default on features:')
  config_info_line('nogcp', 'Disable GCP support.')
  config_info_line('nonccl', 'Disable NVIDIA NCCL support.')


if __name__ == '__main__':
  main()
