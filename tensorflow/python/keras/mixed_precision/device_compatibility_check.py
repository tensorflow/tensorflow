# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Contains function to log if devices are compatible with mixed precision."""

import itertools

from tensorflow.python.framework import config
from tensorflow.python.platform import tf_logging


_COMPAT_CHECK_PREFIX = 'Mixed precision compatibility check (mixed_float16): '
_COMPAT_CHECK_OK_PREFIX = _COMPAT_CHECK_PREFIX + 'OK'
_COMPAT_CHECK_WARNING_PREFIX = _COMPAT_CHECK_PREFIX + 'WARNING'
_COMPAT_CHECK_WARNING_SUFFIX = (
    'If you will use compatible GPU(s) not attached to this host, e.g. by '
    'running a multi-worker model, you can ignore this warning. This message '
    'will only be logged once')


def _dedup_strings(device_strs):
  """Groups together consecutive identical strings.

  For example, given:
      ['GPU 1', 'GPU 2', 'GPU 2', 'GPU 3', 'GPU 3', 'GPU 3']
  This function returns:
      ['GPU 1', 'GPU 2 (x2)', 'GPU 3 (x3)']

  Args:
    device_strs: A list of strings, each representing a device.

  Returns:
    A copy of the input, but identical consecutive strings are merged into a
    single string.
  """
  new_device_strs = []
  for device_str, vals in itertools.groupby(device_strs):
    num = len(list(vals))
    if num == 1:
      new_device_strs.append(device_str)
    else:
      new_device_strs.append('%s (x%d)' % (device_str, num))
  return new_device_strs


def _log_device_compatibility_check(policy_name, gpu_details_list):
  """Logs a compatibility check if the devices support the policy.

  Currently only logs for the policy mixed_float16.

  Args:
    policy_name: The name of the dtype policy.
    gpu_details_list: A list of dicts, one dict per GPU. Each dict
      is the device details for a GPU, as returned by
      `tf.config.experimental.get_device_details()`.
  """
  if policy_name != 'mixed_float16':
    # TODO(b/145686977): Log if the policy is 'mixed_bfloat16'. This requires
    # checking if a TPU is available.
    return
  supported_device_strs = []
  unsupported_device_strs = []
  for details in gpu_details_list:
    name = details.get('device_name', 'Unknown GPU')
    cc = details.get('compute_capability')
    if cc:
      device_str = '%s, compute capability %s.%s' % (name, cc[0], cc[1])
      if cc >= (7, 0):
        supported_device_strs.append(device_str)
      else:
        unsupported_device_strs.append(device_str)
    else:
      unsupported_device_strs.append(
          name + ', no compute capability (probably not an Nvidia GPU)')

  if unsupported_device_strs:
    warning_str = _COMPAT_CHECK_WARNING_PREFIX + '\n'
    if supported_device_strs:
      warning_str += ('Some of your GPUs may run slowly with dtype policy '
                      'mixed_float16 because they do not all have compute '
                      'capability of at least 7.0. Your GPUs:\n')
    elif len(unsupported_device_strs) == 1:
      warning_str += ('Your GPU may run slowly with dtype policy mixed_float16 '
                      'because it does not have compute capability of at least '
                      '7.0. Your GPU:\n')
    else:
      warning_str += ('Your GPUs may run slowly with dtype policy '
                      'mixed_float16 because they do not have compute '
                      'capability of at least 7.0. Your GPUs:\n')
    for device_str in _dedup_strings(supported_device_strs +
                                     unsupported_device_strs):
      warning_str += '  ' + device_str + '\n'
    warning_str += ('See https://developer.nvidia.com/cuda-gpus for a list of '
                    'GPUs and their compute capabilities.\n')
    warning_str += _COMPAT_CHECK_WARNING_SUFFIX
    tf_logging.warning(warning_str)
  elif not supported_device_strs:
    tf_logging.warning(
        '%s\n'
        'The dtype policy mixed_float16 may run slowly because '
        'this machine does not have a GPU. Only Nvidia GPUs with '
        'compute capability of at least 7.0 run quickly with '
        'mixed_float16.\n%s' % (_COMPAT_CHECK_WARNING_PREFIX,
                                _COMPAT_CHECK_WARNING_SUFFIX))
  elif len(supported_device_strs) == 1:
    tf_logging.info('%s\n'
                    'Your GPU will likely run quickly with dtype policy '
                    'mixed_float16 as it has compute capability of at least '
                    '7.0. Your GPU: %s' % (_COMPAT_CHECK_OK_PREFIX,
                                           supported_device_strs[0]))
  else:
    tf_logging.info('%s\n'
                    'Your GPUs will likely run quickly with dtype policy '
                    'mixed_float16 as they all have compute capability of at '
                    'least 7.0' % _COMPAT_CHECK_OK_PREFIX)


_logged_compatibility_check = False


def log_device_compatibility_check(policy_name):
  """Logs a compatibility check if the devices support the policy.

  Currently only logs for the policy mixed_float16. A log is shown only the
  first time this function is called.

  Args:
    policy_name: The name of the dtype policy.
  """
  global _logged_compatibility_check
  if _logged_compatibility_check:
    return
  _logged_compatibility_check = True
  gpus = config.list_physical_devices('GPU')
  gpu_details_list = [config.get_device_details(g) for g in gpus]
  _log_device_compatibility_check(policy_name, gpu_details_list)
