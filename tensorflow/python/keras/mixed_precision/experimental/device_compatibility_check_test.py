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
"""Tests the device compatibility check."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.core.framework import device_attributes_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.keras.mixed_precision.experimental import device_compatibility_check
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def _get_device_attrs(device_type, device_name=None, cc_major=None,
                      cc_minor=None):
  if device_type == 'CPU':
    return device_attributes_pb2.DeviceAttributes(device_type='CPU')
  assert device_type == 'GPU', 'Invalid device type: %s' % (device_type,)
  if not device_name:
    return device_attributes_pb2.DeviceAttributes(device_type='GPU')
  physical_device_desc = (
      'device: 0, name: %s, pci bus id: 0:0:0.0' % device_name)
  if cc_major:
    physical_device_desc += ', compute capability: %d.%d' % (cc_major, cc_minor)
  return device_attributes_pb2.DeviceAttributes(
      device_type='GPU', physical_device_desc=physical_device_desc)


@test_util.run_all_in_graph_and_eager_modes
class DeviceCompatibilityCheckTest(test.TestCase):

  def _test_compat_check(self, device_attr_list, should_warn, expected_regex,
                         policy_name='mixed_float16'):
    with test.mock.patch.object(tf_logging, 'warn') as mock_warn, \
         test.mock.patch.object(tf_logging, 'info') as mock_info:
      device_compatibility_check._log_device_compatibility_check(
          policy_name, device_attr_list)
    if should_warn:
      self.assertRegexpMatches(mock_warn.call_args[0][0], expected_regex)
      mock_info.assert_not_called()
    else:
      self.assertRegexpMatches(mock_info.call_args[0][0], expected_regex)
      mock_warn.assert_not_called()

  def test_supported(self):
    device_attrs_list = [_get_device_attrs('GPU', 'GPU 1', 7, 1)]
    regex = re.compile(
        r'.*compatibility check \(mixed_float16\): OK\n'
        r'Your GPU will likely run quickly with dtype policy mixed_float16 as '
        r'it has compute capability of at least 7.0. Your GPU: GPU 1, compute '
        r'capability 7.1', flags=re.MULTILINE)
    self._test_compat_check(device_attrs_list, False, regex)
    device_attrs_list.append(_get_device_attrs('CPU'))
    self._test_compat_check(device_attrs_list, False, regex)

    device_attrs_list = [
        _get_device_attrs('CPU', 'CPU'),
        _get_device_attrs('GPU', 'GPU 1', 7, 0),
        _get_device_attrs('GPU', 'GPU 2', 7, 1),
        _get_device_attrs('GPU', 'GPU 3', 8, 0),
    ]
    regex = re.compile(
        r'.*compatibility check \(mixed_float16\): OK\n'
        r'Your GPUs will likely run quickly with dtype policy mixed_float16 as '
        r'they all have compute capability of at least 7.0', flags=re.MULTILINE)
    self._test_compat_check(device_attrs_list, False, regex)

  def test_unsupported(self):
    device_attrs_list = [
        _get_device_attrs('GPU', 'GPU 1', 6, 0)
    ]
    regex = re.compile(
        r'.*compatibility check \(mixed_float16\): WARNING\n'
        r'Your GPU may run slowly with dtype policy mixed_float16.*\n'
        r'  GPU 1, compute capability 6.0\n'
        r'See.*', flags=re.MULTILINE)
    self._test_compat_check(device_attrs_list, True, regex)
    device_attrs_list.append(_get_device_attrs('CPU'))
    self._test_compat_check(device_attrs_list, True, regex)

    device_attrs_list = [
        _get_device_attrs('GPU')
    ]
    regex = re.compile(
        r'.*compatibility check \(mixed_float16\): WARNING\n'
        r'Your GPU may run slowly with dtype policy mixed_float16.*\n'
        r'  Unknown GPU, no compute capability \(probably not an Nvidia GPU\)\n'
        r'See.*', flags=re.MULTILINE)
    self._test_compat_check(device_attrs_list, True, regex)
    device_attrs_list.append(_get_device_attrs('CPU'))
    self._test_compat_check(device_attrs_list, True, regex)

    device_attrs_list = [
        _get_device_attrs('CPU', 'CPU'),
        _get_device_attrs('GPU', 'GPU 1', 6, 0),
        _get_device_attrs('GPU', 'GPU 2', 3, 10),
    ]
    regex = re.compile(
        r'.*compatibility check \(mixed_float16\): WARNING\n'
        r'Your GPUs may run slowly with dtype policy mixed_float16.*\n'
        r'  GPU 1, compute capability 6.0\n'
        r'  GPU 2, compute capability 3.10\n'
        r'See.*', flags=re.MULTILINE)
    self._test_compat_check(device_attrs_list, True, regex)

    device_attrs_list = [
        _get_device_attrs('CPU', 'CPU'),
        _get_device_attrs('GPU', 'GPU 1', 6, 0),
        _get_device_attrs('GPU', 'GPU 1', 6, 0),
        _get_device_attrs('GPU', 'GPU 1', 6, 0),
        _get_device_attrs('GPU', 'GPU 2', 3, 10),
    ]
    regex = re.compile(
        r'.*compatibility check \(mixed_float16\): WARNING\n'
        r'Your GPUs may run slowly with dtype policy mixed_float16.*\n'
        r'  GPU 1, compute capability 6.0 \(x3\)\n'
        r'  GPU 2, compute capability 3.10\n'
        r'See.*', flags=re.MULTILINE)
    self._test_compat_check(device_attrs_list, True, regex)

    device_attrs_list = [_get_device_attrs('CPU')]
    regex = re.compile(
        r'.*compatibility check \(mixed_float16\): WARNING\n'
        r'The dtype policy mixed_float16 may run slowly because this machine '
        r'does not have a GPU', flags=re.MULTILINE)
    self._test_compat_check(device_attrs_list, True, regex)

  def test_mix_of_supported_and_unsupported(self):
    device_attrs_list = [
        _get_device_attrs('GPU', 'GPU 1', 7, 0),
        _get_device_attrs('GPU', 'GPU 1', 7, 0),
        _get_device_attrs('GPU', 'GPU 2', 6, 0)
    ]
    regex = re.compile(
        r'.*compatibility check \(mixed_float16\): WARNING\n'
        r'Some of your GPUs may run slowly with dtype policy mixed_float16.*\n'
        r'  GPU 1, compute capability 7.0 \(x2\)\n'
        r'  GPU 2, compute capability 6.0\n'
        r'See.*', flags=re.MULTILINE)
    self._test_compat_check(device_attrs_list, True, regex)
    device_attrs_list.append(_get_device_attrs('CPU'))
    self._test_compat_check(device_attrs_list, True, regex)


if __name__ == '__main__':
  test.main()
