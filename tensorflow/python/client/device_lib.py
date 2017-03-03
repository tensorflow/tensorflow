# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A Python interface for creating TensorFlow servers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import device_attributes_pb2
from tensorflow.python import pywrap_tensorflow


def list_local_devices():
  """List the available devices available in the local process.

  Returns:
    A list of `DeviceAttribute` protocol buffers.
  """
  def _convert(pb_str):
    m = device_attributes_pb2.DeviceAttributes()
    m.ParseFromString(pb_str)
    return m

  return [_convert(s) for s in pywrap_tensorflow.list_devices()]
