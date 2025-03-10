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

from tensorflow.core.framework import device_attributes_pb2
# pylint: disable=invalid-import-order, g-bad-import-order, wildcard-import, unused-import, undefined-variable
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import _pywrap_device_lib
from tensorflow.python.platform import tf_logging as logging


def list_local_devices(session_config=None):
  """List the available devices available in the local process.

  Args:
    session_config: a session config proto or None to use the default config.

  Returns:
    A list of `DeviceAttribute` protocol buffers.
  """
  def _convert(pb_str):
    try:
      m = device_attributes_pb2.DeviceAttributes()
      m.ParseFromString(pb_str)
      return m
    except Exception as e:
      logging.error("Failed to parse device %s",e)
      return None

  serialized_config = None
  if session_config is not None:
    serialized_config = session_config.SerializeToString()

  try:
    devices = _pywrap_device_lib.list_devices(serialized_config)
  except Exception as e:
    logging.error("Failed to list devices %s",e)
    return []
  
  return [
      _convert(s) for s in devices if s is not None
  ]
