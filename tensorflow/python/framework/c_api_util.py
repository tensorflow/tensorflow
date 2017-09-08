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

"""Utilities for using the TensorFlow C API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow as c_api


class ScopedTFStatus(object):
  """Wrapper around TF_Status that handles deletion."""

  def __init__(self):
    self.status = c_api.TF_NewStatus()

  def __del__(self):
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api.TF_DeleteStatus is not None:
      c_api.TF_DeleteStatus(self.status)


class ScopedTFGraph(object):
  """Wrapper around TF_Graph that handles deletion."""

  def __init__(self):
    self.graph = c_api.TF_NewGraph()

  def __del__(self):
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api.TF_DeleteGraph is not None:
      c_api.TF_DeleteGraph(self.graph)
