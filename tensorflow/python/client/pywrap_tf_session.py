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
"""Python module for Session ops, vars, and functions exported by pybind11."""

# pylint: disable=invalid-import-order,g-bad-import-order, wildcard-import, unused-import
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client._pywrap_tf_session import *
from tensorflow.python.client._pywrap_tf_session import _TF_SetTarget
from tensorflow.python.client._pywrap_tf_session import _TF_SetConfig
from tensorflow.python.client._pywrap_tf_session import _TF_NewSessionOptions

# Convert versions to strings for Python2 and keep api_compatibility_test green.
# We can remove this hack once we remove Python2 presubmits. pybind11 can only
# return unicode for Python2 even with py::str.
# https://pybind11.readthedocs.io/en/stable/advanced/cast/strings.html#returning-c-strings-to-python
# pylint: disable=undefined-variable
__version__ = str(get_version())
__git_version__ = str(get_git_version())
__compiler_version__ = str(get_compiler_version())
__cxx11_abi_flag__ = get_cxx11_abi_flag()
__monolithic_build__ = get_monolithic_build()

# User getters to hold attributes rather than pybind11's m.attr due to
# b/145559202.
GRAPH_DEF_VERSION = get_graph_def_version()
GRAPH_DEF_VERSION_MIN_CONSUMER = get_graph_def_version_min_consumer()
GRAPH_DEF_VERSION_MIN_PRODUCER = get_graph_def_version_min_producer()
TENSOR_HANDLE_KEY = get_tensor_handle_key()

# pylint: enable=undefined-variable


# Disable pylint invalid name warnings for legacy functions.
# pylint: disable=invalid-name
def TF_NewSessionOptions(target=None, config=None):
  # NOTE: target and config are validated in the session constructor.
  opts = _TF_NewSessionOptions()
  if target is not None:
    _TF_SetTarget(opts, target)
  if config is not None:
    config_str = config.SerializeToString()
    _TF_SetConfig(opts, config_str)
  return opts


# Disable pylind undefined-variable as the variable is exported in the shared
# object via pybind11.
# pylint: disable=undefined-variable
def TF_Reset(target, containers=None, config=None):
  opts = TF_NewSessionOptions(target=target, config=config)
  try:
    TF_Reset_wrapper(opts, containers)
  finally:
    TF_DeleteSessionOptions(opts)
