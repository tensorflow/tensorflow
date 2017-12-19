# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""TensorFlow versions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow

__version__ = pywrap_tensorflow.__version__
__git_version__ = pywrap_tensorflow.__git_version__
__compiler_version__ = pywrap_tensorflow.__compiler_version__
__cxx11_abi_flag__ = pywrap_tensorflow.__cxx11_abi_flag__
__monolithic_build__ = pywrap_tensorflow.__monolithic_build__

VERSION = __version__
GIT_VERSION = __git_version__
COMPILER_VERSION = __compiler_version__
CXX11_ABI_FLAG = __cxx11_abi_flag__
MONOLITHIC_BUILD = __monolithic_build__

GRAPH_DEF_VERSION = pywrap_tensorflow.GRAPH_DEF_VERSION
GRAPH_DEF_VERSION_MIN_CONSUMER = (
    pywrap_tensorflow.GRAPH_DEF_VERSION_MIN_CONSUMER)
GRAPH_DEF_VERSION_MIN_PRODUCER = (
    pywrap_tensorflow.GRAPH_DEF_VERSION_MIN_PRODUCER)

__all__ = [
    "__version__",
    "__git_version__",
    "__compiler_version__",
    "__cxx11_abi_flag__",
    "__monolithic_build__",
    "COMPILER_VERSION",
    "CXX11_ABI_FLAG",
    "GIT_VERSION",
    "GRAPH_DEF_VERSION",
    "GRAPH_DEF_VERSION_MIN_CONSUMER",
    "GRAPH_DEF_VERSION_MIN_PRODUCER",
    "VERSION",
    "MONOLITHIC_BUILD",
]
