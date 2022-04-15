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

from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.util.tf_export import tf_export

__version__ = pywrap_tf_session.__version__
__git_version__ = pywrap_tf_session.__git_version__
__compiler_version__ = pywrap_tf_session.__compiler_version__
__cxx11_abi_flag__ = pywrap_tf_session.__cxx11_abi_flag__
__monolithic_build__ = pywrap_tf_session.__monolithic_build__

VERSION = __version__
tf_export(
    "version.VERSION",
    "__version__",
    v1=["version.VERSION", "VERSION", "__version__"]).export_constant(
        __name__, "VERSION")
GIT_VERSION = __git_version__
tf_export(
    "version.GIT_VERSION",
    "__git_version__",
    v1=["version.GIT_VERSION", "GIT_VERSION",
        "__git_version__"]).export_constant(__name__, "GIT_VERSION")
COMPILER_VERSION = __compiler_version__
tf_export(
    "version.COMPILER_VERSION",
    "__compiler_version__",
    v1=["version.COMPILER_VERSION", "COMPILER_VERSION",
        "__compiler_version__"]).export_constant(__name__, "COMPILER_VERSION")

CXX11_ABI_FLAG = __cxx11_abi_flag__
tf_export(
    "sysconfig.CXX11_ABI_FLAG",
    "__cxx11_abi_flag__",
    v1=["sysconfig.CXX11_ABI_FLAG", "CXX11_ABI_FLAG",
        "__cxx11_abi_flag__"]).export_constant(__name__, "CXX11_ABI_FLAG")
MONOLITHIC_BUILD = __monolithic_build__
tf_export(
    "sysconfig.MONOLITHIC_BUILD",
    "__monolithic_build__",
    v1=[
        "sysconfig.MONOLITHIC_BUILD", "MONOLITHIC_BUILD", "__monolithic_build__"
    ]).export_constant(__name__, "MONOLITHIC_BUILD")

GRAPH_DEF_VERSION = pywrap_tf_session.GRAPH_DEF_VERSION
tf_export(
    "version.GRAPH_DEF_VERSION",
    v1=["version.GRAPH_DEF_VERSION", "GRAPH_DEF_VERSION"]).export_constant(
        __name__, "GRAPH_DEF_VERSION")
GRAPH_DEF_VERSION_MIN_CONSUMER = (
    pywrap_tf_session.GRAPH_DEF_VERSION_MIN_CONSUMER)
tf_export(
    "version.GRAPH_DEF_VERSION_MIN_CONSUMER",
    v1=[
        "version.GRAPH_DEF_VERSION_MIN_CONSUMER",
        "GRAPH_DEF_VERSION_MIN_CONSUMER"
    ]).export_constant(__name__, "GRAPH_DEF_VERSION_MIN_CONSUMER")
GRAPH_DEF_VERSION_MIN_PRODUCER = (
    pywrap_tf_session.GRAPH_DEF_VERSION_MIN_PRODUCER)
tf_export(
    "version.GRAPH_DEF_VERSION_MIN_PRODUCER",
    v1=[
        "version.GRAPH_DEF_VERSION_MIN_PRODUCER",
        "GRAPH_DEF_VERSION_MIN_PRODUCER"
    ]).export_constant(__name__, "GRAPH_DEF_VERSION_MIN_PRODUCER")

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
