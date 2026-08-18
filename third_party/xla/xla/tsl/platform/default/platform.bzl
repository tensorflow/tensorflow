# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

"""Platform specific paths for various libraries and utilities."""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

CUDA_VERSION = ""

CUDNN_VERSION = ""

PLATFORM = ""

def cuda_sdk_version():
    return CUDA_VERSION

def cudnn_sdk_version():
    return CUDNN_VERSION

# buildifier: disable=function-docstring
def cuda_library_path(name, version = cuda_sdk_version()):
    if PLATFORM == "Darwin":
        if not version:
            return "lib/lib{}.dylib".format(name)
        else:
            return "lib/lib{}.{}.dylib".format(name, version)
    elif not version:
        return "lib64/lib{}.so".format(name)
    else:
        return "lib64/lib{}.so.{}".format(name, version)

def cuda_static_library_path(name):
    if PLATFORM == "Darwin":
        return "lib/lib{}_static.a".format(name)
    else:
        return "lib64/lib{}_static.a".format(name)

# buildifier: disable=function-docstring
def cudnn_library_path(version = cudnn_sdk_version()):
    if PLATFORM == "Darwin":
        if not version:
            return "lib/libcudnn.dylib"
        else:
            return "lib/libcudnn.{}.dylib".format(version)
    elif not version:
        return "lib64/libcudnn.so"
    else:
        return "lib64/libcudnn.so.{}".format(version)

# buildifier: disable=function-docstring
def cupti_library_path(version = cuda_sdk_version()):
    if PLATFORM == "Darwin":
        if not version:
            return "extras/CUPTI/lib/libcupti.dylib"
        else:
            return "extras/CUPTI/lib/libcupti.{}.dylib".format(version)
    elif not version:
        return "extras/CUPTI/lib64/libcupti.so"
    else:
        return "extras/CUPTI/lib64/libcupti.so.{}".format(version)

def readlink_command():
    if PLATFORM == "Darwin":
        return "greadlink"
    else:
        return "readlink"
