# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""External-only build rules for mini-benchmark."""

load("//tensorflow:tensorflow.bzl", "clean_dep")

def libjpeg_hdrs_deps():
    """Returns the deps for the jpeg header used in the mini-benchmark."""
    return [clean_dep("//tensorflow/core/platform:jpeg")]

def libjpeg_deps():
    """Returns the deps for the jpeg lib used in the mini-benchmark."""
    return [clean_dep("//tensorflow/core/platform:jpeg")]

def libjpeg_handle_deps():
    """Returns the deps for the jpeg handle used in the mini-benchmark."""
    return ["//tensorflow/lite/experimental/acceleration/mini_benchmark:libjpeg_handle_static_link"]

def minibenchmark_visibility_allowlist():
    """Returns a list of packages that can depend on mini_benchmark."""
    return [
        "//tensorflow/lite/core/experimental/acceleration/mini_benchmark/c:__subpackages__",
        "//tensorflow/lite/tools/benchmark/experimental/delegate_performance:__subpackages__",
    ]

def register_selected_ops_deps():
    """Return a list of dependencies for registering selected ops."""
    return [
        "//tensorflow/lite/tools/benchmark:register_custom_op",
    ]
