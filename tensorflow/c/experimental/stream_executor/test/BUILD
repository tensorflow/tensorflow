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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Description:
# test for stream_executor
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_shared_object",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cc_shared_object(
    name = "test_pluggable_device.so",
    srcs = ["test_pluggable_device.cc"],
    visibility = ["//tensorflow/c:__subpackages__"],
    deps = [
        "//tensorflow/c/experimental/stream_executor:stream_executor_hdrs",
        "//tensorflow/c/experimental/stream_executor:stream_executor_test_util",
    ],
)

cc_library(
    name = "test_pluggable_device",
    srcs = ["test_pluggable_device.cc"],
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        "//tensorflow/c/experimental/stream_executor:stream_executor_hdrs",
        "//tensorflow/c/experimental/stream_executor:stream_executor_test_util",
    ],
)
