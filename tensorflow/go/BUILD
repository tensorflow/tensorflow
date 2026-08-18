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

# Description:
# Go API for TensorFlow.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_shared_library_deps",
)

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

sh_test(
    name = "test",
    size = "small",
    srcs = ["test.sh"],
    data = [
        ":all_files",  # Go sources
        "//tensorflow/c:headers",  # C library header
        "//tensorflow/c/eager:headers",  # Eager C library header
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",  # Testdata for LoadSavedModel
    ] + tf_shared_library_deps(),
    # TODO: Enable this test again once protos are supported by bazel.
    tags = ["manual"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
