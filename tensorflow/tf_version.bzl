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

"""Provides project and wheel version data for TensorFlow."""

load(
    "//tensorflow:tf_version.default.bzl",
    "SEMANTIC_VERSION_SUFFIX",
    "VERSION_SUFFIX",
)

# These constants are used by the targets //third_party/tensorflow/core/public:release_version,
# //third_party/tensorflow:tensorflow_bzl and //third_party/tensorflow/tools/pip_package:setup_py.
TF_VERSION = "2.22.0"
MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION = TF_VERSION.split(".")
TF_WHEEL_VERSION_SUFFIX = VERSION_SUFFIX
TF_SEMANTIC_VERSION_SUFFIX = SEMANTIC_VERSION_SUFFIX
