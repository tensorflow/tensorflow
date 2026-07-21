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

load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

py_binary(
    name = "generate_v2_renames_map",
    srcs = ["generate_v2_renames_map.py"],
    strict_deps = True,
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:modules_with_exports",
        "//tensorflow/python:no_contrib",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
        "//tensorflow/tools/compatibility:all_renames_v2",
        "@absl_py//absl:app",
    ],
)

py_binary(
    name = "generate_v2_reorders_map",
    srcs = ["generate_v2_reorders_map.py"],
    strict_deps = True,
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:no_contrib",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
        "//tensorflow/tools/compatibility:tf_upgrade_v2_lib",
        "@absl_py//absl:app",
    ],
)
