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

# Cloud TPU Client.

load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_library(
    name = "client",
    srcs = [
        "client.py",
        "version.py",
    ],
    strict_deps = True,
    deps = ["@absl_py//absl/flags"],
)

py_library(
    name = "client_lib",
    srcs = [
        "__init__.py",
    ],
    strict_deps = True,
    deps = [":client"],
)

tf_py_strict_test(
    name = "client_py_test",
    size = "small",
    srcs = ["client_test.py"],
    grpc_enabled = True,
    main = "client_test.py",
    deps = [
        ":client",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/flags",
    ],
)
