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

load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "v2_compat",
    srcs = ["v2_compat.py"],
    strict_deps = True,
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:registry",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:control_flow_v2_toggles",
        "//tensorflow/python/ops:resource_variables_toggle",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "compat",
    srcs = ["compat.py"],
    strict_deps = True,
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "compat_test",
    size = "small",
    srcs = ["compat_test.py"],
    tags = ["nofwdcompat"],
    deps = [
        ":compat",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "disable_v2_behavior_test",
    size = "small",
    srcs = ["disable_v2_behavior_test.py"],
    deps = [
        ":v2_compat",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:_pywrap_tf2",
        "//tensorflow/python/platform:client_testlib",
    ],
)
