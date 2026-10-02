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
    name = "api",
    srcs = ["api.py"],
    strict_deps = True,
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":conversion",
        "//tensorflow/python/autograph/converters:asserts",
        "//tensorflow/python/autograph/converters:break_statements",
        "//tensorflow/python/autograph/converters:call_trees",
        "//tensorflow/python/autograph/converters:conditional_expressions",
        "//tensorflow/python/autograph/converters:continue_statements",
        "//tensorflow/python/autograph/converters:control_flow",
        "//tensorflow/python/autograph/converters:directives",
        "//tensorflow/python/autograph/converters:functions",
        "//tensorflow/python/autograph/converters:lists",
        "//tensorflow/python/autograph/converters:logical_expressions",
        "//tensorflow/python/autograph/converters:return_statements",
        "//tensorflow/python/autograph/converters:slices",
        "//tensorflow/python/autograph/converters:variables",
        "//tensorflow/python/autograph/core:ag_ctx",
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/core:function_wrappers",
        "//tensorflow/python/autograph/core:unsupported_features_checker",
        "//tensorflow/python/autograph/lang:special_functions",
        "//tensorflow/python/autograph/operators:__init__",
        "//tensorflow/python/autograph/operators:py_builtins",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:cfg",
        "//tensorflow/python/autograph/pyct:error_utils",
        "//tensorflow/python/autograph/pyct:errors",
        "//tensorflow/python/autograph/pyct:inspect_utils",
        "//tensorflow/python/autograph/pyct:origin_info",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:transpiler",
        "//tensorflow/python/autograph/pyct/static_analysis:activity",
        "//tensorflow/python/autograph/pyct/static_analysis:reaching_definitions",
        "//tensorflow/python/autograph/utils:__init__",
        "//tensorflow/python/autograph/utils:ag_logging",
        "//tensorflow/python/eager/polymorphic_function:tf_method_target",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//tensorflow/python/util:tf_stack",
    ],
)

py_library(
    name = "conversion",
    srcs = ["conversion.py"],
    strict_deps = True,
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/python/autograph/core:config",
        "//tensorflow/python/autograph/pyct:cache",
        "//tensorflow/python/autograph/pyct:inspect_utils",
        "//tensorflow/python/autograph/utils:ag_logging",
        "//tensorflow/python/eager/polymorphic_function:tf_method_target",
        "//tensorflow/python/util:tf_inspect",
    ],
)

tf_py_strict_test(
    name = "api_test",
    srcs = ["api_test.py"],
    deps = [
        ":api",
        ":conversion",
        "//tensorflow/python/autograph/core:ag_ctx",
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/core:test_lib",
        "//tensorflow/python/autograph/pyct:errors",
        "//tensorflow/python/autograph/pyct:inspect_utils",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/utils:ag_logging",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:_errors_test_helper",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:function_utils",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_inspect",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "conversion_test",
    srcs = ["conversion_test.py"],
    deps = [
        ":api",
        ":conversion",
        "//tensorflow/python/autograph/core:config",
        "//tensorflow/python/autograph/core:converter",
        "//tensorflow/python/autograph/impl/testing:pybind_for_testing",
        "//tensorflow/python/autograph/utils:__init__",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:client_testlib",
    ],
)
