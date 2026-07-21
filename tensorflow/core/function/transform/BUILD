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

load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core/function/runtime_client:__subpackages__",
    ],
)

licenses(["notice"])

pytype_strict_library(
    name = "transform",
    srcs = [
        "transform.py",
    ],
    visibility = ["//smartass/brain/configure:__subpackages__"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/framework:function_proto_py",
        "//tensorflow/core/function/capture:restore_captures",
        "//tensorflow/core/function/runtime_client:runtime_client_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:function_def_to_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:default_gradient",
        "//tensorflow/python/ops:handle_data_util",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
    ],
)

py_test(
    name = "transform_test",
    srcs = ["transform_test.py"],
    strict_deps = True,
    tags = ["no_oss"],  # TODO(b/219089812)
    deps = [
        ":transform",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//tensorflow/core/function/testing:test_pass_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
    ],
)
