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

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.default.bzl", "pybind_extension")
load("//tensorflow/core/platform:build_config_root.bzl", "if_pywrap")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

pybind_extension(
    name = "format_converter_wrapper_pybind11",
    srcs = ["format_converter_wrapper_pybind11.cc"],
    common_lib_packages = [
        "litert/python",
        "tensorflow/lite/python",
    ],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    data = [
        "format_converter_wrapper_pybind11.pyi",
    ],
    enable_stub_generation = True,
    features = ["-use_header_modules"],
    wrap_py_init = True,
    deps = [
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels/internal/utils:sparsity_format_converter",
        "@pybind11",
        "@xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

py_test(
    name = "format_converter_wrapper_pybind11_test",
    srcs = ["format_converter_wrapper_pybind11_test.py"],
    strict_deps = True,
    deps = [
        ":format_converter_wrapper_pybind11",
        "@absl_py//absl/testing:absltest",
        #internal proto upb dep
        "//third_party/py/numpy",
    ] + if_pywrap(
        if_true = ["//tensorflow/lite/python:pywrap_tflite"],
    ),
)
