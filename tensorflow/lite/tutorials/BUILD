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
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_binary(
    name = "mnist_tflite",
    srcs = ["mnist_tflite.py"],
    strict_deps = True,
    deps = [
        ":mnist_tflite_lib",
    ],
)

py_library(
    name = "mnist_tflite_lib",
    srcs = ["mnist_tflite.py"],
    strict_deps = True,
    deps = [
        ":dataset",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "mnist_tflite_test",
    srcs = ["mnist_tflite_test.py"],
    strict_deps = True,
    deps = [
        ":mnist_tflite_lib",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    strict_deps = True,
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)
