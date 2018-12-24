# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
#   Keras saving and loading libraries.
load("//tensorflow:tensorflow.bzl", "py_test")

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "saving",
    srcs = ["__init__.py"],
    deps = [":saved_model"],
)

py_library(
    name = "saved_model",
    srcs = [
        "saved_model.py",
        "saving_utils.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:metrics",
        "//tensorflow/python:mode_keys",
        "//tensorflow/python:platform",
        "//tensorflow/python:saver",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/saved_model/model_utils",
    ],
)

py_test(
    name = "saved_model_test",
    size = "medium",
    srcs = ["saved_model_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_oss",  # TODO(b/119349471): Re-enable
        "no_windows",
    ],
    deps = [
        ":saved_model",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:mode_keys",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
