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
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "common_transformers",
    srcs = [
        "anf.py",
    ],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/pyct:gast_util",
        "//tensorflow/python/autograph/pyct:templates",
        "//tensorflow/python/autograph/pyct:transformer",
        "@pypi//gast",
    ],
)

py_test(
    name = "anf_test",
    srcs = ["anf_test.py"],
    strict_deps = True,
    tags = ["no_oss"],
    deps = [
        ":common_transformers",
        "@pypi//gast",
        #internal proto upb dep
        "//tensorflow/python/autograph/pyct:loader",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:transformer",
        "//tensorflow/python/platform:client_testlib",
    ],
)
