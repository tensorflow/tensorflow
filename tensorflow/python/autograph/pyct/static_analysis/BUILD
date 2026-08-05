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
    name = "liveness",
    srcs = ["liveness.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":annos",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:cfg",
        "//tensorflow/python/autograph/pyct:transformer",
        "@pypi//gast",
    ],
)

py_library(
    name = "reaching_fndefs",
    srcs = ["reaching_fndefs.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:cfg",
        "//tensorflow/python/autograph/pyct:transformer",
        "@pypi//gast",
    ],
)

py_test(
    name = "reaching_fndefs_test",
    srcs = ["reaching_fndefs_test.py"],
    strict_deps = True,
    deps = [
        ":activity",
        ":reaching_definitions",
        ":reaching_fndefs",
        #internal proto upb dep
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:cfg",
        "//tensorflow/python/autograph/pyct:naming",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:transformer",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "activity",
    srcs = ["activity.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":annos",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:transformer",
        "@pypi//gast",
    ],
)

py_library(
    name = "type_inference",
    srcs = ["type_inference.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":activity",
        ":annos",
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:cfg",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:transformer",
        "@pypi//gast",
    ],
)

py_library(
    name = "reaching_definitions",
    srcs = ["reaching_definitions.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:cfg",
        "//tensorflow/python/autograph/pyct:transformer",
        "@pypi//gast",
    ],
)

py_library(
    name = "annos",
    srcs = ["annos.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
)

py_test(
    name = "activity_test",
    srcs = ["activity_test.py"],
    strict_deps = True,
    deps = [
        ":activity",
        ":annos",
        "@pypi//gast",
        #internal proto upb dep
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:naming",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:transformer",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "liveness_test",
    testonly = True,
    srcs = ["liveness_test.py"],
    strict_deps = True,
    deps = [
        ":activity",
        ":liveness",
        ":reaching_fndefs",
        #internal proto upb dep
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:cfg",
        "//tensorflow/python/autograph/pyct:naming",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:transformer",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "reaching_definitions_test",
    srcs = ["reaching_definitions_test.py"],
    strict_deps = True,
    deps = [
        ":activity",
        ":reaching_definitions",
        #internal proto upb dep
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:cfg",
        "//tensorflow/python/autograph/pyct:naming",
        "//tensorflow/python/autograph/pyct:parser",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:transformer",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "type_inference_test",
    srcs = ["type_inference_test.py"],
    strict_deps = True,
    deps = [
        ":activity",
        ":reaching_definitions",
        ":reaching_fndefs",
        ":type_inference",
        #internal proto upb dep
        "//tensorflow/python/autograph/pyct:anno",
        "//tensorflow/python/autograph/pyct:cfg",
        "//tensorflow/python/autograph/pyct:qual_names",
        "//tensorflow/python/autograph/pyct:transpiler",
        "//tensorflow/python/platform:client_testlib",
    ],
)
