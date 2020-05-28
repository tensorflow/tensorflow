# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "interpreter_utils",
    srcs = ["interpreter_utils.cc"],
    hdrs = ["interpreter_utils.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    linkopts = tflite_linkopts(),
    linkstatic = 1,
    deps = [
        ":utils",
        "//tensorflow/lite/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "delegate_test",
    size = "small",
    srcs = ["delegate_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    tags = [
        "tflite_not_portable_ios",  # TODO(b/117786830)
    ],
    deps = [
        ":interpreter_utils",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:version",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)
