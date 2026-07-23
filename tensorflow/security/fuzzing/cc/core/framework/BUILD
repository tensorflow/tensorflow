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

# Fuzztest TensorShape domains

load("@rules_cc//cc:cc_library.bzl", "cc_library")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "tensor_shape_domains",
    testonly = 1,
    srcs = ["tensor_shape_domains.cc"],
    hdrs = ["tensor_shape_domains.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "@com_google_absl//absl/status:statusor",
        "@com_google_fuzztest//fuzztest",
        "@xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "tensor_domains",
    testonly = 1,
    srcs = ["tensor_domains.cc"],
    hdrs = ["tensor_domains.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensor_shape_domains",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:tstring",
        "@com_google_absl//absl/log",
        "@com_google_fuzztest//fuzztest",
        "@xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "datatype_domains",
    testonly = 1,
    srcs = ["datatype_domains.cc"],
    hdrs = ["datatype_domains.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_fuzztest//fuzztest",
    ],
)
