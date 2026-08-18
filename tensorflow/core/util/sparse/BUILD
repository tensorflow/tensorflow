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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "higher_level_tests_group",
    srcs = ["sparse_tensor_test.cc"],
    visibility = ["//visibility:private"],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "group_iterator.cc",
        "sparse_tensor.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "mobile_hdrs_only_runtime",
    srcs = [
        "dim_comparator.h",
        "group_iterator.h",
        "sparse_tensor.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "framework_group",
    srcs = [
        "group_iterator.h",
        "sparse_tensor.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "framework_internal_private_headers_group",
    srcs = [
        "dim_comparator.h",
        "group_iterator.h",
        "sparse_tensor.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "framework_internal_impl_group",
    srcs = [
        "group_iterator.cc",
        "sparse_tensor.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

cc_library(
    name = "sparse_tensor",
    srcs = ["sparse_tensor.cc"],
    hdrs = [
        "dim_comparator.h",
        "sparse_tensor.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "sparse_tensor_test",
    size = "small",
    srcs = [
        "sparse_tensor_test.cc",
    ],
    deps = [
        ":sparse_tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@xla//xla/tsl/platform:test_main",
        "@xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)
