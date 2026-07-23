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

# Description:
#   C++ implementation code for the summary writing APIs.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_copts",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "schema",
    srcs = ["schema.cc"],
    hdrs = ["schema.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/lib/db:sqlite",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "schema_test",
    size = "small",
    srcs = ["schema_test.cc"],
    deps = [
        ":schema",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "summary_db_writer",
    srcs = ["summary_db_writer.cc"],
    hdrs = ["summary_db_writer.h"],
    copts = tf_copts(),
    deps = [
        ":summary_converter",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:summary_interface",
        "//tensorflow/core/lib/db:sqlite",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@xla//xla/tsl/protobuf:histogram_proto_cc",
    ],
)

tf_cc_test(
    name = "summary_db_writer_test",
    size = "small",
    srcs = ["summary_db_writer_test.cc"],
    deps = [
        ":schema",
        ":summary_db_writer",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/db:sqlite",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/protobuf:histogram_proto_cc",
    ],
)

cc_library(
    name = "summary_file_writer",
    srcs = ["summary_file_writer.cc"],
    hdrs = ["summary_file_writer.h"],
    copts = tf_copts(),
    deps = [
        ":summary_converter",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:summary_interface",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_test(
    name = "summary_file_writer_test",
    size = "medium",  # file i/o
    timeout = "short",
    srcs = ["summary_file_writer_test.cc"],
    deps = [
        ":summary_file_writer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "summary_converter",
    srcs = ["summary_converter.cc"],
    hdrs = ["summary_converter.h"],
    copts = tf_copts(),
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/png:png_io",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_binary(
    name = "loader",
    srcs = ["loader.cc"],
    linkstatic = 1,
    deps = [
        ":schema",
        ":summary_db_writer",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/db:sqlite",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_binary(
    name = "vacuum",
    srcs = ["vacuum.cc"],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/lib/db:sqlite",
        "@com_google_absl//absl/log",
    ],
)
