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
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:special_rules.bzl", "nonportable_visibility_allowlist")

# Kernel for custom parse_example
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "parse_example",
    srcs = [
        "example_proto_fast_parsing.cc",
        "parse_example.cc",
    ],
    hdrs = [
        "example_proto_fast_parsing.h",
        "parse_example.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "//third_party/protobuf/io",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:feature_util",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/platform:hash",
        ],
    }),
)

tf_cc_test(
    name = "parse_example_test",
    srcs = ["parse_example_test.cc"],
    deps = [
        ":parse_example",
        "//tensorflow/core/example:example_protos_cc",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api:op_resolver",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/kernels:test_main",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//third_party/protobuf",
        "//third_party/protobuf/io",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:ios": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/example:feature_util",
            "//tensorflow/core/platform:protobuf",
            "//tensorflow/core/platform:tstring",
        ],
    }),
)

# Same as "parse_example" above, but doesn't use the "select" statements
# to use "portable_tensorflow_lib_lite" for portable builds.
cc_library(
    name = "nonportable_parse_example",
    srcs = [
        "example_proto_fast_parsing.cc",
        "parse_example.cc",
    ],
    hdrs = [
        "example_proto_fast_parsing.h",
        "parse_example.h",
    ],
    visibility = nonportable_visibility_allowlist(),
    deps = [
        "//tensorflow/core:feature_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "//third_party/protobuf/io",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "nonportable_parse_example_test",
    srcs = ["parse_example_test.cc"],
    deps = [
        ":nonportable_parse_example",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/example:example_protos_cc",
        "//tensorflow/core/example:feature_util",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:tstring",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api:op_resolver",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/kernels:test_main",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//third_party/protobuf",
        "//third_party/protobuf/io",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)
