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

# buildifier: disable=out-of-order-load

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//bazel:java_lite_proto_library.bzl", "java_lite_proto_library")
load("@flatbuffers//:build_defs.bzl", "DEFAULT_FLATC_ARGS", "flatbuffer_android_library", "flatbuffer_cc_library", "flatbuffer_java_library", "flatc_path")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite", "cc_test_with_tflite")

# copybara:uncomment load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")
load(":build_defs.bzl", "flatbuffer_schema_compat_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# We generate a FlatBuffer schema from the Protobuf schema.
genrule(
    name = "configuration_schema",
    srcs = ["configuration.proto"],
    outs = ["configuration.fbs"],
    # We rename the namespace since otherwise the proto classes and flatbuffer
    # classes would have the same names.
    cmd = """
        $(location {}) --proto -o $(@D) $(location :configuration.proto)
        perl -p -i -e 's/tflite.proto/tflite/' $@
    """.format(flatc_path),
    compatible_with = get_compatible_with_portable(),
    tools = [flatc_path],
)

# We also do the same transformation for the _previous_
# version of the schema -- this is used to test that changes
# to the schema preserve binary backwards compatibility.
genrule(
    name = "configuration_prev_schema",
    srcs = ["testdata/configuration.proto_prev"],  # Must NOT end in '.proto'.
    outs = ["configuration.prev.fbs"],  # MUST end in '.fbs'.
    # We rename the namespace since otherwise the proto classes and flatbuffer
    # classes would have the same names.
    cmd = """
        cp $(location :testdata/configuration.proto_prev) $(@D)/configuration.prev.proto
        $(location {}) --proto -o $(@D) $(@D)/configuration.prev.proto
        perl -p -i -e 's/tflite.proto/tflite/' $@
    """.format(flatc_path),
    compatible_with = get_compatible_with_portable(),
    tools = [flatc_path],
)

# Test that changes to the proto file preserve binary backwards compatibility
# of the generated FlatBuffer schema, relative to the one generated from the
# previous proto file.
flatbuffer_schema_compat_test(
    name = "configuration_abi_stability_test",
    ref_schema = ":configuration.prev.fbs",
    schema = ":configuration.fbs",
)

# Test that changes to the proto file OR to the FlatBuffer proto-to-flatbuffer
# schema conversion itself will preserve binary backwards compatibility of the
# generated FlatBuffer schema, relative to an older snapshot of the
# generated FlatBuffer schema.
flatbuffer_schema_compat_test(
    name = "configuration_flatbuffer_abi_stability_test",
    ref_schema = "testdata/configuration.old.fbs",
    schema = ":configuration.fbs",
)

# Test that the previous version of the proto is different than the current version.
sh_test(
    name = "prev_is_different_than_current_test",
    srcs = ["prev_is_different_than_current_test.sh"],
    data = [
        "configuration.proto",
        "testdata/configuration.proto_prev",
    ],
)

# Generate a C++ header containing the contents of the FlatBuffer schema
# as a char array literal.  This is potentially useful for embedding in programs
# (e.g. for JSON parsing using that schema).
genrule(
    name = "configuration_fbs_contents_cc",
    srcs = ["configuration.fbs"],
    outs = ["configuration_fbs_contents-inl.h"],
    cmd = """
      echo 'constexpr char configuration_fbs_contents[] = R"Delimiter(' > $(@)
      cat < $(<) >> $(@)
      echo ')Delimiter";' >> $(@)
    """,
)

exports_files(["configuration.proto"])

proto_library(
    name = "configuration_proto",
    srcs = [
        "configuration.proto",
    ],
    compatible_with = get_compatible_with_portable(),
)

cc_proto_library(
    name = "configuration_cc_proto",
    compatible_with = get_compatible_with_portable(),
    deps = [":configuration_proto"],
)

java_lite_proto_library(
    name = "configuration_java_proto_lite",
    deps = [":configuration_proto"],
)

flatbuffer_cc_library(
    name = "configuration_fbs",
    srcs = [":configuration.fbs"],
    compatible_with = get_compatible_with_portable(),
    flatc_args = DEFAULT_FLATC_ARGS + ["--gen-compare"],
)

flatbuffer_java_library(
    name = "configuration_fbs_java",
    srcs = [":configuration.fbs"],
)

flatbuffer_android_library(
    name = "configuration_fbs_android",
    srcs = [":configuration.fbs"],
)

cc_library(
    name = "proto_to_flatbuffer",
    srcs = [
        "proto_to_flatbuffer.cc",
    ],
    hdrs = ["proto_to_flatbuffer.h"],
    deps = [
        ":configuration_cc_proto",
        ":configuration_fbs",
        "//tensorflow/lite:minimal_logging",
        "@flatbuffers",
    ],
)

cc_test(
    name = "proto_to_flatbuffer_test",
    srcs = ["proto_to_flatbuffer_test.cc"],
    deps = [
        ":configuration_cc_proto",
        ":proto_to_flatbuffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "flatbuffer_to_proto",
    srcs = [
        "flatbuffer_to_proto.cc",
    ],
    hdrs = ["flatbuffer_to_proto.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":configuration_cc_proto",
        ":configuration_fbs",
        "//tensorflow/lite:minimal_logging",
        "@flatbuffers",
    ],
)

cc_test(
    name = "flatbuffer_to_proto_test",
    srcs = ["flatbuffer_to_proto_test.cc"],
    deps = [
        ":configuration_cc_proto",
        ":configuration_fbs",
        ":flatbuffer_to_proto",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite(
    name = "delegate_registry",
    hdrs = ["delegate_registry.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    deps = [
        ":configuration_fbs",
        "//tensorflow/lite/core/acceleration/configuration:delegate_registry",
        "//tensorflow/lite/core/c:common",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library_with_tflite(
    name = "delegate_plugin_converter",
    srcs = ["delegate_plugin_converter.cc"],
    hdrs = ["delegate_plugin_converter.h"],
    tflite_deps = [
        "//tensorflow/lite/acceleration/configuration/c:delegate_plugin",
        "//tensorflow/lite/acceleration/configuration:delegate_registry",
        "//tensorflow/lite/c:common",
    ],
    deps = ["@com_google_absl//absl/memory"],
)

cc_library_with_tflite(
    name = "nnapi_plugin",
    compatible_with = get_compatible_with_portable(),
    deps = ["//tensorflow/lite/core/acceleration/configuration:nnapi_plugin"],
)

# Commented under b/279852433 because caused an error in the OSS
# TODO(zhurakovskyi): Uncomment when fixed.
# copybara:uncomment_begin
# cc_library(
#     name = "hexagon_plugin",
#     srcs = ["hexagon_plugin.cc"],
#     deps = [
#         ":configuration_fbs",
#         "@com_google_absl//absl/memory",
#         "//tensorflow/lite/core/acceleration/configuration:delegate_registry",
#     ] + select({
#         "//third_party/bazel_platforms/cpu:aarch64": [
#             "//tensorflow/lite/delegates/hexagon:hexagon_delegate",
#         ],
#         "//third_party/bazel_platforms/cpu:armv7": [
#             "//tensorflow/lite/delegates/hexagon:hexagon_delegate",
#         ],
#         "//conditions:default": [],
#     }),
#     alwayslink = 1,  # For registration to always run.
# )
# copybara:uncomment_end

cc_library(
    name = "gpu_plugin",
    deps = [
        ":gpu_plugin_impl",
    ],
)

common_copts = tflite_copts() + tflite_copts_warnings()

cc_library(
    name = "gpu_plugin_impl",
    srcs = ["gpu_plugin.cc"],
    hdrs = ["gpu_plugin.h"],
    copts = common_copts + select({
        "//tensorflow:ios": [
            "-xobjective-c++",
        ],
        "//tensorflow:macos_arm64": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    visibility = [
        "//tensorflow/lite/core/acceleration/configuration/c:__pkg__",
        "//tensorflow/lite/experimental/acceleration/configuration:__pkg__",
    ],
    deps = [
        ":configuration_fbs",
        "//tensorflow/lite/core/acceleration/configuration:delegate_registry",
        "@com_google_absl//absl/memory",
    ] + select({
        "//tensorflow/lite/delegates/gpu:supports_gpu_delegate": [
            "//tensorflow/lite/delegates/gpu:delegate",
        ],
        "//conditions:default": [],
    }) + select({
        "//tensorflow:ios": [
            "//tensorflow/lite/delegates/gpu:metal_delegate",
        ],
        "//tensorflow:macos_arm64": [
            "//tensorflow/lite/delegates/gpu:metal_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,  # For registration to always run.
)

cc_test(
    name = "gpu_plugin_test",
    srcs = ["gpu_plugin_test.cc"],
    tags = [
        # TODO(b/214492180): Enable the tests for mac/ios too.  This most likely
        # needs TAP to recognize the xobjective-c++ flag.
        "no_mac",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":configuration_cc_proto",
        ":configuration_fbs",
        ":gpu_plugin_impl",
        ":proto_to_flatbuffer",
        "//tensorflow/lite/core/acceleration/configuration:delegate_registry",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library_with_tflite(
    name = "xnnpack_plugin",
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/lite/core/acceleration/configuration:xnnpack_plugin"],
)

cc_test_with_tflite(
    name = "xnnpack_plugin_with_tflite_test",
    srcs = ["xnnpack_plugin_test.cc"],
    # The variant of this test that links against TF Lite in Play services
    # isn't portable to iOS / Mac or Android, because it relies on a separate
    # shared library that isn't included in the executable, and the testing
    # infrastructure for iOS and Android doesn't propagate data dependencies
    # to the test device.  So we disable this test on those devices.
    # TODO(b/306161304): ideally we ought to apply these tags only to the
    # variant for TF Lite in Play services.  In the mean time, we apply those
    # tags to the whole test, but also duplicate the test below using cc_test
    # without the tags.
    tags = [
        "no_mac",
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    tflite_deps = [
        ":xnnpack_plugin",
        "//tensorflow/lite:test_util",
        "//tensorflow/lite/acceleration/configuration:delegate_registry",
    ],
    deps = [
        ":configuration_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@pthreadpool",
    ],
)

# This duplicates xnnnpack_plugin_with_tflite_test above, but without the tags,
# to ensure that this test does get run on iOS and Android.
cc_test(
    name = "xnnpack_plugin_test",
    srcs = ["xnnpack_plugin_test.cc"],
    deps = [
        ":configuration_fbs",
        ":xnnpack_plugin",
        "//tensorflow/lite:test_util",
        "//tensorflow/lite/acceleration/configuration:delegate_registry",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@pthreadpool",
    ],
)

cc_library(
    name = "coreml_plugin",
    srcs = ["coreml_plugin.cc"],
    deps = [
        ":configuration_fbs",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/core/acceleration/configuration:delegate_registry",
        "@com_google_absl//absl/memory",
    ] + select({
        "//tensorflow:macos": [
            "//tensorflow/lite/delegates/coreml:coreml_delegate",
        ],
        "//tensorflow:ios": [
            "//tensorflow/lite/delegates/coreml:coreml_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,  # For registration to always run.
)

# TODO(b/260582614): Add support for TF Lite in Play Services.
cc_library(
    name = "stable_delegate_plugin",
    srcs = ["stable_delegate_plugin.cc"],
    hdrs = ["stable_delegate_plugin.h"],
    deps = [
        ":configuration_fbs",
        "//tensorflow/lite/acceleration/configuration/c:delegate_plugin",
        "//tensorflow/lite/acceleration/configuration/c:stable_delegate",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/acceleration/configuration:delegate_registry",
        "//tensorflow/lite/delegates/utils/experimental/stable_delegate:delegate_loader",
        "//tensorflow/lite/tools:logging",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,  # For registration to always run.
)

cc_test(
    name = "stable_delegate_plugin_test",
    srcs = ["stable_delegate_plugin_test.cc"],
    data = ["//tensorflow/lite/delegates/utils/experimental/stable_delegate:libtensorflowlite_stable_xnnpack_delegate.so"],
    tags = [
        # TODO(b/259303511): Propagate build config to data correctly to enable the test on x86 platforms.
        "no_test_android_x86",
    ],
    deps = [
        ":configuration_fbs",
        ":stable_delegate_plugin",
        "//tensorflow/lite/core/acceleration/configuration:delegate_registry",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@pthreadpool",
    ],
)

tflite_portable_test_suite()
