# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

# C API for delegate plugins.

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load("//tensorflow/lite:build_def.bzl", "tflite_cc_library_with_c_headers_test", "tflite_copts", "tflite_copts_warnings")
load(
    "//tensorflow/lite/core/acceleration/configuration/c:special_rules.bzl",
    "delegate_plugin_visibility_allowlist",
    "gpu_plugin_visibility_allowlist",
    "xnnpack_plugin_visibility_allowlist",
)
load("//tensorflow/lite/core/c:special_rules.bzl", "experimental_acceleration_api_allowlist")
load("//tensorflow/lite/core/shims:build_defs.bzl", "build_test")
load("//tensorflow/lite/delegates/gpu:build_defs.bzl", "gpu_delegate_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/lite:__subpackages__",
    ] + experimental_acceleration_api_allowlist(),
    licenses = ["notice"],
)

filegroup(
    name = "tflite_internal_cc_3p_api_deps_src",
    srcs = [
        "stable_delegate.h",
    ],
    visibility = [
        "//tensorflow/lite:__pkg__",
    ],
)

# LINT.IfChange(tflite_acceleration_exported_headers)
exports_files([
    "delegate_plugin.h",
    "gpu_plugin.h",
    "xnnpack_plugin.h",
])
# LINT.ThenChange(
#     ../../../../acceleration/configuration/c/BUILD:tflite_acceleration_exported_headers,
#     ../../../../java/BUILD:tflite_acceleration_exported_headers
# )

tflite_cc_library_with_c_headers_test(
    name = "delegate_plugin",
    hdrs = ["delegate_plugin.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ] + delegate_plugin_visibility_allowlist(),
    deps = [
        "//tensorflow/lite/core/c:common",
    ],
)

common_copts = tflite_copts() + tflite_copts_warnings()

tflite_cc_library_with_c_headers_test(
    name = "gpu_plugin",
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
        "//tensorflow/lite:__subpackages__",
    ] + gpu_plugin_visibility_allowlist(),
    deps = [
        ":delegate_plugin",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/acceleration/configuration:gpu_plugin_impl",
        "//tensorflow/lite/core/c:common",
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
)

# For non-Android platforms, this should be built with '--copt=-DCL_DELEGATE_NO_GL'.
# On non-supported platforms (i.e. non-Android platforms if -DCL_DELEGATE_NO_GL wasn't specified),
# the test srcs are set to the empty list, so the test will succeed without testing anything.
cc_test(
    name = "gpu_plugin_test",
    srcs = select({
        "//tensorflow/lite/delegates/gpu:supports_gpu_delegate": ["gpu_plugin_test.cc"],
        "//conditions:default": [],
    }),
    linkopts = gpu_delegate_linkopts(),
    tags = tf_gpu_tests_tags(),
    deps = select({
        "//tensorflow/lite/delegates/gpu:supports_gpu_delegate": [":gpu_plugin"],
        "//conditions:default": [],
    }) + [
        "//tensorflow/lite/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/core/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "nnapi_plugin",
    srcs = ["nnapi_plugin.cc"],
    hdrs = ["nnapi_plugin.h"],
    deps = [
        ":delegate_plugin",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/core/acceleration/configuration:nnapi_plugin",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
    ],
)

cc_test(
    name = "nnapi_plugin_test",
    srcs = ["nnapi_plugin_test.cc"],
    deps = [
        ":nnapi_plugin",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/core/c:common",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "xnnpack_plugin",
    srcs = ["xnnpack_plugin.cc"],
    hdrs = ["xnnpack_plugin.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ] + xnnpack_plugin_visibility_allowlist(),
    deps = [
        ":delegate_plugin",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
    ],
)

cc_test(
    name = "xnnpack_plugin_test",
    srcs = ["xnnpack_plugin_test.cc"],
    deps = [
        ":xnnpack_plugin",
        "//tensorflow/lite/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@pthreadpool",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "stable_delegate",
    hdrs = ["stable_delegate.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite/core/acceleration/configuration/c:delegate_plugin",
    ],
)

# Commented out under the (b/279852433) because caused an error in the OSS
# TODO(zhurakovskyi): Uncomment when fixed.
#
# copybara:uncomment_begin
# # This rule invokes the "flatcc" FlatBuffer C API compiler to generate the sources
# # use by the ":configuration_c_fbs" C library rule below.
# genrule(
#     name = "configuration_c_fbs_gen",
#     srcs = ["//tensorflow/lite/acceleration/configuration:configuration.fbs"],
#     outs = [
#         "configuration_builder.h",
#         "configuration_reader.h",
#     ],
#     cmd = "$(location //third_party/flatcc:flatcc) -o$(RULEDIR) --builder --reader $(SRCS)",
#     tools = ["//third_party/flatcc"],
#     # Currently this only enables the API for _building_ configuration flatbuffer objects,
#     # not the APIs for reading them, verifying them, or converting them to/from JSON.
#     # [If you need to enable those, replace the two lines above with the following
#     #   outs = ["configuration_builder.h", "configuration_reader.h", "configuration_verifier.h",
#     #           "configuration_json_parser.h", "configuration_json_printer.h"],
#     #   cmd = "$(location //third_party/flatcc:flatcc) -o$(RULEDIR) " +
#     #         "--builder --reader --verifier --json $(SRCS)",
#     # and then in the rule below -- or preferably in a separate target --
#     # add the additional header files in "hdrs" and fix the dependencies.]
# )
#
# # This rule defines a C library containing the Flatbuffer-generated C API for constructing objects
# # using the FlatBuffer schema generated from tensorflow/lite/acceleration/configuration/configuration.proto,
# # which defines the 'TFLiteSettings' FlatBuffer table and related types.
# tflite_cc_library_with_c_headers_test(
#     name = "configuration_c_fbs",
#     hdrs = [
#         "configuration_builder.h",
#         "configuration_reader.h",
#     ],
#     deps = ["//third_party/flatcc:runtime"],
# )
#
# build_test(
#     name = "configuration_c_fbs_build_test",
#     targets = [
#         ":configuration_c_fbs",
#     ],
# )
# copybara:uncomment_end
