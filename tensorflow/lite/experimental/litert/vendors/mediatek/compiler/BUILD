# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//tensorflow/lite/experimental/litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib", "litert_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert/vendors/mediatek/compiler:__subpackages__"],
)

litert_dynamic_lib(
    name = "compiler_plugin",
    srcs = ["compiler_plugin.cc"],
    hdrs = ["//tensorflow/lite/experimental/litert/vendors/c:litert_compiler_plugin.h"],
    export_litert_only = True,
    shared_lib_name = "compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_MediaTek.so",
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    ungrte = True,
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":compile_model",
        ":create_model",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:common_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/schema:mediatek_litert_schema",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/schema:neuron_litert_schema",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "create_model",
    srcs = ["create_model.cc"],
    hdrs = ["create_model.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        # copybara:uncomment "//third_party/neuro_pilot:latest_host_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:add_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:batch_matmul_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:common_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:concat_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:fully_connected_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:gelu_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:mean_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:mul_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:operand_map",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:reshape_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:rsqrt_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:softmax_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:sub_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations:transpose_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/schema:mediatek_litert_schema",
    ],
)

cc_library(
    name = "compile_model",
    srcs = ["compile_model.cc"],
    hdrs = ["compile_model.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        # copybara:uncomment "//third_party/neuro_pilot:latest_host_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
        "//tensorflow/lite/experimental/litert/vendors/mediatek/schema:mediatek_litert_schema",
    ],
)

litert_test(
    name = "compiler_plugin_test",
    srcs = [
        "compiler_plugin_test.cc",
    ],
    data = [
        "//tensorflow/lite/experimental/litert/test:mlir_test_data",
        "//tensorflow/lite/experimental/litert/test:tflite_test_data",
    ],
    linkstatic = True,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        # Don't build/test in OS until qnn is available.
        "nobuilder",
        "no_oss",
        # Sanitizer runtime doesn't work with anything that loads libQnnHtp.so.
        "nosan",
    ],
    # Currently this test can only be run on Android because we don't have x86 shared libraries for
    # MTK.
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    ungrte = True,
    use_sys_malloc = True,
    deps = [
        ":compiler_plugin",  # buildcleaner: keep
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:matchers_oss",
        "//tensorflow/lite/experimental/litert/test:test_models",
        "//tensorflow/lite/experimental/litert/vendors/cc:litert_compiler_plugin",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
    ],
)
