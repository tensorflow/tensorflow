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

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "litert_environment",
    hdrs = ["litert_environment.h"],
    deps = [
        ":litert_any",
        ":litert_expected",
        ":litert_handle",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_environment",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_environment_test",
    srcs = [
        "litert_environment_test.cc",
    ],
    linkopts = select({
        "//tensorflow:android": ["-llog"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_any",
        ":litert_compiled_model",
        ":litert_environment",
        ":litert_expected",
        ":litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:matchers",
        "//tensorflow/lite/experimental/litert/test:simple_model",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_event",
    hdrs = ["litert_event.h"],
    deps = [
        ":litert_expected",
        ":litert_handle",
        ":litert_macros",
        "//tensorflow/lite/experimental/litert/c:litert_event",
        "//tensorflow/lite/experimental/litert/c:litert_event_type",
    ],
)

cc_library(
    name = "litert_any",
    hdrs = ["litert_any.h"],
    deps = [
        ":litert_expected",
        ":litert_macros",
        "//tensorflow/lite/experimental/litert/c:litert_any",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_any_test",
    srcs = [
        "litert_any_test.cc",
    ],
    linkopts = select({
        "//tensorflow:android": ["-llog"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_any",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_consts",
    hdrs = [
        "litert_consts.h",
    ],
)

cc_library(
    name = "litert_model",
    srcs = ["litert_model.cc"],
    hdrs = [
        "litert_model.h",
    ],
    deps = [
        ":litert_buffer_ref",
        ":litert_consts",
        ":litert_detail",
        ":litert_element_type",
        ":litert_expected",
        ":litert_handle",
        ":litert_layout",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_model_test",
    srcs = [
        "litert_model_test.cc",
    ],
    data = [
        "//tensorflow/lite/experimental/litert/test:mlir_test_data",
    ],
    deps = [
        ":litert_element_type",
        ":litert_layout",
        ":litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/test:common",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_handle",
    hdrs = ["litert_handle.h"],
)

cc_library(
    name = "litert_tensor_buffer",
    hdrs = [
        "litert_tensor_buffer.h",
        "litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":litert_detail",
        ":litert_event",
        ":litert_expected",
        ":litert_handle",
        ":litert_macros",
        ":litert_model",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_event",
        "//tensorflow/lite/experimental/litert/c:litert_gl_types",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer_types",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ],
)

cc_test(
    name = "litert_tensor_buffer_test",
    srcs = [
        "litert_tensor_buffer_test.cc",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_element_type",
        ":litert_event",
        ":litert_layout",
        ":litert_macros",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer_types",
        "//tensorflow/lite/experimental/litert/runtime:tensor_buffer",
        "//tensorflow/lite/experimental/litert/test:matchers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/lite/delegates/gpu/gl:egl_environment",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "litert_tensor_buffer_requirements",
    hdrs = [
        "litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":litert_detail",
        ":litert_handle",
        ":litert_macros",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_tensor_buffer_requirements_test",
    srcs = [
        "litert_tensor_buffer_requirements_test.cc",
    ],
    deps = [
        ":litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_buffer_ref",
    hdrs = [
        "litert_buffer_ref.h",
    ],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_macros",
    srcs = ["litert_macros.cc"],
    hdrs = ["litert_macros.h"],
    deps = [
        ":litert_expected",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "litert_macros_test",
    srcs = ["litert_macros_test.cc"],
    deps = [
        ":litert_expected",
        ":litert_macros",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_expected",
    hdrs = ["litert_expected.h"],
    deps = [
        ":litert_detail",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "litert_expected_test",
    srcs = ["litert_expected_test.cc"],
    deps = [
        ":litert_buffer_ref",
        ":litert_expected",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_detail",
    hdrs = ["litert_detail.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "@com_google_absl//absl/log:absl_check",
    ],
)

# Dispatch Delegate of LiteRt.
# Warning: This API is not ABI stable and is subject to change.
cc_library(
    name = "litert_dispatch_delegate",
    hdrs = [
        "litert_dispatch_delegate.h",
    ],
    deps = [
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils:simple_opaque_delegate",
        "//tensorflow/lite/experimental/litert/c:litert_environment_options",
        "//tensorflow/lite/experimental/litert/runtime/dispatch:dispatch_delegate",
    ],
)

cc_test(
    name = "litert_buffer_ref_test",
    srcs = ["litert_buffer_ref_test.cc"],
    deps = [
        ":litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_element_type",
    hdrs = ["litert_element_type.h"],
    deps = ["//tensorflow/lite/experimental/litert/c:litert_model"],
)

cc_test(
    name = "litert_element_type_test",
    srcs = ["litert_element_type_test.cc"],
    deps = [
        ":litert_element_type",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_model_predicates",
    srcs = ["litert_model_predicates.cc"],
    hdrs = ["litert_model_predicates.h"],
    deps = [
        ":litert_detail",
        ":litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_layout",
    hdrs = ["litert_layout.h"],
    deps = [
        ":litert_consts",
        "//tensorflow/lite/experimental/litert/c:litert_layout",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "litert_model_predicates_test",
    srcs = ["litert_model_predicates_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:mlir_test_data",
    ],
    deps = [
        ":litert_element_type",
        ":litert_model",
        ":litert_model_predicates",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/test:common",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "litert_layout_test",
    srcs = ["litert_layout_test.cc"],
    deps = [
        ":litert_layout",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_compiled_model",
    srcs = ["litert_compiled_model.cc"],
    hdrs = ["litert_compiled_model.h"],
    deps = [
        ":litert_compilation_options",
        ":litert_environment",
        ":litert_expected",
        ":litert_handle",
        ":litert_macros",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_compiled_model",
        "//tensorflow/lite/experimental/litert/c:litert_environment",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "litert_compilation_options",
    hdrs = ["litert_compilation_options.h"],
    deps = [
        ":litert_accelerator_compilation_options",
        ":litert_environment",
        ":litert_expected",
        ":litert_handle",
        ":litert_macros",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/experimental/litert/c:litert_accelerator_compilation_options",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_compilation_options",
        "//tensorflow/lite/experimental/litert/c:litert_compiled_model",
        "//tensorflow/lite/experimental/litert/c:litert_environment",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_compiled_model_test",
    srcs = ["litert_compiled_model_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":litert_compiled_model",
        ":litert_environment",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer_types",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:matchers",
        "//tensorflow/lite/experimental/litert/test:simple_model",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "litert_compiled_model_integration_test",
    srcs = ["litert_compiled_model_integration_test.cc"],
    deps = [
        ":litert_buffer_ref",
        ":litert_compiled_model",
        ":litert_environment",
        ":litert_event",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_dispatch_headers",
        "//tensorflow/lite/experimental/litert/core/model:model_buffer",
        "//tensorflow/lite/experimental/litert/runtime:tensor_buffer",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:matchers",
        "//tensorflow/lite/experimental/litert/test:simple_model",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/lite/delegates/gpu/gl:egl_environment",
        ],
        "//conditions:default": [],
    }),
)

# copybara:uncomment_begin(google-only)
# cc_test(
#     name = "litert_compiled_model_gpu_test",
#     srcs = ["litert_compiled_model_gpu_test.cc"],
#     data = [
#         "//tensorflow/lite/experimental/litert/test:testdata/simple_model.tflite",
#     ],
#     env = {
#         "ASAN_OPTIONS": "detect_odr_violation=0",
#     },
#     tags = [
#         "requires-gpu-nvidia",
#     ],
#     deps = [
#         ":litert_compiled_model",
#         ":litert_environment",
#         ":litert_event",
#         ":litert_model",
#         ":litert_tensor_buffer",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/debugging:leak_check",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/strings:string_view",
#         "@com_google_absl//absl/types:span",
#         "//third_party/odml/infra/ml_drift_delegate/litert:ml_drift_cl_accelerator",  # buildcleaner: keep
#         "//tensorflow/lite:framework",
#         "//tensorflow/lite/c:c_api_opaque",
#         "//tensorflow/lite/c:common",
#         "//tensorflow/lite/experimental/litert/c:litert_common",
#         "//tensorflow/lite/experimental/litert/c:litert_event",
#         "//tensorflow/lite/experimental/litert/c:litert_event_type",
#         "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer_types",
#         "//tensorflow/lite/experimental/litert/test:common",
#         "//tensorflow/lite/experimental/litert/test:matchers",
#         "//tensorflow/lite/experimental/litert/test:simple_model",
#         "//tensorflow/lite/kernels:builtin_ops",
#     ],
# )
#
# # The same test as above, but for Android.
# # This test doesn't run on TAP.
# # libLiteRtGpuAccelerator.so and libLiteRtRuntimeCApi.so are required to run this test.
# cc_test(
#     name = "litert_compiled_model_gpu_android_test",
#     srcs = ["litert_compiled_model_gpu_test.cc"],
#     data = [
#         "//tensorflow/lite/experimental/litert/test:testdata/simple_model.tflite",
#     ],
#     tags = [
#         "manual",
#         "notap",
#     ],
#     deps = [
#         ":litert_compiled_model",
#         ":litert_environment",
#         ":litert_event",
#         ":litert_model",
#         ":litert_tensor_buffer",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/debugging:leak_check",
#         "@com_google_absl//absl/log:absl_log",
#         "@com_google_absl//absl/strings:string_view",
#         "@com_google_absl//absl/types:span",
#         "//third_party/odml/infra/ml_drift_delegate/litert:ml_drift_cl_accelerator_shared_lib",  # buildcleaner: keep
#         "//tensorflow/lite:framework",
#         "//tensorflow/lite/c:c_api_opaque",
#         "//tensorflow/lite/c:common",
#         "//tensorflow/lite/experimental/litert/c:litert_runtime_c_api_shared_lib",
#         "//tensorflow/lite/experimental/litert/test:common",
#         "//tensorflow/lite/experimental/litert/test:matchers",
#         "//tensorflow/lite/experimental/litert/test:simple_model",
#         "//tensorflow/lite/kernels:builtin_ops",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "litert_tensor_buffer_utils",
    srcs = ["litert_tensor_buffer_utils.cc"],
    hdrs = ["litert_tensor_buffer_utils.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer_types",
    ],
)

cc_library(
    name = "litert_op_options",
    srcs = ["litert_op_options.cc"],
    hdrs = ["litert_op_options.h"],
    deps = [
        ":litert_expected",
        ":litert_macros",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_op_options_test",
    srcs = ["litert_op_options_test.cc"],
    deps = [
        ":litert_op_options",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_shared_library",
    srcs = ["litert_shared_library.cc"],
    hdrs = ["litert_shared_library.h"],
    deps = [
        ":litert_expected",
        ":litert_macros",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "test_litert_shared_library_impl",
    srcs = ["test_shared_library.cc"],
)

cc_shared_library(
    name = "test_litert_shared_library",
    shared_lib_name = "test_shared_library.so",
    deps = [":test_litert_shared_library_impl"],
)

cc_test(
    name = "litert_shared_library_test",
    srcs = ["litert_shared_library_test.cc"],
    data = [":test_litert_shared_library"],
    defines = ["LITERT_DEFINE_GTEST_STATUS_PRINTER"],
    deps = [
        ":litert_shared_library",
        "//tensorflow/lite/experimental/litert/test:matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "litert_event_test",
    srcs = ["litert_event_test.cc"],
    deps = [
        ":litert_event",
        "//tensorflow/lite/experimental/litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_accelerator_compilation_options",
    hdrs = [
        "litert_accelerator_compilation_options.h",
    ],
    deps = [
        ":litert_expected",
        ":litert_handle",
        ":litert_macros",
        "//tensorflow/lite/experimental/litert/c:litert_accelerator_compilation_options",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "@com_google_absl//absl/strings:string_view",
    ],
)

exports_files(srcs = glob(["litert_*.h"]))
