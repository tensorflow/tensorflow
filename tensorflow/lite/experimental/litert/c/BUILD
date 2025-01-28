load("//tensorflow/lite/experimental/litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment "//third_party/mediapipe/calculators/tensor:__subpackages__",
        # copybara:uncomment "//third_party/odml/infra:__subpackages__",
        "//tensorflow/lite/experimental/litert:__subpackages__",
    ],
)

cc_library(
    name = "litert_common",
    srcs = ["litert_common.cc"],
    hdrs = ["litert_common.h"],
)

cc_test(
    name = "litert_common_test",
    srcs = ["litert_common_test.cc"],
    deps = [
        ":litert_common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_any",
    hdrs = ["litert_any.h"],
)

cc_library(
    name = "litert_environment",
    srcs = ["litert_environment.cc"],
    hdrs = ["litert_environment.h"],
    deps = [
        ":litert_any",
        ":litert_common",
        ":litert_logging",
        "//tensorflow/lite/experimental/litert/core:environment",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "litert_logging",
    srcs = [
        "litert_logging.cc",
    ],
    hdrs = [
        "litert_logging.h",
    ],
    deps = [
        ":litert_common",
        "//tensorflow/lite:minimal_logging",
    ],
)

cc_test(
    name = "litert_logging_test",
    srcs = [
        "litert_logging_test.cc",
    ],
    deps = [
        ":litert_common",
        ":litert_logging",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_layout",
    hdrs = ["litert_layout.h"],
    deps = [
        ":litert_common",
        ":litert_op_code",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "litert_model",
    srcs = ["litert_model.cc"],
    hdrs = ["litert_model.h"],
    deps = [
        ":litert_common",
        ":litert_layout",
        ":litert_op_code",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/core/model:model_load",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "litert_model_test",
    srcs = ["litert_model_test.cc"],
    deps = [
        ":litert_common",
        ":litert_model",
        ":litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/core/util:flatbuffer_tools",
        "//tensorflow/lite/experimental/litert/test:test_macros",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_op_code",
    hdrs = ["litert_op_code.h"],
    deps = ["//tensorflow/lite:builtin_ops"],
)

cc_library(
    name = "litert_options",
    srcs = ["litert_options.cc"],
    hdrs = [
        "litert_options.h",
    ],
    deps = [
        ":litert_common",
        ":litert_op_code",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/litert/core/model",
    ],
)

cc_test(
    name = "litert_options_test",
    srcs = ["litert_options_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:mlir_test_data",
    ],
    tags = ["no_oss"],
    deps = [
        ":litert_options",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:test_macros",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_event",
    srcs = ["litert_event.cc"],
    hdrs = ["litert_event.h"],
    deps = [
        ":litert_common",
        ":litert_logging",
        "//tensorflow/lite/experimental/litert/runtime:event",
    ],
)

cc_library(
    name = "litert_tensor_buffer",
    srcs = [
        "litert_tensor_buffer.cc",
        "litert_tensor_buffer_requirements.cc",
    ],
    hdrs = [
        "litert_tensor_buffer.h",
        "litert_tensor_buffer_requirements.h",
    ],
    deps = [
        ":litert_common",
        ":litert_event",
        ":litert_logging",
        ":litert_model",
        "//tensorflow/lite/experimental/litert/runtime:tensor_buffer",
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
    # require GPU to run OpenCL tests.
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        ":litert_common",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_layout",
        "//tensorflow/lite/experimental/litert/runtime:tensor_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "litert_tensor_buffer_requirements_test",
    srcs = [
        "litert_tensor_buffer_requirements_test.cc",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    deps = [
        ":litert_common",
        ":litert_tensor_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_dispatch_delegate",
    hdrs = [
        "litert_dispatch_delegate.h",
    ],
    deps = [
        ":litert_environment",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils:simple_opaque_delegate",
        "//tensorflow/lite/experimental/litert/runtime/dispatch:dispatch_delegate",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_dispatch_c_api",
    ],
)

cc_library(
    name = "litert_compiled_model_options",
    srcs = ["litert_compiled_model_options.cc"],
    hdrs = [
        "litert_compiled_model_options.h",
    ],
    deps = [
        ":litert_accelerator_options",
        ":litert_common",
        ":litert_logging",
    ],
)

cc_test(
    name = "litert_compiled_model_options_test",
    srcs = ["litert_compiled_model_options_test.cc"],
    deps = [
        ":litert_accelerator_options",
        ":litert_common",
        ":litert_compiled_model_options",
        "//tensorflow/lite/experimental/litert/core:accelerator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_compiled_model",
    srcs = ["litert_compiled_model.cc"],
    hdrs = [
        "litert_compiled_model.h",
    ],
    deps = [
        ":litert_common",
        ":litert_compiled_model_options",
        ":litert_environment",
        ":litert_logging",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/litert/runtime:compiled_model",
    ],
)

cc_test(
    name = "litert_compiled_model_test",
    srcs = [
        "litert_compiled_model_test.cc",
    ],
    data = [
        "//tensorflow/lite/experimental/litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":litert_common",
        ":litert_compiled_model",
        ":litert_compiled_model_options",
        ":litert_environment",
        ":litert_model",
        ":litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:simple_model",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_accelerator_options",
    srcs = ["litert_accelerator_options.cc"],
    hdrs = ["litert_accelerator_options.h"],
    deps = [
        ":litert_common",
        "//tensorflow/lite/experimental/litert/core:accelerator",
    ],
)

cc_test(
    name = "litert_accelerator_options_test",
    srcs = ["litert_accelerator_options_test.cc"],
    deps = [
        ":litert_accelerator_options",
        ":litert_common",
        "//tensorflow/lite/experimental/litert/core:accelerator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_accelerator",
    srcs = ["litert_accelerator.cc"],
    hdrs = ["litert_accelerator.h"],
    deps = [
        ":litert_common",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/core:accelerator",
        "//tensorflow/lite/experimental/litert/core:environment",
    ],
)

cc_test(
    name = "litert_accelerator_test",
    srcs = ["litert_accelerator_test.cc"],
    deps = [
        ":litert_accelerator",
        ":litert_accelerator_registration",
        ":litert_common",
        ":litert_environment",
        "//tensorflow/lite/experimental/litert/core:accelerator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "litert_accelerator_registration",
    srcs = ["litert_accelerator_registration.cc"],
    hdrs = ["litert_accelerator_registration.h"],
    deps = [
        ":litert_accelerator",
        ":litert_common",
        ":litert_compiled_model",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/core:accelerator",
        "//tensorflow/lite/experimental/litert/core:environment",
    ],
)

cc_test(
    name = "litert_accelerator_registration_test",
    srcs = ["litert_accelerator_registration_test.cc"],
    deps = [
        ":litert_accelerator",
        ":litert_accelerator_registration",
        ":litert_common",
        ":litert_compiled_model",
        ":litert_environment",
        "//tensorflow/lite/experimental/litert/core:accelerator",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "litert_model_srcs",
    srcs = ["litert_model.cc"],
    visibility = ["//tensorflow/lite/experimental/litert/core/model:__pkg__"],
)

filegroup(
    name = "litert_model_hdrs",
    srcs = ["litert_model.h"],
    visibility = ["//tensorflow/lite/experimental/litert/core/model:__pkg__"],
)

# Collection of all C API targets.
LITERT_C_API_COMMON_DEPS = [
    ":litert_accelerator",
    ":litert_accelerator_registration",
    ":litert_any",
    ":litert_common",
    ":litert_compiled_model",
    ":litert_compiled_model_options",
    ":litert_dispatch_delegate",
    ":litert_event",
    ":litert_layout",
    ":litert_logging",
    ":litert_model",
    ":litert_op_code",
    ":litert_options",
    ":litert_tensor_buffer",
]

# This test verifies that the C API header files can build via C compiler.
cc_test(
    name = "litert_c_api_common_test",
    srcs = ["litert_c_api_common_test.c"],
    copts = ["--std=c11"],
    linkopts = ["-ldl"],
    deps = LITERT_C_API_COMMON_DEPS,
)

# Build `litert/c:litert_runtime_c_api_so` for `libLiteRtRuntimeCApi.so`.
litert_dynamic_lib(
    name = "litert_runtime_c_api",
    export_litert_only = True,
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    shared_lib_name = "litert_runtime_c_api_so",
    so_name = "libLiteRtRuntimeCApi.so",
    deps = LITERT_C_API_COMMON_DEPS,
)

exports_files(srcs = glob(["litert_*.h"]))
