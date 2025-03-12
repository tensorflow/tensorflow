# copybara:uncomment_begin(google-only)
# load("//devtools/deps/check:deps_check.bzl", "check_dependencies")
#
# copybara:uncomment_end
load("//tensorflow/lite/experimental/litert/build_common:litert_build_defs.bzl", "copy_file", "export_lrt_runtime_only_linkopt", "export_lrt_runtime_only_script")
load("//tensorflow/lite/experimental/litert/build_common:special_rule.bzl", "gles_deps", "gles_headers", "gles_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
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
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/core:environment",
        "//tensorflow/lite/experimental/litert/runtime/accelerators:auto_registration",
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
        "//tensorflow/lite/experimental/litert/core/model:model_serialize",
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
        "//tensorflow/lite/experimental/litert/test:matchers",
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
        ":litert_common",
        ":litert_options",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:matchers",
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
    name = "litert_tensor_buffer_types",
    srcs = [],
    hdrs = ["litert_tensor_buffer_types.h"],
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
    linkopts = select({
        "//tensorflow:android": [
            "-landroid",
        ],
        "//conditions:default": [],
    }) + gles_linkopts(),
    deps = [
        ":litert_common",
        ":litert_event",
        ":litert_logging",
        ":litert_model",
        ":litert_tensor_buffer_types",
        "//tensorflow/lite/experimental/litert/runtime:tensor_buffer",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ] + gles_deps(),
)

cc_test(
    name = "litert_tensor_buffer_test",
    srcs = [
        "litert_tensor_buffer_test.cc",
    ],
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
    name = "litert_compilation_options",
    srcs = ["litert_compilation_options.cc"],
    hdrs = [
        "litert_compilation_options.h",
    ],
    deps = [
        ":litert_accelerator_options",
        ":litert_common",
        ":litert_logging",
        "//tensorflow/lite/experimental/litert/runtime:compilation_options",
    ],
)

cc_test(
    name = "litert_compilation_options_test",
    srcs = ["litert_compilation_options_test.cc"],
    deps = [
        ":litert_accelerator_options",
        ":litert_common",
        ":litert_compilation_options",
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
        ":litert_compilation_options",
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
        ":litert_compilation_options",
        ":litert_compiled_model",
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

# The same test as `litert_compiled_model_test` but using the shared library `libLiteRtRuntimeCApi.so`.
cc_test(
    name = "litert_compiled_model_shared_lib_test",
    srcs = [
        "litert_compiled_model_test.cc",
    ],
    data = [
        "//tensorflow/lite/experimental/litert/test:testdata/simple_model.tflite",
    ],
    deps = [
        ":litert_runtime_c_api_shared_lib",
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
        "//tensorflow/lite/experimental/litert/test:matchers",
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
        ":litert_accelerator_options",
        ":litert_common",
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
        ":litert_accelerator_options",
        ":litert_accelerator_registration",
        ":litert_common",
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
    ":litert_compilation_options",
    ":litert_dispatch_delegate",
    ":litert_event",
    ":litert_environment",
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
cc_shared_library(
    name = "litert_runtime_c_api_so",
    additional_linker_inputs = export_lrt_runtime_only_script(),
    shared_lib_name = "libLiteRtRuntimeCApi.so",
    user_link_flags = export_lrt_runtime_only_linkopt() + select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }) + ["-Wl,-soname=libLiteRtRuntimeCApi.so"],
    deps = LITERT_C_API_COMMON_DEPS,
)

cc_library(
    name = "litert_dispatch_headers",
    hdrs = [
        ":litert_environment.h",
        ":litert_accelerator.h",
        ":litert_accelerator_options.h",
        ":litert_any.h",
        ":litert_common.h",
        ":litert_compiled_model.h",
        ":litert_compilation_options.h",
        ":litert_event.h",
        ":litert_layout.h",
        ":litert_logging.h",
        ":litert_model.h",
        ":litert_op_code.h",
        ":litert_options.h",
        ":litert_tensor_buffer.h",
        ":litert_tensor_buffer_requirements.h",
        ":litert_tensor_buffer_types.h",
        # Needed for litert/c/litert_op_code.h
        "//tensorflow/lite:builtin_ops.h",
        # Neeeded for litert/c/litert_model.h
        "//tensorflow/lite/c:tensorflowlite_c_api_hdrs_filegroup",
        "//tensorflow/lite/core/c:headers_filegroup",
    ],  # Export all header files (.h) in this directory
    deps = [
        "@opencl_headers",
    ],
)

copy_file(
    name = "copy_litert_runtime_c_api_so",
    src = "//tensorflow/lite/experimental/litert/c:litert_runtime_c_api_so",
    target = "libLiteRtRuntimeCApi.so",
)

# This is cc_library target based on `libLiteRtRuntimeCApi.so`.
cc_library(
    name = "litert_runtime_c_api_shared_lib",
    srcs = [":litert_runtime_c_api_so"],
    hdrs = glob(["litert_*.h"]) + [
        # Needed for litert/c/litert_op_code.h
        "//tensorflow/lite:builtin_ops.h",
        # Neeeded for litert/c/litert_model.h
        "//tensorflow/lite/c:tensorflowlite_c_api_hdrs_filegroup",
        "//tensorflow/lite/core/c:headers_filegroup",
    ],
    linkstatic = 1,
    deps = [
        # only depend on headers
        "@opencl_headers",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_dispatch_c_api",
    ] + gles_headers(),
)

# copybara:uncomment_begin(google-only)
# # Check that litert runtime doesn't depend on MLIR.
# check_dependencies(
#     of = [":litert_runtime_c_api_shared_lib"],
#     dont_match_regexp = "^//third_party/llvm/llvm-project/mlir",
# )
# copybara:uncomment_end

exports_files(srcs = glob(["litert_*.h"]))
