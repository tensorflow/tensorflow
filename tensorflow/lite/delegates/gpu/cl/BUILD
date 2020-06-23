load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    deps = [
        ":cl_command_queue",
        ":cl_errors",
        ":cl_event",
        ":egl_sync",
        ":environment",
        ":gl_interop",
        ":inference_context",
        ":opencl_wrapper",
        ":precision",
        ":tensor",
        ":tensor_type",
        ":tensor_type_util",
        "//tensorflow/lite/delegates/gpu:api",
        "//tensorflow/lite/delegates/gpu/cl/kernels:converter",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "arguments",
    srcs = ["arguments.cc"],
    hdrs = ["arguments.h"],
    deps = [
        ":cl_device",
        ":gpu_object",
        ":opencl_wrapper",
        ":tensor_type",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:access_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "arguments_test",
    srcs = ["arguments_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":arguments",
        ":gpu_object",
        ":tensor_type",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "buffer",
    srcs = ["buffer.cc"],
    hdrs = ["buffer.h"],
    deps = [
        ":cl_command_queue",
        ":cl_context",
        ":gpu_object",
        ":opencl_wrapper",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "buffer_test",
    srcs = ["buffer_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":buffer",
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cl_test",
    testonly = 1,
    hdrs = ["cl_test.h"],
    deps = [
        ":environment",
        ":opencl_wrapper",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "cl_command_queue",
    srcs = ["cl_command_queue.cc"],
    hdrs = ["cl_command_queue.h"],
    deps = [
        ":cl_context",
        ":cl_device",
        ":cl_event",
        ":cl_kernel",
        ":opencl_wrapper",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "cl_context",
    srcs = ["cl_context.cc"],
    hdrs = ["cl_context.h"],
    deps = [
        ":cl_device",
        ":cl_image_format",
        ":opencl_wrapper",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cl_device",
    srcs = ["cl_device.cc"],
    hdrs = ["cl_device.h"],
    deps = [
        ":opencl_wrapper",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cl_errors",
    hdrs = ["cl_errors.h"],
    deps = [
        ":util",
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "cl_event",
    srcs = ["cl_event.cc"],
    hdrs = ["cl_event.h"],
    deps = [
        ":opencl_wrapper",
    ],
)

cc_library(
    name = "cl_image_format",
    srcs = ["cl_image_format.cc"],
    hdrs = ["cl_image_format.h"],
    deps = [
        ":opencl_wrapper",
        "//tensorflow/lite/delegates/gpu/common:data_type",
    ],
)

cc_library(
    name = "cl_kernel",
    srcs = ["cl_kernel.cc"],
    hdrs = ["cl_kernel.h"],
    deps = [
        ":cl_context",
        ":cl_device",
        ":cl_program",
        ":opencl_wrapper",
        ":util",
        "//tensorflow/lite/delegates/gpu/cl/kernels:flt_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cl_memory",
    srcs = ["cl_memory.cc"],
    hdrs = ["cl_memory.h"],
    deps = [
        ":opencl_wrapper",
        "//tensorflow/lite/delegates/gpu/common:access_type",
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "cl_program",
    srcs = ["cl_program.cc"],
    hdrs = ["cl_program.h"],
    deps = [
        ":cl_context",
        ":cl_device",
        ":opencl_wrapper",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

flatbuffer_cc_library(
    name = "compiled_program_cache_cc_fbs",
    srcs = ["compiled_program_cache.fbs"],
    flatc_args = [
        "--scoped-enums",
    ],
)

cc_library(
    name = "egl_sync",
    srcs = ["egl_sync.cc"],
    hdrs = ["egl_sync.h"],
    defines = [
        "EGL_EGLEXT_PROTOTYPES",
    ],
    deps = [
        ":cl_device",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/gl:gl_call",
    ],
)

cc_library(
    name = "environment",
    srcs = ["environment.cc"],
    hdrs = ["environment.h"],
    deps = [
        ":cl_command_queue",
        ":cl_context",
        ":cl_device",
        ":cl_kernel",
        ":precision",
        ":program_cache",
        ":tensor",
        ":tensor_type",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
    ],
)

cc_library(
    name = "gl_interop",
    srcs = ["gl_interop.cc"],
    hdrs = ["gl_interop.h"],
    deps = [
        ":cl_command_queue",
        ":cl_context",
        ":cl_device",
        ":cl_errors",
        ":cl_event",
        ":cl_memory",
        ":egl_sync",
        ":environment",
        ":opencl_wrapper",
        "//tensorflow/lite/delegates/gpu:spi",
        "//tensorflow/lite/delegates/gpu/common:access_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/gl:gl_call",
        "//tensorflow/lite/delegates/gpu/gl:gl_sync",
        "//tensorflow/lite/delegates/gpu/gl:portable",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_api_delegate",
    srcs = ["gpu_api_delegate.cc"],
    hdrs = ["gpu_api_delegate.h"],
    linkopts = select({
        "//tensorflow:android": [
            "-lEGL",
            "-lGLESv3",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":api",
        ":opencl_wrapper",
        ":tensor_type_util",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/gpu:api",
        "//tensorflow/lite/delegates/gpu:delegate",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_builder",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/transformations:general_transformations",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gpu_object",
    hdrs = ["gpu_object.h"],
    deps = [
        ":opencl_wrapper",
        "//tensorflow/lite/delegates/gpu/common:access_type",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "inference_context",
    srcs = ["inference_context.cc"],
    hdrs = ["inference_context.h"],
    deps = [
        ":buffer",
        ":cl_command_queue",
        ":cl_device",
        ":environment",
        ":model_hints",
        ":opencl_wrapper",
        ":precision",
        ":storage_type_util",
        ":tensor_type",
        "//tensorflow/lite/delegates/gpu/cl/kernels:gpu_operation",
        "//tensorflow/lite/delegates/gpu/cl/selectors:operation_selector",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:memory_management",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/transformations:add_bias",
        "//tensorflow/lite/delegates/gpu/common/transformations:merge_padding_with",
    ],
)

cc_library(
    name = "linear_storage",
    srcs = ["linear_storage.cc"],
    hdrs = ["linear_storage.h"],
    deps = [
        ":buffer",
        ":gpu_object",
        ":opencl_wrapper",
        ":tensor_type",
        ":texture2d",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "model_hints",
    hdrs = ["model_hints.h"],
)

cc_library(
    name = "opencl_wrapper",
    srcs = ["opencl_wrapper.cc"],
    hdrs = ["opencl_wrapper.h"],
    linkopts = select({
        "//tensorflow:android": [
            "-ldl",  # opencl_wrapper calls dlopen()
            "-lm",
        ],
        "//conditions:default": ["-ldl"],  # opencl_wrapper calls dlopen()
    }),
    deps = [
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
        "@opencl_headers",
    ],
)

cc_library(
    name = "precision",
    srcs = ["precision.cc"],
    hdrs = ["precision.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
    ],
)

cc_library(
    name = "program_cache",
    srcs = ["program_cache.cc"],
    hdrs = ["program_cache.h"],
    deps = [
        ":cl_context",
        ":cl_device",
        ":cl_kernel",
        ":cl_program",
        ":compiled_program_cache_cc_fbs",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/types:span",
        "@farmhash_archive//:farmhash",
        "@flatbuffers",
    ],
)

cc_library(
    name = "storage_type_util",
    srcs = ["storage_type_util.cc"],
    hdrs = ["storage_type_util.h"],
    deps = [
        ":cl_context",
        ":cl_device",
        ":tensor_type",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:shape",
    ],
)

cc_library(
    name = "tensor",
    srcs = ["tensor.cc"],
    hdrs = ["tensor.h"],
    deps = [
        ":cl_command_queue",
        ":cl_context",
        ":cl_device",
        ":cl_image_format",
        ":cl_memory",
        ":gpu_object",
        ":tensor_type",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":tensor",
        ":tensor_type",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tensor_type",
    srcs = ["tensor_type.cc"],
    hdrs = ["tensor_type.h"],
    deps = [
        ":gpu_object",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tensor_type_util",
    srcs = ["tensor_type_util.cc"],
    hdrs = ["tensor_type_util.h"],
    deps = [
        ":tensor_type",
        "//tensorflow/lite/delegates/gpu:api",
    ],
)

cc_library(
    name = "texture2d",
    srcs = ["texture2d.cc"],
    hdrs = ["texture2d.h"],
    deps = [
        ":cl_command_queue",
        ":cl_context",
        ":gpu_object",
        ":opencl_wrapper",
        ":tensor_type",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "texture2d_test",
    srcs = ["texture2d_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":texture2d",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        ":opencl_wrapper",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
