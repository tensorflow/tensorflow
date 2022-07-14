load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load("//tensorflow:tensorflow.bzl", "if_google", "workspace_root")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

config_setting(
    name = "opencl_delegate_no_gl",
    values = {"copt": "-DCL_DELEGATE_NO_GL"},
)

cc_library(
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    deps = select({
        ":opencl_delegate_no_gl": [],
        "//conditions:default": [
            ":egl_sync",
            ":gl_interop",
        ],
    }) + [
        ":cl_command_queue",
        ":cl_errors",
        ":cl_event",
        ":environment",
        ":inference_context",
        ":opencl_wrapper",
        ":tensor",
        ":tensor_type_util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "//tensorflow/lite/delegates/gpu:api",
        "//tensorflow/lite/delegates/gpu/cl/kernels:converter",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
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
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "@com_google_absl//absl/status",
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
    name = "cl_arguments",
    srcs = ["cl_arguments.cc"],
    hdrs = ["cl_arguments.h"],
    deps = [
        ":buffer",
        ":cl_context",
        ":gpu_object",
        ":linear_storage",
        ":tensor",
        ":texture2d",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/task:arguments",
        "//tensorflow/lite/delegates/gpu/common/task:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "cl_arguments_test",
    srcs = ["cl_arguments_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":buffer",
        ":cl_arguments",
        ":gpu_object",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
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
        "//tensorflow/lite/delegates/gpu/common/task:profiling_info",
        "@com_google_absl//absl/strings",
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
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/experimental/acceleration/compatibility:android_info",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "cl_device_test",
    srcs = ["cl_device_test.cc"],
    deps = [
        ":cl_device",
        "@com_google_googletest//:gtest_main",
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
        "//tensorflow/lite/delegates/gpu/common:kernel_info",
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
    name = "cl_operation",
    srcs = ["cl_operation.cc"],
    hdrs = ["cl_operation.h"],
    deps = [
        ":cl_arguments",
        ":cl_command_queue",
        ":cl_context",
        ":cl_device",
        ":cl_kernel",
        ":program_cache",
        ":tensor",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
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
        "//tensorflow/lite/delegates/gpu/common/task:compiler_options",
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
        ":program_cache",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
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
        "//tensorflow/lite/delegates/gpu/common/transformations:model_transformations",
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
        "//tensorflow/lite/delegates/gpu/common/task:gpu_object_desc",
    ],
)

cc_library(
    name = "inference_context",
    srcs = [
        "inference_context.cc",
    ],
    hdrs = [
        "inference_context.h",
    ],
    deps = [
        ":buffer",
        ":cl_command_queue",
        ":cl_device",
        ":cl_operation",
        ":environment",
        ":gpu_object",
        ":opencl_wrapper",
        ":recordable_queue_builder",
        ":serialization_cc_fbs",
        ":tensor",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:gpu_model",
        "//tensorflow/lite/delegates/gpu/common:gpu_model_cc_fbs",
        "//tensorflow/lite/delegates/gpu/common:memory_management",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_hints",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/selectors:operation_selector",
        "//tensorflow/lite/delegates/gpu/common/selectors:special_selector",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:serialization_base",
        "//tensorflow/lite/delegates/gpu/common/task:storage_type_util",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "linear_storage",
    srcs = ["linear_storage.cc"],
    hdrs = ["linear_storage.h"],
    deps = [
        ":cl_context",
        ":cl_image_format",
        ":gpu_object",
        ":opencl_wrapper",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
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
        "//tensorflow/lite/delegates/gpu/cl/" + if_google("google", "default") + ":qcom_wrapper",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@farmhash_archive//:farmhash",
        "@flatbuffers",
    ],
)

cc_library(
    name = "recordable_queue",
    hdrs = ["recordable_queue.h"],
    deps = [
        ":cl_command_queue",
        ":cl_context",
        ":cl_device",
        ":cl_operation",
        ":opencl_wrapper",
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "recordable_queue_builder",
    hdrs = ["recordable_queue_builder.h"],
    deps = [
        ":cl_context",
        ":cl_device",
        ":cl_operation",
        ":recordable_queue",
        "//tensorflow/lite/delegates/gpu/cl/" + if_google("google", "default") + ":recordable_queue",
    ],
)

flatbuffer_cc_library(
    name = "serialization_cc_fbs",
    srcs = ["serialization.fbs"],
    flatc_args = [
        "--scoped-enums",
        "-I " + workspace_root,
    ],
    includes = [
        "//tensorflow/lite/delegates/gpu/common:gpu_model_cc_fbs_includes",
        "//tensorflow/lite/delegates/gpu/common/task:serialization_base_cc_fbs_includes",
    ],
)

cc_library(
    name = "tensor",
    srcs = ["tensor.cc"],
    hdrs = ["tensor.h"],
    deps = [
        ":buffer",
        ":cl_command_queue",
        ":cl_context",
        ":cl_device",
        ":cl_image_format",
        ":cl_memory",
        ":gpu_object",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_tensor",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:texture2d_desc",
        "@com_google_absl//absl/strings",
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
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tensor_type_util",
    srcs = ["tensor_type_util.cc"],
    hdrs = ["tensor_type_util.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu:api",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
    ],
)

cc_library(
    name = "texture2d",
    srcs = ["texture2d.cc"],
    hdrs = ["texture2d.h"],
    deps = [
        ":cl_command_queue",
        ":cl_context",
        ":cl_image_format",
        ":gpu_object",
        ":opencl_wrapper",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:texture2d_desc",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
