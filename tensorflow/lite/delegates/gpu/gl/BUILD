package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

cc_library(
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    deps = [
        ":command_queue",
        ":compiler",
        ":compiler_options",
        ":gl_call",
        ":gpu_info",
        ":node_shader",
        ":object",
        ":object_manager",
        ":portable",
        ":runtime",
        ":runtime_options",
        ":stats",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl/workgroups:calculator",
    ] + select({
        "//tensorflow/lite/delegates/gpu:tflite_gpu_binary_release": [],
        "//conditions:default": [
            ":serialization",
        ],
    }),
)

cc_library(
    name = "command_queue",
    srcs = ["command_queue.cc"],
    hdrs = ["command_queue.h"],
    deps = [
        ":gl_call",
        ":gl_program",
        ":gl_sync",
        ":gpu_info",
        ":portable",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/memory",
    ],
)

flatbuffer_cc_library(
    name = "common_cc_fbs",
    srcs = ["common.fbs"],
)

# Generic schema for inference on GPU device.
flatbuffer_cc_library(
    name = "compiled_model_cc_fbs",
    srcs = ["compiled_model.fbs"],
    flatc_args = [
        "--scoped-enums",
    ],
    includes = [
        "//tensorflow/lite/delegates/gpu/gl:common_cc_fbs_includes",
    ],
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    deps = [
        ":compiler_options",
        ":float16_conversions",
        ":gpu_info",
        ":node_shader",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl/compiler:compiled_node",
        "//tensorflow/lite/delegates/gpu/gl/compiler:fuse_auto_input",
        "//tensorflow/lite/delegates/gpu/gl/compiler:fuse_inline",
        "//tensorflow/lite/delegates/gpu/gl/compiler:fuse_inplace",
        "//tensorflow/lite/delegates/gpu/gl/compiler:shader_code",
        "//tensorflow/lite/delegates/gpu/gl/compiler:shader_codegen",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "compiler_options",
    hdrs = ["compiler_options.h"],
    deps = [
        ":gpu_info",
        ":object",
    ],
)

cc_library(
    name = "egl_context",
    srcs = ["egl_context.cc"],
    hdrs = ["egl_context.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":portable",
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "egl_environment",
    srcs = ["egl_environment.cc"],
    hdrs = ["egl_environment.h"],
    deps = [
        ":egl_context",
        ":egl_surface",
        ":gl_call",
        ":gpu_info",
        ":portable",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "egl_surface",
    srcs = ["egl_surface.cc"],
    hdrs = ["egl_surface.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":portable",
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "float16_conversions",
    srcs = ["float16_conversions.cc"],
    hdrs = ["float16_conversions.h"],
    deps = [
        ":object",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "@FP16",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "gl_buffer",
    srcs = ["gl_buffer.cc"],
    hdrs = ["gl_buffer.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":portable",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "gl_buffer_test",
    srcs = ["gl_buffer_test.cc"],
    linkopts = [
        "-lGLESv3",
        "-lEGL",
    ],
    tags = [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":egl_environment",
        ":gl_buffer",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gl_call",
    hdrs = ["gl_call.h"],
    deps = [
        ":gl_errors",
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "gl_errors",
    srcs = ["gl_errors.cc"],
    hdrs = ["gl_errors.h"],
    deps = [
        ":portable",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gl_program",
    srcs = ["gl_program.cc"],
    hdrs = ["gl_program.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":gl_shader",
        ":portable",
        ":uniform_parameter",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "gl_shader",
    srcs = ["gl_shader.cc"],
    hdrs = ["gl_shader.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":portable",
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "gl_texture",
    srcs = ["gl_texture.cc"],
    hdrs = ["gl_texture.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":portable",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gl_sync",
    srcs = ["gl_sync.cc"],
    hdrs = ["gl_sync.h"],
    deps = [
        ":gl_call",
        ":gl_errors",
        ":portable",
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

cc_library(
    name = "gpu_info",
    srcs = ["gpu_info.cc"],
    hdrs = ["gpu_info.h"],
    deps = [
        ":gl_errors",
        ":portable",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)

flatbuffer_cc_library(
    name = "metadata_cc_fbs",
    srcs = ["metadata.fbs"],
    includes = [
        "//tensorflow/lite/delegates/gpu/gl:common_cc_fbs_includes",
        "//tensorflow/lite/delegates/gpu/gl:workgroups_cc_fbs_includes",
    ],
)

cc_library(
    name = "node_shader",
    hdrs = ["node_shader.h"],
    deps = [
        ":compiler_options",
        ":gpu_info",
        ":object",
        ":uniform_parameter",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_library(
    name = "object",
    hdrs = ["object.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "object_manager",
    srcs = ["object_manager.cc"],
    hdrs = ["object_manager.h"],
    deps = [
        ":gl_buffer",
        ":gl_texture",
        ":stats",
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "portable",
    hdrs = [
        "portable_egl.h",
        "portable_gl31.h",
    ],
)

cc_library(
    name = "runtime",
    srcs = ["runtime.cc"],
    hdrs = ["runtime.h"],
    deps = [
        ":command_queue",
        ":gl_buffer",
        ":gl_call",
        ":gl_errors",
        ":gl_program",
        ":gl_shader",
        ":gl_texture",
        ":gpu_info",
        ":object",
        ":object_manager",
        ":portable",
        ":runtime_options",
        ":stats",
        ":uniform_parameter",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl/runtime:shared_buffer",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "runtime_options",
    hdrs = ["runtime_options.h"],
)

cc_library(
    name = "serialization",
    srcs = ["serialization.cc"],
    hdrs = ["serialization.h"],
    deps = [
        ":common_cc_fbs",
        ":compiled_model_cc_fbs",
        ":object",
        ":uniform_parameter",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@flatbuffers",
    ],
)

cc_test(
    name = "serialization_test",
    srcs = ["serialization_test.cc"],
    tags = [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":object",
        ":serialization",
        ":uniform_parameter",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "stats",
    hdrs = ["stats.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "uniform_parameter",
    hdrs = ["uniform_parameter.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/types:variant",
    ],
)

flatbuffer_cc_library(
    name = "workgroups_cc_fbs",
    srcs = ["workgroups.fbs"],
    includes = [
        "//tensorflow/lite/delegates/gpu/gl:common_cc_fbs_includes",
    ],
)

tflite_portable_test_suite()
