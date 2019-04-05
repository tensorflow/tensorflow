package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

cc_library(
    name = "add",
    srcs = ["add.cc"],
    hdrs = ["add.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "add_test",
    srcs = ["add_test.cc"],
    linkopts = [
        "-lEGL",
        "-lGLESv3",
    ],
    tags = [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":add",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "concat",
    srcs = ["concat.cc"],
    hdrs = ["concat.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "conv",
    srcs = ["conv.cc"],
    hdrs = ["conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl/workgroups:ideal_workgroup_picker",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "depthwise_conv",
    srcs = ["depthwise_conv.cc"],
    hdrs = ["depthwise_conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl/workgroups:ideal_workgroup_picker",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "elementwise",
    srcs = ["elementwise.cc"],
    hdrs = ["elementwise.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "elementwise_test",
    srcs = ["elementwise_test.cc"],
    linkopts = [
        "-lEGL",
        "-lGLESv3",
    ],
    tags = [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":elementwise",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fully_connected",
    srcs = ["fully_connected.cc"],
    hdrs = ["fully_connected.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "lstm",
    srcs = ["lstm.cc"],
    hdrs = ["lstm.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "max_unpooling",
    srcs = ["max_unpooling.cc"],
    hdrs = ["max_unpooling.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "mul",
    srcs = ["mul.cc"],
    hdrs = ["mul.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "pad",
    srcs = ["pad.cc"],
    hdrs = ["pad.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "pooling",
    srcs = ["pooling.cc"],
    hdrs = ["pooling.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "prelu",
    srcs = ["prelu.cc"],
    hdrs = ["prelu.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "relu",
    srcs = ["relu.cc"],
    hdrs = ["relu.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "relu_test",
    srcs = ["relu_test.cc"],
    linkopts = [
        "-lEGL",
        "-lGLESv3",
    ],
    tags = [
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":relu",
        ":test_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "reshape",
    srcs = ["reshape.cc"],
    hdrs = ["reshape.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "slice",
    srcs = ["slice.cc"],
    hdrs = ["slice.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "softmax",
    srcs = ["softmax.cc"],
    hdrs = ["softmax.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/gl:api",
        "//tensorflow/lite/delegates/gpu/gl:compiler_options",
        "//tensorflow/lite/delegates/gpu/gl:egl_environment",
        "//tensorflow/lite/delegates/gpu/gl:gl_buffer",
        "//tensorflow/lite/delegates/gpu/gl:gpu_info",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "//tensorflow/lite/delegates/gpu/gl:object_manager",
        "//tensorflow/lite/delegates/gpu/gl:runtime_options",
        "//tensorflow/lite/delegates/gpu/gl/workgroups:default_calculator",
    ],
)

cc_library(
    name = "transpose_conv",
    srcs = ["transpose_conv.cc"],
    hdrs = ["transpose_conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "upsampling_bilinear",
    srcs = ["upsampling_bilinear.cc"],
    hdrs = ["upsampling_bilinear.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

TFLITE_GPU_BINARY_RELEASE_OPERATORS = [
    "add",
    "concat",
    "conv",
    "depthwise_conv",
    "elementwise",
    "fully_connected",
    "lstm",
    "mul",
    "pad",
    "pooling",
    "prelu",
    "relu",
    "reshape",
    "slice",
    "softmax",
    "transpose_conv",
    "upsampling_bilinear",
]

NON_TFLITE_GPU_BINARY_RELEASE_OPERATORS = [
    "max_unpooling",
]

cc_library(
    name = "registry",
    srcs = ["registry.cc"],
    hdrs = ["registry.h"],
    visibility = ["//visibility:public"],
    deps = [":" + op_name for op_name in TFLITE_GPU_BINARY_RELEASE_OPERATORS] +
           select({
               "//tensorflow/lite/delegates/gpu:tflite_gpu_binary_release": [],
               "//conditions:default": NON_TFLITE_GPU_BINARY_RELEASE_OPERATORS,
           }) + [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tflite_portable_test_suite()
