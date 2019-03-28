package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "kernels",
    deps = [
        ":abs",
        ":add",
        ":concat",
        ":convolution",
        ":convolution1x1",
        ":convolution_generic",
        ":depth_wise_conv3x3_stride1x1",
        ":depth_wise_convolution",
        ":fully_connected",
        ":max_unpooling",
        ":mul",
        ":padding",
        ":pooling",
        ":prelu",
        ":relu",
        ":reshape",
        ":sigmoid",
        ":slice",
        ":softmax",
        ":sub",
        ":transpose_conv",
        ":upsample",
    ],
)

cc_library(
    name = "abs",
    srcs = ["abs.cc"],
    hdrs = ["abs.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
    ],
)

cc_library(
    name = "add",
    srcs = ["add.cc"],
    hdrs = ["add.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "concat",
    srcs = ["concat.cc"],
    hdrs = ["concat.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
    ],
)

cc_library(
    name = "convolution",
    srcs = ["convolution.cc"],
    hdrs = ["convolution.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "convolution1x1",
    srcs = ["convolution1x1.cc"],
    hdrs = ["convolution1x1.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "convolution_generic",
    srcs = ["convolution_generic.cc"],
    hdrs = ["convolution_generic.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "depth_wise_conv3x3_stride1x1",
    srcs = ["depth_wise_conv3x3_stride1x1.cc"],
    hdrs = ["depth_wise_conv3x3_stride1x1.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "depth_wise_convolution",
    srcs = ["depth_wise_convolution.cc"],
    hdrs = ["depth_wise_convolution.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fully_connected",
    srcs = ["fully_connected.cc"],
    hdrs = ["fully_connected.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:environment",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "max_unpooling",
    srcs = ["max_unpooling.cc"],
    hdrs = ["max_unpooling.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "mul",
    srcs = ["mul.cc"],
    hdrs = ["mul.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "padding",
    srcs = ["padding.cc"],
    hdrs = ["padding.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "pooling",
    srcs = ["pooling.cc"],
    hdrs = ["pooling.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "prelu",
    srcs = ["prelu.cc"],
    hdrs = ["prelu.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "relu",
    srcs = ["relu.cc"],
    hdrs = ["relu.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reshape",
    srcs = ["reshape.cc"],
    hdrs = ["reshape.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "sigmoid",
    srcs = ["sigmoid.cc"],
    hdrs = ["sigmoid.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "slice",
    srcs = ["slice.cc"],
    hdrs = ["slice.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "softmax",
    srcs = ["softmax.cc"],
    hdrs = ["softmax.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "sub",
    srcs = ["sub.cc"],
    hdrs = ["sub.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "transpose_conv",
    srcs = ["transpose_conv.cc"],
    hdrs = ["transpose_conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:environment",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "upsample",
    srcs = ["upsample.cc"],
    hdrs = ["upsample.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "@com_google_absl//absl/strings",
    ],
)
