# Defines data structures that store autotuning results.
# The autotuning results can be serialized as a string, allowing you to save
# and later restore them.

# TODO(ruochengw): Currently only supports contrib's fused_conv2d_bias_activation_op.
# We plan to add more ops and move fused_conv2d_bias_activation_op back into core library.
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda_or_rocm",
    "tf_cuda_library",
)

package(
    default_visibility = ["//tensorflow:__subpackages__"],
    features = ["layering_check"],
    licenses = ["notice"],
)

cc_library(
    name = "conv_autotune_maps",
    hdrs = [
        "conv_autotune_maps.h",
    ],
    deps = [
        "//tensorflow/core/kernels:gpu_util_hdrs",
    ],
)

tf_proto_library(
    name = "conv_parameters_proto",
    srcs = [
        "conv_parameters.proto",
    ],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow/core/framework:types_proto",
        "//tensorflow/stream_executor:dnn_proto",
    ],
)

# This ensures the compilation of custom op fused_conv2d_bias_activation_op.
# TODO(ruochengw): Remove this target once we move fused_conv2d_bias_activation_op back in core library.
cc_library(
    name = "conv_parameters_hdrs",
    hdrs = ["conv_parameters.h"],
    deps = [
        ":conv_parameters_proto_cc",
    ],
)

tf_cuda_library(
    name = "conv_parameters",
    srcs = ["conv_parameters.cc"],
    hdrs = [
        "conv_parameters.h",
    ],
    deps = [
        ":autotune_maps_utils",
        ":conv_parameters_proto_cc",
        "//tensorflow/core/platform:hash",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:stream_executor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cuda_library(
    name = "autotune_maps_utils",
    srcs = ["autotune_maps_utils.cc"],
    hdrs = ["autotune_maps_utils.h"],
    visibility = ["//tensorflow/core/utils:__subpackages__"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:hash",
        "//tensorflow/core/platform:errors",
    ] + if_cuda_or_rocm(
        [
            "//tensorflow/stream_executor/gpu:gpu_driver_header",
        ],
    ),
)
