# Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert/vendors/qualcomm:__subpackages__"],
)

cc_library(
    name = "op_builder",
    srcs = ["op_builder.cc"],
    hdrs = ["op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "elementwise_op_builder",
    srcs = ["elementwise_op_builder.cc"],
    hdrs = ["elementwise_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:param_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "cast_op_builder",
    srcs = ["cast_op_builder.cc"],
    hdrs = ["cast_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "concatenation_op_builder",
    srcs = ["concatenation_op_builder.cc"],
    hdrs = ["concatenation_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "embedding_lookup_op_builder",
    srcs = ["embedding_lookup_op_builder.cc"],
    hdrs = ["embedding_lookup_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils:log",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "fully_connected_op_builder",
    srcs = ["fully_connected_op_builder.cc"],
    hdrs = ["fully_connected_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "gather_op_builder",
    srcs = ["gather_op_builder.cc"],
    hdrs = ["gather_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils:log",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "gelu_op_builder",
    srcs = ["gelu_op_builder.cc"],
    hdrs = ["gelu_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "matmul_op_builder",
    srcs = ["matmul_op_builder.cc"],
    hdrs = ["matmul_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "mean_op_builder",
    srcs = ["mean_op_builder.cc"],
    hdrs = ["mean_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils:log",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "quantize_op_builder",
    srcs = ["quantize_op_builder.cc"],
    hdrs = ["quantize_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "reduce_op_builder",
    srcs = ["reduce_op_builder.cc"],
    hdrs = ["reduce_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils:log",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "reshape_op_builder",
    srcs = ["reshape_op_builder.cc"],
    hdrs = ["reshape_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "select_op_builder",
    srcs = ["select_op_builder.cc"],
    hdrs = ["select_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "slice_op_builder",
    srcs = ["slice_op_builder.cc"],
    hdrs = ["slice_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils:log",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "softmax_op_builder",
    srcs = ["softmax_op_builder.cc"],
    hdrs = ["softmax_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "split_op_builder",
    srcs = ["split_op_builder.cc"],
    hdrs = ["split_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils:log",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "tanh_op_builder",
    srcs = ["tanh_op_builder.cc"],
    hdrs = ["tanh_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "transpose_op_builder",
    srcs = ["transpose_op_builder.cc"],
    hdrs = ["transpose_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils:log",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "pack_op_builder",
    srcs = ["pack_op_builder.cc"],
    hdrs = ["pack_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "dynamic_update_slice_op_builder",
    srcs = ["dynamic_update_slice_op_builder.cc"],
    hdrs = ["dynamic_update_slice_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "rms_norm_op_builder",
    srcs = ["rms_norm_op_builder.cc"],
    hdrs = ["rms_norm_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "conv2d_op_builder",
    srcs = ["conv2d_op_builder.cc"],
    hdrs = ["conv2d_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils:log",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "pool2d_op_builder",
    srcs = ["pool2d_op_builder.cc"],
    hdrs = ["pool2d_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "spatial_transform_op_builder",
    srcs = ["spatial_transform_op_builder.cc"],
    hdrs = ["spatial_transform_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "resize_op_builder",
    srcs = ["resize_op_builder.cc"],
    hdrs = ["resize_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils:log",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "leaky_relu_op_builder",
    srcs = ["leaky_relu_op_builder.cc"],
    hdrs = ["leaky_relu_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils:log",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "hard_swish_op_builder",
    srcs = ["hard_swish_op_builder.cc"],
    hdrs = ["hard_swish_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "depthwise_conv2d_op_builder",
    srcs = ["depthwise_conv2d_op_builder.cc"],
    hdrs = ["depthwise_conv2d_op_builder.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":op_builder",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)
