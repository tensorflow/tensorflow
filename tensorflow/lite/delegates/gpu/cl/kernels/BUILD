load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "add",
    srcs = ["add.cc"],
    hdrs = ["add.h"],
    deps = [
        ":gpu_operation",
        ":util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "add_test",
    srcs = ["add_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":add",
        ":cl_test",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cl_test",
    testonly = 1,
    srcs = ["cl_test.cc"],
    hdrs = ["cl_test.h"],
    deps = [
        ":gpu_operation",
        "//tensorflow/lite/delegates/gpu/cl:environment",
        "//tensorflow/lite/delegates/gpu/cl:opencl_wrapper",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "concat_test",
    srcs = ["concat_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":concat_xy",
        ":concat_z",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "concat_xy",
    srcs = ["concat_xy.cc"],
    hdrs = ["concat_xy.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_library(
    name = "concat_z",
    srcs = ["concat_z.cc"],
    hdrs = ["concat_z.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_library(
    name = "conv_3d",
    srcs = ["conv_3d.cc"],
    hdrs = ["conv_3d.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:cl_device",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_buffer_1x1",
    srcs = ["conv_buffer_1x1.cc"],
    hdrs = ["conv_buffer_1x1.h"],
    deps = [
        ":conv_common",
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:cl_device",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:winograd_util",
    ],
)

cc_library(
    name = "conv_common",
    hdrs = ["conv_common.h"],
)

cc_test(
    name = "conv_buffer_1x1_test",
    srcs = ["conv_buffer_1x1_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":conv_buffer_1x1",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "conv_constants",
    srcs = ["conv_constants.cc"],
    hdrs = ["conv_constants.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "conv_constants_test",
    srcs = ["conv_constants_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":conv_constants",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "conv_powervr",
    srcs = ["conv_powervr.cc"],
    hdrs = ["conv_powervr.h"],
    deps = [
        ":conv_common",
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:cl_device",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:winograd_util",
    ],
)

cc_test(
    name = "conv_powervr_test",
    srcs = ["conv_powervr_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":conv_powervr",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "conv_texture",
    srcs = ["conv_texture.cc"],
    hdrs = ["conv_texture.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:cl_command_queue",
        "//tensorflow/lite/delegates/gpu/cl:cl_context",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/cl:texture2d",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:winograd_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "conv_texture_test",
    srcs = ["conv_texture_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":conv_texture",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "conv_weights_converter",
    srcs = ["conv_weights_converter.cc"],
    hdrs = ["conv_weights_converter.h"],
    deps = [
        ":conv_common",
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:cl_command_queue",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_library(
    name = "converter",
    srcs = ["converter.cc"],
    hdrs = ["converter.h"],
    deps = [
        ":util",
        "//tensorflow/lite/delegates/gpu:spi",
        "//tensorflow/lite/delegates/gpu/cl:cl_command_queue",
        "//tensorflow/lite/delegates/gpu/cl:cl_errors",
        "//tensorflow/lite/delegates/gpu/cl:environment",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type_util",
        "//tensorflow/lite/delegates/gpu/common:util",
    ],
)

cc_library(
    name = "convolution_transposed",
    srcs = ["convolution_transposed.cc"],
    hdrs = ["convolution_transposed.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/cl:texture2d",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "convolution_transposed_test",
    srcs = ["convolution_transposed_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":convolution_transposed",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "convolution_transposed_3d",
    srcs = ["convolution_transposed_3d.cc"],
    hdrs = ["convolution_transposed_3d.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/cl:texture2d",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "convolution_transposed_3x3",
    srcs = ["convolution_transposed_3x3.cc"],
    hdrs = ["convolution_transposed_3x3.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "convolution_transposed_3x3_test",
    srcs = ["convolution_transposed_3x3_test.cc"],
    linkstatic = True,
    tags = [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":convolution_transposed_3x3",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "convolution_transposed_3x3_thin",
    srcs = ["convolution_transposed_3x3_thin.cc"],
    hdrs = ["convolution_transposed_3x3_thin.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "convolution_transposed_3x3_thin_test",
    srcs = ["convolution_transposed_3x3_thin_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":convolution_transposed_3x3_thin",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "convolution_transposed_4x4",
    srcs = ["convolution_transposed_4x4.cc"],
    hdrs = ["convolution_transposed_4x4.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "convolution_transposed_4x4_test",
    srcs = ["convolution_transposed_4x4_test.cc"],
    linkstatic = True,
    tags = [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":convolution_transposed_4x4",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "convolution_transposed_thin",
    srcs = ["convolution_transposed_thin.cc"],
    hdrs = ["convolution_transposed_thin.h"],
    deps = [
        ":flt_type",
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:texture2d",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "convolution_transposed_thin_test",
    srcs = ["convolution_transposed_thin_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":convolution_transposed_thin",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "depthwise_conv",
    srcs = ["depthwise_conv.cc"],
    hdrs = ["depthwise_conv.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:cl_device",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:texture2d",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_library(
    name = "depthwise_conv_3d",
    srcs = ["depthwise_conv_3d.cc"],
    hdrs = ["depthwise_conv_3d.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:cl_device",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:texture2d",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "depthwise_conv_test",
    srcs = ["depthwise_conv_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":depthwise_conv",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "depthwise_conv_3x3",
    srcs = ["depthwise_conv_3x3.cc"],
    hdrs = ["depthwise_conv_3x3.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:texture2d",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "depthwise_conv_3x3_test",
    srcs = ["depthwise_conv_3x3_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":depthwise_conv_3x3",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "elementwise",
    srcs = ["elementwise.cc"],
    hdrs = ["elementwise.h"],
    deps = [
        ":gpu_operation",
        ":util",
        "//tensorflow/lite/delegates/gpu/cl:storage_type_util",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "elementwise_test",
    srcs = ["elementwise_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":elementwise",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "flt_type",
    srcs = ["flt_type.cc"],
    hdrs = ["flt_type.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fully_connected",
    srcs = ["fully_connected.cc"],
    hdrs = ["fully_connected.h"],
    deps = [
        ":gpu_operation",
        ":util",
        "//tensorflow/lite/delegates/gpu/cl:buffer",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:util",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "fully_connected_test",
    srcs = ["fully_connected_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":fully_connected",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_operation",
    srcs = ["gpu_operation.cc"],
    hdrs = ["gpu_operation.h"],
    deps = [
        ":tuning_parameters",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:arguments",
        "//tensorflow/lite/delegates/gpu/cl:cl_context",
        "//tensorflow/lite/delegates/gpu/cl:cl_device",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:program_cache",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/common:access_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_library(
    name = "lstm",
    srcs = ["lstm.cc"],
    hdrs = ["lstm.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "lstm_test",
    srcs = ["lstm_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":lstm",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "max_unpooling",
    srcs = ["max_unpooling.cc"],
    hdrs = ["max_unpooling.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "max_unpooling_test",
    srcs = ["max_unpooling_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":max_unpooling",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mean",
    srcs = ["mean.cc"],
    hdrs = ["mean.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
    ],
)

cc_test(
    name = "mean_test",
    srcs = ["mean_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":mean",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "padding",
    srcs = ["padding.cc"],
    hdrs = ["padding.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "padding_test",
    srcs = ["padding_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":padding",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "pooling",
    srcs = ["pooling.cc"],
    hdrs = ["pooling.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "pooling_test",
    srcs = ["pooling_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":pooling",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "prelu",
    srcs = ["prelu.cc"],
    hdrs = ["prelu.h"],
    deps = [
        ":flt_type",
        ":gpu_operation",
        ":util",
        "//tensorflow/lite/delegates/gpu/cl:cl_context",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "prelu_test",
    srcs = ["prelu_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":prelu",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "quantize_and_dequantize",
    srcs = ["quantize_and_dequantize.cc"],
    hdrs = ["quantize_and_dequantize.h"],
    deps = [
        ":flt_type",
        ":gpu_operation",
        ":util",
        "//tensorflow/lite/delegates/gpu/cl:cl_context",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "quantize_and_dequantize_test",
    srcs = ["quantize_and_dequantize_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":quantize_and_dequantize",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/kernels/internal:quantization_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "relu",
    srcs = ["relu.cc"],
    hdrs = ["relu.h"],
    deps = [
        ":flt_type",
        ":gpu_operation",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "relu_test",
    srcs = ["relu_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":relu",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "reshape",
    srcs = ["reshape.cc"],
    hdrs = ["reshape.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "reshape_test",
    srcs = ["reshape_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":reshape",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "reshapex4",
    srcs = ["reshapex4.cc"],
    hdrs = ["reshapex4.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:cl_command_queue",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "reshapex4_test",
    srcs = ["reshapex4_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":reshapex4",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "softmax",
    srcs = ["softmax.cc"],
    hdrs = ["softmax.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "softmax_test",
    srcs = ["softmax_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":softmax",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "softmax1x1",
    srcs = ["softmax1x1.cc"],
    hdrs = ["softmax1x1.h"],
    deps = [
        ":gpu_operation",
        ":util",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

cc_test(
    name = "softmax1x1_test",
    srcs = ["softmax1x1_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":softmax1x1",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "space_to_depth",
    srcs = ["space_to_depth.cc"],
    hdrs = ["space_to_depth.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "space_to_depth_test",
    srcs = ["space_to_depth_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":space_to_depth",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "strided_slice",
    srcs = ["strided_slice.cc"],
    hdrs = ["strided_slice.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "strided_slice_test",
    srcs = ["strided_slice_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":strided_slice",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "transpose",
    srcs = ["transpose.cc"],
    hdrs = ["transpose.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:types",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "transpose_test",
    srcs = ["transpose_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":transpose",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tuning_parameters",
    hdrs = ["tuning_parameters.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl:cl_command_queue",
        "//tensorflow/lite/delegates/gpu/cl:cl_device",
    ],
)

cc_library(
    name = "resize",
    srcs = ["resize.cc"],
    hdrs = ["resize.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

cc_test(
    name = "resize_test",
    srcs = ["resize_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":resize",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl:cl_device",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/common:access_type",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "winograd",
    srcs = ["winograd.cc"],
    hdrs = ["winograd.h"],
    deps = [
        ":gpu_operation",
        ":util",
        ":work_group_picking",
        "//tensorflow/lite/delegates/gpu/cl:cl_device",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/cl:linear_storage",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:winograd_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "winograd_test",
    srcs = ["winograd_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "linux",
        "local",
    ],
    deps = [
        ":cl_test",
        ":util",
        ":winograd",
        "//tensorflow/lite/delegates/gpu/cl:precision",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:winograd_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "work_group_picking",
    srcs = ["work_group_picking.cc"],
    hdrs = ["work_group_picking.h"],
    deps = [
        ":tuning_parameters",
        "//tensorflow/lite/delegates/gpu/cl:cl_command_queue",
        "//tensorflow/lite/delegates/gpu/cl:cl_kernel",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common:workgroup_selection",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        "add_test",
        "concat_test",
        "conv_buffer_1x1_test",
        "conv_constants_test",
        "conv_powervr_test",
        "conv_texture_test",
        "convolution_transposed_3x3_thin_test",
        "convolution_transposed_4x4_test",
        "convolution_transposed_test",
        "convolution_transposed_thin_test",
        "depthwise_conv_3x3_test",
        "depthwise_conv_test",
        "elementwise_test",
        "fully_connected_test",
        "lstm_test",
        "max_unpooling_test",
        "padding_test",
        "pooling_test",
        "prelu_test",
        "relu_test",
        "reshape_test",
        "reshapex4_test",
        "resize_test",
        "softmax1x1_test",
        "softmax_test",
        "strided_slice_test",
        "transpose_test",
        "winograd_test",
    ],
)
