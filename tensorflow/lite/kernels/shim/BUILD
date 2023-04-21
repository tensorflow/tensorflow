# This package is a shim layer over TF and TFLite op kernels.

load("//tensorflow:tensorflow.bzl", "if_mobile", "if_not_mobile", "tf_cc_test", "tf_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    # The public targets are marked public individually
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],
)

filegroup(
    name = "tf_headers",
    srcs = [
        "op_kernel.h",
        "shape.h",
        "status_macros.h",
        "tensor_view.h",
        "tf_op_shim.h",
        "tf_tensor_view.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

cc_library(
    name = "tf_kernel_shim",
    deps = [
        ":op_kernel",
        ":shape",
        ":status_macros",
        ":tensor_view",
        ":tf_op_shim",
        ":tf_tensor_view",
    ],
)

cc_library(
    name = "tensor_view",
    hdrs = ["tensor_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/core/platform:tstring",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "shape",
    srcs = ["shape.cc"],
    hdrs = ["shape.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "shape_test",
    srcs = ["shape_test.cc"],
    deps = [
        ":shape",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tf_tensor_view",
    srcs = ["tf_tensor_view.cc"],
    hdrs = ["tf_tensor_view.h"],
    copts = tf_copts(),
    deps = [
        ":tensor_view",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ] + if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
    ]),
)

tf_cc_test(
    name = "tf_tensor_view_test",
    srcs = ["tf_tensor_view_test.cc"],
    deps = [
        ":tf_tensor_view",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ] + if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:tstring",
    ]),
)

cc_library(
    name = "tflite_tensor_view",
    srcs = ["tflite_tensor_view.cc"],
    hdrs = ["tflite_tensor_view.h"],
    deps = [
        ":tensor_view",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/core/c:common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_test(
    name = "tflite_tensor_view_test",
    srcs = ["tflite_tensor_view_test.cc"],
    deps = [
        ":test_util",
        ":tflite_tensor_view",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core:framework_stable",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "op_kernel",
    hdrs = ["op_kernel.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":shape",
        ":status_macros",
        ":tensor_view",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "tf_op_shim",
    srcs = ["tf_op_shim.cc"],
    hdrs = ["tf_op_shim.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":op_kernel",
        ":shape",
        ":status_macros",
        ":tensor_view",
        ":tf_tensor_view",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ] + if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
    ]),
)

cc_library(
    name = "tflite_op_shim",
    srcs = ["tflite_op_shim.cc"],
    hdrs = ["tflite_op_shim.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":op_kernel",
        ":shape",
        ":status_macros",
        ":tensor_view",
        ":tflite_tensor_view",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

cc_library(
    name = "tflite_op_wrapper",
    hdrs = ["tflite_op_wrapper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":op_kernel",
        ":status_macros",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/c:common",
    ],
)

cc_test(
    name = "tflite_op_wrapper_test",
    srcs = ["tflite_op_wrapper_test.cc"],
    deps = [
        ":op_kernel",
        ":tflite_op_shim",
        ":tflite_op_wrapper",
        "//tensorflow/core/platform:tstring",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "status_macros",
    hdrs = ["status_macros.h"],
    visibility = ["//visibility:public"],
    deps = ["@com_google_absl//absl/status"],
)

cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "test_util_test",
    srcs = ["test_util_test.cc"],
    deps = [
        ":test_util",
        "//tensorflow/lite:cc_api_experimental",
        "//tensorflow/lite/core:framework_stable",
        "@com_google_googletest//:gtest_main",
    ],
)
