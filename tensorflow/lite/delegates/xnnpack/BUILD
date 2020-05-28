load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite_combined")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

EMSCRIPTEN_LINKOPTS = [
    "-s ASSERTIONS=2",
    "-s ERROR_ON_UNDEFINED_SYMBOLS=1",
    "-s DEMANGLE_SUPPORT=1",
    "-s EXIT_RUNTIME=1",
    "-s ALLOW_MEMORY_GROWTH=1",
    "-s TOTAL_MEMORY=134217728",
]

cc_library(
    name = "xnnpack_delegate",
    srcs = ["xnnpack_delegate.cc"],
    hdrs = ["xnnpack_delegate.h"],
    linkstatic = True,
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@XNNPACK",
    ],
)

cc_library(
    name = "xnnpack_delegate_test_mode",
    srcs = ["xnnpack_delegate.cc"],
    hdrs = ["xnnpack_delegate.h"],
    copts = ["-DXNNPACK_DELEGATE_TEST_MODE=1"],
    linkstatic = True,
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@XNNPACK",
    ],
)

################################ Tester classes ################################

cc_library(
    name = "binary_elementwise_tester",
    testonly = 1,
    srcs = ["binary_elementwise_tester.cc"],
    hdrs = ["binary_elementwise_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "depthwise_conv_2d_tester",
    testonly = 1,
    srcs = ["depthwise_conv_2d_tester.cc"],
    hdrs = ["depthwise_conv_2d_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "fully_connected_tester",
    testonly = 1,
    srcs = ["fully_connected_tester.cc"],
    hdrs = ["fully_connected_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "pad_tester",
    testonly = 1,
    srcs = ["pad_tester.cc"],
    hdrs = ["pad_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "pool_2d_tester",
    testonly = 1,
    srcs = ["pool_2d_tester.cc"],
    hdrs = ["pool_2d_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "softmax_tester",
    testonly = 1,
    srcs = ["softmax_tester.cc"],
    hdrs = ["softmax_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "unary_elementwise_tester",
    testonly = 1,
    srcs = ["unary_elementwise_tester.cc"],
    hdrs = ["unary_elementwise_tester.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

############################## Integration tests ###############################

cc_library(
    name = "test_main",
    testonly = 1,
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "add_test",
    srcs = ["add_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "average_pool_2d_test",
    srcs = ["average_pool_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":pool_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "conv_2d_test",
    srcs = ["conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":xnnpack_delegate",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_test(
    name = "depthwise_conv_2d_test",
    srcs = ["depthwise_conv_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    tags = ["nomsan"],  # b/145129478
    deps = [
        ":depthwise_conv_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "fully_connected_test",
    srcs = ["fully_connected_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":fully_connected_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "hard_swish_test",
    srcs = ["hard_swish_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "max_pool_2d_test",
    srcs = ["max_pool_2d_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    tags = [
        "notsan",  # TODO(b/155404603)
    ],
    deps = [
        ":pool_2d_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "logistic_test",
    srcs = ["logistic_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "mul_test",
    srcs = ["mul_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":binary_elementwise_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "pad_test",
    srcs = ["pad_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":pad_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "relu_test",
    srcs = ["relu_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "relu6_test",
    srcs = ["relu6_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "relu_n1_to_1_test",
    srcs = ["relu_n1_to_1_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":test_main",
        ":unary_elementwise_tester",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "softmax_test",
    srcs = ["softmax_test.cc"],
    linkopts = select({
        "//tensorflow:emscripten": EMSCRIPTEN_LINKOPTS,
        "//conditions:default": [],
    }),
    deps = [
        ":softmax_tester",
        ":test_main",
        ":xnnpack_delegate_test_mode",
        "@com_google_googletest//:gtest",
    ],
)

tflite_portable_test_suite_combined(combine_conditions = {"deps": [":test_main"]})
