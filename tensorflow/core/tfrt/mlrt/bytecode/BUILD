load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        # copybara:uncomment "//learning/brain/tfrt:__subpackages__",
        # copybara:uncomment "//learning/infra/mira/distributed:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt/transforms/mlrt:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt/translate/mlrt:__subpackages__",
        "//tensorflow/core/tfrt:__subpackages__",
        "//tensorflow/core/tfrt/mlrt:__subpackages__",
    ],
)

cc_library(
    name = "bytecode",
    hdrs = ["bytecode.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "span",
    hdrs = ["span.h"],
    deps = [":bytecode"],
)

cc_library(
    name = "function",
    hdrs = ["function.h"],
    deps = [
        ":bytecode",
        ":kernel",
    ],
)

cc_library(
    name = "kernel",
    hdrs = ["kernel.h"],
    deps = [":bytecode"],
)

cc_library(
    name = "executable",
    hdrs = ["executable.h"],
    deps = [":function"],
)

tf_cc_test(
    name = "bytecode_test",
    srcs = ["bytecode_test.cc"],
    deps = [
        ":bytecode",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "kernel_test",
    srcs = ["kernel_test.cc"],
    deps = [
        ":kernel",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "function_test",
    srcs = ["function_test.cc"],
    deps = [
        ":function",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "executable_test",
    srcs = ["executable_test.cc"],
    deps = [
        ":executable",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "span_test",
    srcs = ["span_test.cc"],
    deps = [
        ":span",
        "@com_google_googletest//:gtest_main",
    ],
)
