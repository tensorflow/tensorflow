package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "simple_delegate",
    srcs = [
        "simple_delegate.cc",
    ],
    hdrs = [
        "simple_delegate.h",
    ],
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates:utils",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "simple_delegate_test",
    srcs = ["simple_delegate_test.cc"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils/dummy_delegate",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_googletest//:gtest_main",
    ],
)
