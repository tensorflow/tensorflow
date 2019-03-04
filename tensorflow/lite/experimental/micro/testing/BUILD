package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["test_linux_binary.sh"])

cc_library(
    name = "micro_test",
    hdrs = [
        "micro_test.h",
        "test_utils.h",
    ],
    deps = [
        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/experimental/micro:micro_framework",
    ],
)
