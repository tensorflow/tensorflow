# BUILD rules for NNAPI delegate compatibility checking.

cc_library(
    name = "nnapi_compatibility_lib",
    srcs = [
        "nnapi_compatibility_lib.cc",
    ],
    hdrs = [
        "nnapi_compatibility_lib.h",
    ],
    deps = [
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
    ],
)

cc_test(
    name = "nnapi_compatibility_lib_test",
    srcs = [
        "nnapi_compatibility_lib_test.cc",
    ],
    deps = [
        ":nnapi_compatibility_lib",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)
