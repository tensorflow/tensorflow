# Experimental extensions to the C API for eager execution of kernels.
licenses(["notice"])  # Apache 2.0

cc_library(
    name = "c_api",
    srcs = ["c_api.cc"],
    hdrs = ["c_api.h"],
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/python/eager:__pkg__",
    ],
    deps = [
        ":runtime",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "c_api_test",
    srcs = ["c_api_test.cc"],
    deps = [
        ":c_api",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "runtime",
    srcs = ["runtime.cc"],
    hdrs = ["runtime.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "runtime_test",
    srcs = ["runtime_test.cc"],
    deps = [
        ":runtime",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
