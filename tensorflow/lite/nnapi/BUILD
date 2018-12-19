licenses(["notice"])  # Apache 2.0

package(default_visibility = [
    "//visibility:public",
])

cc_library(
    name = "nnapi_lib",
    srcs = [
        "NeuralNetworksShim.cc",
    ],
    hdrs = [
        "NeuralNetworksShim.h",
    ],
    linkopts = ["-ldl"],
)

cc_test(
    name = "nnapi_lib_test",
    srcs = ["nnapi_lib_test.cc"],
    deps = [
        "//tensorflow/lite/nnapi:nnapi_lib",
        "@com_google_googletest//:gtest_main",
    ],
)
