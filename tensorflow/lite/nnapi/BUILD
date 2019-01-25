licenses(["notice"])  # Apache 2.0

package(default_visibility = [
    "//visibility:public",
])

cc_library(
    name = "nnapi_lib",
    hdrs = [
        "NeuralNetworksShim.h",
        "NeuralNetworksTypes.h",
    ],
    linkopts = ["-ldl"],
)

cc_library(
    name = "nnapi_implementation",
    srcs = [
        "nnapi_implementation.cc",
    ],
    hdrs = [
        "nnapi_implementation.h",
    ],
    linkopts = ["-ldl"],
    deps = [
        "//tensorflow/lite/nnapi:nnapi_lib",
    ],
)

cc_test(
    name = "nnapi_implementation_test",
    srcs = ["nnapi_implementation_test.cc"],
    deps = [
        "//tensorflow/lite/nnapi:nnapi_implementation",
        "@com_google_googletest//:gtest_main",
    ],
)
