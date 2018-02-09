licenses(["notice"])  # Apache 2.0

package(default_visibility = [
    "//visibility:public",
])

cc_library(
    name = "nnapi_lib",
    hdrs = [
        "NeuralNetworksShim.h",
    ],
    linkopts = ["-ldl"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
