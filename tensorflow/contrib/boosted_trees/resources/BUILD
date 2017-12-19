licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_cc_library")

package(
    default_visibility = [
        "//tensorflow/contrib/boosted_trees:__subpackages__",
        "//tensorflow/contrib/boosted_trees:friends",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

tf_cc_library(
    name = "stamped_resource",
    hdrs = ["stamped_resource.h"],
    is_external = True,
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
)

tf_cc_library(
    name = "quantile_stream_resource",
    hdrs = ["quantile_stream_resource.h"],
    is_external = True,
    deps = [
        ":stamped_resource",
        "//tensorflow/contrib/boosted_trees/lib:weighted_quantiles",
        "//tensorflow/contrib/boosted_trees/proto:quantiles_proto_cc",
        "//tensorflow/core:framework_headers_lib",
    ],
)

tf_cc_library(
    name = "decision_tree_ensemble_resource",
    hdrs = ["decision_tree_ensemble_resource.h"],
    is_external = True,
    deps = [
        ":stamped_resource",
        "//tensorflow/contrib/boosted_trees/lib:trees",
        "//tensorflow/core:framework_headers_lib",
    ],
)
