licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(
    default_visibility = [
        "//tensorflow/contrib/boosted_trees:__subpackages__",
        "//tensorflow/contrib/boosted_trees:friends",
    ],
)

cc_library(
    name = "stamped_resource",
    hdrs = ["stamped_resource.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "quantile_stream_resource",
    hdrs = ["quantile_stream_resource.h"],
    deps = [
        ":stamped_resource",
        "//tensorflow/contrib/boosted_trees/lib:weighted_quantiles",
        "//tensorflow/contrib/boosted_trees/proto:quantiles_proto_cc",
        "//tensorflow/core:framework_headers_lib",
    ],
)

cc_library(
    name = "decision_tree_ensemble_resource",
    hdrs = ["decision_tree_ensemble_resource.h"],
    deps = [
        ":stamped_resource",
        "//tensorflow/contrib/boosted_trees/lib:trees",
        "//tensorflow/core:framework_headers_lib",
    ],
)
