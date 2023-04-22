# Parts of new C++ API that interface with MLIR.

load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        "//tensorflow/cc/experimental/libtf:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "transform",
    srcs = [
        "mlir_transform.cc",
    ],
    hdrs = ["mlir_transform.h"],
    deps = [
        "//tensorflow/cc/experimental/libtf",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
    ],
)
