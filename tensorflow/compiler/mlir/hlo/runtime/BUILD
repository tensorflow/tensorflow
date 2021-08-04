load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# TODO(b/160617323): Decouple MLIR HLO from TensorFlow/XLA
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//third_party/mlir:subpackages"],
    packages = [
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/tf2xla/...",
        "//tensorflow/compiler/xla/...",
        "//third_party/iree/...",
        "//third_party/mlir_edge/...",
        "//third_party/py/mlir/...",
    ],
)

cc_library(
    name = "ral_base",
    hdrs = [
        "ral/ral_base.h",
    ],
    includes = ["."],
    alwayslink = 1,
)

cc_library(
    name = "ral_logging",
    srcs = [
        "ral/ral_logging.cc",
    ],
    hdrs = [
        "ral/ral_logging.h",
    ],
    includes = ["."],
    alwayslink = 1,
)

tf_cc_test(
    name = "ral_logging_test",
    srcs = ["ral/ral_logging_test.cc"],
    deps = [
        ":ral_base",
        ":ral_logging",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "ral_registry",
    srcs = [
        "ral/ral_registry.cc",
    ],
    hdrs = [
        "ral/ral_registry.h",
    ],
    includes = ["."],
    deps = [
        ":ral_base",
        ":ral_logging",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "ral_registry_test",
    srcs = ["ral/ral_registry_test.cc"],
    deps = [
        ":ral_registry",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
