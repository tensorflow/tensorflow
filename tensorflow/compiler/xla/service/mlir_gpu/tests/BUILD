# TODO(herhut): describe this package.

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

tf_cc_test(
    name = "mlir_gpu_lhlo_gen_test",
    srcs = ["mlir_gpu_lhlo_gen_test.cc"],
    tags = tf_cuda_tests_tags() + ["no_rocm"],
    deps = [
        "//tensorflow/compiler/xla/service:mlir_gpu_plugin",
        "//tensorflow/compiler/xla/service/mlir_gpu:mlir_irgen_test_base",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor/lib",
    ],
)
