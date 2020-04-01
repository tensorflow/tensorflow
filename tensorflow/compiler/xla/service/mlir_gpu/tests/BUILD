# TODO(herhut): describe this package.

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load(
    "//tensorflow/core/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
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
    srcs = if_cuda_is_configured(["mlir_gpu_lhlo_gen_test.cc"]),
    data = [
        "abs.hlo",
        "add.hlo",
        "add_as_kernel.hlo",
        "add_in_gpu_dialect.hlo",
        "add_multiply.hlo",
        "add_multiply_gpu.hlo",
        "add_reduce.hlo",
        "broadcast.hlo",
        "broken_add.hlo",
        "ceil.hlo",
        "compare.hlo",
        "const.hlo",
        "copy.hlo",
        "copy_transpose.hlo",
        "cos.hlo",
        "exp.hlo",
        "fused_reduce.hlo",
        "iota.hlo",
        "iota_add_multiply.hlo",
        "log.hlo",
        "neg.hlo",
        "reduce_window.hlo",
        "rem.hlo",
        "rsqrt.hlo",
        "select.hlo",
        "select_and_scatter.hlo",
        "sign.hlo",
        "sqrt.hlo",
        "tanh.hlo",
    ],
    tags = tf_cuda_tests_tags() + ["no_rocm"],
    deps = [
        "//tensorflow/core:test_main",
        "//tensorflow/core:test",
    ] + if_cuda_is_configured([
        "//tensorflow/core:lib",
        "//tensorflow/compiler/xla/service:gpu_plugin_mlir",
        "//tensorflow/compiler/xla/service/mlir_gpu:mlir_irgen_test_base",
        "//tensorflow/stream_executor/lib",
    ]),
)
