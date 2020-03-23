load(
    "//third_party/mlir:tblgen.bzl",
    "gentbl",
)

package(
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//third_party/mlir:subpackages"],
    packages = [
        "//tensorflow/compiler/aot/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/mlir/lite/...",
    ],
)

cc_library(
    name = "hlo_xla_quantization_passes",
    srcs = [
        "cpu_kernel_fusion.cc",
        "generated_cpu_kernel_fusion.inc",
        "materialize.cc",
        "op_quant_spec.inc",
        "propagate.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_lib",
        "//tensorflow/compiler/mlir/xla:hlo",
        "//tensorflow/compiler/xla/client/lib:quantize",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "quantize",
    srcs = [
        "quantize.cc",
    ],
    hdrs = [
        "quantize.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/xla:hlo",
        "//tensorflow/compiler/mlir/xla:hlo_to_mlir_hlo",
        "//tensorflow/compiler/tf2xla",
        "//tensorflow/compiler/tf2xla:tf2xla_proto_cc",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_dummy_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/core/platform:status",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl(
    name = "cpu_kernel_fusion_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "generated_cpu_kernel_fusion.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "cpu_kernel_fusion.td",
    td_srcs = [
        "@llvm-project//mlir:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/xla:hlo_ops_td_files",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_td_files",
    ],
)
