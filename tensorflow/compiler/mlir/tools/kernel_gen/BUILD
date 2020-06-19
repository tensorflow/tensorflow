load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

licenses(["notice"])

cc_library(
    name = "cubin_creator",
    srcs = ["cubin_creator.cc"],
    hdrs = ["cubin_creator.h"],
    copts = if_cuda(["-DGOOGLE_CUDA=1"]),
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TargetNVVMIR",
        "@llvm-project//mlir:Transforms",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/xla:hlo",
        "//tensorflow/compiler/mlir/xla:lhlo",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "//tensorflow/compiler/mlir/xla:xla_materialize_broadcasts",  # buildcleaner: keep
        "//tensorflow/compiler/mlir/xla:xla_unfuse_batch_norm",  # buildcleaner: keep
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
        "//tensorflow/compiler/xla/service/gpu:target_constants",
        "//tensorflow/compiler/xla/service/gpu/llvm_gpu_backend",
        "//tensorflow/compiler/xla/service/mlir_gpu:kernel_lowering",
        "//tensorflow/core:cuda_libdevice_path",
        "//tensorflow/core:lib",
    ] + if_cuda(["//tensorflow/stream_executor/gpu:asm_compiler"]),
)

tf_cc_binary(
    name = "tf_to_cubin",
    srcs = ["tf_to_cubin.cc"],
    visibility = ["//tensorflow/core/kernels/cubin_headers:__pkg__"],
    deps = [
        ":cubin_creator",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)
