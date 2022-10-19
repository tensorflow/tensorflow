load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "translate_cl_registration",
    testonly = True,
    srcs = ["xla_mlir_translate_registration.cc"],
    deps = [
        "//tensorflow/compiler/jit:xla_cpu_jit",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/compiler/mlir/xla:mhlo_to_lhlo_with_xla",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:translate",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:translate",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TranslateLib",
    ],
    alwayslink = 1,
)
