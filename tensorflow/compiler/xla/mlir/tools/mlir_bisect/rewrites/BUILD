load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")

cc_library(
    name = "rewrites",
    srcs = [
        "func.cc",
        "general.cc",
        "scf.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/mlir/tools/mlir_bisect:bisect_lib",
        "//tensorflow/compiler/xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)
