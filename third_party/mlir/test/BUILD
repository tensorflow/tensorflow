licenses(["notice"])

package(default_visibility = [":test_friends"])

# Please only depend on this from MLIR tests.
package_group(
    name = "test_friends",
    includes = ["@org_tensorflow//tensorflow/compiler/mlir:subpackages"],
    packages = ["//..."],
)

load("@local_config_mlir//:tblgen.bzl", "gentbl")

cc_library(
    name = "IRProducingAPITest",
    hdrs = ["APITest.h"],
    includes = ["."],
)

gentbl(
    name = "TestOpsIncGen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "lib/TestDialect/TestOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "lib/TestDialect/TestOps.cpp.inc",
        ),
        (
            "-gen-rewriters",
            "lib/TestDialect/TestPatterns.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "lib/TestDialect/TestOps.td",
    td_srcs = [
        "@local_config_mlir//:include/mlir/IR/OpBase.td",
    ],
)

cc_library(
    name = "TestDialect",
    srcs = [
        "lib/TestDialect/TestDialect.cpp",
        "lib/TestDialect/TestOps.cpp.inc",
        "lib/TestDialect/TestOps.h.inc",
        "lib/TestDialect/TestPatterns.cpp",
        "lib/TestDialect/TestPatterns.inc",
    ],
    hdrs = [
        "lib/TestDialect/TestDialect.h",
    ],
    includes = ["lib/TestDialect"],
    deps = [
        ":TestOpsIncGen",
        "@llvm//:support",
        "@local_config_mlir//:Dialect",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TransformUtils",
        "@local_config_mlir//:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "TestPass",
    srcs = [
        "lib/Pass/TestPassManager.cpp",
    ],
    deps = [
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "TestTransforms",
    srcs = [
        "lib/Transforms/TestConstantFold.cpp",
        "lib/Transforms/TestInlining.cpp",
        "lib/Transforms/TestLoopFusion.cpp",
        "lib/Transforms/TestLoopMapping.cpp",
        "lib/Transforms/TestLoopParametricTiling.cpp",
        "lib/Transforms/TestVectorizationUtils.cpp",
    ],
    includes = ["lib/TestDialect"],
    deps = [
        ":TestDialect",
        "@llvm//:support",
        "@local_config_mlir//:AffineOps",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:EDSC",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:LoopOps",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TransformUtils",
        "@local_config_mlir//:Transforms",
        "@local_config_mlir//:VectorOps",
    ],
    alwayslink = 1,
)
