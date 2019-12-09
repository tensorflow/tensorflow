load("@local_config_mlir//:tblgen.bzl", "gentbl")

licenses(["notice"])

package(default_visibility = [":test_friends"])

# Please only depend on this from MLIR tests.
package_group(
    name = "test_friends",
    includes = ["@org_tensorflow//tensorflow/compiler/mlir:subpackages"],
    packages = ["//..."],
)

cc_library(
    name = "IRProducingAPITest",
    hdrs = ["APITest.h"],
    includes = ["."],
)

gentbl(
    name = "TestLinalgTransformPatternsIncGen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "lib/DeclarativeTransforms/TestLinalgTransformPatterns.h.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "lib/DeclarativeTransforms/TestLinalgTransformPatterns.td",
    td_srcs = [
        "@local_config_mlir//:LinalgTransformPatternsTdFiles",
    ],
)

gentbl(
    name = "TestLinalgTilePermuteTransformPatternsIncGen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "lib/DeclarativeTransforms/TestLinalgTilePermutePatterns.h.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "lib/DeclarativeTransforms/TestLinalgTilePermutePatterns.td",
    td_srcs = [
        "@local_config_mlir//:LinalgTransformPatternsTdFiles",
    ],
)

gentbl(
    name = "TestOpsIncGen",
    strip_include_prefix = "lib/TestDialect",
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
            "-gen-enum-decls",
            "lib/TestDialect/TestOpEnums.h.inc",
        ),
        (
            "-gen-enum-defs",
            "lib/TestDialect/TestOpEnums.cpp.inc",
        ),
        (
            "-gen-rewriters",
            "lib/TestDialect/TestPatterns.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "lib/TestDialect/TestOps.td",
    td_srcs = [
        "@local_config_mlir//:OpBaseTdFiles",
        "@local_config_mlir//:include/mlir/IR/OpAsmInterface.td",
        "@local_config_mlir//:include/mlir/Analysis/CallInterfaces.td",
        "@local_config_mlir//:include/mlir/Analysis/InferTypeOpInterface.td",
    ],
    test = True,
)

cc_library(
    name = "TestDialect",
    srcs = [
        "lib/TestDialect/TestDialect.cpp",
        "lib/TestDialect/TestPatterns.cpp",
    ],
    hdrs = [
        "lib/TestDialect/TestDialect.h",
    ],
    includes = [
        "lib/DeclarativeTransforms",
        "lib/TestDialect",
    ],
    deps = [
        ":TestOpsIncGen",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:Dialect",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:TransformUtils",
        "@local_config_mlir//:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "TestIR",
    srcs = [
        "lib/IR/TestFunc.cpp",
        "lib/IR/TestMatchers.cpp",
        "lib/IR/TestSymbolUses.cpp",
    ],
    deps = [
        ":TestDialect",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
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
        "lib/Transforms/TestCallGraph.cpp",
        "lib/Transforms/TestConstantFold.cpp",
        "lib/Transforms/TestInlining.cpp",
        "lib/Transforms/TestLinalgTilePermuteTransforms.cpp",
        "lib/Transforms/TestLinalgTransforms.cpp",
        "lib/Transforms/TestLoopFusion.cpp",
        "lib/Transforms/TestLoopMapping.cpp",
        "lib/Transforms/TestLoopParametricTiling.cpp",
        "lib/Transforms/TestMemRefStrideCalculation.cpp",
        "lib/Transforms/TestOpaqueLoc.cpp",
        "lib/Transforms/TestVectorToLoopsConversion.cpp",
        "lib/Transforms/TestVectorToVectorConversion.cpp",
        "lib/Transforms/TestVectorizationUtils.cpp",
    ],
    includes = ["lib/TestDialect"],
    deps = [
        ":TestDialect",
        ":TestLinalgTilePermuteTransformPatternsIncGen",
        ":TestLinalgTransformPatternsIncGen",
        "@llvm//:support",
        "@local_config_mlir//:AffineOps",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:EDSC",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Linalg",
        "@local_config_mlir//:LoopOps",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TransformUtils",
        "@local_config_mlir//:Transforms",
        "@local_config_mlir//:VectorOps",
        "@local_config_mlir//:VectorToLLVM",
        "@local_config_mlir//:VectorToLoops",
    ],
    alwayslink = 1,
)
