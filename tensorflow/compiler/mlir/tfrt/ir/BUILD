# BUILD targets for runtime fallback opdefs.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

# Note: keep the following lines separate due to the way copybara works
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "tfrt_fallback_opdefs",
    srcs = [
        "tfrt_fallback.cc",
    ],
    hdrs = ["tfrt_fallback.h"],
    compatible_with = get_compatible_with_portable(),  # copybara: comment
    visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        # copybara:uncomment "//learning/brain/tfrt/tpu/compiler:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
        "//tensorflow/core/runtime_fallback:internal",
        "//tensorflow/core/tfrt/saved_model:friends",
        # copybara:uncomment "//third_party/tf_runtime_google:__subpackages__",
    ],
    deps = [
        ":tfrt_fallback_opdefs_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tfrt_fallback_async_opdefs",
    srcs = [
        "tfrt_fallback_async.cc",
    ],
    hdrs = ["tfrt_fallback_async.h"],
    compatible_with = get_compatible_with_portable(),  # copybara: comment
    visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        # copybara:uncomment "//learning/brain/tfrt/tpu/compiler/mlir:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
        "//tensorflow/core/tfrt/saved_model:friends",
    ],
    deps = [
        ":tfrt_fallback_async_opdefs_inc_gen",
        ":tfrt_fallback_common",
        ":tfrt_fallback_opdefs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:compiler_tfrt_op_interfaces",
        "@tf_runtime//:compiler_tfrt_traits",
        "@tf_runtime//:core_runtime_opdefs",
        "@tf_runtime//:core_runtime_sync_opdefs",
    ],
)

cc_library(
    name = "tfrt_fallback_sync_opdefs",
    srcs = [
        "tfrt_fallback_sync.cc",
    ],
    hdrs = ["tfrt_fallback_sync.h"],
    visibility = [
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
        "//tensorflow/core/runtime_fallback:internal",
    ],
    deps = [
        ":tfrt_fallback_common",
        ":tfrt_fallback_opdefs",
        ":tfrt_fallback_sync_opdefs_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
        "@tf_runtime//:tensor_opdefs",
    ],
)

cc_library(
    name = "tfrt_fallback_common",
    srcs = ["tfrt_fallback_common.cc"],
    hdrs = ["tfrt_fallback_common.h"],
    compatible_with = get_compatible_with_portable(),  # copybara: comment
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tf_runtime//:basic_kernels_opdefs",
    ],
)

cc_library(
    name = "tfrt_fallback_util",
    srcs = ["tfrt_fallback_util.cc"],
    hdrs = ["tfrt_fallback_util.h"],
    visibility = ["//tensorflow/compiler/mlir/tfrt:__subpackages__"],
    deps = [
        ":tfrt_fallback_async_opdefs",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

td_library(
    name = "tfrt_fallback_td_files",
    srcs = [
        "tfrt_fallback.td",
        "tfrt_fallback_async.td",
        "tfrt_fallback_sync.td",
    ],
    compatible_with = get_compatible_with_portable(),  # copybara: comment
    includes = ["."],
    visibility = [
        "//learning/brain/experimental/tfrt:__subpackages__",
        "//learning/brain/tfrt/tpu/compiler:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@tf_runtime//:CoreRTTdFiles",
        "@tf_runtime//:OpBaseTdFiles",
        "@tf_runtime//:compiler_td_files",
    ],
)

gentbl_cc_library(
    name = "tfrt_fallback_opdefs_inc_gen",
    compatible_with = get_compatible_with_portable(),  # copybara: comment
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "tfrt_fallback.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "tfrt_fallback.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tfrt_fallback.td",
    deps = [":tfrt_fallback_td_files"],
)

gentbl_cc_library(
    name = "tfrt_fallback_async_opdefs_inc_gen",
    compatible_with = get_compatible_with_portable(),  # copybara: comment
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "tfrt_fallback_async.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "tfrt_fallback_async.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tfrt_fallback_async.td",
    deps = [":tfrt_fallback_td_files"],
)

gentbl_cc_library(
    name = "tfrt_fallback_sync_opdefs_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "tfrt_fallback_sync.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "tfrt_fallback_sync.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=tfrt_fallback_sync",
            ],
            "tfrt_fallback_sync_dialect.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tfrt_fallback_sync.td",
    test = True,
    visibility = [
        "//learning/brain/experimental/tfrt:__subpackages__",
    ],
    deps = [":tfrt_fallback_td_files"],
)

td_library(
    name = "gpu_ops_td_file",
    srcs = [
        "gpu_ops.td",
    ],
    includes = ["."],
    visibility = [
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
    ],
    deps = [
        ":tfrt_fallback_td_files",
        "@tf_runtime//:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tfrt_gpu_opdefs_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "gpu_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "gpu_ops.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=gpurt",
            ],
            "gpurt_dialect.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "gpu_ops.td",
    test = True,
    visibility = [
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
    ],
    deps = [":gpu_ops_td_file"],
)

cc_library(
    name = "tfrt_gpu_opdefs",
    srcs = [
        "gpu_ops.cc",
    ],
    hdrs = ["gpu_ops.h"],
    visibility = [
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
    ],
    deps = [
        ":tfrt_fallback_opdefs",
        ":tfrt_gpu_opdefs_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:IR",
    ],
)
