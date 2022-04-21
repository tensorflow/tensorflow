# BUILD targets for runtime fallback opdefs.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    licenses = ["notice"],
)

cc_library(
    name = "tfrt_fallback_opdefs",
    srcs = [
        "tfrt_fallback.cc",
    ],
    hdrs = ["tfrt_fallback.h"],
    visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        # copybara:uncomment "//learning/brain/tfrt/tpu/compiler:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
        # copybara:uncomment "//tensorflow/core/runtime_fallback:internal",
        # copybara:uncomment "//third_party/tf_runtime_google:__subpackages__",
    ],
    deps = [
        ":tfrt_fallback_opdefs_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tfrt_fallback_async_opdefs",
    srcs = [
        "tfrt_fallback_async.cc",
    ],
    hdrs = ["tfrt_fallback_async.h"],
    visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
        # copybara:uncomment "//tensorflow/core/runtime_fallback:internal",
        # copybara:uncomment "//third_party/tf_runtime_google:__subpackages__",
    ],
    deps = [
        ":tfrt_fallback_async_opdefs_inc_gen",
        ":tfrt_fallback_common",
        ":tfrt_fallback_opdefs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
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
        # copybara:uncomment "//tensorflow/core/runtime_fallback:internal",
    ],
    deps = [
        ":tfrt_fallback_common",
        ":tfrt_fallback_opdefs",
        ":tfrt_fallback_sync_opdefs_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
        "@tf_runtime//:tensor_opdefs",
    ],
)

cc_library(
    name = "tfrt_fallback_common",
    srcs = ["tfrt_fallback_common.cc"],
    hdrs = ["tfrt_fallback_common.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@tf_runtime//:basic_kernels_opdefs",
    ],
)

cc_library(
    name = "tfrt_fallback_util",
    srcs = ["tfrt_fallback_util.cc"],
    hdrs = ["tfrt_fallback_util.h"],
    visibility = [
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
        # copybara:uncomment "//tensorflow/core/runtime_fallback:internal",
        "//tensorflow/core/tfrt/graph_executor:__pkg__",
        "//tensorflow/core/tfrt/saved_model:__pkg__",
    ],
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
    includes = ["."],
    visibility = [
        "//learning/brain/tfrt/tpu/compiler:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
    ],
    deps = [
        "@llvm-project//mlir:SideEffectTdFiles",
        "@tf_runtime//:CoreRTTdFiles",
        "@tf_runtime//:OpBaseTdFiles",
        "@tf_runtime//:compiler_td_files",
    ],
)

gentbl_cc_library(
    name = "tfrt_fallback_opdefs_inc_gen",
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
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tfrt_fallback_sync.td",
    test = True,
    deps = [":tfrt_fallback_td_files"],
)
