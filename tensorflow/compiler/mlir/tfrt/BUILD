load("//third_party/mlir:tblgen.bzl", "gentbl")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library_cc")

# TF to TFRT kernels conversion.
package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    packages = [
        "//learning/brain/experimental/tfrt/...",
        "//tensorflow/compiler/...",
        "//tensorflow/core/runtime_fallback/...",
        "//tensorflow/core/tfrt/experimental/saved_model/...",
        "//third_party/tf_runtime_google/...",
    ],
)

cc_library(
    name = "tf_legalize_to_tfrt",
    srcs = [
        "tf_legalize_to_hex.cc",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/memory",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

filegroup(
    name = "runtime_fallback_ops_td_files",
    srcs = [
        "runtime_fallback/runtime_fallback_ops.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
        "@tf_runtime//:OpBaseTdFiles",
    ],
)

gentbl(
    name = "runtime_fallback_ops_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "runtime_fallback_ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "runtime_fallback_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "runtime_fallback/runtime_fallback_ops.td",
    td_includes = [
        "external/tf_runtime/include",
    ],
    td_srcs = [
        ":runtime_fallback_ops_td_files",
    ],
)

cc_library(
    name = "runtime_fallback_opdefs_alwayslink",
    srcs = [
        "runtime_fallback/dialect_static_registration.cc",
        "runtime_fallback/runtime_fallback_combine.cc",
        "runtime_fallback/runtime_fallback_ops.cc",
    ],
    hdrs = [
        "runtime_fallback/runtime_fallback_ops.h",
    ],
    deps = [
        ":runtime_fallback_ops_inc_gen",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Support",
        "@tf_runtime//:basic_kernels_opdefs_alwayslink",
        "@tf_runtime//:tensor_opdefs_alwayslink",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lower_tf_to_tfd_alwayslink",
    srcs = ["runtime_fallback/lower_tf_to_tfd.cc"],
    deps = [
        "runtime_fallback_opdefs_alwayslink",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:basic_kernels_opdefs_alwayslink",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_to_corert",
    srcs = [
        "transforms/optimize.cc",
        "transforms/tf_to_corert.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:tstring",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:basic_kernels_opdefs_alwayslink",
        "@tf_runtime//:core_runtime_opdefs_alwayslink",
    ],
    alwayslink = 1,
)

cc_library(
    name = "saved_model",
    srcs = [
        "saved_model/saved_model.cc",
    ],
    hdrs = [
        "saved_model/saved_model.h",
    ],
    deps = [
        ":tf_to_corert",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_lib",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:mlirtobef",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

cc_library(
    name = "compatibility_analysis",
    srcs = [
        "analysis/compatibility_analysis.cc",
    ],
    hdrs = [
        "analysis/compatibility_analysis.h",
    ],
    deps = [
        ":analysis/analysis_proto_cc",
        ":tf_to_corert",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:lib_proto_parsing",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Translation",
    ],
    alwayslink = 1,
)

tf_proto_library_cc(
    name = "analysis/analysis_proto",
    srcs = ["analysis/analysis.proto"],
    cc_api_version = 2,
)
