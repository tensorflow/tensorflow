load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    # By default, these targets should only be used within the quantization library.
    default_visibility = [
        "//learning/brain/mlir/quantization:__subpackages__",
        "//platforms/darwinn/compiler:__subpackages__",
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

td_library(
    name = "quantization_td_files",
    srcs = [
        "quantization.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantizationOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "quantization_interfaces_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "quantization_interface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "quantization_interface.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "quantization.td",
    deps = [
        ":quantization_td_files",
    ],
)
