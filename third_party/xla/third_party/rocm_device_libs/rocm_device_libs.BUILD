load("build_defs.bzl", "bitcode_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

exports_files([
    "LICENSE.TXT",
])

cc_binary(
    name = "prepare_builtins",
    srcs = glob([
        "utils/prepare-builtins/*.cpp",
        "utils/prepare-builtins/*.h",
    ]),
    copts = [
        "-fno-rtti -fno-exceptions",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "@llvm-project//llvm:BitReader",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:config",
    ],
)

bitcode_library(
    name = "ocml",
    srcs = glob([
        "ocml/src/*.cl",
    ]),
    hdrs = glob([
        "ocml/src/*.h",
        "ocml/inc/*.h",
        "irif/inc/*.h",
        "oclc/inc/*.h",
    ]),
    file_specific_flags = {
        "native_logF.cl": ["-fapprox-func"],
        "native_expF.cl": ["-fapprox-func"],
        "sqrtF.cl": ["-cl-fp32-correctly-rounded-divide-sqrt"],
    },
)

bitcode_library(
    name = "ockl",
    srcs = glob([
        "ockl/src/*.cl",
        "ockl/src/*.ll",
    ]),
    hdrs = glob([
        "ockl/inc/*.h",
        "irif/inc/*.h",
        "oclc/inc/*.h",
    ]),
    file_specific_flags = {
        "gaaf.cl": ["-munsafe-fp-atomics"],
    },
)
