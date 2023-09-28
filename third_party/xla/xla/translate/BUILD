load("//xla:xla.bzl", "xla_cc_binary")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

build_test(
    name = "xla-translate_build_test",
    targets = [
        ":xla-translate",
    ],
)

xla_cc_binary(
    name = "xla-translate",
    testonly = True,
    srcs = ["xla_translate_main.cc"],
    deps = [
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:cpu_transfer_manager",
        "//xla/stream_executor/host:host_platform",
        "//xla/stream_executor/platform",
        "//xla/translate/hlo_to_mhlo:translate_registration",
        "//xla/translate/mhlo_to_hlo:translate_registration",
        "//xla/translate/mhlo_to_lhlo_with_xla:translate_registration",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
        "@local_tsl//tsl/platform:platform_port",
    ],
)
