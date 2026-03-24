load("//xla:lit.bzl", "lit_test_suite")
load("//xla:xla.default.bzl", "xla_cc_binary")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package_group(
    name = "internal",
    packages = [
        "//xla/python/ifrt/ir/tests/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        ":internal",
    ]),
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = glob(["*.mlir"]),
    cfg = "//xla:lit.cfg.py",
    tools = [
        ":ifrt-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

xla_cc_binary(
    name = "ifrt-opt",
    testonly = True,
    srcs = ["ifrt-opt.cc"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt:mock",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir:atom_program_compiler",
        "//xla/python/ifrt/ir:ifrt_ir_program",
        "//xla/python/ifrt/ir/conversions/mpmd:lower_to_ifrt",
        "//xla/python/ifrt/ir/transforms:passes",
        "//xla/python/ifrt/support:module_parsing",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:RegisterAllDialects",  # buildcleaner: keep
        "@llvm-project//mlir:RegisterAllPasses",  # buildcleaner: keep
        "@shardy//shardy/dialect/mpmd/ir:dialect",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_cc_binary(
    name = "ifrt-translate",
    testonly = True,
    srcs = ["ifrt-translate.cc"],
    deps = [
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:serdes_proto_cc",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir:ifrt_ir_program",
        "//xla/python/ifrt/ir:ifrt_ir_program_serdes",  # build_cleaner: keep
        "//xla/python/ifrt/ir:version",
        "//xla/python/ifrt/ir:vifrt",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:TranslateLib",
        "@stablehlo//:register",
        "@stablehlo//:version",
    ],
)
