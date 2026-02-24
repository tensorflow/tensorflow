load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package_group(
    name = "internal",
    packages = [
        "//xla/python/ifrt/ir/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        ":internal",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "xla_computation",
    srcs = ["xla_computation.cc"],
    hdrs = ["xla_computation.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility(["//xla/python/ifrt:users"]),
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/builder/lib:constants",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_sharding",
        "//xla/hlo/ir:tile_assignment",
        "//xla/hlo/translate:stablehlo",
        "//xla/python/ifrt",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)
