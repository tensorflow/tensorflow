load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla/python/ifrt/ir:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "module_parsing",
    srcs = ["module_parsing.cc"],
    hdrs = ["module_parsing.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//xla/python/ifrt/ir:users",
    ]),
    deps = [
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/python/ifrt/ir",
        "//xla/python/ifrt/ir:vifrt",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:ShapeDialect",
        "@shardy//shardy/dialect/sdy/ir:register",
        "@stablehlo//:register",
    ],
)

cc_library(
    name = "sharding_conversions",
    srcs = ["sharding_conversions.cc"],
    hdrs = ["sharding_conversions.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//xla/python/ifrt/ir:friends",
    ]),
    deps = [
        "//xla:array",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "sharding_conversions_test",
    srcs = ["sharding_conversions_test.cc"],
    deps = [
        ":sharding_conversions",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:tile_assignment",
        "//xla/python/ifrt",
        "//xla/python/ifrt:basic_device_list",
        "//xla/python/ifrt:mock",
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util/proto:parse_text_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)
