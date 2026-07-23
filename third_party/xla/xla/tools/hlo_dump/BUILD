load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/util:cc_embed_data.bzl", "cc_embed_data")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla:internal"],
)

cc_embed_data(
    name = "hlo_dump_assets",
    srcs = [
        "assets/hlo_dump_template.html",
        "assets/hlo_dump_style.css",
        "assets/hlo_dump_ui_bin_sanitized.js",
    ],
    outs = [
        "hlo_dump_assets.cc",
        "hlo_dump_assets.h",
    ],
    compatible_with = get_compatible_with_portable(),
    embedopts = ["--namespace=xla::tools::hlo_dump"],
)

cc_library(
    name = "hlo_lexer",
    srcs = ["hlo_lexer.cc"],
    hdrs = ["hlo_lexer.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "hlo_dump_utils",
    srcs = ["hlo_dump_utils.cc"],
    hdrs = ["hlo_dump_utils.h"],
    deps = [
        ":hlo_dump_assets",
        ":hlo_lexer",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
        "@riegeli//riegeli/bytes:string_writer",
        "@riegeli//riegeli/zlib:zlib_writer",
        "@tsl//tsl/platform:coding",
        "@tsl//tsl/platform:path",
    ],
)

xla_cc_test(
    name = "hlo_lexer_test",
    srcs = ["hlo_lexer_test.cc"],
    deps = [
        ":hlo_lexer",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "hlo_dump_utils_test",
    size = "small",
    srcs = ["hlo_dump_utils_test.cc"],
    deps = [
        ":hlo_dump_utils",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
