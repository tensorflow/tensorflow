load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//xla:xla_js.default.bzl",
    "xla_js_binary",
    "xla_ts_library",
)
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl/util:cc_embed_data.bzl", "cc_embed_data")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "hlo_lexer",
    srcs = ["hlo_lexer.cc"],
    hdrs = ["hlo_lexer.h"],
    compatible_with = if_google(["//buildenv/target:libtpu"]),
    deps = [
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
    ],
)

xla_ts_library(
    name = "hlo_dump_ui",
    srcs = [
        "hlo_dump_ui.ts",
    ],
    compatible_with = if_google(["//buildenv/target:libtpu"]),
    tags = ["TPU_SDK_DEPS_ANALYSIS_IGNORE"],
    deps = [
        "//xla/tools/debug_info/graph_view:graph_renderer",
    ],
)

xla_js_binary(
    name = "hlo_dump_ui_bin",
    compatible_with = if_google(["//buildenv/target:libtpu"]),
    tags = ["TPU_SDK_DEPS_ANALYSIS_IGNORE"],
    deps = [":hlo_dump_ui"],
)

genrule(
    name = "hlo_dump_ui_bin_sanitized",
    srcs = [":hlo_dump_ui_bin.js"],
    outs = ["hlo_dump_ui_bin_sanitized.js"],
    cmd = "sed 's/google" + "3/g3/g' $(location :hlo_dump_ui_bin.js) > $@",
    compatible_with = if_google(["//buildenv/target:libtpu"]),
)

cc_embed_data(
    name = "hlo_dump_assets",
    srcs = [
        "hlo_dump_template.html",
        "hlo_dump_style.css",
        "hlo_dump_ui_bin_sanitized.js",
    ],
    outs = [
        "hlo_dump_assets.cc",
        "hlo_dump_assets.h",
    ],
    compatible_with = if_google(["//buildenv/target:libtpu"]),
    embedopts = ["--namespace=xla::tools::debug_info"],
)

cc_library(
    name = "hlo_dump_utils",
    srcs = ["hlo_dump_utils.cc"],
    hdrs = ["hlo_dump_utils.h"],
    compatible_with = if_google(["//buildenv/target:libtpu"]),
    deps = [
        ":hlo_dump_assets",
        ":hlo_lexer",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "//xla/tsl/util:file_toc",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_googlesource_code_re2//:re2",
        "@nlohmann_json_lib//:json",
        "@riegeli//riegeli/bytes:string_writer",
        "@riegeli//riegeli/zlib:zlib_writer",
        "@tsl//tsl/platform:path",
    ],
)
