load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "templates",
    srcs = ["templates.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":anno",
        ":ast_util",
        ":parser",
        ":qual_names",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "transpiler",
    srcs = ["transpiler.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cache",
        ":inspect_utils",
        ":loader",
        ":naming",
        ":origin_info",
        ":parser",
        ":templates",
        ":transformer",
        "//tensorflow/python/autograph/utils:ag_logging",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "ast_util",
    srcs = ["ast_util.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":anno",
        ":parser",
        ":qual_names",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "loader",
    srcs = ["loader.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":origin_info",
        ":parser",
    ],
)

py_strict_library(
    name = "gast_util",
    srcs = ["gast_util.py"],
    visibility = ["//visibility:public"],
    deps = ["@pypi_gast//:pkg"],
)

py_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
)

py_strict_library(
    name = "parser",
    srcs = ["parser.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":errors",
        ":inspect_utils",
        "//tensorflow/python/util:tf_inspect",
        "@pypi_astunparse//:pkg",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "naming",
    srcs = ["naming.py"],
    visibility = ["//visibility:public"],
    deps = [":qual_names"],
)

py_strict_library(
    name = "inspect_utils",
    srcs = ["inspect_utils.py"],
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/python/util:tf_inspect"],
)

py_strict_library(
    name = "origin_info",
    srcs = ["origin_info.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":anno",
        ":ast_util",
        ":parser",
        ":pretty_printer",
        "//tensorflow/python/util:tf_inspect",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "anno",
    srcs = ["anno.py"],
    visibility = ["//visibility:public"],
    deps = ["@pypi_gast//:pkg"],
)

py_strict_library(
    name = "errors",
    srcs = ["errors.py"],
    visibility = ["//visibility:public"],
)

py_strict_library(
    name = "transformer",
    srcs = ["transformer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":anno",
        ":parser",
        ":pretty_printer",
        ":templates",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "qual_names",
    srcs = ["qual_names.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":anno",
        ":parser",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "cfg",
    srcs = ["cfg.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":anno",
        "@pypi_astunparse//:pkg",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "error_utils",
    srcs = ["error_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":origin_info",
        "//tensorflow/python/util:traceback_utils",
    ],
)

py_strict_library(
    name = "cache",
    srcs = ["cache.py"],
    visibility = ["//visibility:public"],
)

py_strict_library(
    name = "pretty_printer",
    srcs = ["pretty_printer.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pypi_gast//:pkg",
        "@pypi_termcolor//:pkg",
    ],
)

py_strict_test(
    name = "anno_test",
    srcs = ["anno_test.py"],
    deps = [
        ":anno",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "ast_util_test",
    srcs = ["ast_util_test.py"],
    deps = [
        ":anno",
        ":ast_util",
        ":loader",
        ":parser",
        ":pretty_printer",
        ":qual_names",
        "@pypi_gast//:pkg",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "cache_test",
    srcs = ["cache_test.py"],
    deps = [
        ":cache",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "cfg_test",
    srcs = ["cfg_test.py"],
    deps = [
        ":cfg",
        ":parser",
        "@pypi_gast//:pkg",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "loader_test",
    srcs = ["loader_test.py"],
    deps = [
        ":ast_util",
        ":loader",
        ":parser",
        ":pretty_printer",
        "@pypi_gast//:pkg",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:tf_inspect",
    ],
)

py_strict_test(
    name = "error_utils_test",
    srcs = ["error_utils_test.py"],
    deps = [
        ":error_utils",
        ":origin_info",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "inspect_utils_test",
    srcs = ["inspect_utils_test.py"],
    deps = [
        ":inspect_utils",
        #internal proto upb dep
        "//tensorflow/python/autograph/pyct/testing:basic_definitions",
        "//tensorflow/python/autograph/pyct/testing:decorators",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/lib:__init__",
        "//tensorflow/python/platform:client_testlib",
    ],
)

sh_test(
    name = "inspect_utils_test_par",
    srcs = ["inspect_utils_test.sh"],
    tags = ["no_oss"],
)

py_strict_test(
    name = "naming_test",
    srcs = ["naming_test.py"],
    deps = [
        ":naming",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "origin_info_test",
    srcs = ["origin_info_test.py"],
    deps = [
        ":anno",
        ":inspect_utils",
        ":origin_info",
        ":parser",
        #internal proto upb dep
        "//tensorflow/python/autograph/pyct/testing:basic_definitions",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:tf_inspect",
    ],
)

py_strict_test(
    name = "parser_test",
    srcs = ["parser_test.py"],
    deps = [
        ":ast_util",
        ":errors",
        ":parser",
        ":pretty_printer",
        "@pypi_gast//:pkg",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "pretty_printer_test",
    srcs = ["pretty_printer_test.py"],
    deps = [
        ":pretty_printer",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "qual_names_test",
    srcs = ["qual_names_test.py"],
    deps = [
        ":anno",
        ":parser",
        ":qual_names",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "templates_test",
    srcs = ["templates_test.py"],
    deps = [
        ":loader",
        ":parser",
        ":qual_names",
        ":templates",
        "@absl_py//absl/testing:parameterized",
        "@pypi_gast//:pkg",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "transformer_test",
    srcs = ["transformer_test.py"],
    deps = [
        ":anno",
        ":origin_info",
        ":parser",
        ":transformer",
        "@pypi_gast//:pkg",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "transpiler_test",
    srcs = ["transpiler_test.py"],
    deps = [
        ":transformer",
        ":transpiler",
        "@pypi_gast//:pkg",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)
