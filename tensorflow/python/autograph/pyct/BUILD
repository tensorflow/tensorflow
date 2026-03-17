load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "templates",
    srcs = ["templates.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":anno",
        ":ast_util",
        ":parser",
        ":qual_names",
        "@pypi//gast",
    ],
)

py_library(
    name = "transpiler",
    srcs = ["transpiler.py"],
    strict_deps = True,
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
        "@pypi//gast",
    ],
)

py_library(
    name = "ast_util",
    srcs = ["ast_util.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":anno",
        ":parser",
        ":qual_names",
        "@pypi//gast",
    ],
)

py_library(
    name = "loader",
    srcs = ["loader.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":origin_info",
        ":parser",
    ],
)

py_library(
    name = "gast_util",
    srcs = ["gast_util.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = ["@pypi//gast"],
)

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
)

py_library(
    name = "parser",
    srcs = ["parser.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":errors",
        ":inspect_utils",
        "//tensorflow/python/util:tf_inspect",
        "@pypi//astunparse",
        "@pypi//gast",
    ],
)

py_library(
    name = "naming",
    srcs = ["naming.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [":qual_names"],
)

py_library(
    name = "inspect_utils",
    srcs = ["inspect_utils.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/python/util:tf_inspect"],
)

py_library(
    name = "origin_info",
    srcs = ["origin_info.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":anno",
        ":ast_util",
        ":parser",
        ":pretty_printer",
        "//tensorflow/python/util:tf_inspect",
        "@pypi//gast",
    ],
)

py_library(
    name = "anno",
    srcs = ["anno.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = ["@pypi//gast"],
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
)

py_library(
    name = "transformer",
    srcs = ["transformer.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":anno",
        ":parser",
        ":pretty_printer",
        ":templates",
        "@pypi//gast",
    ],
)

py_library(
    name = "qual_names",
    srcs = ["qual_names.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":anno",
        ":parser",
        "@pypi//gast",
    ],
)

py_library(
    name = "cfg",
    srcs = ["cfg.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":anno",
        "@pypi//astunparse",
        "@pypi//gast",
    ],
)

py_library(
    name = "error_utils",
    srcs = ["error_utils.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":origin_info",
        "//tensorflow/python/util:traceback_utils",
    ],
)

py_library(
    name = "cache",
    srcs = ["cache.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
)

py_library(
    name = "pretty_printer",
    srcs = ["pretty_printer.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        "@pypi//gast",
        "@pypi//termcolor",
    ],
)

py_test(
    name = "anno_test",
    srcs = ["anno_test.py"],
    strict_deps = True,
    deps = [
        ":anno",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "ast_util_test",
    srcs = ["ast_util_test.py"],
    strict_deps = True,
    deps = [
        ":anno",
        ":ast_util",
        ":loader",
        ":parser",
        ":pretty_printer",
        ":qual_names",
        "@pypi//gast",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "cache_test",
    srcs = ["cache_test.py"],
    strict_deps = True,
    deps = [
        ":cache",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "cfg_test",
    srcs = ["cfg_test.py"],
    strict_deps = True,
    deps = [
        ":cfg",
        ":parser",
        "@pypi//gast",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "loader_test",
    srcs = ["loader_test.py"],
    strict_deps = True,
    deps = [
        ":ast_util",
        ":loader",
        ":parser",
        ":pretty_printer",
        "@pypi//gast",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:tf_inspect",
    ],
)

py_test(
    name = "error_utils_test",
    srcs = ["error_utils_test.py"],
    strict_deps = True,
    deps = [
        ":error_utils",
        ":origin_info",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "inspect_utils_test",
    srcs = ["inspect_utils_test.py"],
    strict_deps = True,
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

py_test(
    name = "naming_test",
    srcs = ["naming_test.py"],
    strict_deps = True,
    deps = [
        ":naming",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "origin_info_test",
    srcs = ["origin_info_test.py"],
    strict_deps = True,
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

py_test(
    name = "parser_test",
    srcs = ["parser_test.py"],
    strict_deps = True,
    deps = [
        ":ast_util",
        ":errors",
        ":parser",
        ":pretty_printer",
        "@pypi//gast",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "pretty_printer_test",
    srcs = ["pretty_printer_test.py"],
    strict_deps = True,
    deps = [
        ":pretty_printer",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "qual_names_test",
    srcs = ["qual_names_test.py"],
    strict_deps = True,
    deps = [
        ":anno",
        ":parser",
        ":qual_names",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "templates_test",
    srcs = ["templates_test.py"],
    strict_deps = True,
    deps = [
        ":loader",
        ":parser",
        ":qual_names",
        ":templates",
        "@absl_py//absl/testing:parameterized",
        "@pypi//gast",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "transformer_test",
    srcs = ["transformer_test.py"],
    strict_deps = True,
    deps = [
        ":anno",
        ":origin_info",
        ":parser",
        ":transformer",
        "@pypi//gast",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_test(
    name = "transpiler_test",
    srcs = ["transpiler_test.py"],
    strict_deps = True,
    deps = [
        ":transformer",
        ":transpiler",
        "@pypi//gast",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
    ],
)
