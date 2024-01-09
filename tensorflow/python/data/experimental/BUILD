load("//tensorflow:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "experimental",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/experimental/ops:dataset_ops",
        "//tensorflow/python/data/experimental/ops:iterator_ops",
        "//tensorflow/python/data/experimental/ops:lookup_ops",
        "//tensorflow/python/data/experimental/ops:parsing_ops",
        "//tensorflow/python/data/experimental/ops:random_ops",
        "//tensorflow/python/data/experimental/service",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:optional_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/util:all_util",
    ],
)
