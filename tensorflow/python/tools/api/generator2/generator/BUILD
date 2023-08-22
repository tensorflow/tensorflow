load("//tensorflow:pytype.default.bzl", "pytype_strict_binary")

pytype_strict_binary(
    name = "generator",
    srcs = ["generator.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/tools/api/generator2/shared:exported_api",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
