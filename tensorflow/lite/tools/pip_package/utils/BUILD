load("//tensorflow:pytype.default.bzl", "pytype_strict_binary")

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "manylinux_compliance_test",
    srcs = ["manylinux_compliance_test.py"],
    visibility = ["//visibility:public"],
)

pytype_strict_binary(
    name = "wheel_builder",
    srcs = [
        "wheel_builder.py",
    ],
    main = "wheel_builder.py",
    deps = [
        "@pypi_setuptools//:pkg",
        "@pypi_wheel//:pkg",
    ],
)
