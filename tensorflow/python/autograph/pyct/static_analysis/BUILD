load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "liveness",
    srcs = ["liveness.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":annos",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "reaching_fndefs",
    srcs = ["reaching_fndefs.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_strict_test(
    name = "reaching_fndefs_test",
    srcs = ["reaching_fndefs_test.py"],
    deps = [
        ":activity",
        ":reaching_definitions",
        ":reaching_fndefs",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "activity",
    srcs = ["activity.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":annos",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "type_inference",
    srcs = ["type_inference.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":activity",
        ":annos",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "reaching_definitions",
    srcs = ["reaching_definitions.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_strict_library(
    name = "annos",
    srcs = ["annos.py"],
    visibility = ["//visibility:public"],
)

py_strict_test(
    name = "activity_test",
    srcs = ["activity_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":activity",
        ":annos",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)

py_strict_test(
    name = "liveness_test",
    testonly = True,
    srcs = ["liveness_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":activity",
        ":liveness",
        ":reaching_fndefs",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct",
    ],
)

py_strict_test(
    name = "reaching_definitions_test",
    srcs = ["reaching_definitions_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":activity",
        ":reaching_definitions",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct",
    ],
)

py_strict_test(
    name = "type_inference_test",
    srcs = ["type_inference_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":activity",
        ":reaching_definitions",
        ":reaching_fndefs",
        ":type_inference",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct",
    ],
)
