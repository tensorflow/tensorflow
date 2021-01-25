# platform package

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_build_info_genrule")
load("//tensorflow/core/platform:build_config.bzl", "pyx_library", "tf_additional_all_protos", "tf_additional_lib_deps", "tf_proto_library", "tf_protos_grappler")  # @unused

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

visibility = [
    "//tensorflow:__subpackages__",
]

package(
    default_visibility = visibility,
    licenses = ["notice"],  # Apache 2.0
)

tf_py_build_info_genrule(
    name = "py_build_info_gen",
    out = "build_info.py",
)

py_library(
    name = "build_info",
    srcs = ["build_info.py"],
    srcs_version = "PY3",
)

py_library(
    name = "platform",
    srcs = glob(
        [
            "*.py",
        ],
        exclude = [
            "*test.py",
            "benchmark.py",  # In platform_benchmark.
            "analytics.py",  # In platform_analytics.
            "device_context.py",  # In platform_device_context.
        ],
    ) + ["build_info.py"],
    srcs_version = "PY3",
    deps = [
        ":build_info",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:util",
        "//tensorflow/python/lib/io:lib",
        "//tensorflow/python/util:_pywrap_util_port",
        "@absl_py//absl/flags",
        "@rules_python//python/runfiles",
        "@six_archive//:six",
    ],
)

py_library(
    name = "benchmark",
    srcs = ["benchmark.py"],
    srcs_version = "PY3",
    deps = [
        ":platform",
        "//tensorflow/python:client",
        "@six_archive//:six",
    ],
)

py_library(
    name = "analytics",
    srcs = ["analytics.py"],
    srcs_version = "PY3",
)

py_library(
    name = "device_context",
    srcs = ["device_context.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
    ],
)

py_library(
    name = "test",
    srcs = ["googletest.py"],
    srcs_version = "PY3",
    deps = [
        ":benchmark",
        "@absl_py//absl/testing:absltest",
    ],
)

tf_py_test(
    name = "resource_loader_test",
    size = "small",
    srcs = ["resource_loader_test.py"],
    data = [
        "resource_loader.py",
    ],
    python_version = "PY3",
    tags = [
        "no_pip",
        "no_windows",
    ],
    deps = [
        ":platform",
        ":test",
    ],
)

tf_py_test(
    name = "sysconfig_test",
    size = "small",
    srcs = ["sysconfig_test.py"],
    data = [
        "sysconfig.py",
    ],
    python_version = "PY3",
    tags = [
        "no_pip",
        "no_windows",
    ],
    deps = [
        ":platform",
        ":test",
    ],
)

tf_py_test(
    name = "flags_test",
    size = "small",
    srcs = ["flags_test.py"],
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":platform",
    ],
)

tf_py_test(
    name = "stacktrace_handler_test",
    size = "small",
    srcs = ["stacktrace_handler_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",
        "nomac",
    ],
    deps = [
        ":client_testlib",
        ":platform",
    ],
)

tf_py_test(
    name = "app_test",
    size = "small",
    srcs = ["app_test.py"],
    python_version = "PY3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [":platform"],
)

tf_python_pybind_extension(
    name = "_pywrap_stacktrace_handler",
    srcs = ["stacktrace_handler_wrapper.cc"],
    hdrs = ["//tensorflow/core/platform:stacktrace_handler_hdrs"],
    module_name = "_pywrap_stacktrace_handler",
    deps = [
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

py_library(
    name = "client_testlib",
    srcs = ["test.py"],
    srcs_version = "PY3",
    deps = [
        ":test",
        "//tensorflow/python:client",
        "//tensorflow/python:cond_v2",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python:util",
        "//tensorflow/python:while_v2",
    ],
)

tf_py_test(
    name = "build_info_test",
    size = "small",
    srcs = [
        "build_info.py",
        "build_info_test.py",
    ],
    main = "build_info_test.py",
    python_version = "PY3",
    tags = [
        "no_pip",
        "notap",
    ],
    deps = [
        ":client_testlib",
        ":platform",
    ],
)

tf_py_test(
    name = "benchmark_test",
    size = "small",
    srcs = [
        "benchmark.py",
        "benchmark_test.py",
    ],
    main = "benchmark_test.py",
    python_version = "PY3",
    tags = [
        "no_pip",
    ],
    deps = [
        ":client_testlib",
        ":platform",
    ],
)

pybind_extension(
    name = "_pywrap_tf2",
    srcs = ["enable_tf2.cc"],
    hdrs = ["//tensorflow/core/platform:enable_tf2_hdr"],
    module_name = "_pywrap_tf2",
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:enable_tf2_utils",
        "@pybind11",
    ],
)
