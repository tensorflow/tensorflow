# python/lib/io package

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test", "tf_python_pybind_extension")
load("//tensorflow:tensorflow.bzl", "if_oss")

# copybara:uncomment_begin(google-only)
# load("//third_party/zlib:BUILD_defs.bzl", "brittle_test_relying_on_stable_zlib_output")
# copybara:uncomment_end

visibility = [
    "//tensorflow:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

tf_python_pybind_extension(
    name = "_pywrap_file_io",
    srcs = ["file_io_wrapper.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_absl",
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

py_strict_library(
    name = "lib",
    srcs = [
        "file_io.py",
        "python_io.py",
        "tf_record.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":_pywrap_file_io",
        ":_pywrap_record_io",
        "//tensorflow/python:errors",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_record_io",
    srcs = ["record_io_wrapper.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core/platform:types",
        "//tensorflow/python/lib/core:pybind11_absl",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/memory",
        "@pybind11",
    ],
)

tf_py_strict_test(
    name = "file_io_test",
    size = "small",
    srcs = ["file_io_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":lib",
        "//tensorflow/python:errors",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = if_oss("tf_record_test", "_tf_record_test"),
    size = "small",
    srcs = ["tf_record_test.py"],
    main = "tf_record_test.py",
    python_version = "PY3",
    tags = [
        # copybara:uncomment_begin(google-only)
        # "manual",
        # "notap",
        # copybara:uncomment_end
    ],
    deps = [
        ":lib",
        "//tensorflow/python:errors",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:compat",
    ],
)

# _tf_record_test relies on stable zlib output only during its execution.
# It should not be brittle with respect to upgrades of zlib software or
# different computers using different zlib software.
# copybara:uncomment_begin(google-only)
# brittle_test_relying_on_stable_zlib_output(
#     name = "tf_record_test",
#     test_target = ":_tf_record_test",
# )
# copybara:uncomment_end
