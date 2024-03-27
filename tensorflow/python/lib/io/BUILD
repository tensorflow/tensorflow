# python/lib/io package

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "if_oss")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test", "tf_python_pybind_extension")

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
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_file_io.pyi",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_absl",
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_record_io",
    srcs = ["record_io_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_record_io.pyi",
    ],
    deps = [
        "//tensorflow/core:core_stringpiece",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:types",
        "//tensorflow/python/lib/core:pybind11_absl",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/memory",
        "@pybind11",
    ],
)

py_strict_library(
    name = "file_io",
    srcs = ["file_io.py"],
    visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow:internal",
        "//third_party/cloud_tpu/convergence_tools:__subpackages__",
        "//third_party/proto_splitter:__subpackages__",  # TODO(b/277279227): remove this dep from proto_splitter
        "//third_party/py/tf_agents:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":_pywrap_file_io",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "tf_record",
    srcs = ["tf_record.py"],
    visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow:internal",
        "//third_party/cloud_tpu/convergence_tools:__subpackages__",
        "//third_party/py/tf_agents:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":_pywrap_record_io",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "python_io",
    srcs = ["python_io.py"],
    visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow:internal",
        "//third_party/cloud_tpu/convergence_tools:__subpackages__",
        "//third_party/py/tf_agents:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [":tf_record"],
)

tf_py_strict_test(
    name = "file_io_test",
    size = "small",
    srcs = ["file_io_test.py"],
    python_version = "PY3",
    tags = [
        "no_rocm",
        "no_windows",
    ],
    deps = [
        ":file_io",
        "//tensorflow/python/framework:errors",
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
        ":tf_record",
        "//tensorflow/python/framework:errors",
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
