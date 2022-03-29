# DTensor Python API and libraries.

load("//tensorflow:tensorflow.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py", "tf_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_pybind_cc_library_wrapper")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

default_visibility = [
    "//tensorflow/dtensor:dtensor-internal",
]

package(
    default_visibility = default_visibility,
    licenses = ["notice"],
)

# -----------------------------------------------------------------------------
# Public API.

pytype_strict_library(
    name = "core",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = default_visibility + [
        "//tensorflow/dtensor:dtensor-users",
    ],
    deps = [
        ":api",
        ":d_checkpoint",
        ":d_variable",
        ":gen_dtensor_ops",
        ":layout",
        ":save_restore",
        "//tensorflow/python/eager:context",
    ],
)

# -----------------------------------------------------------------------------
# Implementations of the public API.

pytype_strict_library(
    name = "api",
    srcs = ["api.py"],
    srcs_version = "PY3",
    deps = [
        ":dtensor_device",
        ":gen_dtensor_ops",
        ":layout",
        "//tensorflow/python:config",
        "//tensorflow/python:device",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_dtensor_ops",
    out = "gen_dtensor_ops.py",
    deps = [
        "//tensorflow/dtensor/cc:dtensor_ops",
        "//tensorflow/dtensor/cc:dtensor_tpu_ops",
    ],
)

pytype_strict_library(
    name = "layout",
    srcs = ["layout.py"],
    deps = [
        ":gen_dtensor_ops",
        "//tensorflow/dtensor/proto:layout_proto_py_pb2",
        "//tensorflow/python:config",
        "//tensorflow/python:device",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "d_variable",
    srcs = ["d_variable.py"],
    srcs_version = "PY3",
    deps = [
        ":api",
        ":layout",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/tracking:base",
    ],
)

pytype_strict_library(
    name = "d_checkpoint",
    srcs = ["d_checkpoint.py"],
    deps = [
        ":api",
        ":d_variable",
        ":gen_dtensor_ops",
        ":layout",
        ":save_restore",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/dtensor/proto:layout_proto_py_pb2",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/training:py_checkpoint_reader",
        "//tensorflow/python/training/saving:checkpoint_options",
        "//tensorflow/python/training/saving:functional_saver",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/training/tracking:base",
        "//tensorflow/python/training/tracking:data_structures",
        "//tensorflow/python/training/tracking:graph_view",
        "//tensorflow/python/training/tracking:util",
    ],
)

pytype_strict_library(
    name = "save_restore",
    srcs = ["save_restore.py"],
    srcs_version = "PY3",
    deps = [
        ":api",
        ":d_variable",
        ":gen_dtensor_ops",
        ":layout",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
    ],
)

# -----------------------------------------------------------------------------
# The DTensor runtime.

pytype_strict_library(
    name = "dtensor_device",
    srcs = ["dtensor_device.py"],
    deps = [
        ":_dtensor_device",
        ":gen_dtensor_ops",
        ":layout",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:device",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//third_party/py/numpy",
    ],
)

tf_python_pybind_extension(
    name = "_dtensor_device",
    srcs = ["pywrap_dtensor_device.cc"],
    features = ["-layering_check"],
    module_name = "_dtensor_device",
    deps = [
        ":pywrap_densor_device_headers",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status_headers",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ],
)

tf_pybind_cc_library_wrapper(
    name = "pywrap_densor_device_headers",
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/dtensor/cc:dtensor_device_cc",
        "//tensorflow/python/eager:pywrap_tfe_lib",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
    ],
)

# -----------------------------------------------------------------------------
# Utilities.

pytype_strict_library(
    name = "mesh_util",
    srcs = ["mesh_util.py"],
    visibility = default_visibility + [
        "//tensorflow/dtensor:dtensor-users",
    ],
    deps = [
        ":core",
        ":tpu_util",
        "//tensorflow/python:config",
        "//tensorflow/python:device",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "tpu_util",
    srcs = ["tpu_util.py"],
    visibility = default_visibility + [
        "//tensorflow/dtensor:dtensor-users",
    ],
    deps = [
        ":core",
        ":dtensor_device",
        ":heartbeat",
        ":multi_client_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:device",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/tpu:topology",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "heartbeat",
    srcs = ["heartbeat.py"],
    deps = [
        ":core",
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:device",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/framework:constant_op",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "multi_client_util",
    srcs = ["multi_client_util.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "@absl_py//absl/logging",
    ],
)

tf_py_test(
    name = "dtensor_tf_api_test",
    size = "small",
    srcs = ["dtensor_tf_api_test.py"],
    deps = [
        "//tensorflow/dtensor/python:core",
        "//tensorflow/python/platform:test",
    ],
)
