# DTensor Python API and libraries.

load("//tensorflow:tensorflow.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py", "tf_py_test")

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
        ":mesh_util",
        ":save_restore",
        ":tpu_util",
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
        "//tensorflow/python/util:tf_export",
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
        "//tensorflow/python/util:tf_export",
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
        "//tensorflow/python/util:tf_export",
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
        "//tensorflow/python/util:tf_export",
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
        "//tensorflow/python/util:tf_export",
    ],
)

# -----------------------------------------------------------------------------
# The DTensor runtime.

pytype_strict_library(
    name = "dtensor_device",
    srcs = ["dtensor_device.py"],
    deps = [
        ":gen_dtensor_ops",
        ":layout",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:_pywrap_dtensor_device",
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

# -----------------------------------------------------------------------------
# Utilities.

pytype_strict_library(
    name = "mesh_util",
    srcs = ["mesh_util.py"],
    visibility = default_visibility + [
        "//tensorflow/dtensor:dtensor-users",
    ],
    deps = [
        ":api",
        ":layout",
        ":multi_client_util",
        ":tpu_util",
        "//tensorflow/python:config",
        "//tensorflow/python:device",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:tf_export",
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
        ":api",
        ":dtensor_device",
        ":gen_dtensor_ops",
        ":heartbeat",
        ":layout",
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
        "//tensorflow/python/framework:tfrt_utils",
        "//tensorflow/python/tpu:topology",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_library(
    name = "heartbeat",
    srcs = ["heartbeat.py"],
    deps = [
        ":api",
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
