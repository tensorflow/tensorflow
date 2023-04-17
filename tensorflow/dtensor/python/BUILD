# DTensor Python API and libraries.

load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

default_visibility = [
    "//tensorflow/dtensor:dtensor-internal",
    "//third_party/py/jax_tpu_embedding:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = default_visibility,
    licenses = ["notice"],
)

# -----------------------------------------------------------------------------
# A list of all modules linked into TensorFlow.

py_library(
    name = "core",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow/python:__pkg__"],
    deps = [
        ":accelerator_util",
        ":api",
        ":config",
        ":d_checkpoint",
        ":d_variable",
        ":input_util",
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
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "config",
    srcs = ["config.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:device",
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
        ":config",
        "//tensorflow/dtensor/proto:layout_proto_py_pb2",
        "//tensorflow/python:_pywrap_dtensor_device",
        "//tensorflow/python:device",
        "//tensorflow/python:framework_ops",
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
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/training/saving:saveable_object",
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
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_options",
        "//tensorflow/python/checkpoint:graph_view",
        "//tensorflow/python/checkpoint:restore",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:data_structures",
        "//tensorflow/python/training:py_checkpoint_reader",
        "//tensorflow/python/training/saving:saveable_object",
        "//tensorflow/python/training/saving:saveable_object_util",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "numpy_util",
    srcs = ["numpy_util.py"],
    deps = [
        ":api",
        ":layout",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/types",
        "//third_party/py/numpy",
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
        ":mesh_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
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
        ":config",
        ":gen_dtensor_ops",
        ":layout",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:_pywrap_dtensor_device",
        "//tensorflow/python:device",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_util",
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
        ":accelerator_util",
        ":api",
        ":config",
        ":layout",
        ":tpu_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:config",
        "//tensorflow/python:device",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:tfrt_utils",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

# TODO(b/245589661): Split accelerator_util to its module after
# The circular dependence is removed with dtensor_initialize_tpu_system.
pytype_strict_library(
    name = "tpu_util",
    srcs = [
        "accelerator_util.py",
        "tpu_util.py",
    ],
    visibility = default_visibility + [
        "//tensorflow/dtensor:dtensor-users",
    ],
    deps = [
        ":api",
        ":config",
        ":dtensor_device",
        ":gen_dtensor_ops",
        ":heartbeat",
        ":layout",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:remote_utils",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/tpu:topology",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "heartbeat",
    srcs = ["heartbeat.py"],
    deps = [
        ":api",
        ":config",
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:device",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "accelerator_util",
    srcs = [],
    deps = [
        ":tpu_util",
    ],
)

pytype_strict_library(
    name = "input_util",
    srcs = ["input_util.py"],
    deps = [
        ":api",
        ":config",
        ":layout",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/types",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)
