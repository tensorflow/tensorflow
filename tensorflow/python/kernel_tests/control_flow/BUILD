# Tests of TensorFlow control flow ops written using the Python API.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(licenses = ["notice"])

cuda_py_test(
    name = "cond_v2_test",
    size = "medium",
    srcs = ["cond_v2_test.py"],
    grpc_enabled = True,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond_v2",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:test_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:while_v2",
        "//tensorflow/python/compat",
    ],
)

cuda_py_test(
    name = "control_flow_ops_py_test",
    size = "medium",
    srcs = ["control_flow_ops_py_test.py"],
    shard_count = 16,
    tags = [
        "no_windows",  # TODO(b/184424727): Re-enable this.
        "notsan",  # TODO(b/132205147): Re-enable this.
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond_v2",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:data_flow_ops_gen",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:logging_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:state_ops_gen",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python:while_v2",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "control_flow_util_test",
    size = "small",
    srcs = ["control_flow_util_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_ops_gen",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:test_ops",
    ],
)

tf_py_test(
    name = "control_flow_util_v2_test",
    size = "small",
    srcs = ["control_flow_util_v2_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond_v2",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util_v2",
        "//tensorflow/python:while_v2",
    ],
)

cuda_py_test(
    name = "functional_ops_test",
    size = "medium",
    srcs = ["functional_ops_test.py"],
    grpc_enabled = True,
    shard_count = 2,
    tags = ["no_windows"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python:while_v2",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/eager:cancellation",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:executor",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "map_fn_test",
    size = "small",
    srcs = ["map_fn_test.py"],
    grpc_enabled = True,
    shard_count = 2,
    tags = ["no_windows"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "py_func_test",
    size = "small",
    srcs = ["py_func_test.py"],
    grpc_enabled = True,
    tags = ["no_windows"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:script_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "scan_ops_test",
    size = "medium",
    srcs = ["scan_ops_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "while_v2_test",
    size = "medium",
    srcs = ["while_v2_test.py"],
    grpc_enabled = True,
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:control_flow_v2_toggles",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:tf_optimizer",
        "//tensorflow/python:while_v2",
        "//tensorflow/python/eager:def_function",
        "@absl_py//absl/testing:parameterized",
    ],
)
