# Tests of TensorFlow array ops kernels written using the Python API.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow/core/platform:build_config_root.bzl", "tf_additional_xla_deps_py")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cuda_py_test(
    name = "array_ops_test",
    size = "medium",
    srcs = ["array_ops_test.py"],
    shard_count = 10,
    tags = [
        "noasan",  # times out
        "optonly",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:test_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "batch_gather_op_test",
    srcs = ["batch_gather_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "batch_scatter_ops_test",
    srcs = ["batch_scatter_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "batchtospace_op_test",
    size = "small",
    srcs = ["batchtospace_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "bcast_ops_test",
    size = "small",
    srcs = ["bcast_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
    ],
)

cuda_py_test(
    name = "bitcast_op_test",
    size = "small",
    srcs = ["bitcast_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "broadcast_to_ops_test",
    size = "medium",
    srcs = ["broadcast_to_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "cast_op_test",
    size = "small",
    srcs = ["cast_op_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "concat_op_test",
    size = "medium",
    srcs = ["concat_op_test.py"],
    tags = ["no_windows"],  # b/126916429
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "constant_op_eager_test",
    size = "small",
    srcs = ["constant_op_eager_test.py"],
    tags = [
        "no_windows",  # TODO(b/207048720): re-enable
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:test",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "constant_op_test",
    size = "small",
    srcs = ["constant_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "denormal_test",
    size = "small",
    srcs = ["denormal_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "depthtospace_op_test",
    size = "medium",
    srcs = ["depthtospace_op_test.py"],
    tags = [
        "no_windows",  # TODO(b/184424727): Re-enable this.
        "no_windows_gpu",
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "diag_op_test",
    size = "medium",
    srcs = ["diag_op_test.py"],
    shard_count = 6,
    tags = ["no_windows_gpu"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "edit_distance_op_test",
    size = "small",
    srcs = ["edit_distance_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "fingerprint_op_test",
    size = "small",
    srcs = ["fingerprint_op_test.py"],
    deps = [
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "gather_nd_op_test",
    size = "small",
    srcs = ["gather_nd_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "gather_op_test",
    size = "medium",
    srcs = ["gather_op_test.py"],
    shard_count = 3,
    deps = [
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:tensor_spec",
    ] + tf_additional_xla_deps_py(),
)

cuda_py_test(
    name = "huge_slice_op_test",
    size = "medium",
    srcs = ["huge_slice_op_test.py"],
    tags = [
        "no_oss",  # Requires 4GB+ RAM
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "identity_n_op_py_test",
    size = "small",
    srcs = ["identity_n_op_py_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "identity_op_py_test",
    size = "small",
    srcs = ["identity_op_py_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "init_ops_test",
    size = "medium",
    srcs = ["init_ops_test.py"],
    shard_count = 4,
    tags = [
        "noasan",
        "notap",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "inplace_ops_test",
    size = "small",
    srcs = ["inplace_ops_test.py"],
    shard_count = 10,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "large_concat_op_test",
    size = "medium",
    srcs = ["large_concat_op_test.py"],
    tags = [
        "nomsan",
        "notsan",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "manip_ops_test",
    size = "small",
    srcs = ["manip_ops_test.py"],
    tags = ["no_windows_gpu"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:manip_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "matrix_band_part_op_test",
    size = "medium",
    srcs = ["matrix_band_part_op_test.py"],
    shard_count = 20,
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "one_hot_op_test",
    size = "small",
    srcs = ["one_hot_op_test.py"],
    tags = ["no_windows_gpu"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "pad_op_test",
    size = "small",
    srcs = ["pad_op_test.py"],
    tags = ["no_mac"],  # test is times out on mac b/186262388
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "reshape_op_test",
    size = "small",
    srcs = ["reshape_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "reverse_sequence_op_test",
    size = "small",
    srcs = ["reverse_sequence_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "scalar_test",
    size = "small",
    srcs = ["scalar_test.py"],
    # b/140221961: Invalid dims for operations
    xla_enable_strict_auto_jit = False,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:io_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:sparse_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "scatter_nd_ops_test",
    size = "medium",
    srcs = ["scatter_nd_ops_test.py"],
    shard_count = 2,
    tags = ["noasan"],  # http://b/32635055
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "scatter_ops_test",
    size = "medium",  # NOTE: This is not run by default.
    srcs = ["scatter_ops_test.py"],
    shard_count = 2,
    tags = ["optonly"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "shape_ops_test",
    size = "medium",
    srcs = ["shape_ops_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "slice_op_test",
    size = "medium",
    srcs = ["slice_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "spacetobatch_op_test",
    size = "small",
    srcs = ["spacetobatch_op_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "spacetodepth_op_test",
    size = "medium",
    srcs = ["spacetodepth_op_test.py"],
    tags = [
        "no_windows",
        "no_windows_gpu",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "split_op_test",
    size = "medium",
    srcs = ["split_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "stack_op_test",
    size = "medium",
    srcs = ["stack_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "unique_op_test",
    size = "small",
    srcs = ["unique_op_test.py"],
    tags = [
        "no_windows",  # TODO(b/192259045)
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "unstack_op_test",
    size = "small",
    srcs = ["unstack_op_test.py"],
    tags = [
        "no_cuda_asan",  # b/173806679
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "weights_broadcast_test",
    size = "small",
    srcs = ["weights_broadcast_test.py"],
    shard_count = 3,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:weights_broadcast_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "where_op_test",
    size = "medium",
    srcs = ["where_op_test.py"],
    tags = ["no_cuda_asan"],  #TODO(b/212580469)
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)
