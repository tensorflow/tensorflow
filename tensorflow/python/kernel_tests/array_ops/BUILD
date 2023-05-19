# Tests of TensorFlow array ops kernels written using the Python API.

load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")
load("//tensorflow/core/platform:build_config_root.bzl", "tf_additional_xla_deps_py")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "array_ops_test",
    size = "medium",
    srcs = ["array_ops_test.py"],
    shard_count = 10,
    tags = [
        "optonly",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:test_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "batch_gather_op_test",
    srcs = ["batch_gather_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "batch_scatter_ops_test",
    srcs = ["batch_scatter_ops_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:indexed_slices",
    ],
)

cuda_py_strict_test(
    name = "batchtospace_op_test",
    size = "small",
    srcs = ["batchtospace_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "bcast_ops_test",
    size = "small",
    srcs = ["bcast_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
    ],
)

cuda_py_strict_test(
    name = "bitcast_op_test",
    size = "small",
    srcs = ["bitcast_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "broadcast_to_ops_test",
    size = "medium",
    srcs = ["broadcast_to_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "cast_op_test",
    size = "small",
    srcs = ["cast_op_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops_gen",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "concat_op_test",
    size = "medium",
    srcs = ["concat_op_test.py"],
    shard_count = 4,
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
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "constant_op_eager_test",
    size = "small",
    srcs = ["constant_op_eager_test.py"],
    tags = [
        "no_windows",  # TODO(b/207048720): re-enable
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/util:compat",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "constant_op_test",
    size = "small",
    srcs = ["constant_op_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/util:compat",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "denormal_test",
    size = "small",
    srcs = ["denormal_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
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
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/client",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "diag_op_test",
    size = "medium",
    srcs = ["diag_op_test.py"],
    shard_count = 6,
    tags = ["no_windows_gpu"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "edit_distance_op_test",
    size = "small",
    srcs = ["edit_distance_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:sparse_tensor",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "fingerprint_op_test",
    size = "small",
    srcs = ["fingerprint_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "gather_nd_op_test",
    size = "small",
    srcs = ["gather_nd_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:ref_variable",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "gather_op_test",
    size = "medium",
    srcs = ["gather_op_test.py"],
    shard_count = 3,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python:gradients",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ref_variable",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)

cuda_py_strict_test(
    name = "huge_slice_op_test",
    size = "medium",
    srcs = ["huge_slice_op_test.py"],
    tags = [
        "no_oss",  # Requires 4GB+ RAM
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python/framework:constant_op",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "identity_n_op_py_test",
    size = "small",
    srcs = ["identity_n_op_py_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/framework:constant_op",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "identity_op_py_test",
    size = "small",
    srcs = ["identity_op_py_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:sparse_tensor",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
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
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "inplace_ops_test",
    size = "small",
    srcs = ["inplace_ops_test.py"],
    shard_count = 10,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "large_concat_op_test",
    size = "medium",
    srcs = ["large_concat_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_strict_test(
    name = "manip_ops_test",
    size = "small",
    srcs = ["manip_ops_test.py"],
    tags = ["no_windows_gpu"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python:manip_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
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
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python:variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:benchmark",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
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
        "//tensorflow/python/framework:constant_op",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "pad_op_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["pad_op_test.py"],
    tags = ["no_mac"],  # test is times out on mac b/186262388
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "reshape_op_test",
    size = "small",
    srcs = ["reshape_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "reverse_sequence_op_test",
    size = "small",
    srcs = ["reverse_sequence_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
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
        "//tensorflow/python:random_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "scatter_nd_ops_test",
    size = "medium",
    srcs = ["scatter_nd_ops_test.py"],
    shard_count = 2,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "scatter_ops_test",
    size = "medium",  # NOTE: This is not run by default.
    srcs = ["scatter_ops_test.py"],
    shard_count = 2,
    tags = ["optonly"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:ref_variable",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
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
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "slice_op_test",
    size = "medium",
    srcs = ["slice_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
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
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "spacetodepth_op_test",
    size = "medium",
    srcs = ["spacetodepth_op_test.py"],
    tags = [
        "no_windows",
        "no_windows_gpu",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "split_op_test",
    size = "medium",
    srcs = ["split_op_test.py"],
    shard_count = 4,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "stack_op_test",
    size = "medium",
    srcs = ["stack_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "unique_op_test",
    size = "small",
    srcs = ["unique_op_test.py"],
    tags = [
        "no_windows",  # TODO(b/192259045)
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
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
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "weights_broadcast_test",
    size = "small",
    srcs = ["weights_broadcast_test.py"],
    shard_count = 3,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:weights_broadcast_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "where_op_test",
    size = "medium",
    srcs = ["where_op_test.py"],
    tags = ["no_cuda_asan"],  #TODO(b/212580469)
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:benchmark",
        "//third_party/py/numpy",
    ],
)
