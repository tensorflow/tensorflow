load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_strict_test")
load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_cuda_cc_test")
load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")
load(
    "//tensorflow/compiler/tests:build_defs.bzl",
    "generate_backend_suites",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    includes = [
        "//tensorflow/compiler/tf2xla:internal",
    ],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/tf2xla:friends",
    ],
    packages = [
        # To pass open source testing in the pip Kokoros.
        "//platforms/xla/tests/neural_nets",
    ],
)

generate_backend_suites()

py_strict_library(
    name = "xla_test",
    testonly = 1,
    srcs = ["xla_test.py"],
    srcs_version = "PY3",
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:session",
        "//tensorflow/python:variables",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/platform:sysconfig",
        "//tensorflow/python/tpu:tpu_py",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "test_utils",
    testonly = 1,
    srcs = [
        "__init__.py",
        "test_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_strict_test(
    name = "xla_test_test",
    size = "small",
    srcs = ["xla_test_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "adadelta_test",
    size = "medium",
    srcs = ["adadelta_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "adagrad_test",
    size = "small",
    srcs = ["adagrad_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "adagrad_da_test",
    size = "small",
    srcs = ["adagrad_da_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_cuda_asan",  # times out
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "adam_test",
    size = "small",
    srcs = ["adam_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_cuda_asan",  # times out
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "add_n_test",
    size = "small",
    srcs = ["add_n_test.py"],
    # TensorList ops are not implemented in the on-demand compilation model yet.
    disabled_backends = ["cpu_ondemand"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:list_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "argminmax_test",
    size = "small",
    srcs = ["argminmax_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_cuda_asan",  # times out
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "binary_ops_test",
    size = "medium",
    srcs = ["binary_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 5,
    tags = [
        "no_oss",  # TODO(b/148108508): Re-enable this test in OSS.
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:bitwise_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "complex_div_test",
    size = "medium",
    srcs = ["complex_div_test.py"],
    enable_mlir_bridge = True,
    enabled_backends = [
        "cpu",
        "gpu",
        "gpu_a100",
    ],
    python_version = "PY3",
    shard_count = 2,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "bucketize_op_test",
    size = "small",
    srcs = ["bucketize_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "categorical_op_test",
    size = "small",
    srcs = ["categorical_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:random_seed",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "cholesky_op_test",
    size = "medium",
    srcs = ["cholesky_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "cond_test",
    size = "small",
    srcs = ["cond_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_switch_case",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/client:session",
        "//tensorflow/python/compiler/xla",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "self_adjoint_eig_op_test",
    size = "medium",
    srcs = ["self_adjoint_eig_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "searchsorted_op_test",
    size = "small",
    timeout = "moderate",
    srcs = ["searchsorted_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "svd_op_test",
    size = "medium",
    srcs = ["svd_op_test.py"],
    disabled_backends = [
        # TODO(b/129396575): Fails on CPU.
        "cpu",
        "cpu_ondemand",
    ],
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly"
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:linalg_ops_gen",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "matrix_inverse_op_test",
    size = "small",
    timeout = "moderate",
    srcs = ["matrix_inverse_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "matrix_solve_op_test",
    size = "small",
    timeout = "moderate",
    srcs = ["matrix_solve_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "matrix_triangular_solve_op_test",
    size = "small",
    timeout = "moderate",
    srcs = ["matrix_triangular_solve_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "clustering_test",
    size = "small",
    srcs = ["clustering_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "concat_ops_test",
    size = "medium",
    srcs = ["concat_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "conv2d_test",
    size = "medium",
    srcs = ["conv2d_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":test_utils",
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "tensor_float_32_test",
    size = "medium",
    srcs = ["tensor_float_32_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    use_xla_device = False,  # Uses tf.function(jit_compile=True)
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:ops",
    ],
)

tf_xla_py_strict_test(
    name = "conv3d_test",
    size = "medium",
    srcs = ["conv3d_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "depthwise_conv_op_test",
    size = "medium",
    srcs = ["depthwise_conv_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "dynamic_slice_ops_test",
    size = "small",
    srcs = ["dynamic_slice_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "einsum_op_test",
    size = "medium",
    srcs = ["einsum_op_test.py"],
    enable_mlir_bridge = True,
    enabled_backends = [
        "cpu",
        "gpu",
        "gpu_a100",
    ],
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "reshape_op_test",
    size = "small",
    srcs = ["reshape_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "dynamic_stitch_test",
    size = "small",
    srcs = ["dynamic_stitch_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "extract_image_patches_op_test",
    size = "small",
    srcs = ["extract_image_patches_op_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "eager_test",
    size = "medium",
    srcs = ["eager_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:cond",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops_gen",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/training:adam",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "fifo_queue_test",
    size = "medium",
    srcs = ["fifo_queue_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "fft_test",
    size = "medium",
    srcs = ["fft_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    shard_count = 12,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "no_rocm",
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops/signal",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "slice_ops_test",
    size = "medium",
    srcs = ["slice_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_cuda_asan",  # times out
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_shape",
    ],
)

tf_xla_py_strict_test(
    name = "ftrl_test",
    size = "medium",
    srcs = ["ftrl_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    shard_count = 16,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "ftrl_ops_test",
    size = "medium",
    srcs = ["ftrl_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "function_test",
    size = "small",
    srcs = ["function_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:function",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "image_ops_test",
    size = "small",
    timeout = "long",
    srcs = ["image_ops_test.py"],
    enable_mlir_bridge = False,
    enabled_backends = [
        "cpu",
        "gpu",
        "gpu_a100",
    ],
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "no_oss",  # TODO(b/282033702): Re-enable this test in OSS.
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:image_ops_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "listdiff_op_test",
    size = "small",
    srcs = ["listdiff_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_cuda_asan",  # times out
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "lrn_ops_test",
    size = "medium",
    srcs = ["lrn_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "manip_ops_test",
    size = "small",
    srcs = ["manip_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:manip_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "matrix_band_part_test",
    size = "medium",
    timeout = "long",
    srcs = ["matrix_band_part_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "matrix_diag_ops_test",
    size = "medium",
    timeout = "long",
    srcs = ["matrix_diag_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "momentum_test",
    size = "small",
    srcs = ["momentum_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "nary_ops_test",
    size = "small",
    srcs = ["nary_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "nullary_ops_test",
    size = "small",
    srcs = ["nullary_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:constant_op",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "pooling_ops_test",
    size = "medium",
    srcs = ["pooling_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 20,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "pooling_ops_3d_test",
    size = "medium",
    srcs = ["pooling_ops_3d_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 20,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "proximal_adagrad_test",
    size = "medium",
    srcs = ["proximal_adagrad_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "proximal_gradient_descent_test",
    size = "medium",
    srcs = ["proximal_gradient_descent_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "qr_op_test",
    size = "medium",
    srcs = ["qr_op_test.py"],
    disabled_backends = [
        # Test is very slow on CPU.
        "cpu",
        "cpu_ondemand",
    ],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly"
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "unstack_test",
    size = "medium",
    srcs = ["unstack_test.py"],
    disabled_backends = [
        # TODO(b/149750262): timeout on CPU.
        "cpu",
        "cpu_ondemand",
    ],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "random_ops_test",
    size = "medium",
    srcs = ["random_ops_test.py"],
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops/distributions:special_math",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "reduce_ops_test",
    size = "medium",
    srcs = ["reduce_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "reduce_window_test",
    size = "small",
    srcs = ["reduce_window_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:function",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "reverse_ops_test",
    size = "medium",
    srcs = ["reverse_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "reverse_sequence_op_test",
    size = "medium",
    srcs = ["reverse_sequence_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

# copybara:uncomment_begin(google-only)
# tf_xla_py_strict_test(
#     name = "reverse_sequence_op_args_test",
#     size = "medium",
#     srcs = ["reverse_sequence_op_args_test.py"],
#     enable_mlir_bridge = False,
#     main = "reverse_sequence_op_args_test.py",
#     python_version = "PY3",
#     tags = [
#         "no_pip",
#         "optonly",
#     ],
#     deps = [
#         ":xla_test",
#         "//tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
#         "//tensorflow/python:array_ops",
#         "//tensorflow/python/compat:v2_compat",
#         "//tensorflow/python/eager:def_function",
#         "//tensorflow/python/framework:errors",
#         "//tensorflow/python/platform:client_testlib",
#     ],
# )
# copybara:uncomment_end

tf_xla_py_strict_test(
    name = "rmsprop_test",
    size = "small",
    srcs = ["rmsprop_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "scan_ops_test",
    size = "medium",
    srcs = ["scan_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 6,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "segment_reduction_ops_test",
    size = "medium",
    srcs = ["segment_reduction_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/client",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "spacetobatch_op_test",
    size = "medium",
    srcs = ["spacetobatch_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 3,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "sparse_to_dense_op_test",
    size = "medium",
    srcs = ["sparse_to_dense_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "stack_ops_test",
    size = "small",
    srcs = ["stack_ops_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "config-cuda-only",
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    use_xla_device = False,
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:data_flow_ops_gen",
        "//tensorflow/python/compiler/xla",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "stateful_random_ops_test",
    size = "medium",
    srcs = ["stateful_random_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "no_oss",  # b/170479349
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",  # b/170479349
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:stateful_random_ops",
        "//tensorflow/python:stateful_random_ops_gen",
        "//tensorflow/python:variables",
        "//tensorflow/python/client",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/kernel_tests/random:util",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:flags",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "stateless_random_ops_test",
    size = "medium",
    srcs = ["stateless_random_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:stateless_random_ops_v2_gen",
        "//tensorflow/python:variables",
        "//tensorflow/python/client",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/kernel_tests/random:util",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "tensor_array_ops_test",
    size = "medium",
    srcs = ["tensor_array_ops_test.py"],
    # TensorArray ops are not implemented in the on-demand compilation model yet.
    disabled_backends = ["cpu_ondemand"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "config-cuda-only",
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "v1only",
    ],
    use_xla_device = False,
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:data_flow_ops_gen",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/compiler/xla",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "tensor_list_ops_test",
    size = "small",
    srcs = ["tensor_list_ops_test.py"],
    # TensorList ops are not implemented in the on-demand compilation model yet.
    disabled_backends = ["cpu_ondemand"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:list_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "ternary_ops_test",
    size = "medium",
    srcs = ["ternary_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 16,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "unary_ops_test",
    size = "medium",
    srcs = ["unary_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 50,
    tags = [
        "no_cuda_asan",  # times out
        "no_pip"  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:bitwise_ops",
        "//tensorflow/python:functional_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

tf_xla_py_strict_test(
    name = "fused_batchnorm_test",
    size = "medium",
    srcs = ["fused_batchnorm_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":test_utils",
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "variable_ops_test",
    size = "small",
    srcs = ["variable_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:state_ops_gen",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "while_test",
    size = "small",
    srcs = ["while_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "case_test",
    size = "small",
    srcs = ["case_test.py"],
    disabled_backends = ["cpu_ondemand"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    use_xla_device = False,  # Uses tf.function(jit_compile=True)
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_switch_case",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "cast_ops_test",
    size = "small",
    srcs = ["cast_ops_test.py"],
    disabled_backends = ["cpu_ondemand"],
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    use_xla_device = False,  # Uses tf.function(jit_compile=True)
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "gather_test",
    size = "medium",
    srcs = ["gather_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:flags",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "gather_nd_op_test",
    size = "medium",
    srcs = ["gather_nd_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "scatter_nd_op_test",
    size = "medium",
    srcs = ["scatter_nd_op_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "sort_ops_test",
    size = "medium",
    srcs = ["sort_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 50,
    # Times out in fastbuild mode.
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "data_format_ops_test",
    size = "small",
    srcs = ["data_format_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:nn_ops",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "xla_device_test",
    size = "small",
    srcs = ["xla_device_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "xla_device_gpu_test",
    size = "small",
    srcs = ["xla_device_gpu_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
    ],
)

cuda_py_strict_test(
    name = "jit_test",
    size = "medium",
    srcs = ["jit_test.py"],
    #shard_count = 5,
    tags = [
        "no_cuda_asan",  # Times out.
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        ":test_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/client:session",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "async_comp_test",
    size = "medium",
    srcs = ["async_comp_test.py"],
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
    ],
)

cuda_py_strict_test(
    name = "dense_layer_test",
    size = "medium",
    srcs = ["dense_layer_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        ":test_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:layers",
        "//tensorflow/python:variables",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

cc_library(
    name = "randomized_tests_library",
    testonly = 1,
    srcs = ["randomized_tests.cc"],
    deps = [
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/jit:common",
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow_opensource",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "randomized_tests",
    size = "medium",
    args = ["--tf_xla_test_use_mlir=false"],
    shard_count = 20,
    # This test is randomized, so only run it if explicitly requested.
    tags = [
        "manual",
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ] + tf_cuda_tests_tags(),
    deps = [":randomized_tests_library"],
)

tf_cuda_cc_test(
    name = "randomized_tests_mlir",
    size = "medium",
    args = ["--tf_xla_test_use_mlir=true"],
    shard_count = 20,
    # This test is randomized, so only run it if explicitly requested.
    tags = [
        "manual",
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ] + tf_cuda_tests_tags(),
    deps = [":randomized_tests_library"],
)

# Create a deterministic version of randomized_tests_mlir with fixed seed.
# This can be used in presubmit checks as it is no longer randomized.
tf_cuda_cc_test(
    name = "randomized_tests_mlir_seeded",
    size = "medium",
    args = [
        "--tf_xla_random_seed=200839030",
        "--tf_xla_test_use_mlir=true",
        "--tf_xla_test_device=GPU:0",
    ],
    shard_count = 20,
    tags = [
        "config-cuda-only",
        "no_cuda_asan",  # TODO(b/201651800)
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "requires-gpu-nvidia",
    ] + tf_cuda_tests_tags(),
    deps = [":randomized_tests_library"],
)

# Create a deterministic version of randomized_tests with fixed seed.
# This can be used in presubmit checks as it is no longer randomized.
tf_cuda_cc_test(
    name = "randomized_tests_seeded",
    size = "medium",
    args = [
        "--tf_xla_random_seed=200839030",
        "--tf_xla_test_use_mlir=false",
        "--tf_xla_test_device=GPU:0",
    ],
    shard_count = 20,
    tags = [
        "config-cuda-only",
        "no_cuda_asan",  # TODO(b/201651800)
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "no_rocm",  # ROCmSoftwarePlatform #958
        "requires-gpu-nvidia",
    ] + tf_cuda_tests_tags(),
    deps = [":randomized_tests_library"],
)

tf_cuda_cc_test(
    name = "unary_ops_composition_test",
    srcs = ["unary_ops_composition_test.cc"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ] + tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/jit:xla_kernel_creator",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "@com_google_absl//absl/synchronization",
    ],
)

py_strict_library(
    name = "lstm",
    testonly = 1,
    srcs = ["lstm.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "@six_archive//:six",
    ],
)

cuda_py_strict_test(
    name = "lstm_test",
    srcs = ["lstm_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        ":lstm",
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

# An example of ahead-of-time compilation using tfcompile.  The
# lstm_layer_inference.pbtxt file was generated by running lstm_test
# --dump_graph_dir, and the config file was written by hand.
#
# Run the following to build a minimal benchmark of the computation on Android:
# $ bazel build -c opt --cxxopt='-std=c++11' --linkopt='-lm' \
#   --cpu=armeabi-v7a \
#   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
#   --crosstool_top=//external:android/crosstool \
#   //tensorflow/compiler/tests:lstm_layer_inference_benchmark

#
# Currently the resulting binary size is ~190KB
tf_library(
    name = "lstm_layer_inference",
    testonly = 1,
    config = "lstm_layer_inference.config.pbtxt",
    cpp_class = "LSTMLayerInference",
    graph = "lstm_layer_inference.pbtxt",
    tags = ["manual"],
    tfcompile_flags = ["--xla_cpu_multi_thread_eigen=false"],
)

tf_xla_py_strict_test(
    name = "fake_quant_ops_test",
    size = "medium",
    srcs = ["fake_quant_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "placeholder_test",
    size = "small",
    srcs = ["placeholder_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
    ],
)

tf_xla_py_strict_test(
    name = "quantized_ops_test",
    size = "medium",
    srcs = ["quantized_ops_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:bitwise_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "xla_ops_test",
    size = "medium",
    srcs = ["xla_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "no_rocm"
    ],
    deps = [
        ":xla_test",
        "//tensorflow/compiler/tf2xla/ops:gen_xla_ops",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:errors",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "xla_custom_call_ops_test",
    size = "small",
    srcs = ["xla_custom_call_ops_test.py"],
    disabled_backends = [
        "gpu",
        "gpu_a100",
    ],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    use_xla_device = False,  # Uses tf.function(jit_compile=True)
    deps = [
        ":xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "runtime_shape_check_test",
    size = "small",
    srcs = ["runtime_shape_check_test.py"],
    disabled_backends = [
        "cpu",
        "cpu_ondemand",
    ],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",
        "notap",
    ],
    use_xla_device = False,
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "conv_node_name_test",
    size = "medium",
    srcs = ["conv_node_name_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    shard_count = 5,
    tags = [
        "no_oss",  # TODO(b/148108508): Re-enable this test in OSS.
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "tridiagonal_solve_ops_test",
    size = "medium",
    srcs = ["tridiagonal_solve_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops/linalg:linalg_impl",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "tridiagonal_matmul_ops_test",
    size = "medium",
    srcs = ["tridiagonal_matmul_ops_test.py"],
    enable_mlir_bridge = True,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops/linalg:linalg_impl",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "special_math_test",
    size = "medium",
    srcs = ["special_math_test.py"],
    enable_mlir_bridge = True,
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:random_ops_gen",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "repeat_op_test",
    size = "medium",
    srcs = ["repeat_op_test.py"],
    enable_mlir_bridge = True,
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "image_ops_jit_compile_test",
    size = "medium",
    srcs = ["image_ops_jit_compile_test.py"],
    disabled_backends = [
        "cpu_ondemand",
    ],
    enable_mlir_bridge = False,
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly"
    ],
    use_xla_device = False,
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "ensure_shape_op_test",
    size = "medium",
    srcs = ["ensure_shape_op_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "where_op_test",
    size = "small",
    srcs = ["where_op_test.py"],
    enable_mlir_bridge = False,
    enabled_backends = [
        "cpu",
        "gpu",
        "gpu_a100",
    ],
    tags = [
        "no_pip",
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/tpu:tpu_py",
    ],
)

tf_xla_py_strict_test(
    name = "where_op_tpu_test",
    size = "small",
    srcs = ["where_op_test.py"],
    args = [
        "--tpu_use_tfrt=true",
        # TODO(b/274633087): Set tf_use_pjrt=true after fixing bug.
    ],
    disabled_backends = [
        "cpu",
        "cpu_ondemand",
        "gpu",
        "gpu_a100",
    ],
    enable_mlir_bridge = False,
    main = "where_op_test.py",
    tags = [
        "no_pip",
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/tpu:tpu_py",
    ],
)

tf_xla_py_strict_test(
    name = "risc_ops_test",
    size = "small",
    srcs = ["risc_ops_test.py"],
    enabled_backends = ["cpu"],
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:is_mlir_bridge_test_true",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops/risc:risc_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "const_arg_test",
    size = "small",
    srcs = ["const_arg_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
    ],
)

cuda_py_strict_test(
    name = "const_test",
    size = "small",
    srcs = ["const_test.py"],
    python_version = "PY3",
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tpu_py_strict_test(
    name = "giant_const_op_test",
    srcs = [
        "giant_const_op_test.py",
    ],
    disable_experimental = True,
    # TODO(b/188995810): Add an optimization in MLIR importer to not
    # materialize giant splat constants.
    disable_mlir_bridge = True,
    python_version = "PY3",
    tags = ["no_oss"],
    deps = [
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/tpu:tpu_lib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "sharding_util_ops_test",
    srcs = ["sharding_util_ops_test.py"],
    disabled_backends = [
        "cpu",
        "cpu_ondemand",
        "gpu",
        "gpu_a100",
    ],
    enable_mlir_bridge = False,
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "notap",
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:tpu_ops_gen",
        "//tensorflow/python:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "approx_topk_test",
    srcs = ["approx_topk_test.py"],
    disable_experimental = False,
    disable_mlir_bridge = False,
    tags = ["no_oss"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "xla_call_module_test",
    size = "small",
    srcs = ["xla_call_module_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    use_xla_device = False,  # Uses tf.function(jit_compile=True)
    deps = [
        ":xla_test",
        "//tensorflow/compiler/mlir/stablehlo",
        "//tensorflow/compiler/tf2xla/ops:gen_xla_ops",
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "bincount_op_test",
    size = "small",
    srcs = ["bincount_op_test.py"],
    enable_mlir_bridge = False,
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:errors",
    ],
)

tf_xla_py_strict_test(
    name = "unique_ops_test",
    size = "small",
    srcs = ["unique_ops_test.py"],
    enable_mlir_bridge = False,
    enabled_backends = [
        "cpu",
        "gpu",
        "gpu_a100",
    ],
    python_version = "PY3",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
    ],
)
