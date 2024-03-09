load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_test")
load("//tensorflow:tensorflow.bzl", "if_google")
load(
    "//tensorflow/dtensor:build_defs.bzl",
    "ALL_BACKENDS",
    "GPU_2DEVS_BACKEND",
    "PATHWAYS",
    "PATHWAYS_V3_DONUT_BACKEND",
    "TPU_V3_DONUT_BACKEND",
    "TPU_V4_DONUT_BACKEND",
    "dtensor_test",
)

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

# File used by internal tests.
exports_files([
    "spmd_test.py",
    "collective_test.py",
])

pytype_strict_library(
    name = "test_util",
    testonly = if_google(
        True,
        oss_value = False,  # build_pip_package depends on this target.
    ),
    srcs = [
        "test_backend_name.py",
        "test_backend_util.py",
        "test_util.py",
        "test_util_ops.py",
    ],
    visibility = [
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/dtensor:dtensor-users",
        "//tensorflow/tools/pip_package:__pkg__",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
    deps = [
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:dtensor_device",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/dtensor/python:tpu_util",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:bitwise_ops_gen",
        "//tensorflow/python/ops:clip_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:nn_ops_gen",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/ops:spectral_ops_gen",
        "//tensorflow/python/ops:stateless_random_ops_gen",
        "//tensorflow/python/ops:stateless_random_ops_v2_gen",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "api_test",
    srcs = [
        "api_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_random",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

# TODO(b/301286466): Investigate why python annotation type mismatch is not catptured by the type
# strict BUILD rules.

dtensor_test(
    name = "array_ops_test",
    srcs = ["array_ops_test.py"],
    additional_backends = [],
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

dtensor_test(
    name = "batchparallel_spmd_test",
    srcs = ["batchparallel_spmd_test.py"],
    additional_backends = [TPU_V4_DONUT_BACKEND],
    main = "batchparallel_spmd_test.py",
    shard_count = {
        "cpu": 4,
        "gpu": 4,
        "tpu": 4,
        TPU_V4_DONUT_BACKEND: 8,
    },
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:image_ops_gen",
        "//tensorflow/python/ops:linalg_ops_gen",
        "//tensorflow/python/ops:nn_impl",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "cache_test",
    srcs = ["cache_test.py"],
    main = "cache_test.py",
    tags = [
        "nomultivm",
    ],
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:stateless_random_ops_gen",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

dtensor_test(
    name = "config_test",
    srcs = ["config_test.py"],
    main = "config_test.py",
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/platform:client_testlib",
    ],
)

dtensor_test(
    name = "collective_test",
    srcs = ["collective_test.py"],
    additional_backends = [
        TPU_V3_DONUT_BACKEND,
        GPU_2DEVS_BACKEND,
        PATHWAYS,
        PATHWAYS_V3_DONUT_BACKEND,
    ],
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:dtensor_device",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "collective_combine_all_reduce_test",
    srcs = [":collective_test.py"],
    args = if_google(
        [
            "--vmodule=dtensor_graph_to_mlir_pass=4",
        ],
        [],
    ),
    env = {
        "DTENSOR_ENABLE_COMBINE_ALL_REDUCES_OPTIMIZATION": "1",
    },
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:dtensor_device",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "conv_test",
    srcs = [
        "conv_test.py",
    ],
    additional_backends = [TPU_V3_DONUT_BACKEND],
    # All tests require 8 TPUs.
    disable = ["tpu"],
    shard_count = {
        "cpu": 4,
        "gpu": 4,
        TPU_V3_DONUT_BACKEND: 4,
    },
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "device_test",
    srcs = ["device_test.py"],
    additional_backends = [TPU_V3_DONUT_BACKEND],
    main = "device_test.py",
    shard_count = {
        TPU_V3_DONUT_BACKEND: 32,
    },
    tags = [
        "nomultivm",
    ],
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:dtensor_device",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:collective_ops_gen",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "input_util_test",
    size = "medium",
    srcs = ["input_util_test.py"],
    shard_count = 8,
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:input_util",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:mesh_util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "multi_client_input_util_test",
    timeout = "long",
    srcs = ["multi_client_input_util_test.py"],
    env = {
        "TF2_BEHAVIOR": "1",
    },
    shard_count = 8,
    tags = [
        # ThreadSanitizer does not support starting new threads after multi-threaded fork.
        "notsan",
        "no_oss",  # Fails on OSS.
        "nosan",  # b/195537906
    ],
    deps = [
        ":multi_client_test_util",
        ":test_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/dtensor/python:accelerator_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:input_util",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:mesh_util",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:device_spec",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/lib/io:tf_record",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/ops:parsing_config",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops:parsing_ops_gen",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "layout_test",
    srcs = ["layout_test.py"],
    disable = [
        "gpu",
        "tpu",
    ],
    main = "layout_test.py",
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "layout_propagation_test",
    srcs = ["layout_propagation_test.py"],
    args = if_google(
        [
            "--vmodule=dtensor_mlir_passes=4",
        ],
        [],
    ),
    disable = [
        "gpu",
        "tpu",
    ],
    main = "layout_propagation_test.py",
    shard_count = {
        "cpu": 5,
    },
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:nn_ops_gen",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "mesh_util_test",
    srcs = ["mesh_util_test.py"],
    additional_backends = [
        TPU_V3_DONUT_BACKEND,
        GPU_2DEVS_BACKEND,
    ],
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:mesh_util",
        "//tensorflow/dtensor/python:tpu_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "multi_client_test_util",
    testonly = if_google(
        True,
        oss_value = False,  # build_pip_package depends on this target.
    ),
    srcs = ["multi_client_test_util.py"],
    visibility = [
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/dtensor:dtensor-users",
        "//tensorflow/tools/pip_package:__pkg__",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
    deps = [
        ":test_util",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/flags",
        "@pypi_portpicker//:pkg",
    ],
)

dtensor_test(
    name = "multi_client_test",
    srcs = ["multi_client_test.py"],
    additional_backends = [
        GPU_2DEVS_BACKEND,
        TPU_V3_DONUT_BACKEND,
    ],
    disable = [
        "gpu",  # multi-client gpu is tested via GPU_2DEVS_BACKEND.
        "tpu",  # multi-client tpu is tested via TPU_V3_DONUT_BACKEND.
    ],
    disable_tfrt = [
        "cpu",  # TODO(b/217969210): Re-enable in TFRT CPU.
        GPU_2DEVS_BACKEND,  # TODO(b/230679405): Re-enable in TFRT GPU.
    ],
    tags = [
        "no_windows",
        "nosan",
    ],  # b/195537906
    deps = [
        ":multi_client_test_util",
        ":test_util",
        "//tensorflow/dtensor/python:accelerator_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:mesh_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
    ],
)

dtensor_test(
    name = "multi_client_test_nccl_local",  # Use a suffix for coverage, b/23027507#comment47
    srcs = ["multi_client_test.py"],
    additional_backends = [
        GPU_2DEVS_BACKEND,
    ],
    args = [
        "--num_clients=0",
        "--num_local_devices=2",
        "--model_dim_size=2",
    ],
    disable = ALL_BACKENDS,
    env = {
        "DTENSOR_GPU_USE_NCCL_COMMUNICATION": "1",
        "NCCL_P2P_DISABLE": "1",  # FIXME(b/251183104): p2p detection in cuda 10.1+ is broken.
    },
    tags = [
        "no_windows",
        "nosan",  # b/195537906
    ],
    deps = [
        ":multi_client_test_util",
        ":test_util",
        "//tensorflow/dtensor/python:accelerator_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:mesh_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
    ],
)

dtensor_test(
    name = "multi_client_test_nccl",  # Use a suffix for coverage, b/23027507#comment47
    srcs = ["multi_client_test.py"],
    additional_backends = [
        GPU_2DEVS_BACKEND,
    ],
    args = [
        "--num_clients=2",
        "--num_local_devices=1",
        "--model_dim_size=2",
    ],
    disable = ALL_BACKENDS,
    env = {
        "DTENSOR_GPU_USE_NCCL_COMMUNICATION": "1",
        "NCCL_P2P_DISABLE": "1",  # FIXME(b/251183104): p2p detection in cuda 10.1+ is broken.
    },
    tags = [
        "no_windows",
        "nosan",  # b/195537906
    ],
    deps = [
        ":multi_client_test_util",
        ":test_util",
        "//tensorflow/dtensor/python:accelerator_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:mesh_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
    ],
)

dtensor_test(
    name = "multi_mesh_test",
    srcs = ["multi_mesh_test.py"],
    disable_tfrt = [
        "gpu",
        "tpu",
    ],  # TODO(b/192095157)
    shard_count = {
        "cpu": 5,
        "gpu": 5,
        "tpu": 10,
        TPU_V3_DONUT_BACKEND: 10,
    },
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:accelerator_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "numpy_util_test",
    srcs = ["numpy_util_test.py"],
    main = "numpy_util_test.py",
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

dtensor_test(
    name = "xla_spmd_test",
    srcs = ["spmd_test.py"],
    additional_backends = [
        TPU_V3_DONUT_BACKEND,
    ],
    disable = ALL_BACKENDS,
    env = {
        "DTENSOR_TEST_USE_XLA_SPMD": "1",
    },
    main = "spmd_test.py",
    shard_count = {
        TPU_V3_DONUT_BACKEND: 32,
    },
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:bitwise_ops_gen",
        "//tensorflow/python/ops:io_ops_gen",
        "//tensorflow/python/ops:linalg_ops_gen",
        "//tensorflow/python/ops:list_ops_gen",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:nn_ops_gen",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:resource_variable_ops_gen",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/ops:spectral_ops_gen",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/ops:stateless_random_ops_gen",
        "//tensorflow/python/ops:string_ops_gen",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "multi_device_spmd_test",
    srcs = ["spmd_test.py"],
    additional_backends = [
        TPU_V3_DONUT_BACKEND,
    ],
    env = {
        "DTENSOR_ENABLE_MULTI_DEVICE_EXPANSION": "1",
    },
    main = "spmd_test.py",
    shard_count = {
        "cpu": 25,
        "gpu": 10,
        "tpu": 10,
        TPU_V3_DONUT_BACKEND: 32,
    },
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:bitwise_ops_gen",
        "//tensorflow/python/ops:io_ops_gen",
        "//tensorflow/python/ops:linalg_ops_gen",
        "//tensorflow/python/ops:list_ops_gen",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:nn_ops_gen",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:resource_variable_ops_gen",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/ops:spectral_ops_gen",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/ops:stateless_random_ops_gen",
        "//tensorflow/python/ops:string_ops_gen",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "spmd_test",
    srcs = ["spmd_test.py"],
    additional_backends = [TPU_V3_DONUT_BACKEND],
    main = "spmd_test.py",
    shard_count = {
        "cpu": 25,
        "gpu": 10,
        "tpu": 10,
        TPU_V3_DONUT_BACKEND: 32,
    },
    tags = [
        "no_oss_py38",  # TODO(b/267017937)
    ],
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:bitwise_ops_gen",
        "//tensorflow/python/ops:io_ops_gen",
        "//tensorflow/python/ops:linalg_ops_gen",
        "//tensorflow/python/ops:list_ops_gen",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:nn_ops_gen",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:resource_variable_ops_gen",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/ops:spectral_ops_gen",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/ops:stateless_random_ops_gen",
        "//tensorflow/python/ops:string_ops_gen",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "rng_test",
    size = "medium",
    srcs = ["rng_test.py"],
    additional_backends = [TPU_V3_DONUT_BACKEND],
    # Requires at least 8 TPUs to run the tests.
    disable = ["tpu"],
    disable_tfrt = [
        "gpu",
        TPU_V3_DONUT_BACKEND,
    ],
    main = "rng_test.py",
    shard_count = {
        "cpu": 20,
        "tpu": 10,
        "gpu": 30,
        TPU_V3_DONUT_BACKEND: 20,
    },
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/distribute/cluster_resolver/tpu:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:bitwise_ops_gen",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:stateful_random_ops_gen",
        "//tensorflow/python/ops:stateless_random_ops_v2_gen",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/tpu:device_assignment",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "save_restore_v2_test",
    srcs = ["save_restore_v2_test.py"],
    additional_backends = [
        TPU_V3_DONUT_BACKEND,
        TPU_V4_DONUT_BACKEND,
    ],
    main = "save_restore_v2_test.py",
    shard_count = {
        "cpu": 8,
        "gpu": 8,
        TPU_V3_DONUT_BACKEND: 8,
    },
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "variable_test",
    srcs = ["variable_test.py"],
    disable_tfrt = [
        "tpu",
        "gpu",
    ],  # b/198521331 timeout on TFRT TPU.
    main = "variable_test.py",
    tags = [
        "nomultivm",
    ],
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

dtensor_test(
    name = "mnist_test",
    size = "large",
    srcs = ["mnist_test.py"],
    shard_count = {
        "tpu": 2,
    },
    tags = ["nosan"],  # Non-opt builds has slow XLA compilation.
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:input_util",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "numerics_test",
    srcs = ["numerics_test.py"],
    additional_backends = [TPU_V3_DONUT_BACKEND],
    disable = ALL_BACKENDS,
    enable = [
        "tpu",
    ],
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:accelerator_util",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "sparse_test",
    srcs = ["sparse_test.py"],
    main = "sparse_test.py",
    shard_count = {
        "cpu": 4,
    },
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "tpu_device_assignment_test",
    srcs = ["tpu_device_assignment_test.py"],
    disable = ALL_BACKENDS,
    enable = [
        "tpu",
    ],
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/dtensor/python:tpu_util",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)
