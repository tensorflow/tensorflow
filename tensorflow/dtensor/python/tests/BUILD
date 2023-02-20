load("//tensorflow:pytype.default.bzl", "pytype_library")
load("//tensorflow:tensorflow.bzl", "if_google")
load(
    "//tensorflow/dtensor:build_defs.bzl",
    "ALL_BACKENDS",
    "GPU_2DEVS_BACKEND",
    "PATHWAYS",
    "TPU_V3_DONUT_BACKEND",
    "dtensor_test",
)

# File used by internal tests.
exports_files([
    "spmd_test.py",
    "collective_test.py",
])

pytype_library(
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
    ],
    deps = [
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/dtensor/python:tpu_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:bitwise_ops_gen",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python:spectral_ops_gen",
        "//tensorflow/python:stateless_random_ops_gen",
        "//tensorflow/python:stateless_random_ops_v2_gen",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "config_test",
    srcs = ["config_test.py"],
    main = "config_test.py",
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:for_generated_wrappers",
    ],
)

dtensor_test(
    name = "collective_test",
    srcs = ["collective_test.py"],
    additional_backends = [
        TPU_V3_DONUT_BACKEND,
        GPU_2DEVS_BACKEND,
        PATHWAYS,
    ],
    disable = [PATHWAYS],
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:dtensor_device",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
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
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:dtensor_device",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
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
        ":test_util",
        "//tensorflow/dtensor/python:accelerator_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:mesh_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
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
        "--num_devices=2",
        "--model_dim_size=2",
    ],
    disable = ALL_BACKENDS,
    env = {
        "DTENSOR_GPU_USE_NCCL_COMMUNICATION": "1",
        "NCCL_P2P_DISABLE": "1",  # FIXME(b/251183104): p2p detection in cuda 10.1+ is broken.
        "DTENSOR_DO_NOT_FUSE_REDUCE_SCATTER": "1",  # FIXME(b/266609048): timeout if fused.
    },
    tags = [
        "no_rocm",
        "no_windows",
        "nosan",  # b/195537906
    ],
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:accelerator_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:mesh_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
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
        "--num_devices=1",
        "--model_dim_size=2",
    ],
    disable = ALL_BACKENDS,
    env = {
        "DTENSOR_GPU_USE_NCCL_COMMUNICATION": "1",
        "NCCL_P2P_DISABLE": "1",  # FIXME(b/251183104): p2p detection in cuda 10.1+ is broken.
        "DTENSOR_DO_NOT_FUSE_REDUCE_SCATTER": "1",  # FIXME(b/266609048): timeout if fused.
    },
    tags = [
        "no_windows",
        "nosan",  # b/195537906
    ],
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:accelerator_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:mesh_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
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
        # TPU_V3_DONUT_BACKEND,  # FIXME(b/269177045): Failing due to RelayoutOp support.
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:bitwise_ops_gen",
        "//tensorflow/python:io_ops_gen",
        "//tensorflow/python:linalg_ops_gen",
        "//tensorflow/python:list_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops_gen",
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:stateless_random_ops_gen",
        "//tensorflow/python:string_ops_gen",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util",
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
	"no_rocm",
    ],
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:numpy_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:bitwise_ops_gen",
        "//tensorflow/python:io_ops_gen",
        "//tensorflow/python:linalg_ops_gen",
        "//tensorflow/python:list_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops_gen",
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:stateless_random_ops_gen",
        "//tensorflow/python:string_ops_gen",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
