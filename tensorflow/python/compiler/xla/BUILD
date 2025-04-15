load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_library(
    name = "compiler_py",
    srcs = [
        "__init__.py",
        "jit.py",
    ],
    deps = [
        ":xla",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "jit_test",
    size = "small",
    srcs = ["tests/jit_test.py"],
    tags = [
        "no_windows",  # TODO(b/171385770)
    ],
    xla_enabled = True,
    deps = [
        ":compiler_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:op_def_registry",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:gradients",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "xla",
    srcs = ["xla.py"],
    deps = [
        "//tensorflow/compiler/jit:xla_ops_py",
        "//tensorflow/compiler/jit/ops:xla_ops_grad",
        "//tensorflow/core:protos_all_py",
        # Do not remove: required to run xla ops on Cloud.
        "//tensorflow/compiler/tf2xla/python:xla",  # build_cleaner: keep
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/distribute:summary_op_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
    ],
)

cuda_py_strict_test(
    name = "xla_test",
    srcs = ["tests/xla_test.py"],
    tags = [
        "no_mac",
        "no_windows",
    ],
    xla_enabled = True,
    deps = [
        ":xla",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:control_flow_util",
        "//tensorflow/python/ops:logging_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/summary:summary_py",
        "//tensorflow/python/tpu:tpu_feed",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "jit_compile_test",
    srcs = ["tests/jit_compile_test.py"],
    tags = [
        "no_mac",
        "no_windows",
    ],
    xla_enabled = True,
    deps = [
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "pjrt_compile_test",
    srcs = ["tests/pjrt_compile_test.py"],
    env = {
        "TF_XLA_FLAGS": "--tf_xla_use_device_api --tf_xla_enable_xla_devices --tf_xla_enable_device_api_for_gpu",
    },
    tags = [
        "config-cuda-only",
        "gpu",
        "no_oss",
        "requires-gpu-nvidia",
        "xla",
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
    ],
)

tf_py_strict_test(
    name = "pjrt_autoclustering_test",
    srcs = ["tests/pjrt_autoclustering_test.py"],
    env = {
        "TF_XLA_FLAGS": "--tf_xla_use_device_api --tf_xla_enable_xla_devices --tf_xla_enable_device_api_for_gpu --tf_xla_auto_jit=2 --tf_xla_enable_lazy_compilation=false --tf_xla_min_cluster_size=0",
    },
    tags = [
        "config-cuda-only",
        "gpu",
        "no_oss",
        "requires-gpu-nvidia",
        "xla",
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:math_ops",
    ],
)

tf_py_strict_test(
    name = "pjrt_compile_virtual_device_test",
    srcs = ["pjrt_compile_virtual_device_test.py"],
    env = {
        "TF_XLA_FLAGS": "--tf_xla_use_device_api --tf_xla_enable_xla_devices --tf_xla_enable_device_api_for_gpu",
    },
    tags = [
        "config-cuda-only",
        "gpu",
        "no_oss",
        "requires-gpu-nvidia",
        "xla",
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
    ],
)
