load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")
load("//tensorflow:tensorflow.bzl", "py_test")  # @unused

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "test_util",
    srcs = ["test_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/lite/python:interpreter",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python:tf_optimizer",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/training:saver",
    ],
)

cuda_py_strict_test(
    name = "dct_ops_test",
    srcs = ["dct_ops_test.py"],
    python_version = "PY3",
    shard_count = 16,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops/signal:dct_ops",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "fft_ops_test",
    size = "medium",
    srcs = ["fft_ops_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_cuda11",  # TODO(b/197522782): reenable test after fixing.
        "optonly",
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:spectral_ops_gen",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops/signal:fft_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "mel_ops_test",
    srcs = ["mel_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops/signal:mel_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "mfcc_ops_test",
    srcs = ["mfcc_ops_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops/signal:mfcc_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "reconstruction_ops_test",
    srcs = ["reconstruction_ops_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/ops/signal:reconstruction_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "shape_ops_test",
    srcs = ["shape_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/ops/signal:shape_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "spectral_ops_test",
    size = "large",
    srcs = ["spectral_ops_test.py"],
    python_version = "PY3",
    tags = [
        "no_rocm",
        "nomac",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradient_checker_v2",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops/signal:spectral_ops",
        "//tensorflow/python/ops/signal:window_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "window_ops_test",
    srcs = ["window_ops_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_rocm",
        "no_windows_gpu",
    ],
    deps = [
        ":test_util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops/signal:window_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
