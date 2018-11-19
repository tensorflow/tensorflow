# Description:
#   A Fused Conv Bias Activation operator wrapper.
#   APIs are meant to change over time.

package(
    default_visibility = ["//visibility:private"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/...",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
)
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_custom_op_py_library(
    name = "fused_conv_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    dso = [":python/ops/_fused_conv2d_bias_activation_op.so"],
    kernels = [
        ":fused_conv2d_bias_activation_op_kernels",
        ":fused_conv2d_bias_activation_op_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":fused_conv2d_bias_activation_op",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

tf_kernel_library(
    name = "fused_conv2d_bias_activation_op_kernels",
    srcs = [
        "kernels/fused_conv2d_bias_activation_op.cc",
        "kernels/fused_conv2d_bias_activation_op.h",
        "kernels/fused_conv_ops_gpu.h",
    ],
    prefix = "fused_conv2d_bias_activation_op",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:stream_executor",
        "//tensorflow/core/kernels:bounds_check",
        "//tensorflow/core/kernels:conv_2d_hdrs",
        "//tensorflow/core/kernels:conv_ops_gpu_hdrs",
        "//tensorflow/core/kernels:gpu_util_hdrs",
        "//tensorflow/core/kernels:ops_util_hdrs",
        "//third_party/eigen3",
        "@local_config_cuda//cuda:cudnn_header",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "python/ops/_fused_conv2d_bias_activation_op.so",
    srcs = [
        "kernels/fused_conv2d_bias_activation_op.cc",
        "kernels/fused_conv2d_bias_activation_op.h",
        "kernels/fused_conv_ops_gpu.h",
        "ops/fused_conv2d_bias_activation_op.cc",
    ],
    deps = [
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/kernels:bounds_check_lib",
        "//tensorflow/core/kernels:conv_2d_hdrs",
        "//tensorflow/core/kernels:conv_ops_gpu_hdrs",
        "//tensorflow/core/kernels:gpu_util_hdrs",
        "//tensorflow/core/kernels:ops_util_hdrs",
        "@local_config_cuda//cuda:cudnn_header",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["fused_conv2d_bias_activation_op"],
    deps = ["//tensorflow/core:lib_proto_parsing"],
)

tf_gen_op_wrapper_py(
    name = "fused_conv2d_bias_activation_op",
    deps = [":fused_conv2d_bias_activation_op_op_lib"],
)

py_library(
    name = "fused_conv2d_bias_activation_op_test_base",
    testonly = 1,
    srcs = ["python/ops/fused_conv2d_bias_activation_op_test_base.py"],
    visibility = ["//tensorflow/compiler/tf2xla:internal"],
    deps = [
        ":fused_conv_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "fused_conv2d_bias_activation_op_test",
    size = "large",
    srcs = ["python/ops/fused_conv2d_bias_activation_op_test.py"],
    additional_deps = [
        ":fused_conv2d_bias_activation_op_test_base",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "manual",  # TODO(b/117128481): re-enable after fixing OSS build
        "no_pip",
        "requires-gpu-sm70",
    ],
)

cuda_py_test(
    name = "fused_conv2d_bias_activation_benchmark",
    srcs = ["python/ops/fused_conv2d_bias_activation_benchmark.py"],
    additional_deps = [
        ":fused_conv_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    main = "python/ops/fused_conv2d_bias_activation_benchmark.py",
    tags = [
        "manual",  # TODO(b/117128481): re-enable after fixing OSS build
        "requires-gpu-sm70",
    ],
)
