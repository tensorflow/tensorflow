# Description:
# Python support for TensorFlow.

package(default_visibility = [
    "//engedu/ml/tf_from_scratch:__pkg__",
    "//tensorflow:internal",
    "//tensorflow_models:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "if_not_windows")
load("//tensorflow:tensorflow.bzl", "tf_cuda_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "py_tests")
load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_tests")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library_py")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_additional_lib_deps")
load("//tensorflow/core:platform/default/build_config_root.bzl", "tf_additional_plugin_deps")
load("//tensorflow/python:build_defs.bzl", "tf_gen_op_wrapper_private_py")
load("//tensorflow/core:platform/default/build_config_root.bzl", "tf_additional_verbs_deps")

py_library(
    name = "python",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/compiler/aot/tests:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/contrib/learn:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/contrib/learn/python/learn/datasets:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python/debug:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python/tools:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/tensorboard/scripts:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/tools/quantization:__pkg__",  # TODO(b/34059704): remove when fixed
    ],
    deps = [
        ":array_ops",
        ":check_ops",
        ":client",
        ":client_testlib",
        ":confusion_matrix",
        ":control_flow_ops",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":functional_ops",
        ":gradient_checker",
        ":histogram_ops",
        ":image_ops",
        ":io_ops",
        ":lib",
        ":math_ops",
        ":nn",
        ":platform",
        ":script_ops",
        ":session_ops",
        ":sets",
        ":sparse_ops",
        ":spectral_ops",
        ":standard_ops",
        ":state_ops",
        ":string_ops",
        ":summary",
        ":metrics",
        ":layers",
        ":tensor_array_ops",
        ":training",
        ":ops",
        ":saver_test_utils",
        ":test_ops",  # TODO: Break testing code out into separate rule.
        ":util",
        ":weights_broadcast_ops",
        "//third_party/py/numpy",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/saved_model",
    ] + if_not_windows([
        "//tensorflow/contrib:contrib_py",
    ]),
)

py_library(
    name = "platform",
    srcs = glob(
        ["platform/*.py"],
        exclude = [
            "**/*test.py",
            "**/benchmark.py",  # In platform_benchmark.
        ],
    ),
    srcs_version = "PY2AND3",
    deps = [
        ":lib",
        ":pywrap_tensorflow",
        ":util",
        "//tensorflow/core:protos_all_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "platform_benchmark",
    srcs = ["platform/benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client",
        ":platform",
        "@six_archive//:six",
    ],
)

py_library(
    name = "platform_test",
    srcs = ["platform/googletest.py"],
    srcs_version = "PY2AND3",
    deps = [":platform_benchmark"],
)

tf_py_test(
    name = "resource_loader_test",
    size = "small",
    srcs = ["platform/resource_loader_test.py"],
    additional_deps = [
        ":platform",
        ":platform_test",
    ],
)

tf_py_test(
    name = "flags_test",
    size = "small",
    srcs = ["platform/flags_test.py"],
    additional_deps = [":platform"],
)

tf_py_test(
    name = "app_test",
    size = "small",
    srcs = ["platform/app_test.py"],
    additional_deps = [":platform"],
    tags = [
        "manual",
        "notap",
    ],
)

cc_library(
    name = "numpy_lib",
    srcs = ["lib/core/numpy.cc"],
    hdrs = ["lib/core/numpy.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/py/numpy:headers",
        "//util/python:python_headers",
    ],
)

cc_library(
    name = "ndarray_tensor_bridge",
    srcs = ["lib/core/ndarray_tensor_bridge.cc"],
    hdrs = ["lib/core/ndarray_tensor_bridge.h"],
    deps = [
        ":numpy_lib",
        "//tensorflow/c:c_api",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "kernel_registry",
    srcs = ["util/kernel_registry.cc"],
    hdrs = ["util/kernel_registry.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "py_func_lib",
    srcs = ["lib/core/py_func.cc"],
    hdrs = ["lib/core/py_func.h"],
    deps = [
        ":ndarray_tensor_bridge",
        ":numpy_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:script_ops_op_lib",
        "//third_party/py/numpy:headers",
        "//util/python:python_headers",
    ],
)

cc_library(
    name = "py_record_reader_lib",
    srcs = ["lib/io/py_record_reader.cc"],
    hdrs = ["lib/io/py_record_reader.h"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "py_record_writer_lib",
    srcs = ["lib/io/py_record_writer.cc"],
    hdrs = ["lib/io/py_record_writer.h"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:lib",
    ],
)

cc_binary(
    name = "framework/test_file_system.so",
    srcs = ["framework/test_file_system.cc"],
    copts = ["-Wno-sign-compare"],
    linkopts = select({
        "//conditions:default": [
            "-lm",
        ],
        "//tensorflow:darwin": [],
    }),
    linkshared = 1,
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "@protobuf//:protobuf_headers",
    ],
)

py_test(
    name = "file_system_test",
    size = "small",
    srcs = ["framework/file_system_test.py"],
    data = [":framework/test_file_system.so"],
    main = "framework/file_system_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":data_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":io_ops",
        ":platform",
        ":util",
    ],
)

py_test(
    name = "decorator_utils_test",
    srcs = ["util/decorator_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":platform",
        ":util",
    ],
)

py_test(
    name = "deprecation_test",
    srcs = ["util/deprecation_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":platform",
        ":util",
    ],
)

py_test(
    name = "keyword_args_test",
    srcs = ["util/keyword_args_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":util",
    ],
)

cc_library(
    name = "python_op_gen",
    srcs = ["framework/python_op_gen.cc"],
    hdrs = ["framework/python_op_gen.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_cc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "python_op_gen_main",
    srcs = ["framework/python_op_gen_main.cc"],
    visibility = ["//visibility:public"],
    deps = [":python_op_gen"],
)

py_library(
    name = "framework_for_generated_wrappers",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":constant_op",
        ":device",
        ":dtypes",
        ":framework_ops",
        ":function",
        ":op_def_library",
        ":op_def_registry",
        ":registry",
        ":tensor_shape",
        ":versions",
    ],
)

# What is needed for tf_gen_op_wrapper_py. This is the same as
# "framework_for_generated_wrappers" minus the "function" dep. This is to avoid
# circular dependencies, as "function" uses generated op wrappers.
py_library(
    name = "framework_for_generated_wrappers_v2",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":constant_op",
        ":device",
        ":dtypes",
        ":framework_ops",
        ":op_def_library",
        ":op_def_registry",
        ":registry",
        ":tensor_shape",
        ":versions",
    ],
)

py_library(
    name = "framework",
    srcs = [
        "framework/framework_lib.py",
        "framework/graph_io.py",
        "framework/importer.py",
        "framework/load_library.py",
        "framework/meta_graph.py",
        "framework/subscribe.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":common_shapes",
        ":cpp_shape_inference_proto_py",
        ":errors",
        ":framework_for_generated_wrappers",
        ":graph_util",
        ":lib",
        ":platform",
        ":pywrap_tensorflow",
        ":random_seed",
        ":sparse_tensor",
        ":tensor_util",
        ":util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "common_shapes",
    srcs = ["framework/common_shapes.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cpp_shape_inference_proto_py",
        ":errors",
        ":framework_ops",
        ":pywrap_tensorflow",
        ":tensor_shape",
        ":tensor_util",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "constant_op",
    srcs = ["framework/constant_op.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":tensor_shape",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "device",
    srcs = ["framework/device.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "dtypes",
    srcs = ["framework/dtypes.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "errors",
    srcs = [
        "framework/errors.py",
        "framework/errors_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [":util"],
)

py_library(
    name = "function",
    srcs = ["framework/function.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":op_def_registry",
        ":util",
        ":variable_scope",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "graph_util",
    srcs = [
        "framework/graph_util.py",
        "framework/graph_util_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":platform",
        ":tensor_util",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "op_def_library",
    srcs = ["framework/op_def_library.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":platform",
        ":tensor_shape",
        ":util",
        "//tensorflow/core:protos_all_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "op_def_registry",
    srcs = ["framework/op_def_registry.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "framework_ops",  # "ops" is already the name of a deprecated target
    srcs = ["framework/ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":device",
        ":dtypes",
        ":op_def_registry",
        ":platform",
        ":registry",
        ":tensor_shape",
        ":util",
        ":versions",
        "//tensorflow/core:protos_all_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "random_seed",
    srcs = ["framework/random_seed.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_ops",
    ],
)

py_library(
    name = "registry",
    srcs = ["framework/registry.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":platform",
        ":util",
    ],
)

py_library(
    name = "sparse_tensor",
    srcs = ["framework/sparse_tensor.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":tensor_util",
    ],
)

py_library(
    name = "tensor_shape",
    srcs = ["framework/tensor_shape.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":util",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "tensor_util",
    srcs = ["framework/tensor_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tensor_shape",
        ":util",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "versions",
    srcs = ["framework/versions.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pywrap_tensorflow",
    ],
)

# load("//third_party/py/cython:build_defs.bzl", "pyx_library")

py_library(
    name = "extra_py_tests_deps",
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "framework_test_lib",
    srcs = ["framework/test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":platform",
        ":platform_test",
        ":pywrap_tensorflow",
        ":util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "client_testlib",
    srcs = ["platform/test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client",
        ":framework_test_lib",
        ":gradient_checker",
        ":platform_test",
        ":util",
    ],
)

py_test(
    name = "framework_registry_test",
    size = "small",
    srcs = ["framework/registry_test.py"],
    main = "framework/registry_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "framework_errors_test",
    size = "small",
    srcs = ["framework/errors_test.py"],
    main = "framework/errors_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":errors",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "framework_subscribe_test",
    size = "small",
    srcs = ["framework/subscribe_test.py"],
    main = "framework/subscribe_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform_test",
        ":script_ops",
    ],
)

py_test(
    name = "contrib_test",
    size = "small",
    srcs = ["framework/contrib_test.py"],
    main = "framework/contrib_test.py",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "proto_test",
    size = "small",
    srcs = ["framework/proto_test.py"],
    main = "framework/proto_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "functional_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

py_library(
    name = "functional_ops",
    srcs = ["ops/functional_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":control_flow_ops",
        ":framework_ops",
        ":functional_ops_gen",
        ":sparse_tensor",
        ":tensor_array_ops",
        ":tensor_shape",
        ":util",
        ":variable_scope",
    ],
)

cuda_py_tests(
    name = "framework_function_test",
    size = "medium",
    srcs = ["framework/function_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":clip_ops",
        ":control_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":functional_ops",
        ":gradients",
        ":init_ops",
        ":logging_ops",
        ":logging_ops_gen",
        ":math_ops",
        ":nn_ops",
        ":platform",
        ":random_ops",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "framework_versions_test",
    size = "small",
    srcs = ["framework/versions_test.py"],
    main = "framework/versions_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
    ],
)

py_test(
    name = "framework_importer_test",
    size = "medium",
    srcs = ["framework/importer_test.py"],
    main = "framework/importer_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":random_ops",
        ":test_ops",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "framework_meta_graph_test",
    size = "small",
    srcs = ["framework/meta_graph_test.py"],
    main = "framework/meta_graph_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":control_flow_ops",
        ":data_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn_ops",
        ":platform",
        ":random_ops",
        ":training",
        ":variables",
    ],
)

tf_gen_op_wrapper_py(
    name = "test_ops",
    out = "framework/test_ops.py",
    deps = [":test_ops_kernels"],
)

cc_library(
    name = "test_ops_kernels",
    srcs = ["framework/test_ops.cc"],
    linkstatic = 1,
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "test_ops_2",
    out = "framework/test_ops_2.py",
    require_shape_functions = True,
    deps = [":test_ops_2_kernels"],
)

cc_library(
    name = "test_ops_2_kernels",
    srcs = ["framework/test_ops_2.cc"],
    linkstatic = 1,
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

py_test(
    name = "framework_common_shapes_test",
    size = "small",
    srcs = ["framework/common_shapes_test.py"],
    main = "framework/common_shapes_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "framework_ops_test",
    size = "small",
    srcs = ["framework/ops_test.py"],
    main = "framework/ops_test.py",
    srcs_version = "PY2AND3",
    tags = ["no_pip"],  # test_ops_2 is not available in pip.
    deps = [
        ":control_flow_ops",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":gradients",
        ":math_ops",
        ":platform_test",
        ":resources",
        ":test_ops",
        ":test_ops_2",
        ":util",
        ":variable_scope",
        ":variables",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "framework_tensor_shape_test",
    size = "small",
    srcs = ["framework/tensor_shape_test.py"],
    main = "framework/tensor_shape_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "framework_sparse_tensor_test",
    size = "small",
    srcs = ["framework/sparse_tensor_test.py"],
    main = "framework/sparse_tensor_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "framework_device_test",
    size = "small",
    srcs = ["framework/device_test.py"],
    main = "framework/device_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "framework_random_seed_test",
    size = "small",
    srcs = ["framework/random_seed_test.py"],
    main = "framework/random_seed_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":framework",
    ],
)

py_test(
    name = "framework_tensor_shape_div_test",
    size = "small",
    srcs = ["framework/tensor_shape_div_test.py"],
    main = "framework/tensor_shape_div_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "framework_tensor_util_test",
    size = "small",
    srcs = ["framework/tensor_util_test.py"],
    main = "framework/tensor_util_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":state_ops_gen",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "framework_test_util_test",
    size = "small",
    srcs = ["framework/test_util_test.py"],
    main = "framework/test_util_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":control_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        ":random_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "framework_dtypes_test",
    size = "small",
    srcs = ["framework/dtypes_test.py"],
    main = "framework/dtypes_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "op_def_library_test",
    size = "small",
    srcs = ["framework/op_def_library_test.py"],
    main = "framework/op_def_library_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "array_ops_gen",
    require_shape_functions = True,
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/contrib/quantization:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "candidate_sampling_ops_gen",
    require_shape_functions = True,
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "control_flow_ops_gen",
    require_shape_functions = True,
    visibility = ["//learning/brain/python/ops:__pkg__"],
    deps = [
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core:no_op_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "ctc_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "data_flow_ops_gen",
    require_shape_functions = True,
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/contrib/lookup:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "image_ops_gen",
    require_shape_functions = True,
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "io_ops_gen",
    require_shape_functions = True,
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "linalg_ops_gen",
    require_shape_functions = True,
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "logging_ops_gen",
    require_shape_functions = True,
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "math_ops_gen",
    require_shape_functions = True,
    visibility = [
        "//learning/brain/google/python/ops:__pkg__",
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/contrib/quantization:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "nn_ops_gen",
    require_shape_functions = True,
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/contrib/quantization:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/tools:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "parsing_ops_gen",
    require_shape_functions = True,
    visibility = [
        "//learning/brain/python/ops:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "random_ops_gen",
    require_shape_functions = True,
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "resource_variable_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "script_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "sdca_ops_gen",
    require_shape_functions = True,
    visibility = ["//tensorflow/contrib/linear_optimizer:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "set_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "state_ops_gen",
    require_shape_functions = True,
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/contrib/framework:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "sparse_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "spectral_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "string_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "user_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "training_ops_gen",
    out = "training/gen_training_ops.py",
    require_shape_functions = True,
)

py_library(
    name = "array_grad",
    srcs = ["ops/array_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":sparse_ops",
    ],
)

py_library(
    name = "array_ops",
    srcs = ["ops/array_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops_gen",
        ":common_shapes",
        ":constant_op",
        ":dtypes",
        ":framework_ops",
        ":math_ops_gen",
        ":sparse_tensor",
        ":tensor_shape",
        ":tensor_util",
        ":util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "sets",
    srcs = [
        "ops/sets.py",
        "ops/sets_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":set_ops_gen",
        ":util",
    ],
)

py_library(
    name = "candidate_sampling_ops",
    srcs = ["ops/candidate_sampling_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":candidate_sampling_ops_gen",
        ":framework",
        ":math_ops",
    ],
)

py_library(
    name = "check_ops",
    srcs = ["ops/check_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "clip_ops",
    srcs = ["ops/clip_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn_ops_gen",
        "@six_archive//:six",
    ],
)

py_library(
    name = "control_flow_grad",
    srcs = ["ops/control_flow_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":control_flow_ops",
        ":control_flow_ops_gen",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
    ],
)

py_library(
    name = "control_flow_ops",
    srcs = ["ops/control_flow_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "tensor_shape",
        ":array_ops",
        ":array_ops_gen",
        ":constant_op",
        ":control_flow_ops_gen",
        ":data_flow_ops_gen",
        ":dtypes",
        ":framework_ops",
        ":logging_ops_gen",
        ":math_ops",
        ":platform",
        ":sparse_tensor",
        ":tensor_array_ops",
        ":util",
        "//tensorflow/core:protos_all_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "ctc_ops",
    srcs = ["ops/ctc_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":ctc_ops_gen",
        ":framework",
        ":framework_for_generated_wrappers",
        ":nn_grad",
    ],
)

py_library(
    name = "data_flow_grad",
    srcs = ["ops/data_flow_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":data_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
    ],
)

py_library(
    name = "data_flow_ops",
    srcs = ["ops/data_flow_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":data_flow_ops_gen",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        "@six_archive//:six",
    ],
)

py_library(
    name = "embedding_ops",
    srcs = ["ops/embedding_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":clip_ops",
        ":data_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":platform",
        ":resource_variable_ops",
        ":variables",
    ],
)

py_library(
    name = "gradients",
    srcs = [
        "ops/gradients.py",
        "ops/gradients_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_grad",
        ":array_ops",
        ":control_flow_grad",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":functional_ops",
        ":image_grad",
        ":linalg_grad",
        ":linalg_ops",
        ":logging_ops",
        ":math_grad",
        ":math_ops",
        ":platform",
        ":spectral_grad",
        ":util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "histogram_ops",
    srcs = ["ops/histogram_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":clip_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
    ],
)

py_library(
    name = "image_grad",
    srcs = ["ops/image_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":image_ops_gen",
    ],
)

py_library(
    name = "image_ops",
    srcs = [
        "ops/image_ops.py",
        "ops/image_ops_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":clip_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":image_ops_gen",
        ":math_ops",
        ":nn_ops_gen",
        ":random_ops",
        ":string_ops",
        ":util",
        ":variables",
    ],
)

py_library(
    name = "init_ops",
    srcs = ["ops/init_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":dtypes",
        ":linalg_ops",
        ":math_ops",
        ":nn_ops",
        ":random_ops",
    ],
)

py_library(
    name = "io_ops",
    srcs = ["ops/io_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":io_ops_gen",
        ":lib",
    ],
)

py_library(
    name = "linalg_grad",
    srcs = ["ops/linalg_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":linalg_ops",
        ":math_ops",
    ],
)

py_library(
    name = "linalg_ops",
    srcs = ["ops/linalg_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":linalg_ops_gen",
        ":math_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "logging_ops",
    srcs = ["ops/logging_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":logging_ops_gen",
        ":util",
    ],
)

py_library(
    name = "math_grad",
    srcs = ["ops/math_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":math_ops_gen",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "math_ops",
    srcs = ["ops/math_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "constant_op",
        ":array_ops",
        ":common_shapes",
        ":control_flow_ops_gen",
        ":data_flow_ops_gen",
        ":dtypes",
        ":framework_ops",
        ":graph_util",
        ":math_ops_gen",
        ":nn_ops_gen",
        ":sparse_ops_gen",
        ":sparse_tensor",
        ":spectral_ops_gen",
        ":state_ops",
        ":state_ops_gen",
        ":tensor_shape",
        ":util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "resources",
    srcs = ["ops/resources.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
    ],
)

py_library(
    name = "resource_variable_ops",
    srcs = ["ops/resource_variable_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework_ops",
        ":resource_variable_ops_gen",
        ":tensor_shape",
        ":util",
        ":variables",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "nn",
    srcs = [
        "ops/nn.py",
        "ops/nn_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":candidate_sampling_ops",
        ":ctc_ops",
        ":embedding_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":nn_ops_gen",
        ":rnn",
        ":sparse_ops",
        ":util",
        ":variables",
    ],
)

py_library(
    name = "nn_grad",
    srcs = ["ops/nn_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn_ops",
        ":nn_ops_gen",
        ":sparse_ops",
    ],
)

py_library(
    name = "nn_ops",
    srcs = ["ops/nn_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":graph_util",
        ":math_ops",
        ":nn_ops_gen",
        ":random_ops",
        ":tensor_shape",
        ":tensor_util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "numerics",
    srcs = ["ops/numerics.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
    ],
)

py_library(
    name = "parsing_ops",
    srcs = ["ops/parsing_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":parsing_ops_gen",
        ":sparse_ops",
    ],
)

py_library(
    name = "partitioned_variables",
    srcs = ["ops/partitioned_variables.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":platform",
        ":variable_scope",
    ],
)

py_library(
    name = "random_ops",
    srcs = ["ops/random_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":random_ops_gen",
        ":random_seed",
    ],
)

py_library(
    name = "rnn",
    srcs = ["ops/rnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":rnn_cell",
        ":tensor_array_ops",
        ":util",
        ":variable_scope",
    ],
)

py_library(
    name = "rnn_cell",
    srcs = [
        "ops/rnn_cell_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":layers_base",
        ":util",
    ],
)

py_library(
    name = "script_ops",
    srcs = ["ops/script_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":script_ops_gen",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "sdca_ops",
    srcs = ["ops/sdca_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":sdca_ops_gen",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "session_ops",
    srcs = ["ops/session_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":data_flow_ops_gen",
        ":framework_for_generated_wrappers",
        ":util",
    ],
)

py_library(
    name = "sparse_grad",
    srcs = ["ops/sparse_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":sparse_ops",
        ":sparse_ops_gen",
    ],
)

py_library(
    name = "sparse_ops",
    srcs = ["ops/sparse_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":sparse_ops_gen",
        ":util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "spectral_grad",
    srcs = ["ops/spectral_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":spectral_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "spectral_ops",
    srcs = ["ops/spectral_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":spectral_ops_gen",
    ],
)

py_library(
    name = "confusion_matrix",
    srcs = ["ops/confusion_matrix.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":sparse_ops",
    ],
)

py_library(
    name = "weights_broadcast_ops",
    srcs = [
        "ops/weights_broadcast_ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
        ":math_ops",
        ":sets",
    ],
)

py_library(
    name = "metrics",
    srcs = [
        "ops/metrics.py",
        "ops/metrics_impl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":confusion_matrix",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn",
        ":sets",
        ":sparse_ops",
        ":state_ops",
        ":util",
        ":variable_scope",
        ":variables",
        ":weights_broadcast_ops",
    ],
)

py_library(
    name = "special_math_ops",
    srcs = ["ops/special_math_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":platform",
    ],
)

py_library(
    name = "standard_ops",
    srcs = ["ops/standard_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_grad",
        ":array_ops",
        ":check_ops",
        ":clip_ops",
        ":confusion_matrix",
        ":control_flow_ops",
        ":data_flow_grad",
        ":data_flow_ops",
        ":framework_for_generated_wrappers",
        ":functional_ops",
        ":gradients",
        ":histogram_ops",
        ":init_ops",
        ":io_ops",
        ":linalg_ops",
        ":logging_ops",
        ":math_grad",
        ":math_ops",
        ":numerics",
        ":parsing_ops",
        ":partitioned_variables",
        ":random_ops",
        ":script_ops",
        ":session_ops",
        ":sparse_grad",
        ":sparse_ops",
        ":special_math_ops",
        ":spectral_grad",
        ":state_grad",
        ":state_ops",
        ":string_ops",
        ":template",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":util",
        ":variable_scope",
        ":variables",
    ],
)

py_library(
    name = "state_grad",
    srcs = ["ops/state_grad.py"],
    srcs_version = "PY2AND3",
    deps = [":framework_for_generated_wrappers"],
)

py_library(
    name = "state_ops",
    srcs = ["ops/state_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_ops",
        ":resource_variable_ops_gen",
        ":state_ops_gen",
        ":tensor_shape",
    ],
)

py_library(
    name = "string_ops",
    srcs = ["ops/string_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":string_ops_gen",
        ":util",
    ],
)

py_library(
    name = "summary_ops",
    srcs = ["ops/summary_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":logging_ops_gen",
    ],
)

py_library(
    name = "template",
    srcs = ["ops/template.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":platform",
        ":util",
        ":variable_scope",
    ],
)

py_library(
    name = "tensor_array_grad",
    srcs = ["ops/tensor_array_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":tensor_array_ops",
    ],
)

py_library(
    name = "tensor_array_ops",
    srcs = ["ops/tensor_array_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":data_flow_ops_gen",
        ":framework_ops",
        ":math_ops",
        ":tensor_shape",
        ":tensor_util",
        ":util",
    ],
)

py_library(
    name = "variable_scope",
    srcs = ["ops/variable_scope.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":init_ops",
        ":platform",
        ":resource_variable_ops",
        ":tensor_shape",
        ":variables",
        "@six_archive//:six",
    ],
)

py_library(
    name = "variables",
    srcs = ["ops/variables.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":state_ops",
        ":tensor_shape",
        ":util",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "gradient_checker",
    srcs = ["ops/gradient_checker.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":platform",
        "//third_party/py/numpy",
    ],
)

# This target is deprecated.
py_library(
    name = "ops",
    srcs = ["user_ops/user_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":user_ops_gen",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "control_flow_ops_test",
    size = "small",
    srcs = ["ops/control_flow_ops_test.py"],
    additional_deps = [
        ":array_ops",
        ":control_flow_ops",
        ":embedding_ops",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":gradients",
        ":init_ops",
        ":math_ops",
        ":platform_test",
        ":state_ops",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":training",
        ":util",
        ":variable_scope",
        ":variables",
    ],
)

cuda_py_test(
    name = "gradient_checker_test",
    size = "medium",
    srcs = ["ops/gradient_checker_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":platform",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "gradients_test",
    size = "small",
    srcs = ["ops/gradients_test.py"],
    additional_deps = [
        ":array_grad",
        ":array_ops",
        ":control_flow_grad",
        ":control_flow_ops",
        ":data_flow_grad",
        ":data_flow_ops",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":functional_ops",
        ":gradients",
        ":math_grad",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":platform_test",
        ":state_grad",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":test_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "histogram_ops_test",
    size = "small",
    srcs = ["ops/histogram_ops_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":histogram_ops",
        ":init_ops",
        ":variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "image_grad_test",
    size = "small",
    srcs = ["ops/image_grad_test.py"],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":image_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "image_ops_test",
    size = "small",
    srcs = ["ops/image_ops_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":image_ops",
        ":io_ops",
        ":math_ops",
        ":platform_test",
        ":random_ops",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    data = ["//tensorflow/core:image_testdata"],
    shard_count = 5,
)

cuda_py_test(
    name = "math_grad_test",
    size = "small",
    srcs = ["ops/math_grad_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "math_ops_test",
    size = "small",
    srcs = ["ops/math_ops_test.py"],
    additional_deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":gradients",
        ":math_ops",
        ":platform_test",
        ":variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "nn_batchnorm_test",
    size = "medium",
    srcs = ["ops/nn_batchnorm_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":math_ops",
        ":nn",
        ":nn_grad",
        ":nn_ops_gen",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "nn_fused_batchnorm_test",
    size = "large",
    srcs = ["ops/nn_fused_batchnorm_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":nn",
        ":nn_grad",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "nn_test",
    size = "medium",
    srcs = ["ops/nn_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":nn",
        ":nn_grad",
        ":nn_ops",
        ":partitioned_variables",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "nn_xent_test",
    size = "medium",
    srcs = ["ops/nn_xent_test.py"],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":nn",
        ":nn_grad",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "special_math_ops_test",
    size = "small",
    srcs = ["ops/special_math_ops_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":special_math_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "training",
    srcs = glob(
        ["training/**/*.py"],
        exclude = ["**/*test*"],
    ),
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client",
        ":control_flow_ops",
        ":data_flow_ops",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":init_ops",
        ":io_ops",
        ":io_ops_gen",
        ":lib",
        ":math_ops",
        ":platform",
        ":protos_all_py",
        ":pywrap_tensorflow",
        ":random_ops",
        ":resource_variable_ops",
        ":resources",
        ":sdca_ops",
        ":sparse_ops",
        ":state_ops",
        ":string_ops",
        ":summary",
        ":training_ops_gen",
        ":util",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "evaluation_test",
    size = "small",
    srcs = ["training/evaluation_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "notap",  # Disabling until b/33000128 and b/33040312 are fixed.
    ],
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":layers",
        ":math_ops",
        ":metrics",
        ":platform",
        ":state_ops",
        ":summary",
        ":training",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/ops/losses",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "client",
    srcs = [
        "client/client_lib.py",
        "client/device_lib.py",
        "client/session.py",
        "client/timeline.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":platform",
        ":session_ops",
        ":util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "util",
    srcs = glob(
        ["util/**/*.py"],
        exclude = [
            "util/example_parser*",
            "util/**/*_test.py",
        ],
    ),
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/numpy",
        "@protobuf//:protobuf_python",
        "@six_archive//:six",
    ],
)

py_test(
    name = "util_nest_test",
    size = "small",
    srcs = ["util/nest_test.py"],
    main = "util/nest_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":util",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "future_api_test",
    size = "small",
    srcs = ["util/future_api_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":util",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "tf_contextlib_test",
    size = "small",
    srcs = ["util/tf_contextlib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":util",
    ],
)

py_test(
    name = "tf_decorator_test",
    size = "small",
    srcs = ["util/tf_decorator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":util",
    ],
)

py_test(
    name = "tf_should_use_test",
    size = "small",
    srcs = ["util/tf_should_use_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":util",
    ],
)

py_test(
    name = "tf_inspect_test",
    size = "small",
    srcs = ["util/tf_inspect_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":util",
    ],
)

py_library(
    name = "util_example_parser_configuration",
    srcs = ["util/example_parser_configuration.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_proto_library(
    name = "protos_all",
    srcs = glob(
        ["**/*.proto"],
        exclude = [
            "util/protobuf/compare_test.proto",
            "framework/cpp_shape_inference.proto",
        ],
    ),
    go_api_version = 2,
)

tf_proto_library_py(
    name = "compare_test_proto",
    testonly = 1,
    srcs = ["util/protobuf/compare_test.proto"],
)

tf_proto_library(
    name = "cpp_shape_inference_proto",
    srcs = ["framework/cpp_shape_inference.proto"],
    cc_api_version = 2,
    protodeps = ["//tensorflow/core:protos_all"],
)

py_test(
    name = "protobuf_compare_test",
    size = "small",
    srcs = ["util/protobuf/compare_test.py"],
    main = "util/protobuf/compare_test.py",
    srcs_version = "PY2AND3",
    tags = ["no_pip"],  # compare_test_pb2 proto is not available in pip.
    deps = [
        ":compare_test_proto_py",
        ":platform_test",
        ":util",
        "@six_archive//:six",
    ],
)

py_test(
    name = "util_example_parser_configuration_test",
    size = "small",
    srcs = ["util/example_parser_configuration_test.py"],
    main = "util/example_parser_configuration_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":parsing_ops",
        ":util_example_parser_configuration",
    ],
)

py_test(
    name = "events_writer_test",
    size = "small",
    srcs = [
        "client/events_writer_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":errors",
        ":framework_test_lib",
        ":lib",
        ":platform_test",
        ":util",
    ],
)

py_library(
    name = "device_lib",
    srcs = ["client/device_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pywrap_tensorflow",
    ],
)

cc_library(
    name = "cpp_shape_inference",
    srcs = ["framework/cpp_shape_inference.cc"],
    hdrs = ["framework/cpp_shape_inference.h"],
    copts = ["-Wno-sign-compare"],
    visibility = ["//visibility:public"],
    deps = [
        ":cpp_shape_inference_proto_cc",
        ":numpy_lib",
        ":py_func_lib",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_cc",
        "//third_party/py/numpy:headers",
        "//util/python:python_headers",
    ],
)

cuda_py_tests(
    name = "device_lib_test",
    size = "small",
    srcs = [
        "client/device_lib_test.py",
    ],
    additional_deps = [
        ":client",
        ":client_testlib",
        ":framework_test_lib",
        ":platform_test",
    ],
)

tf_cuda_library(
    name = "tf_session_helper",
    srcs = ["client/tf_session_helper.cc"],
    hdrs = ["client/tf_session_helper.h"],
    deps = [
        ":construction_fails_op",
        ":ndarray_tensor_bridge",
        ":numpy_lib",
        ":test_ops_kernels",
        "//tensorflow/c:c_api",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_cc",
        "//third_party/py/numpy:headers",
        "//util/python:python_headers",
    ],
)

py_library(
    name = "pywrap_tensorflow",
    srcs = ["pywrap_tensorflow.py"],
    srcs_version = "PY2AND3",
    deps = [":pywrap_tensorflow_internal"],
)

tf_py_wrap_cc(
    name = "pywrap_tensorflow_internal",
    srcs = ["tensorflow.i"],
    swig_includes = [
        "client/device_lib.i",
        "client/events_writer.i",
        "client/tf_session.i",
        "framework/cpp_shape_inference.i",
        "framework/python_op_gen.i",
        "grappler/tf_optimizer.i",
        "lib/core/py_func.i",
        "lib/core/strings.i",
        "lib/io/file_io.i",
        "lib/io/py_record_reader.i",
        "lib/io/py_record_writer.i",
        "platform/base.i",
        "training/quantize_training.i",
        "training/server_lib.i",
        "util/kernel_registry.i",
        "util/port.i",
        "util/py_checkpoint_reader.i",
        "util/stat_summarizer.i",
        "util/transform_graph.i",
    ],
    deps = [
        ":cpp_shape_inference",
        ":kernel_registry",
        ":numpy_lib",
        ":py_func_lib",
        ":py_record_reader_lib",
        ":py_record_writer_lib",
        ":python_op_gen",
        ":tf_session_helper",
        "//tensorflow/c:c_api",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core:lib",
        "//tensorflow/core:reader_base",
        "//tensorflow/core/debug",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/tools/graph_transforms:transform_graph_lib",
        "//tensorflow/tools/tfprof/internal:print_model_analysis",
        "//util/python:python_headers",
    ] + (tf_additional_lib_deps() +
         tf_additional_plugin_deps() +
         tf_additional_verbs_deps()),
)

py_library(
    name = "lib",
    srcs = [
        "lib/io/file_io.py",
        "lib/io/python_io.py",
        "lib/io/tf_record.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":errors",
        ":pywrap_tensorflow",
        ":util",
    ],
)

py_library(
    name = "session",
    srcs = ["client/session.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":platform",
        ":pywrap_tensorflow",
        ":session_ops",
        ":util",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "server_lib_test",
    size = "small",
    srcs = ["training/server_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":data_flow_ops",
        ":errors",
        ":extra_py_tests_deps",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":training",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "server_lib_multiple_containers_test",
    size = "small",
    srcs = ["training/server_lib_multiple_containers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":data_flow_ops",
        ":errors",
        ":extra_py_tests_deps",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":training",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "server_lib_same_variables_clear_container_test",
    size = "small",
    srcs = ["training/server_lib_same_variables_clear_container_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":data_flow_ops",
        ":errors",
        ":extra_py_tests_deps",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":training",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "server_lib_same_variables_clear_test",
    size = "small",
    srcs = ["training/server_lib_same_variables_clear_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":data_flow_ops",
        ":errors",
        ":extra_py_tests_deps",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":training",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "server_lib_same_variables_no_clear_test",
    size = "small",
    srcs = ["training/server_lib_same_variables_no_clear_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":data_flow_ops",
        ":errors",
        ":extra_py_tests_deps",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":training",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "server_lib_sparse_job_test",
    size = "small",
    srcs = ["training/server_lib_sparse_job_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":data_flow_ops",
        ":errors",
        ":extra_py_tests_deps",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":training",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "localhost_cluster_performance_test",
    size = "medium",
    srcs = [
        "training/localhost_cluster_performance_test.py",
    ],
    additional_deps = [
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":partitioned_variables",
        ":training",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sync_replicas_optimizer_test",
    size = "medium",
    srcs = [
        "training/sync_replicas_optimizer_test.py",
    ],
    additional_deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":training",
        ":variables",
    ],
)

py_library(
    name = "timeline",
    srcs = ["client/timeline.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":platform",
    ],
)

# Just used by tests.
tf_cuda_library(
    name = "construction_fails_op",
    srcs = ["client/test_construction_fails_op.cc"],
    deps = [
        "//tensorflow/core",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_cc",
    ],
    alwayslink = 1,
)

py_test(
    name = "session_test",
    size = "small",
    srcs = ["client/session_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_gpu",
        "no_pip_gpu",  # testInteractivePlacePrunedGraph fails on invalid assumption about GPU ops.
    ],
    deps = [
        ":array_ops",
        ":client",
        ":control_flow_ops",
        ":data_flow_ops",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform_test",
        ":state_ops",
        ":training",
        ":util",
        ":variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "timeline_test",
    size = "small",
    srcs = ["client/timeline_test.py"],
    additional_deps = [
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "graph_util_test",
    size = "small",
    srcs = ["framework/graph_util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client",
        ":client_testlib",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":state_ops_gen",
        ":variables",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "file_io_test",
    size = "small",
    srcs = ["lib/io/file_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":errors",
        ":lib",
    ],
)

cuda_py_test(
    name = "adam_test",
    size = "small",
    srcs = ["training/adam_test.py"],
    additional_deps = [
        ":array_ops",
        ":framework",
        ":math_ops",
        ":platform",
        ":training",
        ":platform_test",
        ":client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "moving_averages_test",
    size = "small",
    srcs = [
        "training/moving_averages_test.py",
    ],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":constant_op",
        ":dtypes",
        ":framework_for_generated_wrappers",
        ":framework_ops",
        ":training",
        ":variable_scope",
        ":variables",
    ],
    tags = ["notsan"],
)

cuda_py_tests(
    name = "training_tests",
    size = "small",
    srcs = [
        "training/adadelta_test.py",
        "training/adagrad_da_test.py",
        "training/adagrad_test.py",
        "training/basic_loops_test.py",
        "training/checkpoint_utils_test.py",
        "training/coordinator_test.py",
        "training/device_setter_test.py",
        "training/ftrl_test.py",
        "training/gradient_descent_test.py",
        "training/learning_rate_decay_test.py",
        "training/momentum_test.py",
        "training/optimizer_test.py",
        "training/proximal_adagrad_test.py",
        "training/proximal_gradient_descent_test.py",
        "training/quantize_training_test.py",
        "training/queue_runner_test.py",
        "training/rmsprop_test.py",
        "training/slot_creator_test.py",
        "training/tensorboard_logging_test.py",
        "training/training_ops_test.py",
    ],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":data_flow_ops",
        ":data_flow_ops_gen",
        ":embedding_ops",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":gradients",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":partitioned_variables",
        ":platform",
        ":platform_test",
        ":pywrap_tensorflow",
        ":random_ops",
        ":resource_variable_ops",
        ":resources",
        ":sparse_ops",
        ":state_ops",
        ":state_ops_gen",
        ":summary",
        ":training",
        ":util",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "saver_test_utils",
    srcs = ["training/saver_test_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":data_flow_ops_gen",
        ":training",
    ],
)

cuda_py_test(
    name = "saver_test",
    size = "medium",
    srcs = [
        "training/saver_test.py",
    ],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":control_flow_ops",
        ":data_flow_ops",
        ":errors",
        ":gradients",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":saver_test_utils",
        ":partitioned_variables",
        ":platform",
        ":platform_test",
        ":pywrap_tensorflow",
        ":random_ops",
        ":resource_variable_ops",
        ":sparse_ops",
        ":summary",
        ":training",
        ":util",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "saver_large_variable_test",
    size = "small",
    srcs = ["training/saver_large_variable_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "noasan",  # http://b/30379628
        "notsan",  # http://b/30379628
    ],
    deps = [
        ":client",
        ":client_testlib",
        ":errors",
        ":framework_for_generated_wrappers",
        ":training",
        ":variables",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "saver_large_partitioned_variable_test",
    size = "medium",
    srcs = ["training/saver_large_partitioned_variable_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "noasan",  # http://b/30782289
        "notsan",  # http://b/30782289
    ],
    deps = [
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":partitioned_variables",
        ":training",
        ":variables",
    ],
)

cuda_py_test(
    name = "session_manager_test",
    size = "medium",  # TODO(irving): Can this be made small?
    srcs = ["training/session_manager_test.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":errors",
        ":framework_for_generated_wrappers",
        ":platform",
        ":training",
        ":variables",
    ],
    main = "training/session_manager_test.py",
)

py_test(
    name = "supervisor_test",
    size = "small",
    srcs = ["training/supervisor_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":errors",
        ":extra_py_tests_deps",
        ":framework",
        ":framework_for_generated_wrappers",
        ":io_ops",
        ":parsing_ops",
        ":platform",
        ":summary",
        ":training",
        ":variables",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "basic_session_run_hooks_test",
    size = "small",
    srcs = ["training/basic_session_run_hooks_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":nn_grad",
        ":platform",
        ":state_ops",
        ":summary",
        ":training",
        ":variable_scope",
        ":variables",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/testing:testing_py",
    ],
)

py_test(
    name = "monitored_session_test",
    size = "small",
    srcs = ["training/monitored_session_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":errors",
        ":framework_for_generated_wrappers",
        ":state_ops",
        ":summary",
        ":training",
        ":variables",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/testing:testing_py",
    ],
)

py_test(
    name = "training_util_test",
    size = "small",
    srcs = ["training/training_util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":framework",
        ":platform",
        ":training",
        ":variables",
    ],
)

tf_py_test(
    name = "input_test",
    size = "small",
    srcs = ["training/input_test.py"],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":platform",
        ":util",
        ":variables",
        ":training",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "summary",
    srcs = glob(
        ["summary/**/*.py"],
        exclude = ["**/*test*"],
    ),
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":lib",
        ":logging_ops_gen",
        ":platform",
        ":protos_all_py",
        ":pywrap_tensorflow",
        ":summary_ops",
        ":util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_tests(
    name = "summary_tests",
    size = "small",
    srcs = [
        "summary/plugin_asset_test.py",
        "summary/summary_test.py",
        "summary/text_summary_test.py",
        "summary/writer/writer_test.py",
    ],
    additional_deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":variables",
        ":framework",
        ":framework_test_lib",
        ":platform",
        ":platform_test",
        ":summary",
        ":training",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "layers_base",
    srcs = [
        "layers/__init__.py",
        "layers/base.py",
        "layers/utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":init_ops",
        ":util",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "layers",
    srcs = [
        "layers/convolutional.py",
        "layers/core.py",
        "layers/layers.py",
        "layers/normalization.py",
        "layers/pooling.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":init_ops",
        ":layers_base",
        ":math_ops",
        ":nn",
        ":standard_ops",
        ":training",
        ":util",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "layers_base_test",
    size = "small",
    srcs = ["layers/base_test.py"],
    main = "layers/base_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":init_ops",
        ":layers",
        ":math_ops",
        ":random_ops",
        ":variable_scope",
    ],
)

py_test(
    name = "layers_core_test",
    size = "small",
    srcs = ["layers/core_test.py"],
    main = "layers/core_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":layers",
        ":math_ops",
        ":nn_ops",
        ":random_ops",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "layers_convolutional_test",
    size = "small",
    srcs = ["layers/convolutional_test.py"],
    main = "layers/convolutional_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":layers",
        ":math_ops",
        ":nn_ops",
        ":random_ops",
    ],
)

py_test(
    name = "layers_utils_test",
    size = "small",
    srcs = ["layers/utils_test.py"],
    main = "layers/utils_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":layers",
    ],
)

py_test(
    name = "layers_pooling_test",
    size = "small",
    srcs = ["layers/pooling_test.py"],
    main = "layers/pooling_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":layers",
        ":random_ops",
    ],
)

py_test(
    name = "layers_normalization_test",
    size = "small",
    srcs = ["layers/normalization_test.py"],
    main = "layers/normalization_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":layers",
        ":math_ops",
        ":random_ops",
        ":variables",
        "//third_party/py/numpy",
    ],
)

# -----------------------------------------------------------------------------
# Quantization

py_test(
    name = "dequantize_op_test",
    size = "small",
    srcs = ["ops/dequantize_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "quantized_conv_ops_test",
    size = "small",
    srcs = ["ops/quantized_conv_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":nn_ops",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

cuda_py_test(
    name = "accumulate_n_benchmark",
    size = "large",
    srcs = ["ops/accumulate_n_benchmark.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops_gen",
        ":data_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":random_ops",
        ":state_ops",
        ":state_ops_gen",
    ],
    main = "ops/accumulate_n_benchmark.py",
)

cuda_py_test(
    name = "batch_norm_benchmark",
    srcs = ["ops/batch_norm_benchmark.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":math_ops",
        ":nn",
        ":nn_grad",
        ":nn_ops_gen",
        ":platform",
        ":random_ops",
        ":variables",
    ],
    main = "ops/batch_norm_benchmark.py",
)

cuda_py_test(
    name = "concat_benchmark",
    srcs = ["ops/concat_benchmark.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":platform",
        ":variables",
        "//tensorflow/core:protos_all_py",
    ],
    main = "ops/concat_benchmark.py",
)

cuda_py_test(
    name = "split_benchmark",
    srcs = ["ops/split_benchmark.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":platform",
        ":platform_benchmark",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    main = "ops/split_benchmark.py",
)

cuda_py_test(
    name = "transpose_benchmark",
    size = "medium",
    srcs = ["ops/transpose_benchmark.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":platform",
        ":platform_benchmark",
        ":variables",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
    ],
    main = "ops/transpose_benchmark.py",
)

cuda_py_test(
    name = "session_benchmark",
    srcs = ["client/session_benchmark.py"],
    additional_deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":random_ops",
        ":training",
        ":variables",
        "//third_party/py/numpy",
    ],
    main = "client/session_benchmark.py",
)

py_library(
    name = "tf_optimizer",
    srcs = [
        "grappler/tf_optimizer.py",
    ],
    srcs_version = "PY2AND3",
    deps = [":pywrap_tensorflow_internal"],
)

py_test(
    name = "tf_optimizer_test",
    size = "small",
    srcs = ["grappler/tf_optimizer_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],  # tf_optimizer is not available in pip.
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":tf_optimizer",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)
