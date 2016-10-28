# Description:
# Python support for TensorFlow.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_cuda_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "py_tests")
load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_tests")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library_py")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_additional_lib_deps")
load("//tensorflow/python:build_defs.bzl", "tf_gen_op_wrapper_private_py")

py_library(
    name = "python",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__pkg__"],
    deps = [
        ":client",
        ":client_testlib",
        ":framework",
        ":framework_test_lib",
        ":gradient_checker",
        ":platform",
        ":platform_test",
        ":summary",
        ":training",
        "//tensorflow/contrib:contrib_py",
    ],
)

py_library(
    name = "platform",
    srcs = glob(["platform/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        ":lib",
        ":pywrap_tensorflow",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "platform_test",
    srcs = [
        "platform/googletest.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":platform",
        ":timeline",
    ],
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
    additional_deps = [
        ":platform",
        ":platform_test",
    ],
)

tf_py_test(
    name = "app_test",
    size = "small",
    srcs = ["platform/app_test.py"],
    additional_deps = [
        ":platform",
        ":platform_test",
    ],
    tags = [
        "manual",
        "notap",
    ],
)

cc_library(
    name = "numpy_lib",
    srcs = ["lib/core/numpy.cc"],
    hdrs = [
        "lib/core/numpy.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/py/numpy:headers",
        "//util/python:python_headers",
    ],
)

cc_library(
    name = "py_func_lib",
    srcs = ["lib/core/py_func.cc"],
    hdrs = [
        "lib/core/py_func.h",
    ],
    deps = [
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
    srcs = [
        "lib/io/py_record_reader.cc",
    ],
    hdrs = [
        "lib/io/py_record_reader.h",
    ],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "py_record_writer_lib",
    srcs = [
        "lib/io/py_record_writer.cc",
    ],
    hdrs = [
        "lib/io/py_record_writer.h",
    ],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:lib",
    ],
)

cc_binary(
    name = "framework/test_file_system.so",
    srcs = ["framework/test_file_system.cc"],
    linkopts = select({
        "//conditions:default": [
            "-lm",
        ],
        "//tensorflow:darwin": [],
    }),
    linkshared = 1,
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "@protobuf//:protobuf",
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
        ":framework_test_lib",
        "//tensorflow:tensorflow_py",
    ],
)

cc_library(
    name = "python_op_gen",
    srcs = ["framework/python_op_gen.cc"],
    hdrs = ["framework/python_op_gen.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_cc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "cpp_shape_inference",
    srcs = ["framework/cpp_shape_inference.cc"],
    hdrs = ["framework/cpp_shape_inference.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":numpy_lib",
        ":py_func_lib",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_cc",
        "//third_party/py/numpy:headers",
        "//util/python:python_headers",
    ],
)

cc_library(
    name = "python_op_gen_main",
    srcs = [
        "framework/python_op_gen_main.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":python_op_gen",
    ],
)

# What is needed for tf_gen_op_wrapper_py.
py_library(
    name = "framework_for_generated_wrappers",
    srcs = [
        "framework/constant_op.py",
        "framework/device.py",
        "framework/dtypes.py",
        "framework/function.py",
        "framework/op_def_library.py",
        "framework/op_def_registry.py",
        "framework/ops.py",
        "framework/registry.py",
        "framework/tensor_shape.py",
        "framework/versions.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":platform",
        ":util",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "framework",
    srcs = [
        "framework/common_shapes.py",
        "framework/errors.py",
        "framework/framework_lib.py",
        "framework/graph_util.py",
        "framework/importer.py",
        "framework/load_library.py",
        "framework/random_seed.py",
        "framework/tensor_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_for_generated_wrappers",
        ":pywrap_tensorflow",
    ],
)

py_library(
    name = "errors",
    srcs = ["framework/errors.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":util",
    ],
)

# load("//third_party/py/cython:build_defs.bzl", "pyx_library")

py_library(
    name = "extra_py_tests_deps",
    srcs_version = "PY2AND3",
)

py_library(
    name = "framework_test_lib",
    srcs = [
        "framework/test_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":platform_test",
        ":pywrap_tensorflow",
        ":session",
        ":util",
    ],
)

py_library(
    name = "client_testlib",
    srcs = [
        "platform/test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":device_lib",
        ":framework_test_lib",
        ":platform_test",
    ],
)

py_test(
    name = "framework_errors_test",
    size = "small",
    srcs = ["framework/errors_test.py"],
    main = "framework/errors_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "contrib_test",
    size = "small",
    srcs = ["framework/contrib_test.py"],
    main = "framework/contrib_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_test_lib",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "proto_test",
    size = "small",
    srcs = ["framework/proto_test.py"],
    main = "framework/proto_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_test_lib",
        "//tensorflow:tensorflow_py",
    ],
)

tf_gen_op_wrapper_private_py(name = "functional_ops_gen")

py_library(
    name = "functional_ops",
    srcs = ["ops/functional_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
        ":functional_ops_gen",
        ":tensor_array_ops",
        ":variable_scope",
    ],
)

cuda_py_tests(
    name = "framework_function_test",
    size = "medium",
    srcs = ["framework/function_test.py"],
    additional_deps = [
        ":functional_ops",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "framework_versions_test",
    size = "small",
    srcs = ["framework/versions_test.py"],
    main = "framework/versions_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "framework_importer_test",
    size = "small",
    srcs = ["framework/importer_test.py"],
    main = "framework/importer_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow:tensorflow_py",
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

py_test(
    name = "framework_common_shapes_test",
    size = "small",
    srcs = ["framework/common_shapes_test.py"],
    main = "framework/common_shapes_test.py",
    srcs_version = "PY2AND3",
    deps = [
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
    deps = [
        ":control_flow_ops",
        ":framework_test_lib",
        ":gradients",
        ":math_ops",
        ":platform_test",
        ":session",
        ":sparse_ops",
        ":test_ops",
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
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "framework_tensor_shape_div_test",
    size = "small",
    srcs = ["framework/tensor_shape_div_test.py"],
    main = "framework/tensor_shape_div_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "framework_tensor_util_test",
    size = "small",
    srcs = ["framework/tensor_util_test.py"],
    main = "framework/tensor_util_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_test_lib",
        ":platform_test",
        ":state_ops",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "framework_test_util_test",
    size = "small",
    srcs = ["framework/test_util_test.py"],
    main = "framework/test_util_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_test_lib",
        ":logging_ops",
        ":platform_test",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "framework_dtypes_test",
    size = "small",
    srcs = ["framework/dtypes_test.py"],
    main = "framework/dtypes_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "op_def_library_test",
    size = "small",
    srcs = ["framework/op_def_library_test.py"],
    main = "framework/op_def_library_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_test_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "array_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "candidate_sampling_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "control_flow_ops_gen",
    require_shape_functions = True,
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
)

tf_gen_op_wrapper_private_py(
    name = "image_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "io_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "linalg_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "logging_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "math_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "nn_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "parsing_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "random_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "script_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "state_ops_gen",
    require_shape_functions = True,
)

tf_gen_op_wrapper_private_py(
    name = "sparse_ops_gen",
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
        ":framework",
        ":logging_ops",
        ":math_ops_gen",
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
        ":logging_ops",
        ":math_ops",
    ],
)

py_library(
    name = "clip_ops",
    srcs = ["ops/clip_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework",
        ":math_ops",
        ":nn_ops",
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
        ":math_ops_gen",
    ],
)

py_library(
    name = "control_flow_ops",
    srcs = ["ops/control_flow_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":control_flow_ops_gen",
        ":data_flow_ops_gen",
        ":framework",
        ":logging_ops",
        ":math_ops",
        ":tensor_array_ops",
    ],
)

py_library(
    name = "ctc_ops",
    srcs = ["ops/ctc_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ctc_ops_gen",
        ":framework",
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
        ":framework",
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
    ],
)

py_library(
    name = "embedding_ops",
    srcs = ["ops/embedding_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":data_flow_ops",
        ":framework",
        ":math_ops",
    ],
)

py_library(
    name = "gradients",
    srcs = ["ops/gradients.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_grad",
        ":array_ops",
        ":control_flow_grad",
        ":control_flow_ops",
        ":framework",
        ":functional_ops",
        ":image_grad",
        ":linalg_grad",
        ":linalg_ops",
        ":logging_ops",
        ":math_grad",
        ":math_ops",
    ],
)

py_library(
    name = "histogram_ops",
    srcs = ["ops/histogram_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":clip_ops",
        ":framework",
        ":math_ops",
    ],
)

py_library(
    name = "image_grad",
    srcs = ["ops/image_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework",
        ":image_ops_gen",
    ],
)

py_library(
    name = "image_ops",
    srcs = ["ops/image_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":clip_ops",
        ":control_flow_ops",
        ":framework",
        ":image_ops_gen",
        ":logging_ops",
        ":math_ops",
        ":nn_ops_gen",
        ":random_ops",
    ],
)

py_library(
    name = "init_ops",
    srcs = ["ops/init_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework",
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
        ":framework",
        ":io_ops_gen",
    ],
)

py_library(
    name = "linalg_grad",
    srcs = ["ops/linalg_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
        ":linalg_ops",
        ":math_ops",
    ],
)

py_library(
    name = "linalg_ops",
    srcs = ["ops/linalg_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":linalg_ops_gen",
    ],
)

py_library(
    name = "logging_ops",
    srcs = ["ops/logging_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":logging_ops_gen",
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
        ":math_ops",
        ":math_ops_gen",
    ],
)

py_library(
    name = "math_ops",
    srcs = ["ops/math_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":data_flow_ops_gen",
        ":framework",
        ":math_ops_gen",
        ":sparse_ops_gen",
        ":state_ops",
        ":state_ops_gen",
    ],
)

py_library(
    name = "nn",
    srcs = ["ops/nn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":candidate_sampling_ops",
        ":control_flow_ops",
        ":ctc_ops",
        ":embedding_ops",
        ":framework",
        ":init_ops",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":numerics",
        ":random_ops",
        ":rnn",
        ":rnn_cell",
        ":seq2seq",
        ":sparse_ops",
        ":variable_scope",
    ],
)

py_library(
    name = "nn_grad",
    srcs = ["ops/nn_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework",
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
        ":array_ops_gen",
        ":framework",
        ":math_ops",
        ":nn_ops_gen",
        ":random_ops",
    ],
)

py_library(
    name = "numerics",
    srcs = ["ops/numerics.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
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
        ":logging_ops",
        ":math_ops",
        ":parsing_ops_gen",
    ],
)

py_library(
    name = "partitioned_variables",
    srcs = ["ops/partitioned_variables.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
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
        ":framework",
        ":logging_ops",
        ":math_ops",
        ":random_ops_gen",
    ],
)

py_library(
    name = "rnn",
    srcs = ["ops/rnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
        ":logging_ops",
        ":math_ops",
        ":rnn_cell",
        ":tensor_array_ops",
        ":variable_scope",
    ],
)

py_library(
    name = "rnn_cell",
    srcs = ["ops/rnn_cell.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":clip_ops",
        ":embedding_ops",
        ":framework",
        ":init_ops",
        ":math_ops",
        ":nn_ops",
        ":variable_scope",
    ],
)

py_library(
    name = "script_ops",
    srcs = ["ops/script_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":script_ops_gen",
    ],
)

py_library(
    name = "seq2seq",
    srcs = ["ops/seq2seq.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":embedding_ops",
        ":framework",
        ":math_ops",
        ":nn_ops",
        ":rnn",
        ":rnn_cell",
        ":variable_scope",
    ],
)

py_library(
    name = "session_ops",
    srcs = ["ops/session_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":data_flow_ops_gen",
        ":framework",
    ],
)

py_library(
    name = "sparse_grad",
    srcs = ["ops/sparse_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework",
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
        ":math_ops",
        ":sparse_ops_gen",
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
        ":framework",
        ":math_ops",
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
        ":control_flow_ops",
        ":data_flow_grad",
        ":data_flow_ops",
        ":framework",
        ":functional_ops",
        ":gradients",
        ":histogram_ops",
        ":init_ops",
        ":io_ops",
        ":linalg_ops",
        ":logging_ops",
        ":math_grad",
        ":numerics",
        ":parsing_ops",
        ":partitioned_variables",
        ":random_ops",
        ":script_ops",
        ":sparse_grad",
        ":sparse_ops",
        ":special_math_ops",
        ":state_grad",
        ":state_ops",
        ":string_ops",
        ":template",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":variable_scope",
        ":variables",
    ],
)

py_library(
    name = "state_grad",
    srcs = ["ops/state_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":state_ops",
    ],
)

py_library(
    name = "state_ops",
    srcs = ["ops/state_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":state_ops_gen",
    ],
)

py_library(
    name = "string_ops",
    srcs = ["ops/string_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":string_ops_gen",
    ],
)

py_library(
    name = "summary_ops",
    srcs = ["ops/summary_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":logging_ops_gen",
    ],
)

py_library(
    name = "template",
    srcs = ["ops/template.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":variable_scope",
    ],
)

py_library(
    name = "tensor_array_grad",
    srcs = ["ops/tensor_array_grad.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
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
        ":framework",
        ":math_ops",
    ],
)

py_library(
    name = "variable_scope",
    srcs = ["ops/variable_scope.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework",
        ":init_ops",
        ":variables",
    ],
)

py_library(
    name = "variables",
    srcs = ["ops/variables.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
        ":math_ops",
        ":state_ops",
    ],
)

py_library(
    name = "gradient_checker",
    srcs = ["ops/gradient_checker.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_ops",
        ":framework",
        ":gradients",
    ],
)

## This target is deprecated, it contributes to massive dependency bloat.
## If you are adding a test to this directory, consider depending on the
## individual library targets your test depends on. If you are a client,
## you should be importing the rules in third_party/py/tensorflow to get
## access to TensorFlow.
py_library(
    name = "ops",
    srcs = ["user_ops/user_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":array_grad",
        ":array_ops",
        ":array_ops_gen",
        ":candidate_sampling_ops",
        ":check_ops",
        ":clip_ops",
        ":control_flow_grad",
        ":control_flow_ops",
        ":control_flow_ops_gen",
        ":ctc_ops",
        ":ctc_ops_gen",
        ":data_flow_grad",
        ":data_flow_ops",
        ":data_flow_ops_gen",
        ":embedding_ops",
        ":gradients",
        ":histogram_ops",
        ":image_grad",
        ":image_ops",
        ":image_ops_gen",
        ":init_ops",
        ":io_ops",
        ":io_ops_gen",
        ":linalg_grad",
        ":linalg_ops",
        ":linalg_ops_gen",
        ":logging_ops",
        ":logging_ops_gen",
        ":math_grad",
        ":math_ops",
        ":math_ops_gen",
        ":nn",
        ":nn_grad",
        ":nn_ops",
        ":nn_ops_gen",
        ":numerics",
        ":parsing_ops",
        ":partitioned_variables",
        ":random_ops",
        ":random_ops_gen",
        ":rnn",
        ":rnn_cell",
        ":script_ops",
        ":seq2seq",
        ":session_ops",
        ":sparse_grad",
        ":sparse_ops",
        ":special_math_ops",
        ":standard_ops",
        ":state_grad",
        ":state_ops",
        ":state_ops_gen",
        ":string_ops",
        ":string_ops_gen",
        ":summary_ops",
        ":template",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":user_ops_gen",
        ":variable_scope",
        ":variables",
    ],
)

cuda_py_test(
    name = "control_flow_ops_test",
    size = "small",
    srcs = ["ops/control_flow_ops_test.py"],
    additional_deps = [
        ":control_flow_ops",
        ":embedding_ops",
        ":framework_test_lib",
        ":lib",
        ":standard_ops",
        ":training",
    ],
)

cuda_py_test(
    name = "gradient_checker_test",
    size = "medium",
    srcs = ["ops/gradient_checker_test.py"],
    additional_deps = [
        ":array_ops",
        ":device_lib",
        ":gradients",
        ":framework_test_lib",
        ":lib",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "gradients_test",
    size = "small",
    srcs = ["ops/gradients_test.py"],
    additional_deps = [
        ":framework_test_lib",
        ":lib",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "histogram_ops_test",
    size = "small",
    srcs = ["ops/histogram_ops_test.py"],
    additional_deps = [
        ":framework_test_lib",
        ":lib",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "image_grad_test",
    size = "small",
    srcs = ["ops/image_grad_test.py"],
    additional_deps = [
        ":framework_test_lib",
        ":lib",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "image_ops_test",
    size = "small",
    srcs = ["ops/image_ops_test.py"],
    additional_deps = [
        ":array_ops",
        ":device_lib",
        ":framework_test_lib",
        ":image_ops",
        ":io_ops",
        ":lib",
    ],
    data = ["//tensorflow/core:image_testdata"],
    shard_count = 5,
    tags = ["notsan"],
)

cuda_py_test(
    name = "math_grad_test",
    size = "small",
    srcs = ["ops/math_grad_test.py"],
    additional_deps = [
        ":framework_test_lib",
        ":lib",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "math_ops_test",
    size = "small",
    srcs = ["ops/math_ops_test.py"],
    additional_deps = [
        ":framework_test_lib",
    ],
)

cuda_py_test(
    name = "nn_batchnorm_test",
    size = "medium",
    srcs = ["ops/nn_batchnorm_test.py"],
    additional_deps = [
        ":framework_test_lib",
        ":lib",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "nn_test",
    size = "medium",
    srcs = ["ops/nn_test.py"],
    additional_deps = [
        ":framework_test_lib",
        ":lib",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "nn_xent_test",
    size = "medium",
    srcs = ["ops/nn_xent_test.py"],
    additional_deps = [
        ":framework_test_lib",
        ":lib",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "special_math_ops_test",
    size = "small",
    srcs = ["ops/special_math_ops_test.py"],
    additional_deps = [
        ":framework_test_lib",
        ":lib",
        "//tensorflow:tensorflow_py",
    ],
    tags = ["notsan"],
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
        ":framework",
        ":gradients",
        ":init_ops",
        ":io_ops",
        ":io_ops_gen",
        ":lib",
        ":logging_ops",
        ":math_ops",
        ":protos_all_py",
        ":pywrap_tensorflow",
        ":random_ops",
        ":sparse_ops",
        ":state_ops",
        ":training_ops_gen",
        ":variable_scope",
        ":variables",
    ],
)

py_library(
    name = "client",
    srcs = glob(
        ["client/**/*.py"],
        exclude = ["**/*test*"],
    ),
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":session",
        ":session_ops",
        ":training_ops_gen",
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
    deps = ["@protobuf//:protobuf_python"],
)

py_test(
    name = "util_nest_test",
    size = "small",
    srcs = ["util/nest_test.py"],
    main = "util/nest_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":util",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "future_api_test",
    size = "small",
    srcs = [
        "util/future_api_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":util",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "util_example_parser_configuration",
    srcs = ["util/example_parser_configuration.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
    ],
)

tf_proto_library(
    name = "protos_all",
    srcs = glob(
        ["**/*.proto"],
        exclude = [
            "util/protobuf/compare_test.proto",
        ],
    ),
    go_api_version = 2,
)

tf_proto_library_py(
    name = "compare_test_proto",
    testonly = 1,
    srcs = ["util/protobuf/compare_test.proto"],
)

py_test(
    name = "protobuf_compare_test",
    size = "small",
    srcs = ["util/protobuf/compare_test.py"],
    main = "util/protobuf/compare_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":compare_test_proto_py",
        ":platform_test",
        ":util",
    ],
)

py_test(
    name = "util_example_parser_configuration_test",
    size = "small",
    srcs = ["util/example_parser_configuration_test.py"],
    main = "util/example_parser_configuration_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":framework_test_lib",
        ":platform_test",
        ":util_example_parser_configuration",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
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
        ":framework_test_lib",
        ":lib",
        ":platform_test",
    ],
)

py_test(
    name = "quantize_training_test",
    size = "small",
    srcs = [
        "client/quantize_training_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":framework_test_lib",
        ":lib",
        ":platform_test",
        "//tensorflow:tensorflow_py",
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

cuda_py_tests(
    name = "device_lib_test",
    size = "small",
    srcs = [
        "client/device_lib_test.py",
    ],
    additional_deps = [
        ":device_lib",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "net_lib",
    testonly = 1,
    srcs = ["util/net_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pywrap_tensorflow",
    ],
)

py_tests(
    name = "net_lib_test",
    size = "small",
    srcs = [
        "util/net_lib_test.py",
    ],
    additional_deps = [
        ":net_lib",
        "//tensorflow:tensorflow_py",
    ],
)

tf_cuda_library(
    name = "tf_session_helper",
    srcs = ["client/tf_session_helper.cc"],
    hdrs = ["client/tf_session_helper.h"],
    deps = [
        ":construction_fails_op",
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

tf_py_wrap_cc(
    name = "pywrap_tensorflow",
    srcs = ["tensorflow.i"],
    swig_includes = [
        "client/device_lib.i",
        "client/events_writer.i",
        "client/net_lib.i",
        "client/quantize_training.i",
        "client/tf_session.i",
        "framework/cpp_shape_inference.i",
        "framework/python_op_gen.i",
        "lib/core/py_func.i",
        "lib/core/strings.i",
        "lib/io/file_io.i",
        "lib/io/py_record_reader.i",
        "lib/io/py_record_writer.i",
        "platform/base.i",
        "training/server_lib.i",
        "util/port.i",
        "util/py_checkpoint_reader.i",
    ],
    deps = [
        ":numpy_lib",
        ":py_func_lib",
        ":py_record_reader_lib",
        ":py_record_writer_lib",
        ":cpp_shape_inference",
        ":python_op_gen",
        ":tf_session_helper",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "//tensorflow/c:c_api",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//util/python:python_headers",
    ] + tf_additional_lib_deps(),
)

py_library(
    name = "lib",
    srcs = glob(["lib/**/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        ":errors",
        ":pywrap_tensorflow",
    ],
)

py_library(
    name = "session",
    srcs = ["client/session.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":pywrap_tensorflow",
        ":session_ops",
    ],
)

py_test(
    name = "server_lib_test",
    size = "small",
    srcs = ["training/server_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":extra_py_tests_deps",
        ":framework",
        ":framework_test_lib",
        ":session",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "localhost_cluster_performance_test",
    size = "medium",
    srcs = [
        "training/localhost_cluster_performance_test.py",
    ],
    additional_deps = ["//tensorflow:tensorflow_py"],
)

py_library(
    name = "timeline",
    srcs = ["client/timeline.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":platform",
    ],
)

# Just used by tests.
tf_cuda_library(
    name = "construction_fails_op",
    testonly = 1,
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
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":data_flow_ops",
        ":framework",
        ":framework_test_lib",
        ":math_ops",
        ":session",
        ":state_ops",
        ":variables",
    ],
)

cuda_py_test(
    name = "timeline_test",
    size = "small",
    srcs = ["client/timeline_test.py"],
    additional_deps = [
        ":timeline",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "graph_util_test",
    size = "small",
    srcs = ["framework/graph_util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":framework",
        ":framework_test_lib",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "file_io_test",
    size = "small",
    srcs = ["lib/io/file_io_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lib",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_tests(
    name = "training_tests",
    size = "small",
    srcs = [
        "training/adadelta_test.py",
        "training/adagrad_test.py",
        "training/adam_test.py",
        "training/basic_loops_test.py",
        "training/coordinator_test.py",
        "training/device_setter_test.py",
        "training/ftrl_test.py",
        "training/gradient_descent_test.py",
        "training/learning_rate_decay_test.py",
        "training/momentum_test.py",
        "training/moving_averages_test.py",
        "training/optimizer_test.py",
        "training/proximal_adagrad_test.py",
        "training/proximal_gradient_descent_test.py",
        "training/queue_runner_test.py",
        "training/rmsprop_test.py",
        "training/saver_test.py",
        "training/slot_creator_test.py",
        "training/summary_writer_test.py",
        "training/tensorboard_logging_test.py",
        "training/training_ops_test.py",
    ],
    additional_deps = [
        ":training",
        "//tensorflow:tensorflow_py",
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
        "//tensorflow:tensorflow_py",
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
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "session_manager_test",
    size = "medium",  # TODO(irving): Can this be made small?
    srcs = ["training/session_manager_test.py"],
    additional_deps = [
        ":training",
        "//tensorflow:tensorflow_py",
    ],
    main = "training/session_manager_test.py",
)

py_test(
    name = "supervisor_test",
    size = "small",
    srcs = ["training/supervisor_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":extra_py_tests_deps",
        ":training",
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "input_test",
    size = "small",
    srcs = ["training/input_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        ":training",
    ],
)

py_library(
    name = "summary",
    srcs = glob(
        ["summary/**/*.py"],
        exclude = ["**/*test*"],
    ),
    srcs_version = "PY2AND3",
    deps = [
        ":client",
        ":framework",
        ":protos_all_py",
        ":pywrap_tensorflow",
        ":summary_ops",
        ":util",
    ],
)

py_tests(
    name = "summary_tests",
    size = "small",
    srcs = [
        "summary/event_accumulator_test.py",
        "summary/event_file_inspector_test.py",
        "summary/event_multiplexer_test.py",
        "summary/impl/directory_watcher_test.py",
        "summary/impl/event_file_loader_test.py",
        "summary/impl/gcs_file_loader_test.py",
        "summary/impl/reservoir_test.py",
        "summary/summary_test.py",
    ],
    additional_deps = [
        ":summary",
        ":training",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "docs",
    srcs = [
        "framework/docs.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":platform",
    ],
)

py_library(
    name = "gen_docs_combined_lib",
    srcs = [
        "framework/gen_docs_combined.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":docs",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/ffmpeg:ffmpeg_ops_py",
    ],
)

py_binary(
    name = "gen_docs_combined",
    srcs = [
        "framework/gen_docs_combined.py",
    ],
    main = "framework/gen_docs_combined.py",
    srcs_version = "PY2AND3",
    deps = [
        ":docs",
        "//tensorflow:tensorflow_py",
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
    srcs = [
        "ops/accumulate_n_benchmark.py",
    ],
    additional_deps = ["//tensorflow:tensorflow_py"],
    main = "ops/accumulate_n_benchmark.py",
)

cuda_py_test(
    name = "batch_norm_benchmark",
    srcs = [
        "ops/batch_norm_benchmark.py",
    ],
    additional_deps = [
        ":nn_ops",
        "//tensorflow:tensorflow_py",
    ],
    main = "ops/batch_norm_benchmark.py",
)
