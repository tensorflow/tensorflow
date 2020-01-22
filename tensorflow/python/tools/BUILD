# Description:
#   Tools for manipulating TensorFlow graphs.

load("//tensorflow:tensorflow.bzl", "if_xla_available", "py_binary", "py_test", "tf_cc_test")
load("//tensorflow/compiler/aot:tfcompile.bzl", "target_llvm_triple")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "tools_pip",
    data = [
        ":freeze_graph",
        ":import_pb_to_tensorboard",
        ":inspect_checkpoint",
        ":optimize_for_inference",
        ":print_selective_registration_header",
        ":saved_model_cli",
        ":strip_unused",
        # Include the TF upgrade script to users can run it directly after install TF
        "//tensorflow/tools/compatibility:tf_upgrade_v2",
    ],
    deps = [
        ":saved_model_utils",
        # The following py_library are needed because
        # py_binary may not depend on them when --define=no_tensorflow_py_deps=true
        # is specified. See https://github.com/tensorflow/tensorflow/issues/22390
        ":freeze_graph_lib",
        ":optimize_for_inference_lib",
        ":selective_registration_header_lib",
        ":strip_unused_lib",
    ],
)

py_library(
    name = "saved_model_utils",
    srcs = ["saved_model_utils.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "saved_model_utils_test",
    size = "small",
    srcs = ["saved_model_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = ["no_windows"],  # TODO: needs investigation on Windows
    visibility = ["//visibility:private"],
    deps = [
        ":saved_model_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/saved_model",
    ],
)

py_library(
    name = "freeze_graph_lib",
    srcs = ["freeze_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":saved_model_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:framework",
        "//tensorflow/python:no_contrib",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:py_checkpoint_reader",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/saved_model:loader",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "freeze_graph",
    srcs = ["freeze_graph.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":freeze_graph_main_lib"],
)

py_library(
    name = "freeze_graph_main_lib",
    srcs = ["freeze_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":freeze_graph_lib",
    ],
)

py_binary(
    name = "import_pb_to_tensorboard",
    srcs = ["import_pb_to_tensorboard.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":import_pb_to_tensorboard_lib"],
)

py_library(
    name = "import_pb_to_tensorboard_lib",
    srcs = ["import_pb_to_tensorboard.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python",
        "//tensorflow/python:client",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
    ],
)

py_test(
    name = "freeze_graph_test",
    size = "small",
    srcs = ["freeze_graph_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":freeze_graph_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_binary(
    name = "inspect_checkpoint",
    srcs = ["inspect_checkpoint.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":inspect_checkpoint_lib"],
)

py_library(
    name = "inspect_checkpoint_lib",
    srcs = ["inspect_checkpoint.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:platform",
        "//tensorflow/python:py_checkpoint_reader",
    ],
)

py_library(
    name = "strip_unused_lib",
    srcs = ["strip_unused_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "module_util",
    srcs = ["module_util.py"],
    srcs_version = "PY2AND3",
)

py_binary(
    name = "strip_unused",
    srcs = ["strip_unused.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":strip_unused_lib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "@six_archive//:six",
    ],
)

py_test(
    name = "strip_unused_test",
    size = "small",
    srcs = ["strip_unused_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = ["notap"],
    deps = [
        ":strip_unused_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
    ],
)

py_library(
    name = "optimize_for_inference_lib",
    srcs = ["optimize_for_inference_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":strip_unused_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "optimize_for_inference",
    srcs = ["optimize_for_inference.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":optimize_for_inference_main_lib"],
)

py_library(
    name = "optimize_for_inference_main_lib",
    srcs = ["optimize_for_inference.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":optimize_for_inference_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "@six_archive//:six",
    ],
)

py_test(
    name = "optimize_for_inference_test",
    size = "small",
    srcs = ["optimize_for_inference_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":optimize_for_inference_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:nn_ops_gen",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "selective_registration_header_lib",
    srcs = ["selective_registration_header_lib.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:_pywrap_kernel_registry",
        "//tensorflow/python:platform",
    ],
)

py_binary(
    name = "print_selective_registration_header",
    srcs = ["print_selective_registration_header.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [":print_selective_registration_header_lib"],
)

py_library(
    name = "print_selective_registration_header_lib",
    srcs = ["print_selective_registration_header.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":selective_registration_header_lib",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "print_selective_registration_header_test",
    srcs = ["print_selective_registration_header_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":selective_registration_header_lib",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
    ],
)

py_binary(
    name = "saved_model_cli",
    srcs = ["saved_model_cli.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":saved_model_cli_lib"],
)

py_library(
    name = "saved_model_cli_lib",
    srcs = ["saved_model_cli.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":saved_model_utils",
        "//tensorflow/python",
        "//tensorflow/python/debug:local_cli_wrapper",
        "//tensorflow/python:tf_optimizer",
    ] + if_xla_available(
        ["//tensorflow/compiler/tf2xla:tf2xla_proto_py"],
    ),
)

py_test(
    name = "saved_model_cli_test",
    srcs = ["saved_model_cli_test.py"],
    data = [
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "no-internal-py3",
        "nosan",
    ],
    deps = [
        ":saved_model_cli_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

genrule(
    name = "aot_compiled_x_plus_y_gen",
    srcs = [
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",
        "//tensorflow/cc/saved_model:testdata/x_plus_y_v2_debuginfo/saved_model.pb",
    ],
    outs = [
        "compiled_model.h",
        "compiled_model.o",
        "compiled_model_metadata.o",
        "compiled_model_makefile.inc",
    ],
    cmd = (
        "$(location :saved_model_cli) aot_compile_cpu " +
        "--dir \"$$(dirname $(location //tensorflow/cc/saved_model:testdata/x_plus_y_v2_debuginfo/saved_model.pb))\" " +
        "--output_prefix $(@D)/compiled_model " +
        "--cpp_class CompiledModel " +
        "--target_triple " + target_llvm_triple() +
        " --tag_set serve "
    ),
    tools = [
        ":saved_model_cli",
    ],
)

cc_library(
    name = "aot_compiled_x_plus_y",
    srcs = if_xla_available([
        ":compiled_model.o",
        ":compiled_model_metadata.o",
    ]),
    hdrs = if_xla_available([
        ":compiled_model.h",
    ]),
    deps = if_xla_available([
        "//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
        "//tensorflow/core/platform:types",
    ]),
)

tf_cc_test(
    name = "binary_using_aot_compiled_x_plus_y_test",
    srcs = if_xla_available([
        "binary_using_aot_compiled_x_plus_y_test.cc",
    ]),
    deps = [
        "//tensorflow/core:test_main",
    ] + if_xla_available([
        ":aot_compiled_x_plus_y",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:logging",
    ]),
)
