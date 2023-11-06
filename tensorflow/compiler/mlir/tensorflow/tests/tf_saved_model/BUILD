load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "common",
    srcs = ["common.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:pywrap_mlir",
        "//tensorflow/python/lib/io:file_io",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "common_v1",
    srcs = ["common_v1.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:pywrap_mlir",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

py_strict_binary(
    name = "basic",
    testonly = 1,
    srcs = ["basic.py"],
    deps = [
        ":common",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "cyclic_object_graph",
    srcs = ["cyclic_object_graph.py"],
    deps = [
        ":common",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "multi_variables_v1",
    srcs = ["multi_variables_v1.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "import_restore_v1",
    srcs = ["import_restore_v1.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "shapes_for_arguments",
    srcs = ["shapes_for_arguments.py"],
    deps = [
        ":common",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "control_flow_upgrade_legacy_v1",
    srcs = ["control_flow_upgrade_legacy_v1.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ops:control_flow_ops",
    ],
)

py_strict_binary(
    name = "exported_python_args",
    srcs = ["exported_python_args.py"],
    deps = [
        ":common",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "basic_v1_no_variable_lifting",
    srcs = ["basic_v1_no_variable_lifting.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "multi_arguments_results_v1",
    srcs = ["multi_arguments_results_v1.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/ops:array_ops",
    ],
)

py_strict_binary(
    name = "no_input_shape_v1",
    srcs = ["no_input_shape_v1.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
    ],
)

py_strict_binary(
    name = "structured_input",
    srcs = ["structured_input.py"],
    deps = [
        ":common",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "defun_export",
    srcs = ["defun_export.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:function",
    ],
)

py_strict_binary(
    name = "basic_v1",
    srcs = ["basic_v1.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "duplicate_method_names_v1",
    srcs = ["duplicate_method_names_v1.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "debug_info",
    srcs = ["debug_info.py"],
    deps = [
        ":common",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "keras",
    srcs = ["keras.py"],
    deps = [
        ":common",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "hash_table_v1",
    srcs = ["hash_table_v1.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "include_variables_in_init_v1",
    srcs = ["include_variables_in_init_v1.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "hash_table_asset_v1",
    srcs = ["hash_table_asset_v1.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "control_flow_duplicate_v1",
    srcs = ["control_flow_duplicate_v1.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "shared_variable_v1",
    srcs = ["shared_variable_v1.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "dag_object_graph",
    srcs = ["dag_object_graph.py"],
    deps = [
        ":common",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "call_to_exported",
    srcs = ["call_to_exported.py"],
    deps = [
        ":common",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "remove_init_variable_v1",
    srcs = ["remove_init_variable_v1.py"],
    deps = [
        ":common_v1",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "partially_shaped_variables",
    srcs = ["partially_shaped_variables.py"],
    deps = [
        ":common",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_binary(
    name = "structured_output",
    srcs = ["structured_output.py"],
    deps = [
        ":common",
        "//tensorflow:tensorflow_py",
    ],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "@llvm-project//llvm:FileCheck",
    ],
)

test_files = glob(
    ["*.py"],
    exclude = [
        "common.py",
        "common_v1.py",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    default_size = "medium",
    default_tags = [
        "no_mac",  # TODO(b/191167848)
        "no_oss",  # TODO(b/190855110)
        "no_rocm",
    ],
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = [
        "common.py",
        "common_v1.py",
    ],
    per_test_extra_data = {
        file: [file[:-3]]
        for file in test_files
    },
    test_file_exts = ["py"],
)
