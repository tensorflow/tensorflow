# Description:
#   Tools for manipulating TensorFlow graphs.

load("//tensorflow:tensorflow.bzl", "if_google", "if_xla_available", "py_binary", "py_test", "tf_cc_test")
load("//tensorflow/python/tools:tools.bzl", "saved_model_compile_aot")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

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
    srcs_version = "PY3",
    deps = [
        ":saved_model_aot_compile",
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
    srcs_version = "PY3",
)

py_test(
    name = "saved_model_utils_test",
    size = "small",
    srcs = ["saved_model_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [":freeze_graph_main_lib"],
)

py_library(
    name = "freeze_graph_main_lib",
    srcs = ["freeze_graph.py"],
    srcs_version = "PY3",
    deps = [
        ":freeze_graph_lib",
    ],
)

py_binary(
    name = "import_pb_to_tensorboard",
    srcs = ["import_pb_to_tensorboard.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":import_pb_to_tensorboard_lib"],
)

py_library(
    name = "import_pb_to_tensorboard_lib",
    srcs = ["import_pb_to_tensorboard.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python",
        "//tensorflow/python:client",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/summary",
    ],
)

py_test(
    name = "freeze_graph_test",
    size = "small",
    srcs = ["freeze_graph_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:variables",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_binary(
    name = "inspect_checkpoint",
    srcs = ["inspect_checkpoint.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":inspect_checkpoint_lib"],
)

py_library(
    name = "inspect_checkpoint_lib",
    srcs = ["inspect_checkpoint.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:platform",
        "//tensorflow/python:py_checkpoint_reader",
    ],
)

py_library(
    name = "strip_unused_lib",
    srcs = ["strip_unused_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "module_util",
    srcs = ["module_util.py"],
    srcs_version = "PY3",
)

py_binary(
    name = "strip_unused",
    srcs = ["strip_unused.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [":optimize_for_inference_main_lib"],
)

py_library(
    name = "optimize_for_inference_main_lib",
    srcs = ["optimize_for_inference.py"],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:platform",
        "//tensorflow/python/util:_pywrap_kernel_registry",
    ],
)

py_binary(
    name = "print_selective_registration_header",
    srcs = ["print_selective_registration_header.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [":print_selective_registration_header_lib"],
)

py_library(
    name = "print_selective_registration_header_lib",
    srcs = ["print_selective_registration_header.py"],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [":saved_model_cli_lib"],
)

py_library(
    name = "saved_model_cli_lib",
    srcs = ["saved_model_cli.py"],
    srcs_version = "PY3",
    deps = [
        # Note: if you make any changes here, make corresponding changes to the
        # deps of the "tools_pip" target in this file.  Otherwise release builds
        # (built with --define=no_tensorflow_py_deps=true) may end up with a
        # broken saved_model_cli.
        ":saved_model_aot_compile",
        ":saved_model_utils",
        "@absl_py//absl/flags:argparse_flags",
        "//tensorflow/python",
        "//tensorflow/python/client:session",
        "//tensorflow/python/debug/wrappers:local_cli_wrapper",
    ] + if_google([
        "//tensorflow/core/tfrt/tfrt_session",
        "//tensorflow_text:tensorflow_text",
    ]),
)

py_library(
    name = "saved_model_aot_compile",
    srcs = ["saved_model_aot_compile.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python",
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
    srcs_version = "PY3",
    deps = [
        ":saved_model_cli_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_binary(
    name = "make_aot_compile_models",
    srcs = ["make_aot_compile_models.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/saved_model",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@six_archive//:six",
    ],
)

EMITTED_AOT_SAVE_MODEL_OBJECTS = [
    "x_matmul_y_large/saved_model.pb",
    "x_matmul_y_large/variables/variables.index",
    "x_matmul_y_small/saved_model.pb",
    "x_matmul_y_small/variables/variables.index",
]

genrule(
    name = "create_models_for_aot_compile",
    outs = EMITTED_AOT_SAVE_MODEL_OBJECTS,
    cmd = (
        "$(location :make_aot_compile_models) --out_dir $(@D)"
    ),
    tags = ["no_rocm"],
    tools = [":make_aot_compile_models"],
)

filegroup(
    name = "aot_saved_models",
    srcs = EMITTED_AOT_SAVE_MODEL_OBJECTS,
)

saved_model_compile_aot(
    name = "aot_compiled_x_matmul_y_large",
    cpp_class = "XMatmulYLarge",
    directory = "//tensorflow/python/tools:x_matmul_y_large",
    filegroups = [":aot_saved_models"],
    force_without_xla_support_flag = False,
    tags = ["no_rocm"],
)

saved_model_compile_aot(
    name = "aot_compiled_x_matmul_y_large_multithreaded",
    cpp_class = "XMatmulYLargeMultithreaded",
    directory = "//tensorflow/python/tools:x_matmul_y_large",
    filegroups = [":aot_saved_models"],
    force_without_xla_support_flag = False,
    multithreading = True,
    tags = ["no_rocm"],
)

saved_model_compile_aot(
    name = "aot_compiled_x_matmul_y_small",
    cpp_class = "XMatmulYSmall",
    directory = "//tensorflow/python/tools:x_matmul_y_small",
    filegroups = [":aot_saved_models"],
    force_without_xla_support_flag = False,
    tags = ["no_rocm"],
)

saved_model_compile_aot(
    name = "aot_compiled_x_plus_y",
    cpp_class = "XPlusY",
    directory = "//tensorflow/cc/saved_model:testdata/x_plus_y_v2_debuginfo",
    filegroups = [
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    force_without_xla_support_flag = False,
)

saved_model_compile_aot(
    name = "aot_compiled_vars_and_arithmetic_frozen",
    cpp_class = "VarsAndArithmeticFrozen",
    directory = "//tensorflow/cc/saved_model:testdata/VarsAndArithmeticObjectGraph",
    filegroups = [
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    force_without_xla_support_flag = False,
)

saved_model_compile_aot(
    name = "aot_compiled_vars_and_arithmetic",
    cpp_class = "VarsAndArithmetic",
    directory = "//tensorflow/cc/saved_model:testdata/VarsAndArithmeticObjectGraph",
    filegroups = [
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    force_without_xla_support_flag = False,
    variables_to_feed = "variable_x",
)

sh_test(
    name = "large_matmul_no_multithread_test",
    srcs = if_xla_available(
        ["no_xla_multithread_symbols_test.sh"],
        if_false = ["skip_test.sh"],
    ),
    args = if_xla_available(["$(location :aot_compiled_x_matmul_y_large.o)"]),
    data = if_xla_available([":aot_compiled_x_matmul_y_large.o"]),
    tags = ["no_windows"],  # TODO(b/171875345)
)

sh_test(
    name = "large_matmul_yes_multithread_test",
    srcs = if_xla_available(
        [
            "xla_multithread_symbols_test.sh",
        ],
        if_false = ["skip_test.sh"],
    ),
    args = if_xla_available(
        ["$(location :aot_compiled_x_matmul_y_large_multithreaded.o)"],
    ),
    data = if_xla_available(
        [":aot_compiled_x_matmul_y_large_multithreaded.o"],
    ),
    tags = ["no_windows"],  # TODO(b/171875345)
)

tf_cc_test(
    name = "aot_compiled_test",
    srcs = if_xla_available([
        "aot_compiled_test.cc",
    ]),
    deps = [
        "//tensorflow/core:test_main",
    ] + if_xla_available([
        # LINT.IfChange
        ":aot_compiled_vars_and_arithmetic",
        ":aot_compiled_vars_and_arithmetic_frozen",
        ":aot_compiled_x_matmul_y_large",
        ":aot_compiled_x_matmul_y_large_multithreaded",
        ":aot_compiled_x_matmul_y_small",
        ":aot_compiled_x_plus_y",
        # LINT.ThenChange(//tensorflow/tools/pip_package/xla_build/pip_test/run_xla_aot_test.sh)
        "//tensorflow/compiler/xla/service/cpu:runtime_matmul_acl",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:logging",
        "//third_party/eigen3",
    ]),
)

# copybara:uncomment_begin(google-only)
# gensignature(
#     name = "inspect_checkpoint.par_sig",
#     srcs = [":inspect_checkpoint.par"],
# )
# copybara:uncomment_end
