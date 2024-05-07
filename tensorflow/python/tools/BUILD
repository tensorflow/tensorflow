# Description:
#   Tools for manipulating TensorFlow graphs.

load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("//tensorflow:tensorflow.bzl", "if_google", "if_xla_available", "tf_cc_test")
load("//tensorflow/python/tools:tools.bzl", "saved_model_compile_aot")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Transitive dependencies of this target will be included in the pip package.
py_strict_library(
    name = "tools_pip",
    data = [
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
        ":freeze_graph",
        ":import_pb_to_tensorboard",
        ":inspect_checkpoint",
        ":optimize_for_inference",
        ":print_selective_registration_header",
        ":saved_model_cli",
        ":strip_unused",
    ],
)

py_strict_library(
    name = "saved_model_utils",
    srcs = ["saved_model_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/saved_model:constants",
        "//tensorflow/python/util:compat",
    ],
)

py_strict_test(
    name = "saved_model_utils_test",
    size = "small",
    srcs = ["saved_model_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_windows"],  # TODO: needs investigation on Windows
    visibility = ["//visibility:private"],
    deps = [
        ":saved_model_utils",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:tag_constants",
    ],
)

py_strict_library(
    name = "freeze_graph_lib",
    srcs = ["freeze_graph.py"],
    srcs_version = "PY3",
    deps = [
        ":saved_model_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:convert_to_constants",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/training:py_checkpoint_reader",
        "//tensorflow/python/training:saver",
        "@absl_py//absl:app",
    ],
)

py_strict_binary(
    name = "freeze_graph",
    srcs = ["freeze_graph.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":freeze_graph_lib"],
)

py_strict_binary(
    name = "import_pb_to_tensorboard",
    srcs = ["import_pb_to_tensorboard.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":import_pb_to_tensorboard_lib"],
)

py_strict_library(
    name = "import_pb_to_tensorboard_lib",
    srcs = ["import_pb_to_tensorboard.py"],
    srcs_version = "PY3",
    deps = [
        ":saved_model_utils",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/summary:summary_py",
        "@absl_py//absl:app",
    ],
)

py_strict_binary(
    name = "tf_import_time",
    srcs = ["tf_import_time.py"],
    deps = ["//tensorflow:tensorflow_py"],
)

py_strict_test(
    name = "freeze_graph_test",
    size = "small",
    srcs = ["freeze_graph_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":freeze_graph_lib",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:graph_io",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops:partitioned_variables",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/training:saver",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_binary(
    name = "inspect_checkpoint",
    srcs = ["inspect_checkpoint.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":inspect_checkpoint_lib"],
)

py_strict_library(
    name = "inspect_checkpoint_lib",
    srcs = ["inspect_checkpoint.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/training:py_checkpoint_reader",
        "//third_party/py/numpy",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "strip_unused_lib",
    srcs = ["strip_unused_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:graph_util",
        "//tensorflow/python/platform:gfile",
    ],
)

py_strict_library(
    name = "module_util",
    srcs = ["module_util.py"],
    srcs_version = "PY3",
)

py_strict_binary(
    name = "strip_unused",
    srcs = ["strip_unused.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":strip_unused_lib",
        "//tensorflow/python/framework:for_generated_wrappers",
        "@absl_py//absl:app",
    ],
)

py_strict_test(
    name = "strip_unused_test",
    size = "small",
    srcs = ["strip_unused_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["notap"],
    deps = [
        ":strip_unused_lib",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:graph_io",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "optimize_for_inference_lib",
    srcs = ["optimize_for_inference_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":strip_unused_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:graph_util",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

py_strict_binary(
    name = "optimize_for_inference",
    srcs = ["optimize_for_inference.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":optimize_for_inference_main_lib"],
)

py_strict_library(
    name = "optimize_for_inference_main_lib",
    srcs = ["optimize_for_inference.py"],
    srcs_version = "PY3",
    deps = [
        ":optimize_for_inference_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:graph_io",
        "//tensorflow/python/platform:gfile",
        "@absl_py//absl:app",
    ],
)

py_strict_test(
    name = "optimize_for_inference_test",
    size = "small",
    srcs = ["optimize_for_inference_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimize_for_inference_lib",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:image_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:nn_ops_gen",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "selective_registration_header_lib",
    srcs = ["selective_registration_header_lib.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        # copybara:comment_begin(oss-only)
        "//tensorflow/python",  # to fix libtensorflow_framework.so.2 import error.
        # copybara:comment_end
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:_pywrap_kernel_registry",
    ],
)

py_strict_binary(
    name = "print_selective_registration_header",
    srcs = ["print_selective_registration_header.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [":print_selective_registration_header_lib"],
)

py_strict_library(
    name = "print_selective_registration_header_lib",
    srcs = ["print_selective_registration_header.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":selective_registration_header_lib",
        "@absl_py//absl:app",
    ],
)

py_strict_test(
    name = "print_selective_registration_header_test",
    srcs = ["print_selective_registration_header_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":selective_registration_header_lib",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
    ],
)

py_strict_binary(
    name = "saved_model_cli",
    srcs = ["saved_model_cli.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":saved_model_cli_lib"],
)

py_strict_library(
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
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/flags:argparse_flags",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python",
        "//tensorflow/python/client:session",
        "//tensorflow/python/compiler/tensorrt:trt_convert_py",
        "//tensorflow/python/debug/wrappers:local_cli_wrapper",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:load_options",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/tpu:tpu_py",
        "//tensorflow/python/util:compat",
    ] + if_google([
        "//learning/brain/contrib/hub/public/file:tfhub_file_register",
        "//tensorflow/core/tfrt/tfrt_session",
        "//tensorflow_text:tensorflow_text",
    ]),
)

py_strict_library(
    name = "saved_model_aot_compile",
    srcs = ["saved_model_aot_compile.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/compiler/tf2xla:tf2xla_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:convert_to_constants",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:versions",
        "//tensorflow/python/grappler:tf_optimizer",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:sysconfig",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:saver",
    ],
)

py_strict_test(
    name = "saved_model_cli_test",
    srcs = ["saved_model_cli_test.py"],
    data = [
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "noasan",  # TODO(b/222716501)
    ],
    deps = [
        ":saved_model_cli_lib",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/debug/wrappers:local_cli_wrapper",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:parsing_config",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/trackable:autotrackable",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_binary(
    name = "make_aot_compile_models",
    srcs = ["make_aot_compile_models.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/trackable:autotrackable",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

# copybara:comment_begin(oss-only)
py_strict_binary(
    name = "grpc_tpu_worker",
    srcs = ["grpc_tpu_worker.py"],
)

py_strict_binary(
    name = "grpc_tpu_worker_service",
    srcs = ["grpc_tpu_worker_service.py"],
)
# copybara:comment_end

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
        "@local_xla//xla/service/cpu:runtime_matmul_acl",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:logging",
        "@eigen_archive//:eigen3",
    ]),
)

# copybara:uncomment_begin(google-only)
# gensignature(
#     name = "inspect_checkpoint.par_sig",
#     srcs = [":inspect_checkpoint.par"],
# )
# copybara:uncomment_end
