load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//tensorflow:pytype.default.bzl", "pytype_strict_binary")
load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "if_google", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "tf_cuda_cc_test")
load(":gen_saved_model.bzl", "gen_saved_model")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [
        # copybara:uncomment "//learning/brain/tfrt/...",
        "//tensorflow/core/tfrt/saved_model/tests/...",
        "//tensorflow/core/tfrt/tfrt_session/...",
        "//tensorflow/core/tfrt/utils/debug/...",
    ],
)

py_strict_library(name = "empty")

alias(
    name = "disable_tf2",
    actual = if_google("//learning/brain/public:disable_tf2", ":empty"),
)

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/**",
    ]) + [
        ":saved_model_gen_control_flow_v1",
        ":saved_model_gen_data",
        ":saved_model_gen_dtype_coverage_v1",
        ":saved_model_gen_error_v1",
        ":saved_model_gen_hash_table_asset_v1",
        ":saved_model_gen_if_v1",
        ":saved_model_gen_matmul_gpu",
        ":saved_model_gen_pow",
        ":saved_model_gen_pow_v2",
        ":saved_model_gen_ref_type_tensor_input",
        ":saved_model_gen_resource_gather_v1",
        ":saved_model_gen_sparse_tensor_input",
        ":saved_model_gen_toy_v1",
        ":saved_model_gen_toy_v2",
        ":saved_model_gen_variable_on_tpu",
        ":saved_model_gen_while_v1",
    ],
)

pytype_strict_binary(
    name = "gen_resource_gather_v1",
    srcs = ["gen_resource_gather_v1.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//tensorflow/python/client:session",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "gen_error_v1",
    srcs = ["gen_error_v1.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//tensorflow/python/client:session",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "gen_pow",
    srcs = ["gen_pow.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/client:session",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "gen_sparse_tensor_input",
    srcs = ["gen_sparse_tensor_input.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/client:session",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "gen_ref_type_tensor_input",
    srcs = ["gen_ref_type_tensor_input.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/client:session",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "gen_pow_v2",
    srcs = ["gen_pow_v2.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_binary(
    name = "gen_data",
    srcs = ["gen_data.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:standard_ops",  # build_cleaner: keep
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_binary(
    name = "gen_saved_model_v1",
    srcs = ["gen_saved_model_v1.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//tensorflow/python/client:session",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_ops",
        # copybara:uncomment "//tensorflow/python/framework:test_ops_kernels",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "gen_dtype_coverage_v1",
    srcs = ["gen_dtype_coverage_v1.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//tensorflow/python/client:session",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "gen_saved_model_v2",
    srcs = ["gen_saved_model_v2.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_binary(
    name = "gen_matmul_gpu",
    srcs = ["gen_matmul_gpu.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_binary(
    name = "gen_variable_on_tpu",
    srcs = ["gen_variable_on_tpu.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:standard_ops",  # build_cleaner: keep
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_binary(
    name = "gen_control_flow_v1",
    srcs = ["gen_control_flow_v1.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//tensorflow/python/client:session",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "gen_hash_table_asset_v1",
    srcs = ["gen_hash_table_asset_v1.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:resource_variables_toggle",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "gen_if_v1",
    srcs = ["gen_if_v1.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//tensorflow/python/client:session",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "gen_while_v1",
    srcs = ["gen_while_v1.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//tensorflow/python/client:session",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:save_options",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

gen_saved_model(
    model_name = "resource_gather_v1",
    script = "gen_resource_gather_v1",
)

gen_saved_model(
    model_name = "toy_v1",
    script = "gen_saved_model_v1",
    version = "1",
)

gen_saved_model(
    model_name = "dtype_coverage_v1",
    script = "gen_dtype_coverage_v1",
)

gen_saved_model(
    model_name = "toy_v2",
    script = "gen_saved_model_v2",
)

gen_saved_model(
    model_name = "matmul_gpu",
    script = "gen_matmul_gpu",
)

gen_saved_model(
    model_name = "variable_on_tpu",
    script = "gen_variable_on_tpu",
)

gen_saved_model(
    model_name = "if_v1",
    script = "gen_if_v1",
)

gen_saved_model(
    model_name = "ref_type_tensor_input",
    script = "gen_ref_type_tensor_input",
)

genrule(
    name = "saved_model_gen_while_v1",
    srcs = [],
    outs = [
        "while_v1/saved_model.pb",
    ],
    cmd = if_google(
        "$(location gen_while_v1) --saved_model_path=$(RULEDIR)/while_v1",
        "touch $(OUTS)",  # TODO(b/188517768): fix model gen.
    ),
    tools = ["gen_while_v1"],
)

genrule(
    name = "saved_model_gen_hash_table_asset_v1",
    srcs = [],
    outs = [
        "hash_table_asset_v1/saved_model.pb",
        "hash_table_asset_v1/assets/tokens.txt",
    ],
    cmd = if_google(
        "$(location gen_hash_table_asset_v1) --saved_model_path=$(RULEDIR)/hash_table_asset_v1",
        "touch $(OUTS)",  # TODO(b/188517768): fix model gen.
    ),
    tools = ["gen_hash_table_asset_v1"],
)

genrule(
    name = "saved_model_gen_error_v1",
    srcs = [],
    outs = [
        "error_v1/saved_model.pb",
    ],
    cmd = if_google(
        "$(location gen_error_v1) --saved_model_path=$(RULEDIR)/error_v1",
        "touch $(OUTS)",  # TODO(b/188517768): fix model gen.
    ),
    tools = ["gen_error_v1"],
)

genrule(
    name = "saved_model_gen_control_flow_v1",
    srcs = [],
    outs = [
        "control_flow_v1/saved_model.pb",
    ],
    cmd = if_google(
        "$(location gen_control_flow_v1) --saved_model_path=$(RULEDIR)/control_flow_v1",
        "touch $(OUTS)",  # TODO(b/188517768): fix model gen.
    ),
    tools = ["gen_control_flow_v1"],
)

genrule(
    name = "saved_model_gen_pow",
    srcs = [],
    outs = [
        "pow/saved_model.pb",
    ],
    cmd = if_google(
        "$(location gen_pow) --saved_model_path=$(RULEDIR)/pow",
        "touch $(OUTS)",  # TODO(b/188517768): fix model gen.
    ),
    tools = ["gen_pow"],
)

genrule(
    name = "saved_model_gen_sparse_tensor_input",
    srcs = [],
    outs = [
        "sparse_tensor_input/saved_model.pb",
    ],
    cmd = if_google(
        "$(location gen_sparse_tensor_input) --saved_model_path=$(RULEDIR)/sparse_tensor_input",
        "touch $(OUTS)",  # TODO(b/188517768): fix model gen.
    ),
    tools = ["gen_sparse_tensor_input"],
)

genrule(
    name = "saved_model_gen_pow_v2",
    srcs = [],
    outs = [
        "pow_v2/saved_model.pb",
    ],
    cmd = "$(location gen_pow_v2) --saved_model_path=$(RULEDIR)/pow_v2",
    tools = ["gen_pow_v2"],
)

genrule(
    name = "saved_model_gen_data",
    srcs = [],
    outs = [
        "data/saved_model.pb",
    ],
    cmd = "$(location gen_data) --saved_model_path=$(RULEDIR)/data",
    tools = ["gen_data"],
)

cc_library(
    name = "saved_model_testlib",
    testonly = 1,
    srcs = ["saved_model_test.cc"],
    data = [
        ":testdata",
    ],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/compiler/mlir/tfrt:backend_compiler",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/tfrt/graph_executor:config",
        "//tensorflow/core/tfrt/graph_executor:test_config_proto_cc",
        "//tensorflow/core/tfrt/run_handler_thread_pool:run_handler_concurrent_work_queue",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/runtime:work_queue_interface",
        "//tensorflow/core/tfrt/saved_model:saved_model_cpu",
        "//tensorflow/core/tfrt/saved_model:saved_model_testutil",
        "//tensorflow/core/tfrt/saved_model:saved_model_util",
        "//tensorflow/python/framework:test_ops_kernels",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:hostcontext",
    ],
    alwayslink = 1,
)

cc_library(
    name = "saved_model_ifrt_testlib",
    testonly = 1,
    srcs = ["saved_model_ifrt_test.cc"],
    data = [
        ":testdata",
    ],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/compiler/mlir/tfrt:tfrt_compile_options",
        "//tensorflow/compiler/mlir/tfrt/transforms/ifrt:ifrt_backend_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/tfrt:ifrt_program_ops_op_lib",
        "//tensorflow/core/tfrt/ifrt:ifrt_model_context",
        "//tensorflow/core/tfrt/ifrt:ifrt_serving_core_selector",
        "//tensorflow/core/tfrt/mlrt/kernel:ifrt_ops_kernel",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/saved_model:saved_model_cpu",
        "//tensorflow/core/tfrt/saved_model:saved_model_testutil",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:env",
        "@local_xla//xla/python/ifrt",
        "@local_xla//xla/python/ifrt:test_util",
        "@local_xla//xla/python/pjrt_ifrt:tfrt_cpu_client_test_lib",
        "@local_xla//xla/tsl/framework/test_util:mock_serving_device_selector",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:test_kernels_alwayslink",
        "@tf_runtime//backends/cpu:core_runtime_alwayslink",
        "@tf_runtime//backends/cpu:tf_ops_alwayslink",
    ],
)

tf_cc_test(
    name = "saved_model_ifrt_test",
    srcs = [],
    tags = ["no_oss"],
    deps = [
        ":saved_model_ifrt_testlib",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "saved_model_ifrt_test_mlrt",
    srcs = [],
    args = ["--enable_mlrt=true"],
    tags = ["no_oss"],
    deps = [
        ":saved_model_ifrt_testlib",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "saved_model_test",
    srcs = [],
    tags = ["no_oss"],
    deps = [
        ":saved_model_testlib",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "saved_model_test_mlrt",
    srcs = [],
    args = ["--enable_mlrt=true"],
    tags = ["no_oss"],
    deps = [
        ":saved_model_testlib",
        "//tensorflow/core:test_main",
    ],
)

tf_cuda_cc_test(
    name = "saved_model_gpu_test",
    srcs = ["saved_model_gpu_test.cc"],
    data = [
        ":testdata",
    ],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:reader",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/tfrt/saved_model:saved_model_testutil",
        "//tensorflow/python/framework:test_ops_kernels",
        "@com_google_googletest//:gtest",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/cpu:core_runtime_alwayslink",
        "@tf_runtime//backends/cpu:tf_ops_alwayslink",
    ],
)

bzl_library(
    name = "gen_saved_model_bzl",
    srcs = ["gen_saved_model.bzl"],
    visibility = ["//visibility:private"],
    deps = ["//tensorflow:tensorflow_bzl"],
)

tf_cuda_cc_test(
    name = "saved_model_aot_test",
    srcs = ["saved_model_aot_test.cc"],
    tags = [
        "no_oss",
        "requires-gpu-nvidia",
    ],
    deps = [
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:math_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/jit:test_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/framework:op",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/tfrt/saved_model:saved_model_aot_compile",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)
