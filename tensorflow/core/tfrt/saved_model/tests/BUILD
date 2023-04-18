load(":gen_saved_model.bzl", "gen_saved_model")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//tensorflow:pytype.default.bzl", "pytype_strict_binary")
load("//tensorflow:tensorflow.bzl", "if_google", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "tf_cuda_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/core/tfrt/saved_model/tests/...",
        "//tensorflow/core/tfrt/tfrt_session/...",
    ],
)

py_library(name = "empty")

alias(
    name = "disable_tf2",
    actual = if_google("//learning/brain/public:disable_tf2", ":empty"),
)

pytype_strict_binary(
    name = "gen_resource_gather_v1",
    srcs = ["gen_resource_gather_v1.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:session",
        "//tensorflow/python/distribute:input_lib",
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python/distribute:input_lib",
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python/distribute:input_lib",
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:input_lib",
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
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/module",
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
        "//tensorflow/python:dtypes",
        "//tensorflow/python:standard_ops",  # build_cleaner: keep
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/module",
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/framework:test_ops",
        "//tensorflow/python/framework:test_ops_kernels",
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
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
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/module",
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
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tf2",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/module",
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
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:standard_ops",  # build_cleaner: keep
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/module",
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:def_function",
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:input_lib",
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
        "control_flow_v1/saved_model.pb",
        "data/saved_model.pb",
        "dtype_coverage_v1/saved_model.pb",
        "dtype_coverage_v1/variables/variables.data-00000-of-00001",
        "dtype_coverage_v1/variables/variables.index",
        "error_v1/saved_model.pb",
        "hash_table_asset_v1/assets/tokens.txt",
        "hash_table_asset_v1/saved_model.pb",
        "if_v1/saved_model.pb",
        "if_v1/variables/variables.data-00000-of-00001",
        "if_v1/variables/variables.index",
        "pow/saved_model.pb",
        "pow_v2/saved_model.pb",
        "ref_type_tensor_input/saved_model.pb",
        "ref_type_tensor_input/variables/variables.data-00000-of-00001",
        "ref_type_tensor_input/variables/variables.index",
        "resource_gather_v1/saved_model.pb",
        "resource_gather_v1/variables/variables.data-00000-of-00001",
        "resource_gather_v1/variables/variables.index",
        "sparse_tensor_input/saved_model.pb",
        "toy_v1/saved_model.pb",
        "toy_v1/variables/variables.data-00000-of-00001",
        "toy_v1/variables/variables.index",
        "toy_v2/saved_model.pb",
        "toy_v2/variables/variables.data-00000-of-00001",
        "toy_v2/variables/variables.index",
        "variable_on_tpu/saved_model.pb",
        "variable_on_tpu/variables/variables.data-00000-of-00001",
        "variable_on_tpu/variables/variables.index",
        "while_v1/saved_model.pb",
    ],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/tfrt/fallback:cost_recorder",
        "//tensorflow/core/tfrt/run_handler_thread_pool:run_handler_concurrent_work_queue",
        "//tensorflow/core/tfrt/saved_model:saved_model_mira_impl",
        "//tensorflow/core/tfrt/saved_model:saved_model_testutil",
        "//tensorflow/python/framework:test_ops_kernels",
        "//testing/base/public:gunit_for_library_testonly",
        "@tf_runtime//:core_runtime_alwayslink",
    ],
    alwayslink = 1,
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
        "matmul_gpu/saved_model.pb",
        "matmul_gpu/variables/variables.data-00000-of-00001",
        "matmul_gpu/variables/variables.index",
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
