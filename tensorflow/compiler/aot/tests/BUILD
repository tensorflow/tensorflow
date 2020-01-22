load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

# We disable some tfcompile tests in the open source build with the
# "manual" tag to avoid making our OSS users build LLVM twice
# (once for host and once for target).

test_suite(
    name = "all_tests",
    tags = ["manual"],
    tests = [
        ":test_graph_tfadd_test",
        ":test_graph_tfadd_with_ckpt_saver_test",
        ":test_graph_tfadd_with_ckpt_test",
        ":test_graph_tfassert_eq_test",
        ":test_graph_tfcond_test",
        ":test_graph_tffunction_test",
        ":test_graph_tfgather_test",
        ":test_graph_tfmatmul_test",
        ":test_graph_tfmatmulandadd_test",
        ":test_graph_tfsplits_test",
        ":test_graph_tftop_k_test",
        ":test_graph_tfvariable_readonly_test",
        ":test_graph_tfvariable_sequential_updates_test",
        ":test_graph_tfvariable_test",
        ":tfcompile_test",
    ],
)

py_binary(
    name = "make_test_graphs",
    testonly = 1,
    srcs = ["make_test_graphs.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "@six_archive//:six",
    ],
)

genrule(
    name = "gen_test_graphs",
    testonly = 1,
    outs = [
        "test_graph_tfadd.pb",
        "test_graph_tfadd_with_ckpt.ckpt",
        "test_graph_tfadd_with_ckpt.pb",
        "test_graph_tfadd_with_ckpt_saver.ckpt",
        "test_graph_tfadd_with_ckpt_saver.pb",
        "test_graph_tfadd_with_ckpt_saver.saver",
        "test_graph_tfassert_eq.pb",
        "test_graph_tfcond.pb",
        "test_graph_tffunction.pb",
        "test_graph_tfgather.pb",
        "test_graph_tfmatmul.pb",
        "test_graph_tfmatmulandadd.pb",
        "test_graph_tfsplits.pb",
        "test_graph_tftop_k.pb",
        "test_graph_tfvariable.pb",
        "test_graph_tfvariable_readonly.pb",
        "test_graph_tfvariable_sequential_updates.pb",
    ],
    # Set CUDA_VISIBLE_DEVICES='' to prevent the code we launch from using any
    # GPUs which might be present.  This is important because builds may run
    # concurrently with tests, and tests need to be able to assume that they
    # have control of the full GPU.
    cmd = "CUDA_VISIBLE_DEVICES='' " +
          "$(location :make_test_graphs) --out_dir $(@D)",
    tags = ["manual"],
    tools = [":make_test_graphs"],
)

tf_library(
    name = "test_graph_tfadd",
    testonly = 1,
    config = "test_graph_tfadd.config.pbtxt",
    cpp_class = "AddComp",
    graph = "test_graph_tfadd.pb",
    # This serves as a test for the list of minimal deps included even when
    # include_standard_runtime_deps is False.  If this target fails to
    # compile but the others in this directory succeed, you may need to
    # expand the "required by all tf_library targets" list in tfcompile.bzl.
    include_standard_runtime_deps = False,
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfadd_with_ckpt",
    testonly = 1,
    config = "test_graph_tfadd_with_ckpt.config.pbtxt",
    cpp_class = "AddWithCkptComp",
    freeze_checkpoint = "test_graph_tfadd_with_ckpt.ckpt",
    graph = "test_graph_tfadd_with_ckpt.pb",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfadd_with_ckpt_saver",
    testonly = 1,
    config = "test_graph_tfadd_with_ckpt.config.pbtxt",
    cpp_class = "AddWithCkptSaverComp",
    freeze_checkpoint = "test_graph_tfadd_with_ckpt_saver.ckpt",
    freeze_saver = "test_graph_tfadd_with_ckpt_saver.saver",
    graph = "test_graph_tfadd_with_ckpt_saver.pb",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfassert_eq",
    testonly = 1,
    config = "test_graph_tfassert_eq.config.pbtxt",
    cpp_class = "AssertComp",
    graph = "test_graph_tfassert_eq.pb",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfcond",
    testonly = 1,
    config = "test_graph_tfcond.config.pbtxt",
    cpp_class = "CondComp",
    graph = "test_graph_tfcond.pb",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tffunction",
    testonly = 1,
    config = "test_graph_tffunction.config.pbtxt",
    cpp_class = "FunctionComp",
    graph = "test_graph_tffunction.pb",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfgather",
    testonly = 1,
    config = "test_graph_tfgather.config.pbtxt",
    cpp_class = "GatherComp",
    graph = "test_graph_tfgather.pb",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfmatmul",
    testonly = 1,
    config = "test_graph_tfmatmul.config.pbtxt",
    cpp_class = "foo::bar::MatMulComp",
    graph = "test_graph_tfmatmul.pb",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfmatmulandadd",
    testonly = 1,
    config = "test_graph_tfmatmulandadd.config.pbtxt",
    cpp_class = "::foo::bar::MatMulAndAddComp",
    graph = "test_graph_tfmatmulandadd.pb",
    tags = [
        "manual",
    ],
    tfcompile_flags = "--gen_name_to_index --gen_program_shape",
)

tf_library(
    name = "test_graph_tfmatmulandadd_with_profiling",
    testonly = 1,
    config = "test_graph_tfmatmulandadd.config.pbtxt",
    cpp_class = "MatMulAndAddCompWithProfiling",
    enable_xla_hlo_profiling = True,
    graph = "test_graph_tfmatmulandadd.pb",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfsplits",
    testonly = 1,
    config = "test_graph_tfsplits.config.pbtxt",
    cpp_class = "SplitsComp",
    graph = "test_graph_tfsplits.pb",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tftop_k",
    testonly = 1,
    config = "test_graph_tftop_k.config.pbtxt",
    cpp_class = "TopKComp",
    graph = "test_graph_tftop_k.pb",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfvariable",
    testonly = 1,
    config = "test_graph_tfvariable.config.pbtxt",
    cpp_class = "VariableComp",
    graph = "test_graph_tfvariable.pb",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfvariable_readonly",
    testonly = 1,
    config = "test_graph_tfvariable_readonly.config.pbtxt",
    cpp_class = "VariableReadonlyComp",
    graph = "test_graph_tfvariable_readonly.pb",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfvariable_sequential_updates",
    testonly = 1,
    config = "test_graph_tfvariable_sequential_updates.config.pbtxt",
    cpp_class = "VariableSequentialUpdatesComp",
    graph = "test_graph_tfvariable_sequential_updates.pb",
    tags = [
        "manual",
    ],
)

tf_cc_test(
    name = "tfcompile_test",
    srcs = ["tfcompile_test.cc"],
    tags = [
        "manual",
    ],
    deps = [
        ":test_graph_tfadd",
        ":test_graph_tfadd_with_ckpt",
        ":test_graph_tfadd_with_ckpt_saver",
        ":test_graph_tfassert_eq",
        ":test_graph_tfcond",
        ":test_graph_tffunction",
        ":test_graph_tfgather",
        ":test_graph_tfmatmul",
        ":test_graph_tfmatmulandadd",
        ":test_graph_tfmatmulandadd_with_profiling",
        ":test_graph_tfsplits",
        ":test_graph_tftop_k",
        ":test_graph_tfvariable",
        ":test_graph_tfvariable_readonly",
        ":test_graph_tfvariable_sequential_updates",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_profile_printer",
        "//tensorflow/core:lib",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
    ],
)

tf_library(
    name = "test_graph_tfadd_mlir_bridge",
    testonly = 1,
    config = "test_graph_tfadd.config.pbtxt",
    cpp_class = "AddComp",
    graph = "test_graph_tfadd.pb",
    include_standard_runtime_deps = False,
    mlir_components = "Bridge",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfadd_with_ckpt_mlir_bridge",
    testonly = 1,
    config = "test_graph_tfadd_with_ckpt.config.pbtxt",
    cpp_class = "AddWithCkptComp",
    freeze_checkpoint = "test_graph_tfadd_with_ckpt.ckpt",
    graph = "test_graph_tfadd_with_ckpt.pb",
    mlir_components = "Bridge",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfadd_with_ckpt_saver_mlir_bridge",
    testonly = 1,
    config = "test_graph_tfadd_with_ckpt.config.pbtxt",
    cpp_class = "AddWithCkptSaverComp",
    freeze_checkpoint = "test_graph_tfadd_with_ckpt_saver.ckpt",
    freeze_saver = "test_graph_tfadd_with_ckpt_saver.saver",
    graph = "test_graph_tfadd_with_ckpt_saver.pb",
    mlir_components = "Bridge",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfcond_mlir_bridge",
    testonly = 1,
    config = "test_graph_tfcond.config.pbtxt",
    cpp_class = "CondComp",
    graph = "test_graph_tfcond.pb",
    mlir_components = "Bridge",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfassert_eq_mlir_bridge",
    testonly = 1,
    config = "test_graph_tfassert_eq.config.pbtxt",
    cpp_class = "AssertComp",
    graph = "test_graph_tfassert_eq.pb",
    mlir_components = "Bridge",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfgather_mlir_bridge",
    testonly = 1,
    config = "test_graph_tfgather.config.pbtxt",
    cpp_class = "GatherComp",
    graph = "test_graph_tfgather.pb",
    mlir_components = "Bridge",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfmatmul_mlir_bridge",
    testonly = 1,
    config = "test_graph_tfmatmul.config.pbtxt",
    cpp_class = "foo::bar::MatMulComp",
    graph = "test_graph_tfmatmul.pb",
    mlir_components = "Bridge",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfmatmulandadd_mlir_bridge",
    testonly = 1,
    config = "test_graph_tfmatmulandadd.config.pbtxt",
    cpp_class = "::foo::bar::MatMulAndAddComp",
    graph = "test_graph_tfmatmulandadd.pb",
    mlir_components = "Bridge",
    tags = [
        "manual",
    ],
    tfcompile_flags = "--gen_name_to_index --gen_program_shape",
)

tf_library(
    name = "test_graph_tfmatmulandadd_with_profiling_mlir_bridge",
    testonly = 1,
    config = "test_graph_tfmatmulandadd.config.pbtxt",
    cpp_class = "MatMulAndAddCompWithProfiling",
    enable_xla_hlo_profiling = True,
    graph = "test_graph_tfmatmulandadd.pb",
    mlir_components = "Bridge",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfsplits_mlir_bridge",
    testonly = 1,
    config = "test_graph_tfsplits.config.pbtxt",
    cpp_class = "SplitsComp",
    graph = "test_graph_tfsplits.pb",
    mlir_components = "Bridge",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tftop_k_mlir_bridge",
    testonly = 1,
    config = "test_graph_tftop_k.config.pbtxt",
    cpp_class = "TopKComp",
    graph = "test_graph_tftop_k.pb",
    mlir_components = "Bridge",
    tags = [
        "manual",
    ],
)

tf_library(
    name = "test_graph_tfvariable_readonly_mlir_bridge",
    testonly = 1,
    config = "test_graph_tfvariable_readonly.config.pbtxt",
    cpp_class = "VariableReadonlyComp",
    graph = "test_graph_tfvariable_readonly.pb",
    mlir_components = "Bridge",
    tags = [
        "manual",
    ],
)

tf_cc_test(
    name = "tfcompile_test_mlir_bridge",
    srcs = ["tfcompile_test.cc"],
    extra_copts = ["-DENABLE_MLIR_BRIDGE_TEST"],
    tags = [
        "manual",
    ],
    deps = [
        ":test_graph_tfadd_mlir_bridge",
        ":test_graph_tfadd_with_ckpt_mlir_bridge",
        ":test_graph_tfadd_with_ckpt_saver_mlir_bridge",
        ":test_graph_tfassert_eq_mlir_bridge",
        ":test_graph_tfcond_mlir_bridge",
        ":test_graph_tfgather_mlir_bridge",
        ":test_graph_tfmatmul_mlir_bridge",
        ":test_graph_tfmatmulandadd_mlir_bridge",
        ":test_graph_tfmatmulandadd_with_profiling_mlir_bridge",
        ":test_graph_tfsplits_mlir_bridge",
        ":test_graph_tftop_k_mlir_bridge",
        ":test_graph_tfvariable_readonly_mlir_bridge",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_profile_printer",
        "//tensorflow/core:lib",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
    ],
)
