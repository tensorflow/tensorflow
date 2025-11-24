load("//tensorflow:strict.default.bzl", "py_strict_binary")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "filegroup", "genrule")
load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_lit_tests",
    data = [":filecheck_test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    tags_override = {
        "test_error_message.lit.pbtxt": ["no_oss"],  # TODO(b/150957738): to be fixed on oss.
    },
    test_file_exts = ["lit.pbtxt"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "filecheck_test_utilities",
    testonly = True,
    srcs = [
        "test_error_message.lit.pbtxt.config.pbtxt",
        "test_error_message.lit.pbtxt.debug.pbtxt",
        "test_error_message.lit.pbtxt.fake_py.debug",
    ],
    data = [
        "//tensorflow/compiler/aot:tfcompile",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

# We disable some tfcompile tests in the open source build with the
# "manual" tag to avoid making our OSS users build LLVM twice
# (once for host and once for target).

test_suite(
    name = "all_tests",
    tags = ["manual"],
    tests = [
        ":test_graph_tfadd_thunks_test",
        ":test_graph_tfadd_with_ckpt_saver_thunks_test",
        ":test_graph_tfadd_with_ckpt_thunks_test",
        ":test_graph_tfassert_eq_thunks_test",
        ":test_graph_tfcond_thunks_test",
        ":test_graph_tffunction_thunks_test",
        ":test_graph_tfgather_thunks_test",
        ":test_graph_tfmatmul_thunks_test",
        ":test_graph_tfmatmulandadd_thunks_test",
        ":test_graph_tfsplits_thunks_test",
        ":test_graph_tftop_k_thunks_test",
        ":test_graph_tfvariable_readonly_thunks_test",
        ":test_graph_tfvariable_sequential_updates_thunks_test",
        ":test_graph_tfvariable_thunks_test",
        ":tfcompile_test_thunks",
    ],
    visibility = ["//visibility:public"],
)

py_strict_binary(
    name = "make_test_graphs",
    testonly = 1,
    srcs = ["make_test_graphs.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:control_flow_util",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/training:saver",
        "@absl_py//absl:app",
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
        "test_graph_tfmatmul_with_constant.pb",
        "test_graph_tfsplits.pb",
        "test_graph_tftop_k.pb",
        "test_graph_tfvariable.pb",
        "test_graph_tfvariable_readonly.pb",
        "test_graph_tfvariable_sequential_updates.pb",
        "test_graph_tfrandom_uniform.pb",
        "test_graph_tfscatter.pb",
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

# This list is used to run tf_compile on the compiled graphs used by
# tfcompile_test. The test is run for several different configurations of
# tf_compile, so this allows the configurations to be specified in one place.
# suffix, mlir_component, xla_flags, tfcompile_flags
tfcompile_test_dep_configs = [
    ("_thunks", "None", "--xla_cpu_use_thunk_runtime=true", ""),
    ("_nanort", "None", "--xla_cpu_use_thunk_runtime=true", "--use_xla_nanort_runtime"),
]

[
    [
        tf_library(
            name = "test_graph_tfadd" + suffix,
            testonly = 1,
            config = "test_graph_tfadd.config.pbtxt",
            cpp_class = "AddComp",
            graph = "test_graph_tfadd.pb",
            # This serves as a test for the list of minimal deps included even when
            # include_standard_runtime_deps is False.  If this target fails to
            # compile but the others in this directory succeed, you may need to
            # expand the "required by all tf_library targets" list in tfcompile.bzl.
            include_standard_runtime_deps = False,
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tfadd_with_ckpt" + suffix,
            testonly = 1,
            config = "test_graph_tfadd_with_ckpt.config.pbtxt",
            cpp_class = "AddWithCkptComp",
            freeze_checkpoint = "test_graph_tfadd_with_ckpt.ckpt",
            graph = "test_graph_tfadd_with_ckpt.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tfadd_with_ckpt_saver" + suffix,
            testonly = 1,
            config = "test_graph_tfadd_with_ckpt.config.pbtxt",
            cpp_class = "AddWithCkptSaverComp",
            freeze_checkpoint = "test_graph_tfadd_with_ckpt_saver.ckpt",
            freeze_saver = "test_graph_tfadd_with_ckpt_saver.saver",
            graph = "test_graph_tfadd_with_ckpt_saver.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tfassert_eq" + suffix,
            testonly = 1,
            config = "test_graph_tfassert_eq.config.pbtxt",
            cpp_class = "AssertComp",
            graph = "test_graph_tfassert_eq.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tfcond" + suffix,
            testonly = 1,
            config = "test_graph_tfcond.config.pbtxt",
            cpp_class = "CondComp",
            graph = "test_graph_tfcond.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tffunction" + suffix,
            testonly = 1,
            config = "test_graph_tffunction.config.pbtxt",
            cpp_class = "FunctionComp",
            graph = "test_graph_tffunction.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tfgather" + suffix,
            testonly = 1,
            config = "test_graph_tfgather.config.pbtxt",
            cpp_class = "GatherComp",
            graph = "test_graph_tfgather.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tfmatmul" + suffix,
            testonly = 1,
            config = "test_graph_tfmatmul.config.pbtxt",
            cpp_class = "foo::bar::MatMulComp",
            graph = "test_graph_tfmatmul.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tfmatmulandadd" + suffix,
            testonly = 1,
            config = "test_graph_tfmatmulandadd.config.pbtxt",
            cpp_class = "::foo::bar::MatMulAndAddComp",
            graph = "test_graph_tfmatmulandadd.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = "--gen_name_to_index --gen_program_shape " + tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tfmatmul_with_constant" + suffix,
            testonly = 1,
            config = "test_graph_tfmatmul_with_constant.config.pbtxt",
            cpp_class = "foo::bar::MatMulWithConstantComp",
            graph = "test_graph_tfmatmul_with_constant.pb",
            include_standard_runtime_deps = True,
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tfsplits" + suffix,
            testonly = 1,
            config = "test_graph_tfsplits.config.pbtxt",
            cpp_class = "SplitsComp",
            graph = "test_graph_tfsplits.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tftop_k" + suffix,
            testonly = 1,
            config = "test_graph_tftop_k.config.pbtxt",
            cpp_class = "TopKComp",
            graph = "test_graph_tftop_k.pb",
            mlir_components = "None",
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tfvariable" + suffix,
            testonly = 1,
            config = "test_graph_tfvariable.config.pbtxt",
            cpp_class = "VariableComp",
            graph = "test_graph_tfvariable.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tfvariable_readonly" + suffix,
            testonly = 1,
            config = "test_graph_tfvariable_readonly.config.pbtxt",
            cpp_class = "VariableReadonlyComp",
            graph = "test_graph_tfvariable_readonly.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tfvariable_sequential_updates" + suffix,
            testonly = 1,
            config = "test_graph_tfvariable_sequential_updates.config.pbtxt",
            cpp_class = "VariableSequentialUpdatesComp",
            graph = "test_graph_tfvariable_sequential_updates.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tfrandom_uniform" + suffix,
            testonly = 1,
            config = "test_graph_tfrandom_uniform.config.pbtxt",
            cpp_class = "RandomUniformComp",
            graph = "test_graph_tfrandom_uniform.pb",
            include_standard_runtime_deps = True,
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
        tf_library(
            name = "test_graph_tfscatter" + suffix,
            testonly = 1,
            config = "test_graph_tfscatter.config.pbtxt",
            cpp_class = "ScatterComp",
            graph = "test_graph_tfscatter.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
            tfcompile_flags = tfcompile_flags,
            xla_flags = xla_flags,
        ),
    ]
    for suffix, mlir_component, xla_flags, tfcompile_flags in tfcompile_test_dep_configs
]

tfcompile_bench_tfmatmul_mkn = [
    # Intentionally empty to avoid running unnecessary tests.
    # Add here your desired (M, K, N) parameters, e.g.
    #    (1, 1, 256),
    #    (1, 2, 256),
]

tfcompile_bench_tfmatmul = [
    (
        "bench_graph_tfmatmul_%sx%sx%s" % (m, k, n),
        "bench_graph_tfmatmul_%sx%sx%s.config.pbtxt" % (m, k, n),
        "bench_graph_tfmatmul.template.pbtxt",
        "-e \"s|<M>|%s|g\" -e \"s|<K>|%s|g\" -e \"s|<N>|%s|g\"" % (m, k, n),
    )
    for (m, k, n) in tfcompile_bench_tfmatmul_mkn
]

test_suite(
    name = "all_tfmatmul_benchmarks",
    tags = ["manual"],
    tests = [
        (":%s_test" % bench_name)
        for (bench_name, _, _, _) in tfcompile_bench_tfmatmul
    ],
    visibility = ["//visibility:public"],
)

[[
    genrule(
        name = "gen_" + config_file,
        testonly = 1,
        srcs = [template_file],
        outs = [config_file],
        cmd = ("sed " + sed_replace + " " +
               "$(location " + template_file + ") " +
               "> $(OUTS)"),
        tags = ["manual"],
    ),
    tf_library(
        name = bench_name,
        testonly = 1,
        config = config_file,
        cpp_class = "foo::bar::MatMulComp",
        graph = "test_graph_tfmatmul.pb",
        tags = [
            "manual",
            "no_mac",  # TODO(b/228273415)
        ],
    ),
] for (bench_name, config_file, template_file, sed_replace) in tfcompile_bench_tfmatmul]

tf_cc_test(
    name = "tfcompile_test_thunks",
    srcs = ["tfcompile_test.cc"],
    extra_copts = ["-DENABLE_XLA_THUNK_TEST"],
    tags = [
        "manual",
        "no_mac",  # TODO(b/228273415)
        "not_run:arm",
    ],
    deps = [
        ":test_graph_tfadd_thunks",
        ":test_graph_tfadd_with_ckpt_saver_thunks",
        ":test_graph_tfadd_with_ckpt_thunks",
        ":test_graph_tfassert_eq_thunks",
        ":test_graph_tfcond_thunks",
        ":test_graph_tffunction_thunks",
        ":test_graph_tfgather_thunks",
        ":test_graph_tfmatmul_thunks",
        ":test_graph_tfmatmul_with_constant_thunks",
        ":test_graph_tfmatmulandadd_thunks",
        ":test_graph_tfrandom_uniform_thunks",
        ":test_graph_tfscatter_thunks",
        ":test_graph_tfsplits_thunks",
        ":test_graph_tftop_k_thunks",
        ":test_graph_tfvariable_readonly_thunks",
        ":test_graph_tfvariable_sequential_updates_thunks",
        ":test_graph_tfvariable_thunks",
        "//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@eigen_archive//:eigen3",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/hlo/testlib:test",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "tfcompile_test_nanort",
    srcs = ["tfcompile_test.cc"],
    extra_copts = ["-DENABLE_XLA_NANORT_TEST"],
    tags = [
        "manual",
        "no_mac",  # TODO(b/228273415)
        "not_run:arm",
    ],
    deps = [
        ":test_graph_tfadd_nanort",
        ":test_graph_tfadd_with_ckpt_nanort",
        ":test_graph_tfadd_with_ckpt_saver_nanort",
        ":test_graph_tfassert_eq_nanort",
        ":test_graph_tfcond_nanort",
        ":test_graph_tffunction_nanort",
        ":test_graph_tfgather_nanort",
        ":test_graph_tfmatmul_nanort",
        ":test_graph_tfmatmul_with_constant_nanort",
        ":test_graph_tfmatmulandadd_nanort",
        ":test_graph_tfrandom_uniform_nanort",
        ":test_graph_tfscatter_nanort",
        ":test_graph_tfsplits_nanort",
        ":test_graph_tftop_k_nanort",
        ":test_graph_tfvariable_nanort",
        ":test_graph_tfvariable_readonly_nanort",
        ":test_graph_tfvariable_sequential_updates_nanort",
        "//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@eigen_archive//:eigen3",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/hlo/testlib:test",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)
