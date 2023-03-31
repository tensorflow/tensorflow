load("//tensorflow:tensorflow.default.bzl", "filegroup", "genrule")
load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

glob_lit_tests(
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
    visibility = ["//visibility:public"],
)

py_binary(
    name = "make_test_graphs",
    testonly = 1,
    srcs = ["make_test_graphs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:cond",
        "//tensorflow/python:control_flow_assert",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "@absl_py//absl:app",
        "@six_archive//:six",
    ],
)

genrule(
    name = "gen_test_graphs",
    testonly = 1,
    outs = [
        "test_graph_tfadd.pb",
        "test_debuginfo_tfadd.pb",
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

# This list is used to run tf_compile on the compiled graphs used by
# tfcompile_test. The test is run for several different configurations of
# tf_compile, so this allows the configurations to be specified in one place.
# suffix, mlir_component
tfcompile_test_dep_configs = [
    ("", "None"),
    ("_mlir_bridge", "Bridge"),
    ("_mhlo_lowering", "HloLowering"),
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
            tfcompile_flags = "--gen_name_to_index --gen_program_shape",
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
        ),
        tf_library(
            name = "test_graph_tftop_k" + suffix,
            testonly = 1,
            config = "test_graph_tftop_k.config.pbtxt",
            cpp_class = "TopKComp",
            graph = "test_graph_tftop_k.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
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
        ),
        tf_library(
            name = "test_graph_tfmatmulandadd_with_profiling" + suffix,
            testonly = 1,
            config = "test_graph_tfmatmulandadd.config.pbtxt",
            cpp_class = "MatMulAndAddCompWithProfiling",
            enable_xla_hlo_profiling = True,
            graph = "test_graph_tfmatmulandadd.pb",
            mlir_components = mlir_component,
            tags = [
                "manual",
                "no_mac",  # TODO(b/228273415)
            ],
        ),
    ]
    for suffix, mlir_component in tfcompile_test_dep_configs
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

test_suite(
    name = "all_tfmatmul_mlir_benchmarks",
    tags = ["manual"],
    tests = [
        (":%s_mlir_test" % bench_name)
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
    name = "tfcompile_test",
    srcs = ["tfcompile_test.cc"],
    tags = [
        "manual",
        "no_mac",  # TODO(b/228273415)
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
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:regexp",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "tfcompile_test_mhlo_lowering",
    srcs = ["tfcompile_test.cc"],
    extra_copts = ["-DMHLO_LOWERING_TEST"],
    tags = [
        "manual",
        "no_mac",  # TODO(b/228273415)
    ],
    deps = [
        ":test_graph_tfadd_mhlo_lowering",
        ":test_graph_tfadd_with_ckpt_mhlo_lowering",
        ":test_graph_tfadd_with_ckpt_saver_mhlo_lowering",
        ":test_graph_tfassert_eq_mhlo_lowering",
        ":test_graph_tfcond_mhlo_lowering",
        ":test_graph_tffunction_mhlo_lowering",
        ":test_graph_tfgather_mhlo_lowering",
        ":test_graph_tfmatmul_mhlo_lowering",
        ":test_graph_tfmatmulandadd_mhlo_lowering",
        ":test_graph_tfsplits_mhlo_lowering",
        ":test_graph_tftop_k_mhlo_lowering",
        ":test_graph_tfvariable_mhlo_lowering",
        ":test_graph_tfvariable_readonly_mhlo_lowering",
        ":test_graph_tfvariable_sequential_updates_mhlo_lowering",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_profile_printer",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:regexp",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "tfcompile_test_mlir_bridge",
    srcs = ["tfcompile_test.cc"],
    extra_copts = ["-DENABLE_MLIR_BRIDGE_TEST"],
    tags = [
        "manual",
        "no_mac",  # TODO(b/228273415)
    ],
    deps = [
        ":test_graph_tfadd_mlir_bridge",
        ":test_graph_tfadd_with_ckpt_mlir_bridge",
        ":test_graph_tfadd_with_ckpt_saver_mlir_bridge",
        ":test_graph_tfassert_eq_mlir_bridge",
        ":test_graph_tfcond_mlir_bridge",
        ":test_graph_tffunction_mlir_bridge",
        ":test_graph_tfgather_mlir_bridge",
        ":test_graph_tfmatmul_mlir_bridge",
        ":test_graph_tfmatmulandadd_mlir_bridge",
        ":test_graph_tfmatmulandadd_with_profiling_mlir_bridge",
        ":test_graph_tfsplits_mlir_bridge",
        ":test_graph_tftop_k_mlir_bridge",
        ":test_graph_tfvariable_mlir_bridge",
        ":test_graph_tfvariable_readonly_mlir_bridge",
        ":test_graph_tfvariable_sequential_updates_mlir_bridge",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_profile_printer",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:regexp",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
    ],
)
