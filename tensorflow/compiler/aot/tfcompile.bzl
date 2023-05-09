"""Build macro that compiles a TensorFlow graph into a cc_library.

To use from your BUILD file, add the following line to load the macro:

load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

Then call the macro like this:

tf_library(
    name = "test_graph_tfmatmul",
    config = "test_graph_tfmatmul.config.pbtxt",
    cpp_class = "MatMulComp",
    graph = ":test_graph_tfmatmul.pb",
)
"""

load(
    "//tensorflow:tensorflow.bzl",
    "if_android",
    "if_google",
    "if_oss",
    "tf_cc_test",
    "tf_copts",
)
load("//tensorflow:tensorflow.default.bzl", "tfcompile_dfsan_abilists", "tfcompile_dfsan_enabled", "tfcompile_target_cpu")

def _tfcompile_model_library_rule_impl(ctx):
    header_file = ctx.outputs.header_out
    metadata_object_file = ctx.actions.declare_file("%s_tfcompile_metadata.o" % ctx.attr.model_name)
    function_object_file = ctx.actions.declare_file("%s_tfcompile_function.o" % ctx.attr.model_name)
    session_module_pb = ctx.actions.declare_file("%s_session_module.pb" % ctx.attr.model_name)
    out_files = [header_file, metadata_object_file, function_object_file, session_module_pb]
    compiler_log_file = None
    if ctx.attr.gen_compiler_log:
        compiler_log_file = ctx.actions.declare_file("%s_compiler.log" % ctx.attr.model_name)
        out_files.append(compiler_log_file)

    output_dict = {}
    output_dict["header_files"] = [header_file]
    output_dict["object_files"] = [metadata_object_file, function_object_file]
    if compiler_log_file:
        output_dict["log_files"] = [compiler_log_file]

    output_flags = [
        "--out_header=" + header_file.path,
        "--out_metadata_object=" + metadata_object_file.path,
        "--out_function_object=" + function_object_file.path,
        "--out_session_module=" + session_module_pb.path,
    ]

    tfcompile_env = {
        "XLA_FLAGS": ("--xla_cpu_enable_fast_math=true " +
                      "--xla_cpu_fast_math_honor_nans=false " +
                      "--xla_cpu_fast_math_honor_infs=false " +
                      "--xla_cpu_fast_math_honor_functions=false " +
                      "--xla_cpu_fast_math_honor_division=false " +
                      "--xla_cpu_enable_fast_min_max=true " +
                      ctx.attr.xla_flags + " " +
                      "$${XLA_FLAGS:-}' "),
        "CUDA_VISIBLE_DEVICES": "",
    }

    dfsan_flags = []
    dfsan_deps = []

    # DFSan is only supported on linux.
    if ctx.attr.is_linux and ctx.attr.dfsan:
        dfsan_flags = [
            "--sanitize_dataflow",
            "--sanitize_abilists_dataflow=" + ",".join([f.path for f in ctx.files.dfsan_abilists]),
        ]
        dfsan_deps = ctx.files.dfsan_abilists

    cpu_flags = ["--target_cpu=" + ctx.attr.target_cpu] if ctx.attr.target_cpu else []

    flags = [
        "--graph=" + ctx.file.tfcompile_graph.path,
        "--config=" + ctx.file.tfcompile_config.path,
        "--entry_point=" + ctx.attr.entry_point,
        "--cpp_class=" + ctx.attr.cpp_class,
        "--target_triple=" + ctx.attr.target_triple,
    ] + cpu_flags + output_flags + ctx.attr.extra_flags + dfsan_flags

    post_command = ""
    if ctx.attr.gen_compiler_log:
        post_command += " --vmodule=cpu_compiler=5 2> >(tee -a " + compiler_log_file.path + " >&2) "

    full_cmd = (
        ctx.executable.tfcompile_tool.path + " " + " ".join(flags) + " " + ctx.attr.flags + post_command
    )
    ctx.actions.run_shell(
        inputs = ctx.files.srcs,
        outputs = out_files,
        tools = [ctx.executable.tfcompile_tool] + dfsan_deps,
        env = tfcompile_env,
        command = full_cmd,
        progress_message = "tfcompile for model %s (%s)" % (ctx.attr.model_name, ctx.file.tfcompile_graph.path),
        mnemonic = "TensorflowCompile",
    )
    return [
        DefaultInfo(
            files = depset(out_files),
        ),
        OutputGroupInfo(**output_dict),
    ]

# Use tf_library macro instead of using this rule directly.
_tfcompile_model_library = rule(
    implementation = _tfcompile_model_library_rule_impl,
    attrs = {
        "model_name": attr.string(),
        "srcs": attr.label_list(mandatory = True, allow_files = True),
        "header_out": attr.output(),
        "cmd": attr.string(),
        "tfcompile_tool": attr.label(cfg = "exec", executable = True, allow_files = True),
        "tfcompile_graph": attr.label(allow_single_file = True),
        "tfcompile_config": attr.label(allow_single_file = True),
        "entry_point": attr.string(),
        "cpp_class": attr.string(),
        "target_triple": attr.string(),
        "target_cpu": attr.string(),
        # The tfcompile_flags passed into tf_library macro may be a string
        # containing multiple flags (and there are cases that do this).
        "flags": attr.string(),
        # Extra flags are built in the tf_library macro as a list.
        "extra_flags": attr.string_list(),
        "dfsan": attr.bool(default = False),
        "dfsan_abilists": attr.label_list(default = [], allow_files = True),
        "is_linux": attr.bool(),
        "gen_compiler_log": attr.bool(),
        "xla_flags": attr.string(),
    },
)

def _tf_library(
        name,
        graph,
        config,
        debug_info = None,
        freeze_checkpoint = None,
        freeze_saver = None,
        cpp_class = None,
        gen_test = True,
        gen_benchmark = True,
        gen_compiler_log = False,
        visibility = None,
        testonly = None,
        tfcompile_flags = None,
        tfcompile_tool = "//tensorflow/compiler/aot:tfcompile",
        include_standard_runtime_deps = True,
        enable_xla_hlo_profiling = False,
        enable_tracemes = False,
        mlir_components = "None",
        deps = None,
        tags = [],
        copts = [],
        xla_flags = None):
    if not cpp_class:
        fail("cpp_class must be specified")

    tfcompile_graph = graph
    if freeze_checkpoint or freeze_saver:
        if not freeze_checkpoint:
            fail("freeze_checkpoint must be specified when freeze_saver is " +
                 "specified")

        freeze_name = "freeze_" + name
        freeze_file = freeze_name + ".pb"

        # First run tfcompile to generate the list of out_nodes.
        #
        # Here and below, we set CUDA_VISIBLE_DEVICES='' to prevent the code we
        # launch from using any GPUs which might be present.  This is important
        # because builds may run concurrently with tests, and tests need to be
        # able to assume that they have control of the full GPU.
        out_nodes_file = "out_nodes_" + freeze_name
        native.genrule(
            name = ("gen_" + out_nodes_file),
            srcs = [config],
            outs = [out_nodes_file],
            cmd = ("CUDA_VISIBLE_DEVICES='' " +
                   "$(location " + tfcompile_tool + ")" +
                   " --config=$(location " + config + ")" +
                   " --dump_fetch_nodes > $@"),
            tools = [tfcompile_tool],
            # Run tfcompile on the build host, rather than forge, since it's
            # typically way faster on the local machine.
            local = 1,
            tags = tags,
        )

        # Now run freeze_graph to convert variables into constants.
        freeze_args = (
            " --input_graph=$(location " + graph + ")" +
            " --checkpoint_version=1" +
            " --input_binary=" + str(not graph.endswith(".pbtxt")) +
            " --input_checkpoint=$(location " + freeze_checkpoint + ")" +
            " --output_graph=$(location " + freeze_file + ")" +
            " --output_node_names=$$(<$(location " + out_nodes_file +
            "))"
        )
        freeze_saver_srcs = []
        if freeze_saver:
            freeze_args += " --input_saver=$(location " + freeze_saver + ")"
            freeze_saver_srcs.append(freeze_saver)
        native.genrule(
            name = freeze_name,
            srcs = [
                graph,
                freeze_checkpoint,
                out_nodes_file,
            ] + freeze_saver_srcs,
            outs = [freeze_file],
            cmd = (
                "CUDA_VISIBLE_DEVICES='' " +
                "$(location " +
                "//tensorflow/python/tools:freeze_graph)" +
                freeze_args
            ),
            tools = ["//tensorflow/python/tools:freeze_graph"],
            tags = tags,
        )
        tfcompile_graph = freeze_file

    # Rule that runs tfcompile to produce the header and object file.
    header_file = name + ".h"

    # The XLA backends morph kernel name prefix __ that is not in the form of
    # __xla_.
    ep = ("__xla_" + native.package_name() + "__" + name).replace("/", "_")
    if type(tfcompile_flags) == type(""):
        flags = tfcompile_flags
    else:
        flags = " ".join([
            "'" + arg.replace("'", "'\\''") + "'"
            for arg in (tfcompile_flags or [])
        ])

    # Do this before we append the `select` into `flags`, because doing so
    # transforms `flags` into a variable of type `select`, and we can't call
    # `find` on such an object.
    need_xla_data_proto = flags and flags.find("--gen_program_shape") != -1

    if enable_xla_hlo_profiling:
        profiling_flags = ["--xla_hlo_profile"]
    else:
        profiling_flags = []

    if enable_tracemes:
        traceme_flags = ["--xla_cpu_enable_xprof_traceme=true"]
    else:
        traceme_flags = ["--xla_cpu_enable_xprof_traceme=false"]

    mlir_flags = ["--mlir_components=" + mlir_components]

    srcs = [tfcompile_graph, config]
    debug_info_flags = []
    if debug_info:
        srcs.append(debug_info)
        debug_info_flags = ["--debug_info=$(location " + debug_info + ")"]

    tfcompile_gen = "gen_" + name
    _tfcompile_model_library(
        name = tfcompile_gen,
        model_name = name,
        srcs = srcs,
        gen_compiler_log = gen_compiler_log,
        header_out = header_file,
        tfcompile_tool = tfcompile_tool,
        tfcompile_graph = tfcompile_graph,
        tfcompile_config = config,
        entry_point = ep,
        cpp_class = cpp_class,
        target_cpu = tfcompile_target_cpu(name),
        target_triple = target_llvm_triple(),
        flags = flags,
        extra_flags = debug_info_flags + profiling_flags + mlir_flags + traceme_flags,
        dfsan = tfcompile_dfsan_enabled(),
        dfsan_abilists = tfcompile_dfsan_abilists(),
        is_linux = select({
            "//tensorflow:linux_x86_64": True,
            "//conditions:default": False,
        }),
        visibility = visibility,
        testonly = testonly,
        tags = tags,
        xla_flags = xla_flags,
    )

    tfcompile_gen_object_files = tfcompile_gen + "_object_files"
    native.filegroup(
        name = tfcompile_gen_object_files,
        srcs = [tfcompile_gen],
        output_group = "object_files",
        visibility = visibility,
        testonly = testonly,
    )

    # The cc_library rule packaging up the header and object file, and needed
    # kernel implementations.
    native.cc_library(
        name = name,
        srcs = [tfcompile_gen_object_files],
        hdrs = [header_file],
        visibility = visibility,
        testonly = testonly,
        deps = [
            # These deps are required by all tf_library targets even if
            # include_standard_runtime_deps is False.  Without them, the
            # generated code will fail to compile.
            "//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
            "//tensorflow/core:framework_lite",
        ] + (need_xla_data_proto and [
            # If we're generating the program shape, we must depend on the
            # proto.
            "//tensorflow/compiler/xla:xla_data_proto_cc",
        ] or []) + (enable_xla_hlo_profiling and [
            "//tensorflow/compiler/xla/service:hlo_profile_printer_data_cc",
        ] or []) + (include_standard_runtime_deps and [
            # TODO(cwhipkey): only depend on kernel code that the model actually
            # needed.
            "//tensorflow/compiler/xla/service/cpu:runtime_conv2d",
            "//tensorflow/compiler/xla/service/cpu:runtime_custom_call_status",
            "//tensorflow/compiler/xla/service/cpu:runtime_key_value_sort",
            "//tensorflow/compiler/xla/service/cpu:runtime_matmul",
            "//tensorflow/compiler/xla/service/cpu:runtime_topk",
            "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_conv2d",
            "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_matmul",
            "//third_party/eigen3",
        ] or []) + (
            mlir_components.count("HloLowering") > 0 and [
                "//tensorflow/compiler/xla/service/cpu:runtime_mlir_utils",
            ] or []
        ) + (
            include_standard_runtime_deps and mlir_components == "HloLowering" and [
                "//tensorflow/compiler/xla/service/cpu/runtime:retain",
            ] or []
        ) + (deps or []),
        tags = tags,
        copts = copts,
    )

    # Variables used for gen_test and gen_benchmark.
    cpp_class_split = cpp_class.rsplit("::", 2)
    if len(cpp_class_split) == 1:
        no_ns_name = cpp_class_split[0]
    else:
        no_ns_name = cpp_class_split[1]
    sed_replace = (
        "-e \"s|{{TFCOMPILE_HEADER}}|$(location " + header_file + ")|g\" " +
        "-e \"s|{{TFCOMPILE_CPP_CLASS}}|" + cpp_class + "|g\" " +
        "-e \"s|{{TFCOMPILE_NAME}}|" + no_ns_name + "|g\" "
    )

    if gen_test:
        test_name = name + "_test"
        test_file = test_name + ".cc"

        template_file = "//tensorflow/compiler/aot:test"
        template_file += if_oss("", "_google") + ".cc"

        # Rule to rewrite the template_file to produce the test_file.
        native.genrule(
            name = ("gen_" + test_name),
            testonly = 1,
            srcs = [
                template_file,
                header_file,
            ],
            outs = [test_file],
            cmd = (
                "sed " + sed_replace +
                " $(location " + template_file + ") " +
                "> $(OUTS)"
            ),
            tags = tags,
        )

        # The cc_test rule for the generated code.  To ensure that this works
        # reliably across build configurations, we must use tf_cc_test instead
        # of native.cc_test.  This is related to how we build
        # //tensorflow/core:lib -- see the note in
        # tensorflow/core/BUILD for more details.
        tf_cc_test(
            name = test_name,
            srcs = [test_file],
            deps = [
                ":" + name,
                "//tensorflow/compiler/aot:tf_library_test_main",
                "//tensorflow/compiler/xla:executable_run_options",
                "//third_party/eigen3",
            ] + if_oss([
                "//tensorflow/core:lib",
                "//tensorflow/core:test",
            ]) + if_google([
                "@com_google_googletest//:gtest",
                "//tensorflow/core/platform:byte_order",
                "//tensorflow/core/platform:platform_port",
            ]),
            tags = tags,
            extra_copts = copts,
            visibility = visibility,
        )

    if gen_benchmark:
        benchmark_name = name + "_benchmark"
        benchmark_file = benchmark_name + ".cc"
        benchmark_main = ("//tensorflow/compiler/aot:" +
                          "benchmark_main.template")

        # Rule to rewrite benchmark.cc to produce the benchmark_file.
        native.genrule(
            name = ("gen_" + benchmark_name),
            srcs = [
                benchmark_main,
                header_file,
            ],
            testonly = testonly,
            outs = [benchmark_file],
            cmd = ("sed " + sed_replace +
                   " $(location " + benchmark_main + ") " +
                   "> $(OUTS)"),
            tags = tags,
        )

        # The cc_benchmark rule for the generated code.  This does not need the
        # tf_cc_binary since we (by deliberate design) do not depend on
        # //tensorflow/core:lib.
        #
        # Note: to get smaller size on android for comparison, compile with:
        #    --copt=-fvisibility=hidden
        #    --copt=-D_LIBCPP_TYPE_VIS=_LIBCPP_HIDDEN
        #    --copt=-D_LIBCPP_EXCEPTION_ABI=_LIBCPP_HIDDEN
        native.cc_binary(
            name = benchmark_name,
            srcs = [benchmark_file],
            testonly = testonly,
            copts = copts + tf_copts(),
            linkopts = if_android(["-pie", "-s"]),
            deps = [
                ":" + name,
                "//tensorflow/compiler/aot:benchmark",
                "//tensorflow/compiler/xla:executable_run_options",
                "//third_party/eigen3",
            ] + if_android([
                "//tensorflow/compiler/aot:benchmark_extra_android",
            ]),
            tags = tags,
            visibility = visibility,
        )

def tf_library(
        name,
        graph,
        config,
        debug_info = None,
        freeze_checkpoint = None,
        freeze_saver = None,
        cpp_class = None,
        gen_test = True,
        gen_benchmark = True,
        gen_compiler_log = False,
        visibility = None,
        testonly = None,
        tfcompile_flags = None,
        tfcompile_tool = "//tensorflow/compiler/aot:tfcompile",
        include_standard_runtime_deps = True,
        enable_xla_hlo_profiling = False,
        enable_tracemes = False,
        mlir_components = "None",
        deps = None,
        tags = [],
        copts = [],
        xla_flags = None):
    """Compiles a TensorFlow graph into an executable with fast math enabled.

    Given an invocation of tf_library(name="foo", ...), generates the following
    build targets:
      foo:           A cc_library containing the generated header and
                      computation.
      foo_test:      A cc_test with simple tests and benchmarks. Only created if
                      gen_test=True.
      foo_benchmark: A cc_binary that runs a minimal-dependency benchmark,
                      useful for mobile devices or other platforms that can't
                      compile the full test libraries. Only created if
                      gen_benchmark=True.
    The output header is called <name>.h.

    Args:
      name: The name of the build rule.
      graph: The TensorFlow GraphDef to compile.  If the file ends in '.pbtxt'
        it is expected to be in the human-readable proto text format, otherwise
        it is expected to be in the proto binary format.
      config: File containing tensorflow.tf2xla.Config proto.  If the file ends
        in '.pbtxt' it is expected to be in the human-readable proto text
        format, otherwise it is expected to be in the proto binary format.
      debug_info: Debug info to include in the output.
      freeze_checkpoint: If provided, run freeze_graph with this checkpoint to
        convert variables into constants.
      freeze_saver: If provided, run freeze_graph with this saver, in SaverDef
        binary form, to convert variables into constants.
      cpp_class: The name of the generated C++ class, wrapping the generated
        function.  The syntax of this flag is
        [[<optional_namespace>::],...]<class_name>.  This mirrors the C++ syntax
        for referring to a class, where multiple namespaces may precede the
        class name, separated by double-colons.  The class will be generated in
        the given namespace(s), or if no namespaces are given, within the global
        namespace.
      gen_test: If True, also generate a cc_test rule that builds a simple
        test and benchmark.
      gen_benchmark: If True, also generate a binary with a simple benchmark.
        Unlike the output of gen_test, this benchmark can be run on android.
      gen_compiler_log: If True, dumps XLA:CPU debug output to a log file.
      visibility: Bazel build visibility.
      testonly:   Bazel testonly attribute.
      tfcompile_flags: Extra flags to pass to tfcompile to control compilation.
      tfcompile_tool: The tfcompile binary. A non-default can be passed to
        use a tfcompile built with extra dependencies.
      include_standard_runtime_deps: If True, the standard list of
        kernel/runtime deps is added to deps.  If False, deps must contain the
        full set of deps needed by the generated library.
      enable_xla_hlo_profiling: Enable XLA HLO profiling in the generated
        program, and emit metadata that lets us pretty-print the gathered
        profile counters.
      enable_tracemes: Tell tfcompile to generate calls to
        TraceMe::Activity{Start|End} around HLO instructions that can be used by
        Xprof to construct profiler timelines.
      mlir_components: When the value is "None", no components use MLIR. When
        the value is "Bridge", use MLIR to translate GraphDef to HLO.
      deps: a list of deps to include on the build rules for the generated
        library, added to the standard deps if standard_runtime_deps is True.
      tags: tags to apply to subsidiary build rules.
      copts: list of copts to pass to cc rules.
    """
    _tf_library(
        name,
        graph,
        config,
        debug_info,
        freeze_checkpoint,
        freeze_saver,
        cpp_class,
        gen_test,
        gen_benchmark,
        gen_compiler_log,
        visibility,
        testonly,
        tfcompile_flags,
        tfcompile_tool,
        include_standard_runtime_deps,
        enable_xla_hlo_profiling,
        enable_tracemes,
        mlir_components,
        deps,
        tags,
        copts,
        xla_flags,
    )
    if mlir_components == "None":
        _tf_library(
            name + "_mlir",
            graph,
            config,
            debug_info,
            freeze_checkpoint,
            freeze_saver,
            cpp_class,
            gen_test,
            gen_benchmark,
            gen_compiler_log,
            visibility,
            testonly,
            tfcompile_flags,
            tfcompile_tool,
            include_standard_runtime_deps,
            enable_xla_hlo_profiling,
            enable_tracemes,
            "HloLowering",
            deps,
            tags + ["notap", "local", "manual"],
            copts,
            xla_flags,
        )

def target_llvm_triple():
    """Returns the target LLVM triple to be used for compiling the target."""

    # TODO(toddw): Add target_triple for other targets.  For details see:
    # http://llvm.org/docs/doxygen/html/Triple_8h_source.html
    return select({
        "//tensorflow:android_armeabi": "armv5-none-android",
        "//tensorflow:android_arm": "armv7-none-android",
        "//tensorflow:android_arm64": "aarch64-none-android",
        "//tensorflow:android_x86": "i686-none-android",
        "//tensorflow:ios": "arm64-none-ios",
        "//tensorflow:ios_x86_64": "x86_64-apple-ios",
        "//tensorflow:linux_ppc64le": "ppc64le-ibm-linux-gnu",
        "//tensorflow:linux_aarch64": "aarch64-none-linux-gnu",
        "//tensorflow:macos_x86_64": "x86_64-none-darwin",
        "//tensorflow:macos_arm64": "aarch64-none-darwin",
        "//tensorflow:windows": "x86_64-none-windows",
        "//tensorflow:linux_s390x": "systemz-none-linux-gnu",
        # internal placeholder,
        "//conditions:default": "x86_64-pc-linux",
    })
