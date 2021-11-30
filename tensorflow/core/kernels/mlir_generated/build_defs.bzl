"""Generates cubin headers for TF dialect ops."""

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_gpu_architectures")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_gpu_architectures",
)
load(
    "//tensorflow/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")

def _lookup_file(filegroup, path):
    """Extracts file at (relative) path in filegroup."""
    for file in filegroup.files.to_list():
        if file.path.endswith(path) or file.path.endswith(path + ".exe"):
            return file
    return None

GpuBinaryInfo = provider(
    "GPU binaries in either cubin format or hsaco format",
    fields = ["gpu_bins"],
)

type_to_mlir = {
    "c64": "complex<f32>",
    "c128": "complex<f64>",
}

type_to_tf_dtype = {
    "i1": "DT_BOOL",
    "i8": "DT_INT8",
    "i16": "DT_INT16",
    "i32": "DT_INT32",
    "i64": "DT_INT64",
    "ui8": "DT_UINT8",
    "ui16": "DT_UINT16",
    "ui32": "DT_UINT32",
    "ui64": "DT_UINT64",
    "f16": "DT_HALF",
    "f32": "DT_FLOAT",
    "f64": "DT_DOUBLE",
    "c64": "DT_COMPLEX64",
    "c128": "DT_COMPLEX128",
}

def _get_mlir_type(type):
    """Return the mlir type corresponding to 'type'"""
    if type in type_to_mlir:
        return type_to_mlir[type]
    return type

def _gen_mlir_op_impl(ctx):
    mlir_type = _get_mlir_type(ctx.attr.type)
    mlir_output_type = _get_mlir_type(ctx.attr.output_type)

    cmd = ctx.actions.run_shell(
        inputs = [ctx.file.template],
        outputs = [ctx.outputs.out],
        command = (
            ("cat %s | sed 's/platform/%s/g' | sed 's/_elem_type/_%s/g' | " +
             "sed 's/elem_type/%s/g' | " + "sed 's/_output_type/_%s/g' | " +
             "sed 's/output_type/%s/g' > %s") % (
                ctx.file.template.path,
                ctx.attr.platform.upper(),
                type_to_tf_dtype[ctx.attr.type],
                mlir_type,
                type_to_tf_dtype[ctx.attr.output_type],
                mlir_output_type,
                ctx.outputs.out.path,
            )
        ),
        use_default_shell_env = True,
    )

_gen_mlir_op_rule = rule(
    attrs = {
        "template": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "type": attr.string(mandatory = True),
        "output_type": attr.string(mandatory = True),
        "platform": attr.string(mandatory = True),
        "out": attr.output(mandatory = True),
    },
    output_to_genfiles = True,
    implementation = _gen_mlir_op_impl,
)

def _gen_mlir_op(op, type, platform, output_type):
    _gen_mlir_op_rule(
        name = "generate_{op}_{platform}_{type}_{output_type}_mlir".format(
            op = op,
            platform = platform,
            type = type,
            output_type = output_type,
        ),
        out = "{op}_{platform}_{type}_{output_type}.mlir".format(
            op = op,
            platform = platform,
            type = type,
            output_type = output_type,
        ),
        output_type = output_type,
        platform = platform,
        template = "op_definitions/{op}.mlir.tmpl".format(op = op),
        type = type,
    )

################################################################################
# Kernels build rules.
################################################################################

def if_mlir_generated_gpu_kernels_enabled(if_true, if_false = []):
    return select({
        "//tensorflow/core/kernels/mlir_generated:is_gpu_enabled": if_true,
        "//conditions:default": if_false,
    })

def if_mlir_generated_cpu_kernels_enabled(if_true, if_false = []):
    return select({
        "//tensorflow/core/kernels/mlir_generated:is_cpu_enabled": if_true,
        "//conditions:default": if_false,
    })

def if_mlir_generated_experimental_kernels_enabled(if_true, if_false = []):
    return select({
        "//tensorflow/core/kernels/mlir_generated:is_experimental_enabled": if_true,
        "//conditions:default": if_false,
    })

def _gen_kernel_bin_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    name = ctx.attr.name
    cmd_args = []
    if ctx.attr.unroll_factors:
        cmd_args.append("--unroll_factors=%s" % ctx.attr.unroll_factors)
    if ctx.attr.extra_args:
        cmd_args.extend(ctx.attr.extra_args)
    tile_sizes = ctx.attr.tile_size.replace("x", ",")
    arch_flag = ",".join(ctx.attr.gpu_archs)
    gpu_bin = ctx.outputs.kernel

    # cc_binary seems not to bring its dependencies with it, so do that explicitly here.
    ctx.actions.run(
        inputs = [ctx.file.mlir_op, ctx.file._tfso],
        outputs = [gpu_bin],
        executable = ctx.executable._tool,
        arguments = cmd_args + [
            "--tile_sizes=%s" % tile_sizes,
            "--max-supported-rank=%s" % ctx.attr.max_supported_rank,
            "--arch=%s" % arch_flag,
            "--input=%s" % ctx.file.mlir_op.path,
            "--output=%s" % gpu_bin.path,
            "--enable_ftz=%s" % (ctx.attr.data_type == "f32"),
            "--cpu_codegen=%s" % ctx.attr.cpu_codegen,
            "--jit=%s" % ctx.attr.jit,
        ],
        use_default_shell_env = True,
        mnemonic = "compile",
        progress_message = "Generating kernel '%{label}'",
    )
    compilation_outputs = cc_common.create_compilation_outputs(
        # We always produce PIC object files, so use the same object files for both.
        objects = depset([gpu_bin]),
        pic_objects = depset([gpu_bin]),
    )
    (linking_context, linking_outputs) = cc_common.create_linking_context_from_compilation_outputs(
        name = ctx.label.name,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        compilation_outputs = compilation_outputs,
    )
    return [CcInfo(linking_context = linking_context)]

_gen_kernel_bin_rule = rule(
    attrs = {
        "mlir_op": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "data_type": attr.string(mandatory = True),
        "tile_size": attr.string(mandatory = True),
        "unroll_factors": attr.string(),
        "max_supported_rank": attr.int(),
        "gpu_archs": attr.string_list(),
        "jit": attr.bool(mandatory = False),
        "cpu_codegen": attr.bool(mandatory = False),
        "extra_args": attr.string_list(),
        # cc_binary seems not to bring its dependencies with it, so do that explicitly here.
        "_tfso": attr.label(
            default = Label("//tensorflow:libtensorflow_framework.so.2"),
            cfg = "exec",
            allow_single_file = True,
        ),
        "_tool": attr.label(
            executable = True,
            default = Label("//tensorflow/compiler/mlir/tools/kernel_gen:tf_to_kernel"),
            cfg = "exec",
        ),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
    },
    fragments = ["cpp"],
    outputs = {"kernel": "%{name}_kernel.o"},
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    implementation = _gen_kernel_bin_impl,
)

def _gen_kernel_library(
        name,
        op,
        types,
        platform,
        tile_size,
        max_supported_rank = 5,
        output_types = None,
        jit_types = None,
        output_jit_types = None,
        gpu_archs = [],
        tags = [],
        unroll_factors = None,
        types_with_unrolling_disabled = [],
        extra_args = [],
        test_tags = [],
        test_size = "medium"):
    """ Generate a library with GPU or CPU kernels for a specific tensorflow op.

    Args:
      name: The name of the produced library with kernels.
      op: The name of the tensorflow op.
      types: The types ("f16", "f32", "f64") for which a kernel should be generated.
      tile_size: The tiling specification, e.g. "16x16".
      max_supported_rank: Maximum supported rank for rank specialization.
      jit_types: The types ("f16", "f32", "f64") for which a kernel should be
                 generated. These kernels are different in that they are only
                 partially compiled and will be JIT compiled at execution time.
      output_types: The output types for which a kernel should be generated. If
                    specified, the i-th entry in types corresponds to the i-th
                    entry in types. By default, output_types = types is
                    assumed.
      output_jit_types: The output types for which a jit kernel should be
                        generated. If specified, the i-th entry in types
                        corresponds to the i-th entry in jit_types. By default,
                        output_jit_types = jit_types is assumed.
      platform: Platform to compile for, i.e. "gpu" or "cpu"
      gpu_archs: The list of GPU architectures to compile for. If empty, then
                 the compilation will happen for CPU.
      tags: The tags which should be added to the library.
      unroll_factors: The unrolling specification, e.g. "4,4"
      types_with_unrolling_disabled: The types for which unrolling should be disabled.
      extra_args: Extra arguments to pass to the generator tool.
      test_tags: The tags to pass to the generated test.
      test_size: The "size" argument to pass to the test.
    """

    enable_cpu = bool(platform == "cpu")
    if not output_types:
        output_types = types
    if not jit_types:
        jit_types = []
    if not output_jit_types:
        output_jit_types = jit_types

    true_jits = [True for i in range(len(jit_types))]
    all_jit_kernels = zip(jit_types, output_jit_types, true_jits)
    false_jits = [False for i in range(len(types))]
    all_precomp_kernels = zip(types, output_types, false_jits)
    all_kernels = all_precomp_kernels
    if if_mlir_generated_experimental_kernels_enabled(True, False):
        all_kernels += all_jit_kernels

    if cuda_gpu_architectures() or rocm_gpu_architectures() or enable_cpu:
        for (type, output_type, jit) in all_kernels:
            # Disable unrolling for integer types while LLVM does not vectorize these.
            # See b/182343395 for context.
            unrolling_disabled = (types_with_unrolling_disabled + ["i1", "i8", "i16", "i32", "i64"])
            filtered_unroll_factors = ""
            if type not in unrolling_disabled:
                filtered_unroll_factors = unroll_factors
            _gen_mlir_op(
                op = op,
                output_type = output_type,
                platform = platform,
                type = type,
            )
            _gen_kernel_bin_rule(
                name = "{op}_{platform}_{type}_{output_type}_kernel_generator".format(
                    op = op,
                    platform = platform,
                    type = type,
                    output_type = output_type,
                ),
                cpu_codegen = enable_cpu,
                data_type = type,
                extra_args = extra_args,
                gpu_archs = gpu_archs,
                jit = jit,
                max_supported_rank = max_supported_rank,
                mlir_op = "{op}_{platform}_{type}_{output_type}.mlir".format(
                    op = op,
                    platform = platform,
                    type = type,
                    output_type = output_type,
                ),
                tile_size = tile_size,
                unroll_factors = filtered_unroll_factors,
            )

            # We have to use a sh_test instead of build_test because it doesn't properly find the dependent targets.
            gpu_arch_option = "sm_70,compute_75" if cuda_gpu_architectures() else ",".join(rocm_gpu_architectures())
            test_args = [
                "$(location //tensorflow/compiler/mlir/tools/kernel_gen:tf_to_kernel)",
                "$(location {op}_{platform}_{type}_{output_type}.mlir)".format(
                    op = op,
                    platform = platform,
                    type = type,
                    output_type = output_type,
                ),
                "--cpu_codegen=true" if enable_cpu else "--arch={}".format(gpu_arch_option),
                "--tile_sizes=%s" % tile_size,
                "--enable_ftz=%s" % (type == "f32"),
            ]
            if filtered_unroll_factors:
                test_args.append("--unroll_factors=%s" % filtered_unroll_factors)
            native.sh_test(
                name = "{op}_{platform}_{type}_{output_type}_gen_test".format(
                    op = op,
                    platform = platform,
                    type = type,
                    output_type = output_type,
                ),
                srcs = ["build_test.sh"],
                tags = ["no_rocm"] + test_tags,
                args = test_args,
                size = test_size,
                data = [
                    ":{op}_{platform}_{type}_{output_type}.mlir".format(
                        op = op,
                        platform = platform,
                        type = type,
                        output_type = output_type,
                    ),
                    "//tensorflow/compiler/mlir/tools/kernel_gen:tf_to_kernel",
                ],
            )

    kernel_deps = [
        ":{op}_{platform}_{type}_{output_type}_kernel_generator".format(
            op = op,
            platform = platform,
            type = type,
            output_type = output_type,
        )
        for (type, output_type, jit) in all_kernels
    ] + ["//tensorflow/compiler/mlir/tools/kernel_gen:tf_framework_c_interface"]

    native.cc_library(
        name = name,
        deps = kernel_deps if enable_cpu else if_gpu_is_configured(kernel_deps + [
            "//tensorflow/compiler/mlir/tools/kernel_gen:tf_gpu_runtime_wrappers",
        ]),
        linkstatic = 1,
        tags = tags,
    )

def gpu_kernel_library(name, **kwargs):
    """ Generate a library with GPU kernels for a specific tensorflow op. """
    _gen_kernel_library(
        name = name,
        platform = "gpu",
        gpu_archs = cuda_gpu_architectures() or rocm_gpu_architectures(),
        **kwargs
    )

def cpu_kernel_library(name, **kwargs):
    """ Generate a library with CPU kernels for a specific tensorflow op. """
    _gen_kernel_library(
        name = name,
        platform = "cpu",
        gpu_archs = [],
        **kwargs
    )
