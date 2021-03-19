"""Generates cubin headers for TF dialect ops."""

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_gpu_architectures")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_gpu_architectures",
)
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load(
    "//tensorflow/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")

def if_mlir_generated_gpu_kernels_enabled(if_true, if_false = []):
    return select({
        "//tensorflow/core/kernels/mlir_generated:is_gpu_enabled": if_true,
        "//conditions:default": if_false,
    })

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
            (("cat %s | sed 's/platform/%s/g' | sed 's/_elem_type/_%s/g' | " +
              "sed 's/elem_type/%s/g' | " + "sed 's/_output_type/_%s/g' | " +
              "sed 's/output_type/%s/g' > %s")) % (
                ctx.file.template.path,
                ctx.attr.platform.upper(),
                type_to_tf_dtype[ctx.attr.type],
                mlir_type,
                type_to_tf_dtype[ctx.attr.output_type],
                mlir_output_type,
                ctx.outputs.out.path,
            )
        ),
    )

_gen_mlir_op_rule = rule(
    implementation = _gen_mlir_op_impl,
    output_to_genfiles = True,
    attrs = {
        "template": attr.label(mandatory = True, allow_single_file = True),
        "type": attr.string(mandatory = True),
        "output_type": attr.string(mandatory = True),
        "platform": attr.string(mandatory = True),
        "out": attr.output(mandatory = True),
    },
)

def _gen_mlir_op(op, type, platform, output_type):
    _gen_mlir_op_rule(
        compatible_with = get_compatible_with_cloud(),
        name = "generate_{op}_{platform}_{type}_{output_type}_mlir".format(
            op = op,
            platform = platform,
            type = type,
            output_type = output_type,
        ),
        template = "op_definitions/{op}.mlir.tmpl".format(op = op),
        platform = platform,
        type = type,
        output_type = output_type,
        out = "{op}_{platform}_{type}_{output_type}.mlir".format(
            op = op,
            platform = platform,
            type = type,
            output_type = output_type,
        ),
    )

################################################################################
# Kernels build rules.
################################################################################

def if_mlir_experimental_kernels_enabled(if_true, if_false = []):
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
            "--arch=%s" % arch_flag,
            "--input=%s" % ctx.file.mlir_op.path,
            "--output=%s" % gpu_bin.path,
            "--enable_ftz=%s" % (ctx.attr.data_type == "f32"),
            "--cpu_codegen=%s" % ctx.attr.cpu_codegen,
        ],
        mnemonic = "compile",
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
        "mlir_op": attr.label(mandatory = True, allow_single_file = True),
        "data_type": attr.string(mandatory = True),
        "tile_size": attr.string(mandatory = True),
        "unroll_factors": attr.string(),
        "gpu_archs": attr.string_list(),
        "cpu_codegen": attr.bool(mandatory = False),
        "extra_args": attr.string_list(),
        # cc_binary seems not to bring its dependencies with it, so do that explicitly here.
        "_tfso": attr.label(
            default = Label("//tensorflow:libtensorflow_framework.so.2"),
            cfg = "host",
            allow_single_file = True,
        ),
        "_tool": attr.label(
            executable = True,
            default = Label("//tensorflow/compiler/mlir/tools/kernel_gen:tf_to_kernel"),
            cfg = "host",
        ),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
    },
    fragments = ["cpp"],
    outputs = {"kernel": "%{name}_kernel.o"},
    implementation = _gen_kernel_bin_impl,
    incompatible_use_toolchain_transition = True,
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
)

def _gen_kernel_library(
        name,
        op,
        types,
        platform,
        tile_size,
        output_types = None,
        gpu_archs = [],
        tags = [],
        unroll_factors = None,
        extra_args = []):
    """ Generate a library with GPU or CPU kernels for a specific tensorflow op.

    Args:
      name: The name of the produced library with kernels.
      op: The name of the tensorflow op.
      types: The types ("f16", "f32", "f64") for which a kernel should be generated.
      tile_size: The tiling specification, e.g. "16x16".
      output_types: The output types for which a kernel should be generated. If
                    specified, the i-th entry in types corresponds to the i-th
                    entry in output_types. By default, output_types = types is
                    assumed.
      platform: Platform to compile for, i.e. "gpu" or "cpu"
      gpu_archs: The list of GPU architectures to compile for. If empty, then
                 the compilation will happen for CPU.
      tags: The tags which should be added to the library.
      unroll_factors: The unrolling specification, e.g. "4,4"
      extra_args: Extra arguments to pass to the generator tool.
    """

    enable_cpu = bool(platform == "cpu")
    if not output_types:
        output_types = types

    if cuda_gpu_architectures() or rocm_gpu_architectures() or enable_cpu:
        for (type, output_type) in zip(types, output_types):
            # Disable unrolling for integer types while LLVM does not vectorize these.
            # See b/182343395 for context.
            filtered_unroll_factors = ""
            if type not in ["i1", "i8", "i16", "i32", "i64"]:
                filtered_unroll_factors = unroll_factors
            _gen_mlir_op(
                op = op,
                platform = platform,
                type = type,
                output_type = output_type,
            )
            _gen_kernel_bin_rule(
                name = "{op}_{platform}_{type}_{output_type}_kernel_generator".format(
                    op = op,
                    platform = platform,
                    type = type,
                    output_type = output_type,
                ),
                mlir_op = "{op}_{platform}_{type}_{output_type}.mlir".format(
                    op = op,
                    platform = platform,
                    type = type,
                    output_type = output_type,
                ),
                data_type = type,
                gpu_archs = gpu_archs,
                cpu_codegen = enable_cpu,
                tile_size = tile_size,
                unroll_factors = filtered_unroll_factors,
                extra_args = extra_args,
                compatible_with = get_compatible_with_cloud(),
            )

            # We have to use a sh_test instead of build_test because it doesn't properly find the dependent targets.
            native.sh_test(
                name = "{op}_{platform}_{type}_{output_type}_gen_test".format(
                    op = op,
                    platform = platform,
                    type = type,
                    output_type = output_type,
                ),
                srcs = ["build_test.sh"],
                tags = ["no_rocm"],
                args = [
                    "$(location //tensorflow/compiler/mlir/tools/kernel_gen:tf_to_kernel)",
                    "$(location {op}_{platform}_{type}_{output_type}.mlir)".format(
                        op = op,
                        platform = platform,
                        type = type,
                        output_type = output_type,
                    ),
                    "--cpu_codegen=true" if enable_cpu else "--arch=sm_70,compute_75",
                ],
                size = "medium",
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
        for (type, output_type) in zip(types, output_types)
    ] + ["//tensorflow/compiler/mlir/tools/kernel_gen:tf_framework_c_interface"]

    native.cc_library(
        name = platform + "_" + op + "_kernels",
        deps = kernel_deps if enable_cpu else if_gpu_is_configured(kernel_deps + [
            "//tensorflow/compiler/mlir/tools/kernel_gen:tf_gpu_runtime_wrappers",
        ]),
        linkstatic = 1,
        tags = tags,
        compatible_with = get_compatible_with_cloud(),
    )

def gpu_kernel_library(**kwargs):
    """ Generate a library with GPU kernels for a specific tensorflow op. """
    _gen_kernel_library(
        platform = "gpu",
        gpu_archs = cuda_gpu_architectures() or rocm_gpu_architectures(),
        **kwargs
    )

def cpu_kernel_library(**kwargs):
    """ Generate a library with CPU kernels for a specific tensorflow op. """
    _gen_kernel_library(
        platform = "cpu",
        gpu_archs = [],
        **kwargs
    )
