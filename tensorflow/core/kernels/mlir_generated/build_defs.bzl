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
        "//tensorflow/core/kernels/mlir_generated:mlir_generated_gpu_kernels_disabled": if_false,
        "//conditions:default": if_true,
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

def _gen_mlir_op_impl(ctx):
    # Map attr.type to MLIR type.
    mlir_type = ctx.attr.type
    if mlir_type in type_to_mlir:
        mlir_type = type_to_mlir[mlir_type]

    cmd = ctx.actions.run_shell(
        inputs = [ctx.file.template],
        outputs = [ctx.outputs.out],
        command = (
            ("cat %s | sed 's/_elem_type/_%s/g' | sed 's/elem_type/%s/g' > %s") % (
                ctx.file.template.path,
                ctx.attr.type,
                mlir_type,
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
        "out": attr.output(mandatory = True),
    },
)

def _gen_mlir_op(name, type):
    _gen_mlir_op_rule(
        name = "generate_{name}_{type}_mlir".format(name = name, type = type),
        template = "op_definitions/{name}.mlir.tmpl".format(name = name),
        type = type,
        out = "{name}_{type}.mlir".format(name = name, type = type),
    )

################################################################################
# Kernels build rules.
################################################################################

def if_mlir_experimental_kernels_enabled(if_true, if_false = []):
    return select({
        "//tensorflow/core/kernels/mlir_generated:mlir_experimental_kernels_enabled": if_true,
        "//conditions:default": if_false,
    })

def _gen_kernel_fatbin_impl(ctx):
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

_gen_kernel_fatbin_rule = rule(
    attrs = {
        "mlir_op": attr.label(mandatory = True, allow_single_file = True),
        "data_type": attr.string(mandatory = True),
        "tile_size": attr.string(mandatory = True),
        "unroll_factors": attr.string(),
        "gpu_archs": attr.string_list(mandatory = True),
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
    implementation = _gen_kernel_fatbin_impl,
)

def gen_kernel_library(name, types, tile_size, tags = [], unroll_factors = None, extra_args = []):
    """ Generate a library with kernels for a specific tensorflow op.

    Args:
      name: The name of the tensorflow op.
      types: The types ("f16", "f32", "f64") for which a kernel should be generated.
      tile_size: The tiling specification, e.g. "16x16".
      unroll_factors: The unrolling specification, e.g. "4,4"
      tags: The tags which should be added to the library.
      extra_args: Extra arguments to pass to the generator tool.
    """

    if cuda_gpu_architectures() or rocm_gpu_architectures():
        for type in types:
            _gen_mlir_op(
                name = name,
                type = type,
            )
            _gen_kernel_fatbin_rule(
                name = "{name}_{type}_kernel_generator".format(name = name, type = type),
                mlir_op = "{name}_{type}.mlir".format(name = name, type = type),
                data_type = type,
                gpu_archs = rocm_gpu_architectures() + cuda_gpu_architectures(),
                tile_size = tile_size,
                unroll_factors = unroll_factors,
                extra_args = extra_args,
            )

            # We have to use a sh_test instead of build_test because it doesn't properly find the dependent targets.
            native.sh_test(
                name = "{name}_{type}_gen_test".format(name = name, type = type),
                srcs = ["build_test.sh"],
                tags = ["no_rocm"],
                args = [
                    "$(location //tensorflow/compiler/mlir/tools/kernel_gen:tf_to_kernel)",
                    "$(location {name}_{type}.mlir)".format(name = name, type = type),
                ],
                size = "medium",
                data = [
                    ":{name}_{type}.mlir".format(name = name, type = type),
                    "//tensorflow/compiler/mlir/tools/kernel_gen:tf_to_kernel",
                ],
            )

    native.cc_library(
        name = name + "_kernels",
        compatible_with = get_compatible_with_cloud(),
        deps = if_gpu_is_configured([":{name}_{type}_kernel_generator".format(name = name, type = type) for type in types]),
        linkstatic = 1,
        tags = tags,
    )
