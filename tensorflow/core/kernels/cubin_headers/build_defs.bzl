"""Generates cubin headers for TF dialect ops."""

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_gpu_architectures", "if_cuda")

def _lookup_file(filegroup, path):
    """Extracts file at (relative) path in filegroup."""
    for file in filegroup.files.to_list():
        if file.path.endswith(path) or file.path.endswith(path + ".exe"):
            return file
    return None

CubinInfo = provider(fields = ["cubins"])

def _gen_kernel_cubin_impl(ctx):
    name = ctx.attr.name
    tile_sizes = ctx.attr.tile_size.replace("x", ",")
    cmd_args = []
    if ctx.attr.same_shape:
        cmd_args.append("--same_shape=%s" % ctx.attr.same_shape)
    if ctx.attr.unroll_factors:
        cmd_args.append("--unroll_factors=%s" % ctx.attr.unroll_factors)

    cubins = []
    for arch in ctx.attr.gpu_archs:
        # TODO(b/152737872): 'compute_' should generate both SASS and PTX.
        arch = arch.replace("compute_", "sm_")
        filename = "%s.%s.cubin" % (name, arch)
        cubin = ctx.actions.declare_file(filename)
        ctx.actions.run(
            inputs = [ctx.file.mlir_op, ctx.file._tfso],
            outputs = [cubin],
            executable = ctx.executable._tool,
            arguments = cmd_args + [
                "--tile_sizes=%s" % tile_sizes,
                "--arch=%s" % arch.split("_")[1],
                "--input=%s" % ctx.file.mlir_op.path,
                "--output=%s" % cubin.path,
            ],
            mnemonic = "compile",
        )
        cubins.append(cubin)
    return [CubinInfo(cubins = cubins)]

_gen_kernel_cubin_rule = rule(
    implementation = _gen_kernel_cubin_impl,
    attrs = {
        "mlir_op": attr.label(mandatory = True, allow_single_file = True),
        "tile_size": attr.string(mandatory = True),
        "same_shape": attr.string(),
        "unroll_factors": attr.string(),
        "gpu_archs": attr.string_list(mandatory = True),
        "_tfso": attr.label(
            default = Label("//tensorflow:libtensorflow_framework.so.2"),
            cfg = "host",
            allow_single_file = True,
        ),
        "_tool": attr.label(
            executable = True,
            default = Label("//tensorflow/compiler/mlir/tools/kernel_gen:tf_to_cubin"),
            cfg = "host",
        ),
    },
    output_to_genfiles = True,
)

def _gen_kernel_image_hdr_impl(ctx):
    images = []
    for cubin in ctx.attr.input[CubinInfo].cubins:
        arch = cubin.path.split(".")[-2]
        images.append("--image=profile=%s,file=%s" % (arch, cubin.path))

    # Generate fatbin file from all cubins.
    fatbin = ctx.actions.declare_file("%s.fatbin" % ctx.attr.name)
    ctx.actions.run(
        outputs = [fatbin],
        inputs = ctx.attr.input[CubinInfo].cubins,
        executable = _lookup_file(ctx.attr._cuda_root, "bin/fatbinary"),
        arguments = [
            "--64",
            "--cmdline=--compile-only",
            "--link",
            "--compress-all",
            "--create=%s" % fatbin.path,
        ] + images,
        mnemonic = "fatbinary",
    )

    bin2c = _lookup_file(ctx.attr._cuda_root, "bin/bin2c")
    ctx.actions.run_shell(
        outputs = [ctx.outputs.out],
        inputs = [fatbin],
        tools = [bin2c],
        command = "%s --static --const --type=char --name=%s %s 1> %s" %
                  (bin2c.path, ctx.attr.symbol, fatbin.path, ctx.outputs.out.path),
        mnemonic = "bin2c",
    )

_gen_kernel_image_hdr_rule = rule(
    implementation = _gen_kernel_image_hdr_impl,
    output_to_genfiles = True,
    attrs = {
        "input": attr.label(mandatory = True, providers = [CubinInfo]),
        "out": attr.output(mandatory = True),
        "symbol": attr.string(mandatory = True),
        "_cuda_root": attr.label(
            default = Label("@local_config_cuda//cuda:cuda_root"),
        ),
    },
)

def _gen_kernel_image_hdr(name, mlir_op, tile_size, same_shape = None, unroll_factors = None):
    """Generates a C header with fatbin data from a Tensorflow op."""
    if cuda_gpu_architectures():
        _gen_kernel_cubin_rule(
            name = name + "_cubin",
            mlir_op = mlir_op,
            tile_size = tile_size,
            same_shape = same_shape,
            unroll_factors = unroll_factors,
            gpu_archs = cuda_gpu_architectures(),
        )
        _gen_kernel_image_hdr_rule(
            name = name,
            input = ":" + name + "_cubin",
            out = "%s.h" % name,
            symbol = "k%s" % name.replace("_", " ").title().replace(" ", ""),
        )

def _gen_mlir_op_impl(ctx):
    ctx.actions.run_shell(
        inputs = [ctx.file.template],
        outputs = [ctx.outputs.out],
        command = "cat %s | sed s/f99/%s/g > %s" % (
            ctx.file.template.path,
            ctx.attr.type,
            ctx.outputs.out.path,
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
        template = "{name}.mlir.tmpl".format(name = name),
        type = type,
        out = "{name}_{type}.mlir".format(name = name, type = type),
    )

def gen_kernel_library(name, types, tile_size, tags = [], same_shape = None, unroll_factors = None):
    """ Generate a library with kernels for a specific tensorflow op.

    Args:
      name: The name of the tensorflow op.
      types: The types ("f16", "f32", "f64") for which a kernel should be generated.
      tile_size: The tiling specification, e.g. "16x16".
      unroll_factors: The unrolling specification, e.g. "4,4"
      tags: The tags which should be added to the library.
      same_shape: The information about which shapes are the same, e.g. "0,1".
    """

    if cuda_gpu_architectures():
        for type in types:
            _gen_mlir_op(
                name = name,
                type = type,
            )
            _gen_kernel_image_hdr(
                name = "{name}_{type}_kernel".format(name = name, type = type),
                mlir_op = "{name}_{type}.mlir".format(name = name, type = type),
                tile_size = tile_size,
                same_shape = same_shape,
                unroll_factors = unroll_factors,
            )

    native.cc_library(
        name = name + "_kernels",
        hdrs = if_cuda(if_true = [":{name}_{type}_kernel".format(name = name, type = type) for type in types]),
        tags = tags,
    )
