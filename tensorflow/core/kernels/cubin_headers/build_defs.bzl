"""Generates cubin headers for TF dialect ops."""

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_gpu_architectures", "if_cuda")

def _lookup_file(filegroup, path):
    """Extracts file at (relative) path in filegroup."""
    for file in filegroup.files.to_list():
        if file.path.endswith(path):
            return file
    return None

def _gen_kernel_image_hdr_impl(ctx):
    if not ctx.attr.gpu_archs:
        fail("No GPU architecture specified, use --config=cuda or similar")

    name = ctx.attr.name
    tile_sizes = ctx.attr.tile_size.replace("x", ",")
    same_shape = []
    if ctx.attr.same_shape:
        same_shape.append("--same_shape=%s" % ctx.attr.same_shape)

    cubins = []
    images = []
    for arch in ctx.attr.gpu_archs:
        # TODO(b/152737872): 'compute_' should generate both SASS and PTX.
        arch = arch.replace("compute_", "sm_")
        filename = "%s.%s.cubin" % (name, arch)
        cubin = ctx.actions.declare_file(filename)
        ctx.actions.run(
            inputs = [ctx.file.mlir_op],
            outputs = [cubin],
            executable = ctx.executable._tool,
            arguments = same_shape + [
                "--tile_sizes=%s" % tile_sizes,
                "--arch=%s" % arch.split("_")[1],
                "--input=%s" % ctx.file.mlir_op.path,
                "--output=%s" % cubin.path,
            ],
            mnemonic = "compile",
        )
        cubins.append(cubin)
        images.append("--image=profile=%s,file=%s" % (arch, cubin.path))

    # Generate fatbin file from all cubins.
    fatbin = ctx.actions.declare_file("%s.fatbin" % name)
    ctx.actions.run(
        outputs = [fatbin],
        inputs = cubins,
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
        "mlir_op": attr.label(mandatory = True, allow_single_file = True),
        "tile_size": attr.string(mandatory = True),
        "same_shape": attr.string(),
        "out": attr.output(mandatory = True),
        "symbol": attr.string(mandatory = True),
        "gpu_archs": attr.string_list(mandatory = True),
        "_cuda_root": attr.label(
            default = Label("@local_config_cuda//cuda:cuda_root"),
        ),
        "_tool": attr.label(
            executable = True,
            default = Label("//tensorflow/compiler/mlir/tools/kernel_gen:tf_to_cubin"),
            cfg = "host",
        ),
    },
)

def _gen_kernel_image_hdr(name, mlir_op, tile_size, tags = [], same_shape = None):
    """Generates a C header with fatbin data from a Tensorflow op."""
    if cuda_gpu_architectures():
        _gen_kernel_image_hdr_rule(
            name = name,
            mlir_op = mlir_op,
            tile_size = tile_size,
            same_shape = same_shape,
            out = "%s.h" % name,
            symbol = "k%s" % name.replace("_", " ").title().replace(" ", ""),
            gpu_archs = cuda_gpu_architectures(),
            tags = tags,
        )

def _gen_mlir_op_impl(ctx):
    type_to_dtype = {
        "f16": "DT_HALF",
        "f32": "DT_FLOAT",
        "f64": "DT_DOUBLE",
    }
    ctx.actions.run_shell(
        inputs = [ctx.file.template],
        outputs = [ctx.outputs.out],
        command = "cat %s | sed s/f99/%s/g | sed s/DT_DTYPE/%s/g > %s" % (
            ctx.file.template.path,
            ctx.attr.type,
            type_to_dtype[ctx.attr.type],
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

def gen_kernel_library(name, types, tile_size, tags = [], same_shape = None):
    """ Generate a library with kernels for a specific tensorflow op.

    Args:
      name: The name of the tensorflow op.
      types: The types ("f16", "f32", "f64") for which a kernel should be generated.
      tile_size: The tiling specification, e.g. "16x16".
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
                tags = tags,
                same_shape = same_shape,
            )

    native.cc_library(
        name = name + "_kernels",
        hdrs = if_cuda(if_true = [":{name}_{type}_kernel".format(name = name, type = type) for type in types]),
        tags = tags,
    )
