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
            outputs = [cubin],
            executable = ctx.executable._tool,
            arguments = same_shape + [
                "--tile_sizes=%s" % tile_sizes,
                "--arch=%s" % arch.split("_")[1],
                "--output=%s" % cubin.path,
                ctx.attr.op,
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
        "op": attr.string(mandatory = True),
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

def _gen_kernel_image_hdr(name, op, tile_size, tags = [], same_shape = None):
    """Generates a C header with fatbin data from a Tensorflow op."""
    if cuda_gpu_architectures():
        _gen_kernel_image_hdr_rule(
            name = name,
            op = op,
            tile_size = tile_size,
            same_shape = same_shape,
            out = "%s.h" % name,
            symbol = "k%s" % name.replace("_", " ").title().replace(" ", ""),
            gpu_archs = cuda_gpu_architectures(),
            tags = tags,
        )

def gen_kernel_library(name, op, types, tile_size, tags = [], same_shape = None):
    if cuda_gpu_architectures():
        type_to_dtype = {
            "f16": "DT_HALF",
            "f32": "DT_FLOAT",
            "f64": "DT_DOUBLE",
        }
        for type in types:
            _gen_kernel_image_hdr(
                name = "{name}_{type}_kernel".format(name = name, type = type),
                op = op.replace("f99", type).replace("DT_TYPE", type_to_dtype[type]),
                tile_size = tile_size,
                tags = tags,
                same_shape = same_shape,
            )

    native.cc_library(
        name = name + "_kernels",
        hdrs = if_cuda(if_true = [":{name}_{type}_kernel".format(name = name, type = type) for type in types]),
        tags = tags,
    )
