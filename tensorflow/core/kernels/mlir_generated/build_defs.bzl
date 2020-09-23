"""Generates cubin headers for TF dialect ops."""

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_gpu_architectures")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_gpu_architectures",
    "rocm_is_configured",
)
load(
    "//tensorflow/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)

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

def _gen_kernel_gpu_bin_impl(ctx):
    name = ctx.attr.name
    tile_sizes = ctx.attr.tile_size.replace("x", ",")
    cmd_args = []
    if ctx.attr.same_shape:
        cmd_args.append("--same_shape=%s" % ctx.attr.same_shape)
    if ctx.attr.unroll_factors:
        cmd_args.append("--unroll_factors=%s" % ctx.attr.unroll_factors)

    if ctx.attr.extra_args:
        cmd_args.extend(ctx.attr.extra_args)

    gpu_bins = []
    for arch in ctx.attr.gpu_archs:
        # TODO(b/152737872): 'compute_' should generate both SASS and PTX.
        arch = arch.replace("compute_", "sm_")
        filename = "%s.%s.bin" % (name, arch)
        gpu_bin = ctx.actions.declare_file(filename)
        ctx.actions.run(
            inputs = [ctx.file.mlir_op, ctx.file._tfso],
            outputs = [gpu_bin],
            executable = ctx.executable._tool,
            arguments = cmd_args + [
                "--tile_sizes=%s" % tile_sizes,
                # For ROCM, remove the "gfx" prefix. For CUDA, remove the "sm_" prefix.
                "--arch=%s" % arch[3:],
                "--input=%s" % ctx.file.mlir_op.path,
                "--output=%s" % gpu_bin.path,
            ],
            mnemonic = "compile",
        )
        gpu_bins.append(gpu_bin)
    return [GpuBinaryInfo(gpu_bins = gpu_bins)]

_gen_kernel_gpu_bin_rule = rule(
    attrs = {
        "mlir_op": attr.label(mandatory = True, allow_single_file = True),
        "tile_size": attr.string(mandatory = True),
        "same_shape": attr.string(),
        "unroll_factors": attr.string(),
        "gpu_archs": attr.string_list(mandatory = True),
        "extra_args": attr.string_list(),
        "_tfso": attr.label(
            default = Label("//tensorflow:libtensorflow_framework.so.2"),
            cfg = "host",
            allow_single_file = True,
        ),
        "_tool": attr.label(
            executable = True,
            default = Label("//tensorflow/compiler/mlir/tools/kernel_gen:tf_to_gpu_binary"),
            cfg = "host",
        ),
    },
    output_to_genfiles = True,
    implementation = _gen_kernel_gpu_bin_impl,
)

def _gen_kernel_image_hdr_impl_cuda(ctx):
    images = []
    for cubin in ctx.attr.input[GpuBinaryInfo].gpu_bins:
        arch = cubin.path.split(".")[-2]
        images.append("--image=profile=%s,file=%s" % (arch, cubin.path))

    # Generate fatbin file from all cubins.
    fatbin = ctx.actions.declare_file("%s.fatbin" % ctx.attr.name)
    ctx.actions.run(
        outputs = [fatbin],
        inputs = ctx.attr.input[GpuBinaryInfo].gpu_bins,
        executable = _lookup_file(ctx.attr._gpu_root, "bin/fatbinary"),
        arguments = [
            "--64",
            "--cmdline=--compile-only",
            "--link",
            "--compress-all",
            "--create=%s" % fatbin.path,
        ] + images,
        mnemonic = "fatbinary",
    )

    bin2c = _lookup_file(ctx.attr._gpu_root, "bin/bin2c")
    ctx.actions.run_shell(
        outputs = [ctx.outputs.out],
        inputs = [fatbin],
        tools = [bin2c],
        command = "%s --static --const --type=char --name=%s %s 1> %s" %
                  (bin2c.path, ctx.attr.symbol, fatbin.path, ctx.outputs.out.path),
        mnemonic = "bin2c",
    )

def _gen_kernel_image_hdr_impl_rocm(ctx):
    hsaco_files = []
    hsaco_targets = []

    # Add a dummy host target triple...clang-offload-bundler requires 1 and only 1 host target triple
    hsaco_files.append("/dev/null")
    hsaco_targets.append("host-x86_64-unknown-linux")

    hsacos = ctx.attr.input[GpuBinaryInfo].gpu_bins
    for hsaco in hsacos:
        gfx_arch = hsaco.path.split(".")[-2]
        hsaco_files.append(hsaco.path)
        hsaco_targets.append("hip-amdgcn-amd-amdhsa-%s" % gfx_arch)

    # Generate fatbin file from all hsacos.
    fatbin = ctx.actions.declare_file("%s.fatbin" % ctx.attr.name)
    ctx.actions.run(
        outputs = [fatbin],
        inputs = hsacos,
        executable = _lookup_file(ctx.attr._gpu_root, "bin/clang-offload-bundler"),
        arguments = [
            "--inputs=%s" % ",".join(hsaco_files),
            "--targets=%s" % ",".join(hsaco_targets),
            "--type=o",
            "--outputs=%s" % fatbin.path,
        ],
        mnemonic = "fatbinary",
    )

    ctx.actions.run_shell(
        outputs = [ctx.outputs.out],
        inputs = [fatbin],
        command = (
            ("echo 'static const unsigned char %s[] = {' > %s && " +
             "hexdump -v -e \'/1 \"0x%%02x, \"\' %s | cat >> %s && " +
             "echo '};' >> %s") % (
                ctx.attr.symbol,
                ctx.outputs.out.path,
                fatbin.path,
                ctx.outputs.out.path,
                ctx.outputs.out.path,
            )
        ),
    )

_gen_kernel_image_hdr_rule = rule(
    implementation = _gen_kernel_image_hdr_impl_rocm if rocm_is_configured() else _gen_kernel_image_hdr_impl_cuda,
    output_to_genfiles = True,
    attrs = {
        "input": attr.label(mandatory = True, providers = [GpuBinaryInfo]),
        "out": attr.output(mandatory = True),
        "symbol": attr.string(mandatory = True),
        "_gpu_root": attr.label(
            default = Label("@local_config_rocm//rocm:rocm_root") if rocm_is_configured() else Label("@local_config_cuda//cuda:cuda_root"),
        ),
    },
)

def _gen_kernel_image_hdr(name, mlir_op, gpu_archs, tile_size, same_shape = None, unroll_factors = None, extra_args = []):
    """Generates a C header with fatbin data from a Tensorflow op."""
    _gen_kernel_gpu_bin_rule(
        name = name + "_cubin",
        mlir_op = mlir_op,
        tile_size = tile_size,
        same_shape = same_shape,
        unroll_factors = unroll_factors,
        gpu_archs = gpu_archs,
        extra_args = extra_args,
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
        command = "cat %s | sed s/elem_type/%s/g > %s" % (
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
        template = "op_definitions/{name}.mlir.tmpl".format(name = name),
        type = type,
        out = "{name}_{type}.mlir".format(name = name, type = type),
    )

def gen_kernel_library(name, types, tile_size, tags = [], same_shape = None, unroll_factors = None, extra_args = []):
    """ Generate a library with kernels for a specific tensorflow op.

    Args:
      name: The name of the tensorflow op.
      types: The types ("f16", "f32", "f64") for which a kernel should be generated.
      tile_size: The tiling specification, e.g. "16x16".
      unroll_factors: The unrolling specification, e.g. "4,4"
      tags: The tags which should be added to the library.
      same_shape: The information about which shapes are the same, e.g. "0,1".
      extra_args: Extra arguments to pass to the generator tool.
    """

    if cuda_gpu_architectures() or rocm_gpu_architectures():
        for type in types:
            _gen_mlir_op(
                name = name,
                type = type,
            )
            _gen_kernel_image_hdr(
                name = "{name}_{type}_kernel".format(name = name, type = type),
                mlir_op = "{name}_{type}.mlir".format(name = name, type = type),
                gpu_archs = rocm_gpu_architectures() if rocm_is_configured() else cuda_gpu_architectures(),
                tile_size = tile_size,
                same_shape = same_shape,
                unroll_factors = unroll_factors,
                extra_args = extra_args,
            )

    native.cc_library(
        name = name + "_kernels",
        hdrs = if_gpu_is_configured([":{name}_{type}_kernel".format(name = name, type = type) for type in types]),
        tags = tags,
    )
