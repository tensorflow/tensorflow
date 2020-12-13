"""Generates cubin headers for TF dialect ops."""

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_gpu_architectures")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_gpu_architectures",
    "rocm_is_configured",
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

def _gen_kernel_gpu_bin_impl(ctx):
    name = ctx.attr.name
    tile_sizes = ctx.attr.tile_size.replace("x", ",")
    cmd_args = []
    if ctx.attr.unroll_factors:
        cmd_args.append("--unroll_factors=%s" % ctx.attr.unroll_factors)

    if ctx.attr.extra_args:
        cmd_args.extend(ctx.attr.extra_args)

    gpu_bins = []
    for arch in ctx.attr.gpu_archs:
        # TODO(b/170283783): 'compute_' should generate both SASS and PTX.
        arch = arch.replace("compute_", "sm_")
        filename = "%s.%s.bin" % (name, arch)
        gpu_bin = ctx.actions.declare_file(filename)
        ctx.actions.run(
            inputs = [ctx.file.mlir_op, ctx.file._tfso],
            outputs = [gpu_bin],
            executable = ctx.executable._tool,
            arguments = cmd_args + [
                "--tile_sizes=%s" % tile_sizes,
                "--arch=%s" % arch,
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
            ("hex=`hexdump -v -e \'/1 \"0x%%02x, \"\' %s` && " +
             "len=`echo $hex | wc -c` && " +
             "echo 'static const unsigned char %s['$len' + 1] = {' > %s && " +
             "echo $hex | cat >> %s && " +
             "echo '};' >> %s") % (
                fatbin.path,
                ctx.attr.symbol,
                ctx.outputs.out.path,
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

def _gen_kernel_image_hdr(name, mlir_op, gpu_archs, tile_size, unroll_factors = None, extra_args = []):
    """Generates a C header with fatbin data from a Tensorflow op."""
    _gen_kernel_gpu_bin_rule(
        name = name + "_cubin",
        mlir_op = mlir_op,
        tile_size = tile_size,
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

type_to_mlir = {
    "c64": "complex<f32>",
    "c128": "complex<f64>",
}

def _gen_mlir_op_impl(ctx):
    # Map attr.type to MLIR type.
    mlir_type = ctx.attr.type
    if mlir_type in type_to_mlir:
        mlir_type = type_to_mlir[mlir_type]

    # In order to generate a ranked kernel we change *xelem_type to ?xelem_type
    # and remove element type from the entry function name.
    convert_to_ranked = ""
    if ctx.attr.unranked == False:
        convert_to_ranked = "sed s/*x/?x/g | sed s/_elem_type//g |"
    cmd = ctx.actions.run_shell(
        inputs = [ctx.file.template],
        outputs = [ctx.outputs.out],
        command = (
            ("cat %s | %s sed 's/_elem_type/_%s/g' | sed 's/elem_type/%s/g' > %s") % (
                ctx.file.template.path,
                convert_to_ranked,
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
        "unranked": attr.bool(mandatory = True),
    },
)

def _gen_mlir_op(name, type, unranked):
    tmpl_name = name.replace("_unranked", "") if unranked else name
    _gen_mlir_op_rule(
        name = "generate_{name}_{type}_mlir".format(name = name, type = type),
        template = "op_definitions/{name}.mlir.tmpl".format(name = tmpl_name),
        type = type,
        out = "{name}_{type}.mlir".format(name = name, type = type),
        unranked = unranked,
    )

def gen_ranked_kernel_library(name, types, tile_size, tags = [], unroll_factors = None, extra_args = []):
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
                unranked = False,
            )
            _gen_kernel_image_hdr(
                name = "{name}_{type}_kernel".format(name = name, type = type),
                mlir_op = "{name}_{type}.mlir".format(name = name, type = type),
                gpu_archs = rocm_gpu_architectures() if rocm_is_configured() else cuda_gpu_architectures(),
                tile_size = tile_size,
                unroll_factors = unroll_factors,
                extra_args = extra_args,
            )

    native.cc_library(
        name = name + "_kernels",
        hdrs = if_gpu_is_configured([":{name}_{type}_kernel".format(name = name, type = type) for type in types]),
        tags = tags,
    )

################################################################################
# Unranked kernels build rules.
################################################################################

def if_mlir_unranked_kernels_enabled(if_true, if_false = []):
    return select({
        "//tensorflow/core/kernels/mlir_generated:mlir_use_unranked_kernels": if_true,
        "//conditions:default": if_false,
    })

def _gen_unranked_kernel_fatbin_impl(ctx):
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

_gen_unranked_kernel_fatbin_rule = rule(
    attrs = {
        "mlir_op": attr.label(mandatory = True, allow_single_file = True),
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
    implementation = _gen_unranked_kernel_fatbin_impl,
)

def gen_unranked_kernel_library(name, types, tile_size, tags = [], unroll_factors = None, extra_args = []):
    """ Generate a library with unranked kernels for a specific tensorflow op.

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
                unranked = True,
            )
            _gen_unranked_kernel_fatbin_rule(
                name = "{name}_{type}_kernel_generator".format(name = name, type = type),
                mlir_op = "{name}_{type}.mlir".format(name = name, type = type),
                gpu_archs = rocm_gpu_architectures() if rocm_is_configured() else cuda_gpu_architectures(),
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
                size = "small",
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

def gen_kernel_library(name, types, tile_size, tags = [], unroll_factors = None, extra_args = [], generate_ranked = True, generate_unranked = False):
    if (generate_ranked):
        gen_ranked_kernel_library(
            name = name,
            types = types,
            tile_size = tile_size,
            tags = tags,
            unroll_factors = unroll_factors,
            extra_args = extra_args,
        )
    if (generate_unranked):
        gen_unranked_kernel_library(
            name = name + "_unranked",
            types = types,
            tile_size = tile_size,
            tags = tags,
            unroll_factors = unroll_factors,
            extra_args = extra_args,
        )
