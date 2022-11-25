"""Repository rule for NCCL."""

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts", "cuda_gpu_architectures")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")

# CUDA toolkit version as tuple (e.g. '(11, 1)').
_cuda_version = %{cuda_version}

def _gen_device_srcs_impl(ctx):
    ops = ["sum", "prod", "min", "max", "premulsum", "sumpostdiv"]
    # TF uses CUDA version > 11.0, so enable bf16 type unconditionally.
    types = ["i8", "u8", "i32", "u32", "i64", "u64", "f16", "bf16", "f32", "f64"]
    hdr_tail = "****************************************/"
    defines = "\n\n#define NCCL_OP %d\n#define NCCL_TYPE %d"

    files = []
    for NCCL_OP, op in enumerate(ops):
        for NCCL_TYPE, dt in enumerate(types):
            substitutions = {
                hdr_tail: hdr_tail + defines % (NCCL_OP, NCCL_TYPE),
            }
            for src in ctx.files.srcs:
                name = "%s_%s_%s" % (op, dt, src.basename)
                file = ctx.actions.declare_file(name, sibling = src)
                ctx.actions.expand_template(
                    output = file,
                    template = src,
                    substitutions = substitutions,
                )
                files.append(file)
    return [DefaultInfo(files = depset(files))]

gen_device_srcs = rule(
    implementation = _gen_device_srcs_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
    },
)
"""Adds prefix to each file name in srcs and adds #define NCCL_OP."""

def _rdc_copts():
    """Returns copts for compiling relocatable device code."""

    # The global functions can not have a lower register count than the
    # device functions. This is enforced by setting a fixed register count.
    # https://github.com/NVIDIA/nccl/blob/f93fe9bfd94884cec2ba711897222e0df5569a53/makefiles/common.mk#L48
    maxrregcount = "-maxrregcount=96"

    return cuda_default_copts() + select({
        "@local_config_cuda//:is_cuda_compiler_nvcc": [
            "-nvcc_options",
            "relocatable-device-code=true",
            "-nvcc_options",
            "ptxas-options=" + maxrregcount,
            "-nvcc_options",
            "extended-lambda",
        ],
        "@local_config_cuda//:is_cuda_compiler_clang": [
            "-fcuda-rdc",
            "-Xcuda-ptxas",
            maxrregcount,
        ],
        "//conditions:default": [],
    })

def _lookup_file(filegroup, path):
    """Extracts file at (relative) path in filegroup."""
    for file in filegroup.files:
        if file.path.endswith(path):
            return file
    return None

def _pic_only(files):
    """Returns the PIC files if there are any in 'files', otherwise 'files'."""
    pic_only = [f for f in files if f.basename.find(".pic.") >= 0]
    return pic_only if pic_only else files

def _device_link_impl(ctx):
    if not ctx.attr.gpu_archs:
        fail("No GPU architecture specified. NCCL requires --config=cuda or similar.")

    inputs = []
    for dep in ctx.attr.deps:
        inputs += dep.files.to_list()
    inputs = _pic_only(inputs)

    # Device-link to cubins for each architecture.
    name = ctx.attr.name
    register_h = None
    cubins = []
    images = []
    for arch in ctx.attr.gpu_archs:
        arch = arch.replace("compute_", "sm_")  # PTX is JIT-linked at runtime.
        cubin = ctx.actions.declare_file("%s_%s.cubin" % (name, arch))
        register_h = ctx.actions.declare_file("%s_register_%s.h" % (name, arch))
        ctx.actions.run(
            outputs = [register_h, cubin],
            inputs = inputs,
            executable = ctx.file._nvlink,
            arguments = ctx.attr.nvlink_args + [
                "--arch=%s" % arch,
                "--register-link-binaries=%s" % register_h.path,
                "--output-file=%s" % cubin.path,
            ] + [file.path for file in inputs],
            mnemonic = "nvlink",
            use_default_shell_env = True,
        )
        cubins.append(cubin)
        images.append("--image=profile=%s,file=%s" % (arch, cubin.path))

    # Generate fatbin header from all cubins.
    tmp_fatbin = ctx.actions.declare_file("%s.fatbin" % name)
    fatbin_h = ctx.actions.declare_file("%s_fatbin.h" % name)
    bin2c = ctx.file._bin2c
    arguments_list = [
        "-64",
        "--cmdline=--compile-only",
        "--link",
        "--compress-all",
        "--create=%s" % tmp_fatbin.path,
        "--embedded-fatbin=%s" % fatbin_h.path,
    ]
    if _cuda_version <= (10, 1):
        arguments_list.append("--bin2c-path=%s" % bin2c.dirname)
    ctx.actions.run(
        outputs = [tmp_fatbin, fatbin_h],
        inputs = cubins,
        executable = ctx.file._fatbinary,
        arguments = arguments_list + images,
        tools = [bin2c],
        mnemonic = "fatbinary",
        use_default_shell_env = True,
    )

    # Generate the source file #including the headers generated above.
    ctx.actions.expand_template(
        output = ctx.outputs.out,
        template = ctx.file._link_stub,
        substitutions = {
            "REGISTERLINKBINARYFILE": '"%s"' % register_h.short_path,
            "FATBINFILE": '"%s"' % fatbin_h.short_path,
        },
    )

    return [DefaultInfo(files = depset([register_h, fatbin_h]))]

_device_link = rule(
    implementation = _device_link_impl,
    attrs = {
        "deps": attr.label_list(),
        "out": attr.output(mandatory = True),
        "gpu_archs": attr.string_list(),
        "nvlink_args": attr.string_list(),
        "_nvlink": attr.label(
            default = Label("@local_config_cuda//cuda:cuda/bin/nvlink"),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
        "_fatbinary": attr.label(
            default = Label("@local_config_cuda//cuda:cuda/bin/fatbinary"),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
        "_bin2c": attr.label(
            default = Label("@local_config_cuda//cuda:cuda/bin/bin2c"),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
        "_link_stub": attr.label(
            default = Label("@local_config_cuda//cuda:cuda/bin/crt/link.stub"),
            allow_single_file = True,
        ),
    },
)
"""Links device code and generates source code for kernel registration."""

def _prune_relocatable_code_impl(ctx):
    """Clears __nv_relfatbin section containing relocatable device code."""

    if _cuda_version < (11, 3):
        # -no-relocatable-elf not supported, return unpruned input.
        return ctx.attr.input[DefaultInfo]

    # nvcc --generate-code options for the active set of cuda architectures.
    gencodes = []
    for code in ctx.attr.gpu_archs:
        arch = code.replace("compute_", "sm_")
        if code != arch:
            gencodes.append((arch, arch))
        gencodes.append((arch, code))

    outputs = []
    for input in ctx.files.input:
        output = ctx.actions.declare_file(
            "pruned_" + input.basename,
            sibling = input,
        )
        arguments = (
            ["--generate-code=arch=%s,code=%s" % code for code in gencodes] +
            ["-no-relocatable-elf", "--output-file=%s" % output.path, str(input.path)]
        )
        ctx.actions.run(
            outputs = [output],
            inputs = [input],
            executable = ctx.file._nvprune,
            arguments = arguments,
            mnemonic = "nvprune",
            use_default_shell_env = True,
        )
        outputs.append(output)

    return DefaultInfo(files = depset(outputs))

_prune_relocatable_code = rule(
    implementation = _prune_relocatable_code_impl,
    attrs = {
        "input": attr.label(mandatory = True, allow_files = True),
        "gpu_archs": attr.string_list(),
        "_nvprune": attr.label(
            default = Label("@local_config_cuda//cuda:cuda/bin/nvprune"),
            allow_single_file = True,
            executable = True,
            cfg = "host",
        ),
    },
)

def _merge_archive_impl(ctx):
    # Generate an mri script to the merge archives in srcs and pass it to 'ar'.
    # See https://stackoverflow.com/a/23621751.
    files = _pic_only(ctx.files.srcs)
    mri_script = "create " + ctx.outputs.out.path
    for f in files:
        mri_script += r"\naddlib " + f.path
    mri_script += r"\nsave\nend"

    cc_toolchain = find_cpp_toolchain(ctx)
    ctx.actions.run_shell(
        inputs = ctx.files.srcs,  # + ctx.files._crosstool,
        outputs = [ctx.outputs.out],
        command = "echo -e \"%s\" | %s -M" % (mri_script, cc_toolchain.ar_executable),
        use_default_shell_env = True,
    )

_merge_archive = rule(
    implementation = _merge_archive_impl,
    attrs = {
        "srcs": attr.label_list(mandatory = True, allow_files = True),
        "_cc_toolchain": attr.label(
            default = "@bazel_tools//tools/cpp:current_cc_toolchain",
        ),
        # "_crosstool": attr.label_list(
        #     cfg = "host",
        #     default = ["@bazel_tools//tools/cpp:crosstool"]
        # ),
    },
    outputs = {"out": "lib%{name}.a"},
)
"""Merges srcs into a single archive."""

def cuda_rdc_library(name, hdrs = None, copts = None, linkstatic = True, **kwargs):
    r"""Produces a cuda_library using separate compilation and linking.

    CUDA separate compilation and linking allows device function calls across
    translation units. This is different from the normal whole program
    compilation where each translation unit contains all device code. For more
    background, see
    https://devblogs.nvidia.com/separate-compilation-linking-cuda-device-code/,
    https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-options-for-separate-compilation

    During separate compilation, the different CUDA source files are compiled
    to 'relocatable device code' (RDC) and embedded in the host object files.
    When using nvcc, linking the device code for each supported GPU
    architecture and generating kernel registration code for the CUDA runtime
    is handled automatically. Clang supports generating relocatable device
    code, but it can't link it. We therefore rely on tools provided by the CUDA
    SDK to link the device code and generate the host code to register the
    kernels.

    The nvlink tool extracts the RDC code from the object files and links it
    into cubin files, one per GPU architecture. It also produces a header file
    with a list of kernel names to register. The cubins are merged into a
    binary blob using the fatbinary tool, and converted to a C header file with
    the help of the bin2c tool. The registration header file, the fatbinary
    header file, and the link.stub file (shipped with the CUDA SDK) are
    compiled as ordinary host code.

    Here is a diagram of the CUDA separate compilation trajectory:

     x.cu.cc    y.cu.cc
           \    /            cc_library (compile RDC and archive)
            xy.a
           /    \            * nvlink
    register.h  xy.cubin
          :      |           * fatbinary and bin2c
          :     xy.fatbin.h
          :      :           * #include
          dlink.cc           * Expanded from crt/dlink.stub template
             |               cc_library (host compile and archive)
          dlink.a

    The steps marked with '*' are implemented in the _device_link rule.

    The intermediate relocatable device code in xy.a is no longer needed at
    this point and the corresponding section is replaced with an empty one using
    objcopy. We do not remove the section completely because it is referenced by
    relocations, and removing those as well breaks fatbin registration.

    The object files in both xy.a and dlink.a reference symbols defined in the
    other archive. The separate archives are a side effect of using two
    cc_library targets to implement a single compilation trajectory. We could
    fix this once bazel supports C++ sandwich. For now, we just merge the two
    archives to avoid unresolved symbols:

                    xy.a
                     |         objcopy --update-section __nv_relfatbin=''
    dlink.a     xy_pruned.a
         \           /         merge archive
          xy_merged.a
              |                cc_library (or alternatively, cc_import)
         final target

    Another complication is that cc_library produces (depending on the
    configuration) both PIC and non-PIC archives, but the distinction
    is hidden from Starlark until C++ sandwich becomes available. We work
    around this by dropping the non-PIC files if PIC files are available.

    Args:
      name: Target name.
      hdrs: Header files.
      copts: Compiler options.
      linkstatic: Must be true.
      **kwargs: Any other arguments.
    """

    if not hdrs:
        hdrs = []
    if not copts:
        copts = []

    # Compile host and device code into library.
    lib = name + "_lib"
    native.cc_library(
        name = lib,
        hdrs = hdrs,
        copts = _rdc_copts() + copts,
        linkstatic = linkstatic,
        **kwargs
    )

    # Generate source file containing linked device code.
    dlink_hdrs = name + "_dlink_hdrs"
    dlink_cc = name + "_dlink.cc"
    _device_link(
        name = dlink_hdrs,
        deps = [lib],
        out = dlink_cc,
        gpu_archs = cuda_gpu_architectures(),
        nvlink_args = select({
            "@org_tensorflow//tensorflow:linux_x86_64": ["--cpu-arch=X86_64"],
            "@org_tensorflow//tensorflow:linux_ppc64le": ["--cpu-arch=PPC64LE"],
            "//conditions:default": [],
        }),
    )

    # Compile the source file into a library.
    dlink = name + "_dlink"
    native.cc_library(
        name = dlink,
        srcs = [dlink_cc],
        textual_hdrs = [dlink_hdrs],
        deps = [
            "@local_config_cuda//cuda:cuda_headers",
        ],
        defines = [
            # Silence warning about including internal header.
            "__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__",
            # Macros that need to be defined starting with CUDA 10.
            "__NV_EXTRA_INITIALIZATION=",
            "__NV_EXTRA_FINALIZATION=",
        ],
        linkstatic = linkstatic,
    )

    # Remove intermediate relocatable device code.
    pruned = name + "_pruned"
    _prune_relocatable_code(
        name = pruned,
        input = lib,
        gpu_archs = cuda_gpu_architectures(),
    )

    # Repackage the two libs into a single archive. This is required because
    # both libs reference symbols defined in the other one. For details, see
    # https://eli.thegreenplace.net/2013/07/09/library-order-in-static-linking
    merged = name + "_merged"
    _merge_archive(
        name = merged,
        srcs = [pruned, dlink],
    )

    # Create cc target from archive.
    native.cc_library(
        name = name,
        srcs = [merged],
        hdrs = hdrs,
        linkstatic = linkstatic,
    )
