"""Build rules for ROCm-Device-Libs bitcode libraries."""

load("@bazel_skylib//lib:paths.bzl", "paths")

BitcodeLibraryInfo = provider(fields = ["bc_file"])

def _bitcode_library_impl(ctx):
    """Implements a bitcode library rule."""
    srcs = ctx.files.srcs
    hdrs = ctx.files.hdrs

    bc_outputs = []

    include_dirs = dict([(paths.dirname(h.path), None) for h in ctx.files.hdrs]).keys()

    # Compile .cl files to .bc
    for src in srcs:
        if src.path.endswith(".cl"):
            out = ctx.actions.declare_file(src.basename + ".bc")
            bc_outputs.append(out)

            extra_flags = ctx.attr.file_specific_flags.get(src.basename, "")
            include_flags = ["-I{}".format(dir) for dir in include_dirs]
            include_flags += ["-I{}".format(ctx.files._clang_header[0].dirname)]
            include_flags += ["-I{}".format(ctx.files._clang_includes[0].dirname)]
            args = [
                "-x",
                "cl",
                "--target=amdgcn-amd-amdhsa",
                "-emit-llvm",
                "-fcolor-diagnostics",
                "-Werror",
                "-Wno-error=atomic-alignment",
                "-Xclang",
                "-cl-std=CL2.0",
                "-fvisibility=hidden",
                "-fomit-frame-pointer",
                "-Xclang",
                "-finclude-default-header",
                "-Xclang",
                "-fexperimental-strict-floating-point",
                "-Xclang",
                "-fdenormal-fp-math=dynamic",
                "-Xclang",
                "-Qn",
                "-nogpulib",
                "-cl-no-stdinc",
                "-Xclang",
                "-mcode-object-version=none",
                "-c",
            ] + include_flags + [src.path, "-o", out.path] + extra_flags.split(" ")

            ctx.actions.run(
                executable = ctx.executable._clang,
                inputs = [src] + hdrs + ctx.files._clang_includes + ctx.files._clang_header,
                outputs = [out],
                arguments = args,
                progress_message = "Compiling {} → bitcode".format(src.basename),
                mnemonic = "RocmBitCodeCompile",
            )

        elif src.path.endswith(".ll"):
            # Directly include .ll files in linking
            bc_outputs.append(src)

    # Link all .bc files into one prelinked .bc
    prelink_out = ctx.actions.declare_file(ctx.label.name + ".link0.lib.bc")
    ctx.actions.run(
        executable = ctx.executable._llvm_link,
        inputs = bc_outputs,
        outputs = [prelink_out],
        arguments = [f.path for f in bc_outputs] + ["-o", prelink_out.path],
        progress_message = "Linking {} bitcode files".format(ctx.label.name),
        mnemonic = "RocmBitCodeLink",
    )

    # Internalize symbols (llvm-link + -internalize)
    internalize_out = ctx.actions.declare_file(ctx.label.name + ".lib.bc")
    ctx.actions.run(
        executable = ctx.executable._llvm_link,
        inputs = [prelink_out],
        outputs = [internalize_out],
        arguments = ["-internalize", "-only-needed", prelink_out.path, "-o", internalize_out.path],
        progress_message = "Internalizing symbols for {}".format(ctx.label.name),
        mnemonic = "RocmBitCodeInternalizingSymbols",
    )

    # Strip unnecessary metadata
    strip_out = ctx.actions.declare_file(ctx.label.name + ".strip.bc")
    ctx.actions.run(
        executable = ctx.executable._opt,
        inputs = [internalize_out],
        outputs = [strip_out],
        arguments = ["-passes=strip", "-o", strip_out.path, internalize_out.path],
        progress_message = "Stripping {}".format(ctx.label.name),
        mnemonic = "RocmBitCodeStripping",
    )

    # Final preparation of bitcode (custom prepare_builtins tool)
    final_bc = ctx.actions.declare_file(ctx.label.name + ".bc")
    ctx.actions.run(
        executable = ctx.executable._prepare_builtins,
        inputs = [strip_out],
        outputs = [final_bc],
        arguments = [strip_out.path, "-o", final_bc.path],
        progress_message = "Preparing final bitcode for {}".format(ctx.label.name),
        mnemonic = "RocmBitCodeFinalize",
    )

    return [
        DefaultInfo(files = depset([final_bc])),
        BitcodeLibraryInfo(bc_file = final_bc),
    ]

bitcode_library = rule(
    implementation = _bitcode_library_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = [".cl", ".ll"]),
        "hdrs": attr.label_list(allow_files = [".h"]),
        "file_specific_flags": attr.string_dict(),
        "_clang": attr.label(
            default = Label("@llvm-project//clang:clang"),
            executable = True,
            cfg = "exec",
        ),
        "_llvm_link": attr.label(
            default = Label("@llvm-project//llvm:llvm-link"),
            executable = True,
            cfg = "exec",
        ),
        "_opt": attr.label(
            default = Label("@llvm-project//llvm:opt"),
            executable = True,
            cfg = "exec",
        ),
        "_prepare_builtins": attr.label(
            default = Label(":prepare_builtins"),
            executable = True,
            cfg = "exec",
        ),
        "_clang_includes": attr.label(
            default = Label("@llvm-project//clang:builtin_headers_gen"),
            allow_files = True,
        ),
        "_clang_header": attr.label(
            default = Label("@llvm-project//clang:staging/include/opencl-c.h"),
            allow_files = True,
        ),
    },
)
