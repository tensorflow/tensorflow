"""Build rules for ROCm-Device-Libs bitcode libraries."""

load("@bazel_skylib//lib:paths.bzl", "paths")

def bitcode_library(
        name,
        srcs = [],
        hdrs = [],
        file_specific_flags = {}):
    """Builds a bitcode library

    Args:
        name: Unique name of the build rule.
        srcs: List of source files (*.cl, *.ll).
        hdrs: List of header files (*.h).
        file_specific_flags: Per-file dict of flags to be passed to clang.
    """
    # Takes the CL sources and compiles them into bitcode files.
    # Merges those bitcode files together with any given .ll files into a single bitcode file.
    # Strips unnecessary metadata and forces linkonce local visibility for symbols.
    # Adapted from:
    #   https://github.com/ROCm/llvm-project/blob/22ee53fa53edc3a5f25feb08dc840f5b0fc362da/amd/device-libs/cmake/OCL.cmake#L73

    clang_tool = "@llvm-project//clang:clang"
    llvm_link_tool = "@llvm-project//llvm:llvm-link"
    opt_tool = "@llvm-project//llvm:opt"
    prepare_builtins_tool = ":prepare_builtins"
    clang_includes = "@llvm-project//clang:builtin_headers_gen"

    # Just for calculating the include path.
    clang_header = "@llvm-project//clang:staging/include/opencl-c.h"

    include_paths = dict([(paths.dirname(h), None) for h in hdrs]).keys()

    #TODO(rocm): Maybe compute this in cmd not to pass dirs as srcs
    includes = " ".join(["-I$(location {})".format(inc) for inc in include_paths])
    flags = ("-fcolor-diagnostics -Werror -Wno-error=atomic-alignment -x cl -Xclang " +
             "-cl-std=CL2.0 --target=amdgcn-amd-amdhsa -fvisibility=hidden -fomit-frame-pointer " +
             "-Xclang -finclude-default-header -Xclang -fexperimental-strict-floating-point " +
             "-Xclang -fdenormal-fp-math=dynamic -Xclang -Qn " +
             "-nogpulib -cl-no-stdinc -Xclang -mcode-object-version=none")

    link_inputs = []

    for src in srcs:
        filename = paths.basename(src)
        (basename, _, ext) = filename.partition(".")

        if (ext == "ll"):
            link_inputs.append(src)
            continue

        out = basename + ".bc"
        link_inputs.append(out)
        extra_flags = " ".join(file_specific_flags.get(filename, []))
        native.genrule(
            name = "compile_" + basename,
            srcs = [src] + hdrs + include_paths + [clang_includes, clang_header],
            outs = [out],
            cmd = "$(location {}) -I$$(dirname $(location {}))  {} {} {} -emit-llvm -c $(location {}) -o $@".format(
                clang_tool,
                clang_header,
                includes,
                flags,
                extra_flags,
                src,
            ),
            tools = [clang_tool],
            message = "Compiling {} ...".format(filename),
        )

    link_message = "Linking {}.bc ...".format(name)

    prelink_out = name + ".link0.lib.bc"
    native.genrule(
        name = "prelink_" + name,
        srcs = link_inputs,
        outs = [prelink_out],
        cmd = "$(location {}) $(SRCS) -o $@".format(llvm_link_tool),
        tools = [llvm_link_tool],
        message = link_message,
    )

    internalize_out = name + ".lib.bc"
    native.genrule(
        name = "internalize_" + name,
        srcs = [prelink_out],
        outs = [internalize_out],
        cmd = "$(location {}) -internalize -only-needed $< -o $@".format(llvm_link_tool),
        tools = [llvm_link_tool],
        message = link_message,
    )

    strip_out = name + ".strip.bc"
    native.genrule(
        name = "strip_" + name,
        srcs = [internalize_out],
        outs = [strip_out],
        cmd = "$(location {}) -passes=strip -o $@ $<".format(opt_tool),
        tools = [opt_tool],
        message = link_message,
    )

    native.genrule(
        name = name,
        srcs = [strip_out],
        outs = [name + ".bc"],
        cmd = "$(location {}) -o $@ $<".format(prepare_builtins_tool),
        tools = [prepare_builtins_tool],
        message = link_message,
    )
