"""
A rule to compile a C++ file to a header containing LLVM IR.
"""

load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

visibility(DEFAULT_LOAD_VISIBILITY)

def to_camel_case(s):
    """Converts a snake_case or kebab-case string to CamelCase."""
    s_with_underscores = s.replace("-", "_")
    return "".join([p.capitalize() for p in s_with_underscores.split("_")])

def _cc_ir_header_impl(ctx):
    """Rule implementation that generates IR for multiple features and embeds them in a header."""
    cc_toolchain = find_cc_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        unsupported_features = ctx.disabled_features +
                               [
                                   "thin_lto",
                                   "per_object_debug_info",
                                   "module_maps",
                                   "use_header_modules",
                                   "layering_check",
                                   "parse_headers",
                                   "fdo_optimize",
                                   "fdo_instrument",
                               ],
    )
    compilation_contexts = [dep[CcInfo].compilation_context for dep in ctx.attr.deps]
    output_header = ctx.outputs.out_header
    temp_ir_output = ctx.actions.declare_file(ctx.label.name + ".ll")

    compiler_path = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = "c++-compile",
    )

    compile_variables = cc_common.create_compile_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        user_compile_flags = ctx.fragments.cpp.cxxopts + ctx.fragments.cpp.copts,
    )

    # We use get_memory_inefficient_command_line to get the full command line including
    # toolchain flags, system includes, etc.
    # We need to manually construct the command line because cc_common.compile
    # doesn't allow us to filter out specific flags (like -gsplit-dwarf) that
    # cause issues when the output is not an object file.
    command_line = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = "c++-compile",
        variables = compile_variables,
    )

    allowed_prefixes = [
        "-I",
        "-isystem",
        "-iquote",
        "--sysroot",
        "-std=",
        "-D",
        "-x",
        "-resource-dir",
        "-stdlib=",
    ]

    filtered_args = []
    skip_next = False
    for i, arg in enumerate(command_line):
        if skip_next:
            skip_next = False
            continue

        # Handle flags with arguments
        if arg in ["-isystem", "-iquote", "-I", "--sysroot", "-resource-dir"]:
            filtered_args.append(arg)
            if i + 1 < len(command_line):
                filtered_args.append(command_line[i + 1])
                skip_next = True
            continue

        # Handle joined flags
        allowed = False
        for prefix in allowed_prefixes:
            if arg.startswith(prefix):
                allowed = True
                break

        if allowed:
            filtered_args.append(arg)

    # Libc++ -> Clang -> System
    libcxx_includes = []
    clang_includes = []
    system_includes = []

    for d in cc_toolchain.built_in_include_directories:
        # Heuristic: If this is a system include directory, it might contain libc++ headers in c++/v1.
        if d.endswith("/usr/include"):
            libcxx_path = d + "/c++/v1"
            libcxx_includes.append("-isystem")
            libcxx_includes.append(libcxx_path)

        if "libcxx" in d or "c++" in d or "include/c++/v1" in d:
            libcxx_includes.append("-isystem")
            libcxx_includes.append(d)
        elif "lib/clang" in d or "lib64/clang" in d:
            clang_includes.append("-isystem")
            clang_includes.append(d)
        else:
            system_includes.append("-isystem")
            system_includes.append(d)

    builtin_args = libcxx_includes + clang_includes + system_includes

    dep_args = []
    for dep in ctx.attr.deps:
        compilation_context = dep[CcInfo].compilation_context
        for include in compilation_context.includes.to_list():
            dep_args.append("-I" + include)
        for quote_include in compilation_context.quote_includes.to_list():
            dep_args.append("-iquote" + quote_include)
        for system_include in compilation_context.system_includes.to_list():
            dep_args.append("-isystem" + system_include)
        for define in compilation_context.defines.to_list():
            filtered_args.append("-D" + define)  # Add defines to filtered_args (User flags)

    # Add our specific flags
    extra_flags = [
        "-S",
        "-emit-llvm",
        "-O3",
        "-DNDEBUG",
        "-mprefer-vector-width=512",
        "-DEIGEN_VECTORIZE_GENERIC",
        "-fno-builtin",
        "-c",
        "-o",
        temp_ir_output.path,
        ctx.file.src.path,
    ]

    # Order: User Flags (filtered_args) + Deps (dep_args) + System (builtin_args) + Extra
    # We put filtered_args first to allow user overrides.
    final_args = filtered_args + dep_args + builtin_args + extra_flags

    ctx.actions.run(
        executable = compiler_path,
        arguments = final_args,
        inputs = depset(
            [ctx.file.src],
            transitive = [
                dep[CcInfo].compilation_context.headers
                for dep in ctx.attr.deps
            ] + [cc_toolchain.all_files],
        ),
        outputs = [temp_ir_output],
        mnemonic = "CompileLlvmIr",
        progress_message = "Compiling %s to LLVM IR" % ctx.label.name,
        env = {
            # Ensure we don't pick up any stray environment variables
            "LC_ALL": "C",
        },
    )

    # Generate the final C++ header file.
    # We use a python one-liner to perform a binary-safe copy.
    # 1. 'wb' mode writes the C++ preamble.
    # 2. We read the input as binary ('rb').
    # 3. We check for the UTF-8 BOM (\xef\xbb\xbf) and strip it if present.
    #    NOTE: We use double backslashes (\\x) so Starlark passes the literal characters
    #    to Python, allowing Python to interpret the hex escape sequence.
    # 4. We append the rest of the file and the C++ closing syntax.
    ctx.actions.run_shell(
        inputs = [temp_ir_output],
        outputs = [output_header],
        mnemonic = "EmbeddingLLVMIR",
        command = """
set -e
input_path=$1
output_path=$2
variable_name=$3
namespace=$4

# 1. Write the header start
cat <<EOF > "$output_path"
#pragma once

// Generated by cc_ir_header rule. DO NOT EDIT.

namespace $namespace {

inline constexpr char ${variable_name}[] = R"IR(
EOF

# 2. Append the IR content using Python for binary safety and BOM stripping.
python -c "import sys; d = open(sys.argv[1], 'rb').read(); d = d[3:] if d.startswith(b'\\xef\\xbb\\xbf') else d; open(sys.argv[2], 'ab').write(d)" "$input_path" "$output_path"

# 3. Append the header end
cat <<EOF >> "$output_path"
)IR";

} // namespace $namespace
EOF
""",
        arguments = [
            temp_ir_output.path,
            output_header.path,
            "k{}Ir".format(to_camel_case(ctx.attr.base_name)),
            ctx.attr.namespace,
        ],
    )

    compilation_context = cc_common.create_compilation_context(headers = depset([output_header]))
    cc_info = CcInfo(compilation_context = compilation_context)

    return [DefaultInfo(files = depset([output_header])), cc_info]

_cc_ir_header_rule = rule(
    implementation = _cc_ir_header_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "The C++ source file to compile.",
        ),
        "deps": attr.label_list(providers = [CcInfo]),
        "out_header": attr.output(
            mandatory = True,
            doc = "The output header file.",
        ),
        "base_name": attr.string(
            mandatory = True,
            doc = "The base name of the generated IR variables.",
        ),
        "namespace": attr.string(
            default = "llvm_ir",
            doc = "The C++ namespace for the generated IR variables.",
        ),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
    },
    toolchains = use_cc_toolchain(),
    fragments = ["cpp"],
)

def cc_ir_header(name, src, deps, **kwargs):
    """A macro that generates an IR header and wraps it in a cc_library.

    Args:
      name: The name of the generated cc_library.
      src: The C++ source file to compile.
      deps: The C++ dependencies of the source file.
      **kwargs: Additional arguments to pass to the generated cc_library.
    """
    out_header = name + ".h"
    generator_name = name + "_generator"

    _cc_ir_header_rule(
        base_name = name,
        name = generator_name,
        tags = ["manual"],
        src = src,
        deps = deps,
        out_header = out_header,
        # copybara_removed compatible_with = ["//buildenv/target:non_prod"],
        **kwargs
    )

    cc_library(
        name = name,
        hdrs = [":" + out_header],
        **kwargs
    )
