"""
Starlark rule for compiling a C++ file to LLVM IR (.ll) in a hermetic way.
"""

load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def _cc_to_llvm_ir_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)

    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
    )
    dep_compilation_contexts = [
        dep[CcInfo].compilation_context
        for dep in ctx.attr.deps
    ]

    compilation_outputs = cc_common.compile(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        srcs = [ctx.file.src],
        cxx_flags = [
            "-S",
            "-emit-llvm",
        ],
        name = "cc_to_llvm_ir",
        compilation_contexts = dep_compilation_contexts,
    )

    # The compile action produces an output file. Even though we requested
    # LLVM IR, the filename will likely end in ".o" or ".pic.o".
    # We take the first (and only) object file from the outputs.
    intermediate_ll_file = compilation_outputs[1].pic_objects[0]
    final_ll_file = ctx.outputs.out

    ctx.actions.run_shell(
        inputs = [intermediate_ll_file],
        outputs = [final_ll_file],
        command = "cp %s %s" % (intermediate_ll_file.path, final_ll_file.path),
        progress_message = "Copying LLVM IR for %s" % ctx.label.name,
        mnemonic = "CopyLlvmIr",
    )

    return [DefaultInfo(files = depset([final_ll_file]))]

cc_to_llvm_ir = rule(
    implementation = _cc_to_llvm_ir_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "The C++ source file to compile.",
        ),
        "out": attr.output(
            mandatory = True,
            doc = "The output .ll (LLVM IR) file.",
        ),
        "deps": attr.label_list(
            providers = [CcInfo],
            mandatory = True,
            doc = "A cc_library target to provide the C++ compilation context. You MUST specify at least one cc_library dependency here so this rule can leech the C++ context required for compilation.",
        ),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
    },
    # This declares that the rule needs a C++ toolchain to be present.
    toolchains = use_cc_toolchain(),
    doc = "Compiles a single C++ file to LLVM IR (.ll) using the C++ toolchain.",
    fragments = ["cpp"],
)
