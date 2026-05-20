"""Contains embed_files build rule."""

load("@bazel_skylib//lib:new_sets.bzl", "sets")
load("@com_google_protobuf//bazel/common:proto_info.bzl", "ProtoInfo")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def _descriptor_set_list(deps, descriptor_set):
    """Makes a list of distinct FileDescriptorSet files."""

    descriptor_set_set = sets.make()
    for dep in deps:
        # Silently drop deps without ProtoInfo.
        if ProtoInfo in dep:
            for descriptor_set in dep[ProtoInfo].transitive_descriptor_sets.to_list():
                sets.insert(descriptor_set_set, descriptor_set)
    if descriptor_set != None:
        sets.insert(descriptor_set_set, descriptor_set)
    return sets.to_list(descriptor_set_set)

def _run_protoc_impl(ctx):
    """Rule to translate text to binary using protocol_compiler."""

    proto_descriptor_sets = _descriptor_set_list(ctx.attr.deps, ctx.file.descriptor_set)

    descriptor_set_in = ("--descriptor_set_in=%s" %
                         ":".join([file.path for file in proto_descriptor_sets]))

    if len(ctx.outputs.outs) != 1:
        fail("Expected exactly one output")
    out = ctx.outputs.outs[0]

    protoc_args = [
        "--encode=%s" % ctx.attr.proto_name,
        "--deterministic_output",
        descriptor_set_in,
    ]
    redirect = [
        "< %s" % ctx.file.src.path,
        "> %s" % out.path,
    ]

    # If command line will be long, use flag file
    if len(descriptor_set_in) > 20000:
        # Unfortunately, we can't use Starlark's flag file support,
        # because we're not using the Args object (because we need
        # to specify redirection using some of the "args")
        flagfile = ctx.actions.declare_file(ctx.attr.name + ".flagfile")
        ctx.actions.write(flagfile, "\n".join(protoc_args))

        ctx.actions.run_shell(
            outputs = ctx.outputs.outs,
            inputs = [ctx.file.src, flagfile] + proto_descriptor_sets,
            tools = [ctx.executable._tool],
            command = " ".join([ctx.executable._tool.path, "@%s" % flagfile.path] + redirect),
            mnemonic = "ProtoDataCompilerFlagfile",
            use_default_shell_env = False,
        )

    else:
        # No flag file necessary
        ctx.actions.run_shell(
            outputs = ctx.outputs.outs,
            inputs = [ctx.file.src] + proto_descriptor_sets,
            tools = [ctx.executable._tool],
            command = " ".join([ctx.executable._tool.path] + protoc_args + redirect),
            mnemonic = "ProtoDataCompiler",
            use_default_shell_env = False,
        )

    return DefaultInfo(runfiles = ctx.runfiles(files = ctx.outputs.outs))

def _run_protoc_rule():
    return rule(
        # Consider removing output_to_genfiles once integrated, is preferred but unclear if it is
        # supported by XLA's use cases.
        output_to_genfiles = True,
        attrs = {
            "src": attr.label(allow_single_file = True, mandatory = True),
            "outs": attr.output_list(mandatory = True),
            "deps": attr.label_list(allow_files = True, default = []),
            "descriptor_set": attr.label(allow_single_file = True),
            "proto_name": attr.string(mandatory = True),
            "_tool": attr.label(
                default = "@com_google_protobuf//:protoc",
                executable = True,
                cfg = "exec",
            ),
        },
        implementation = _run_protoc_impl,
    )

_run_protoc = _run_protoc_rule()

def text_to_binary_proto(
        name,
        src,
        proto_name,
        proto_deps = None,
        out = None,
        descriptor_set = None,
        **kwargs):
    """Converts a protocol buffer in text format into binary format using protoc_minimal.

    This is a stripped-down version of proto_data that only supports protobuf output.
    """

    _run_protoc(
        name = name,
        src = src,
        outs = [out or (name + ".binarypb")],
        deps = proto_deps or [],
        descriptor_set = descriptor_set,
        proto_name = proto_name,
        **kwargs
    )
