"""BUILD rules for converting proto text files into binary format."""

load("@bazel_skylib//lib:new_sets.bzl", "sets")

def _descriptor_set_list(deps):
    """Makes a list of distinct FileDescriptorSet files of deps's transitive dependencies"""
    descriptor_set_set = sets.make()
    for dep in deps:
        if ProtoInfo in dep:
            for descriptor_set in dep[ProtoInfo].transitive_descriptor_sets.to_list():
                sets.insert(descriptor_set_set, descriptor_set)
    return sets.to_list(descriptor_set_set)

def _proto_data_impl(ctx):
    output = ctx.actions.declare_file(
        ctx.attr.out if ctx.attr.out else "%s.pb" % (ctx.attr.name,),
    )

    args = []
    args.append("--deterministic_output")
    args.append("--encode=%s" % (ctx.attr.proto_name,))

    # Determine all proto descriptor set dependencies, as well as transitive dependencies. Pass
    # them via --descriptor_set_in flag to the protoc tool.
    #
    # If descriptor_set_in exceeds 20000 characters, this implementation will need to be ported to
    # support passing descriptor_set_in as a flagfile argument.
    descriptor_set_in = []
    descriptor_set_list = _descriptor_set_list(ctx.attr.proto_deps)
    for file in ctx.files.proto_deps:
        descriptor_set_in.append(file.path)
    if descriptor_set_list:
        args.append("--descriptor_set_in=%s" % ":".join([d.path for d in descriptor_set_list]))

    redirect = [
        # textproto input is passed via stdin.
        "< '%s\'" % ctx.file.src.path,
        # binaryproto output is emitted via stdout.
        "> '%s'" % output.path,
    ]

    ctx.actions.run_shell(
        outputs = [output],
        inputs = [ctx.file.src] + descriptor_set_list,
        tools = [ctx.executable._tool],
        command = " ".join([ctx.executable._tool.path] + args + redirect),
        use_default_shell_env = False,
    )
    return DefaultInfo(
        files = depset([output]),
        runfiles = ctx.runfiles(files = [output]),
    )

_TOOL = "@com_google_protobuf//:protoc"

# BUILD rule to convert a protocol buffer in text format into the standard binary format.
#
# Args:
#   name: The name of the build target.
#   src: A text formatted protocol buffer.
#   proto_name: The name of the message type in the .proto files that "src" file represents.
#   proto_deps: The list of proto_library targets where "proto" is defined.
#               Transitive dependencies are pulled in automatically.
#   out: (optional) The name of output file. If out is not set then name of
#        output file is name + ".binarypb" extension.
proto_data = rule(
    implementation = _proto_data_impl,
    output_to_genfiles = True,
    attrs = {
        "src": attr.label(
            allow_single_file = True,
        ),
        "proto_name": attr.string(),
        "proto_deps": attr.label_list(
            providers = [ProtoInfo],
        ),
        "out": attr.string(),
        "_tool": attr.label(
            executable = True,
            cfg = "exec",
            allow_files = True,
            default = Label(_TOOL),
        ),
    },
)
