"""Generates C++ grpc stubs from proto_library rules.

This is an internal rule used by cc_grpc_library, and shouldn't be used
directly.
"""

load(
    "@com_github_grpc_grpc//bazel:protobuf.bzl",
    "get_include_directory",
    "get_plugin_args",
    "get_proto_root",
    "proto_path_to_generated_filename",
)

_GRPC_PROTO_HEADER_FMT = "{}.grpc.pb.h"
_GRPC_PROTO_SRC_FMT = "{}.grpc.pb.cc"
_GRPC_PROTO_MOCK_HEADER_FMT = "{}_mock.grpc.pb.h"
_PROTO_HEADER_FMT = "{}.pb.h"
_PROTO_SRC_FMT = "{}.pb.cc"

def _strip_package_from_path(label_package, file):
    prefix_len = 0
    if not file.is_source and file.path.startswith(file.root.path):
        prefix_len = len(file.root.path) + 1

    path = file.path
    if len(label_package) == 0:
        return path
    if not path.startswith(label_package + "/", prefix_len):
        fail("'{}' does not lie within '{}'.".format(path, label_package))
    return path[prefix_len + len(label_package + "/"):]

def _get_srcs_file_path(file):
    if not file.is_source and file.path.startswith(file.root.path):
        return file.path[len(file.root.path) + 1:]
    return file.path

def _join_directories(directories):
    massaged_directories = [directory for directory in directories if len(directory) != 0]
    return "/".join(massaged_directories)

def generate_cc_impl(ctx):
    """Implementation of the generate_cc rule."""
    protos = [f for src in ctx.attr.srcs for f in src[ProtoInfo].check_deps_sources.to_list()]
    includes = [
        f
        for src in ctx.attr.srcs
        for f in src[ProtoInfo].transitive_imports.to_list()
    ]
    outs = []
    proto_root = get_proto_root(
        ctx.label.workspace_root,
    )

    label_package = _join_directories([ctx.label.workspace_root, ctx.label.package])
    if ctx.executable.plugin:
        outs += [
            proto_path_to_generated_filename(
                _strip_package_from_path(label_package, proto),
                _GRPC_PROTO_HEADER_FMT,
            )
            for proto in protos
        ]
        outs += [
            proto_path_to_generated_filename(
                _strip_package_from_path(label_package, proto),
                _GRPC_PROTO_SRC_FMT,
            )
            for proto in protos
        ]
        if ctx.attr.generate_mocks:
            outs += [
                proto_path_to_generated_filename(
                    _strip_package_from_path(label_package, proto),
                    _GRPC_PROTO_MOCK_HEADER_FMT,
                )
                for proto in protos
            ]
    else:
        outs += [
            proto_path_to_generated_filename(
                _strip_package_from_path(label_package, proto),
                _PROTO_HEADER_FMT,
            )
            for proto in protos
        ]
        outs += [
            proto_path_to_generated_filename(
                _strip_package_from_path(label_package, proto),
                _PROTO_SRC_FMT,
            )
            for proto in protos
        ]
    out_files = [ctx.actions.declare_file(out) for out in outs]
    dir_out = str(ctx.genfiles_dir.path + proto_root)

    arguments = []
    if ctx.executable.plugin:
        arguments += get_plugin_args(
            ctx.executable.plugin,
            ctx.attr.flags,
            dir_out,
            ctx.attr.generate_mocks,
        )
        tools = [ctx.executable.plugin]
    else:
        arguments += ["--cpp_out=" + ",".join(ctx.attr.flags) + ":" + dir_out]
        tools = []

    arguments += [
        "--proto_path={}".format(get_include_directory(i))
        for i in includes
    ]

    # Include the output directory so that protoc puts the generated code in the
    # right directory.
    arguments += ["--proto_path={0}{1}".format(dir_out, proto_root)]
    arguments += [_get_srcs_file_path(proto) for proto in protos]

    # create a list of well known proto files if the argument is non-None
    well_known_proto_files = []
    if ctx.attr.well_known_protos:
        f = ctx.attr.well_known_protos.files.to_list()[0].dirname
        if f != "external/com_google_protobuf/src/google/protobuf":
            print(
                "Error: Only @com_google_protobuf//:well_known_protos is supported",
            )
        else:
            # f points to "external/com_google_protobuf/src/google/protobuf"
            # add -I argument to protoc so it knows where to look for the proto files.
            arguments += ["-I{0}".format(f + "/../..")]
            well_known_proto_files = [
                f
                for f in ctx.attr.well_known_protos.files.to_list()
            ]

    ctx.actions.run(
        inputs = protos + includes + well_known_proto_files,
        tools = tools,
        outputs = out_files,
        executable = ctx.executable._protoc,
        arguments = arguments,
    )

    return struct(files = depset(out_files))

_generate_cc = rule(
    attrs = {
        "srcs": attr.label_list(
            mandatory = True,
            allow_empty = False,
            providers = [ProtoInfo],
        ),
        "plugin": attr.label(
            executable = True,
            providers = ["files_to_run"],
            cfg = "host",
        ),
        "flags": attr.string_list(
            mandatory = False,
            allow_empty = True,
        ),
        "well_known_protos": attr.label(mandatory = False),
        "generate_mocks": attr.bool(
            default = False,
            mandatory = False,
        ),
        "_protoc": attr.label(
            default = Label("@com_google_protobuf//:protoc"),
            executable = True,
            cfg = "host",
        ),
    },
    # We generate .h files, so we need to output to genfiles.
    output_to_genfiles = True,
    implementation = generate_cc_impl,
)

def generate_cc(well_known_protos, **kwargs):
    if well_known_protos:
        _generate_cc(
            well_known_protos = "@com_google_protobuf//:well_known_protos",
            **kwargs
        )
    else:
        _generate_cc(**kwargs)
