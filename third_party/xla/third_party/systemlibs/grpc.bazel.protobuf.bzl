"""Utility functions for generating protobuf code."""

load("@rules_proto//proto:defs.bzl", "ProtoInfo")

_PROTO_EXTENSION = ".proto"
_VIRTUAL_IMPORTS = "/_virtual_imports/"

def well_known_proto_libs():
    return [
        "@com_google_protobuf//:any_proto",
        "@com_google_protobuf//:api_proto",
        "@com_google_protobuf//:compiler_plugin_proto",
        "@com_google_protobuf//:descriptor_proto",
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:empty_proto",
        "@com_google_protobuf//:field_mask_proto",
        "@com_google_protobuf//:source_context_proto",
        "@com_google_protobuf//:struct_proto",
        "@com_google_protobuf//:timestamp_proto",
        "@com_google_protobuf//:type_proto",
        "@com_google_protobuf//:wrappers_proto",
    ]

def get_proto_root(workspace_root):
    """Gets the root protobuf directory.

    Args:
      workspace_root: context.label.workspace_root

    Returns:
      The directory relative to which generated include paths should be.
    """
    if workspace_root:
        return "/{}".format(workspace_root)
    else:
        return ""

def _strip_proto_extension(proto_filename):
    if not proto_filename.endswith(_PROTO_EXTENSION):
        fail('"{}" does not end with "{}"'.format(
            proto_filename,
            _PROTO_EXTENSION,
        ))
    return proto_filename[:-len(_PROTO_EXTENSION)]

def proto_path_to_generated_filename(proto_path, fmt_str):
    """Calculates the name of a generated file for a protobuf path.

    For example, "examples/protos/helloworld.proto" might map to
      "helloworld.pb.h".

    Args:
      proto_path: The path to the .proto file.
      fmt_str: A format string used to calculate the generated filename. For
        example, "{}.pb.h" might be used to calculate a C++ header filename.

    Returns:
      The generated filename.
    """
    return fmt_str.format(_strip_proto_extension(proto_path))

def get_include_directory(source_file):
    """Returns the include directory path for the source_file.

    I.e. all of the include statements within the given source_file
    are calculated relative to the directory returned by this method.

    The returned directory path can be used as the "--proto_path=" argument
    value.

    Args:
      source_file: A proto file.

    Returns:
      The include directory path for the source_file.
    """
    directory = source_file.path
    prefix_len = 0

    if is_in_virtual_imports(source_file):
        root, relative = source_file.path.split(_VIRTUAL_IMPORTS, 2)
        result = root + _VIRTUAL_IMPORTS + relative.split("/", 1)[0]
        return result

    if not source_file.is_source and directory.startswith(source_file.root.path):
        prefix_len = len(source_file.root.path) + 1

    if directory.startswith("external", prefix_len):
        external_separator = directory.find("/", prefix_len)
        repository_separator = directory.find("/", external_separator + 1)
        return directory[:repository_separator]
    else:
        return source_file.root.path if source_file.root.path else "."

def get_plugin_args(
        plugin,
        flags,
        dir_out,
        generate_mocks,
        plugin_name = "PLUGIN"):
    """Returns arguments configuring protoc to use a plugin for a language.

    Args:
      plugin: An executable file to run as the protoc plugin.
      flags: The plugin flags to be passed to protoc.
      dir_out: The output directory for the plugin.
      generate_mocks: A bool indicating whether to generate mocks.
      plugin_name: A name of the plugin, it is required to be unique when there
      are more than one plugin used in a single protoc command.
    Returns:
      A list of protoc arguments configuring the plugin.
    """
    augmented_flags = list(flags)
    if generate_mocks:
        augmented_flags.append("generate_mock_code=true")

    augmented_dir_out = dir_out
    if augmented_flags:
        augmented_dir_out = ",".join(augmented_flags) + ":" + dir_out

    return [
        "--plugin=protoc-gen-{plugin_name}={plugin_path}".format(
            plugin_name = plugin_name,
            plugin_path = plugin.path,
        ),
        "--{plugin_name}_out={dir_out}".format(
            plugin_name = plugin_name,
            dir_out = augmented_dir_out,
        ),
    ]

def _get_staged_proto_file(context, source_file):
    if (source_file.dirname == context.label.package or
        is_in_virtual_imports(source_file)):
        return source_file
    else:
        copied_proto = context.actions.declare_file(source_file.basename)
        context.actions.run_shell(
            inputs = [source_file],
            outputs = [copied_proto],
            command = "cp {} {}".format(source_file.path, copied_proto.path),
            mnemonic = "CopySourceProto",
        )
        return copied_proto

def protos_from_context(context):
    """Copies proto files to the appropriate location.

    Args:
      context: The ctx object for the rule.

    Returns:
      A list of the protos.
    """
    protos = []
    for src in context.attr.deps:
        for file in src[ProtoInfo].direct_sources:
            protos.append(_get_staged_proto_file(context, file))
    return protos

def includes_from_deps(deps):
    """Get includes from rule dependencies."""
    return [
        file
        for src in deps
        for file in src[ProtoInfo].transitive_imports.to_list()
    ]

def get_proto_arguments(protos, genfiles_dir_path):
    """Get the protoc arguments specifying which protos to compile."""
    arguments = []
    for proto in protos:
        strip_prefix_len = 0
        if is_in_virtual_imports(proto):
            incl_directory = get_include_directory(proto)
            if proto.path.startswith(incl_directory):
                strip_prefix_len = len(incl_directory) + 1
        elif proto.path.startswith(genfiles_dir_path):
            strip_prefix_len = len(genfiles_dir_path) + 1

        arguments.append(proto.path[strip_prefix_len:])

    return arguments

def declare_out_files(protos, context, generated_file_format):
    """Declares and returns the files to be generated."""

    out_file_paths = []
    for proto in protos:
        if not is_in_virtual_imports(proto):
            out_file_paths.append(proto.basename)
        else:
            path = proto.path[proto.path.index(_VIRTUAL_IMPORTS) + 1:]
            out_file_paths.append(path)

    return [
        context.actions.declare_file(
            proto_path_to_generated_filename(
                out_file_path,
                generated_file_format,
            ),
        )
        for out_file_path in out_file_paths
    ]

def get_out_dir(protos, context):
    """ Returns the calculated value for --<lang>_out= protoc argument based on
    the input source proto files and current context.

    Args:
        protos: A list of protos to be used as source files in protoc command
        context: A ctx object for the rule.
    Returns:
        The value of --<lang>_out= argument.
    """
    at_least_one_virtual = 0
    for proto in protos:
        if is_in_virtual_imports(proto):
            at_least_one_virtual = True
        elif at_least_one_virtual:
            fail("Proto sources must be either all virtual imports or all real")
    if at_least_one_virtual:
        out_dir = get_include_directory(protos[0])
        ws_root = protos[0].owner.workspace_root
        if ws_root and out_dir.find(ws_root) >= 0:
            out_dir = "".join(out_dir.rsplit(ws_root, 1))
        return struct(
            path = out_dir,
            import_path = out_dir[out_dir.find(_VIRTUAL_IMPORTS) + 1:],
        )
    return struct(path = context.genfiles_dir.path, import_path = None)

def is_in_virtual_imports(source_file, virtual_folder = _VIRTUAL_IMPORTS):
    """Determines if source_file is virtual (is placed in _virtual_imports
    subdirectory). The output of all proto_library targets which use
    import_prefix  and/or strip_import_prefix arguments is placed under
    _virtual_imports directory.

    Args:
        source_file: A proto file.
        virtual_folder: The virtual folder name (is set to "_virtual_imports"
            by default)
    Returns:
        True if source_file is located under _virtual_imports, False otherwise.
    """
    return not source_file.is_source and virtual_folder in source_file.path
