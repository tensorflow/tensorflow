"""Generates and compiles C++ grpc stubs from proto_library rules."""

load("@com_github_grpc_grpc//bazel:generate_cc.bzl", "generate_cc")
load("@com_github_grpc_grpc//bazel:protobuf.bzl", "well_known_proto_libs")

def cc_grpc_library(
        name,
        srcs,
        deps,
        proto_only = False,
        well_known_protos = False,
        generate_mocks = False,
        use_external = False,
        grpc_only = False,
        **kwargs):
    """Generates C++ grpc classes for services defined in a proto file.

    If grpc_only is True, this rule is compatible with proto_library and
    cc_proto_library native rules such that it expects proto_library target
    as srcs argument and generates only grpc library classes, expecting
    protobuf messages classes library (cc_proto_library target) to be passed in
    deps argument. By default grpc_only is False which makes this rule to behave
    in a backwards-compatible mode (trying to generate both proto and grpc
    classes).

    Assumes the generated classes will be used in cc_api_version = 2.

    Args:
        name (str): Name of rule.
        srcs (list): A single .proto file which contains services definitions,
          or if grpc_only parameter is True, a single proto_library which
          contains services descriptors.
        deps (list): A list of C++ proto_library (or cc_proto_library) which
          provides the compiled code of any message that the services depend on.
        proto_only (bool): If True, create only C++ proto classes library,
          avoid creating C++ grpc classes library (expect it in deps).
          Deprecated, use native cc_proto_library instead. False by default.
        well_known_protos (bool): Should this library additionally depend on
          well known protos. Deprecated, the well known protos should be
          specified as explicit dependencies of the proto_library target
          (passed in srcs parameter) instead. False by default.
        generate_mocks (bool): when True, Google Mock code for client stub is
          generated. False by default.
        use_external (bool): Not used.
        grpc_only (bool): if True, generate only grpc library, expecting
          protobuf messages library (cc_proto_library target) to be passed as
          deps. False by default (will become True by default eventually).
        **kwargs: rest of arguments, e.g., compatible_with and visibility
    """
    if len(srcs) > 1:
        fail("Only one srcs value supported", "srcs")
    if grpc_only and proto_only:
        fail("A mutualy exclusive configuration is specified: grpc_only = True and proto_only = True")

    extra_deps = []
    proto_targets = []

    if not grpc_only:
        proto_target = "_" + name + "_only"
        cc_proto_target = name if proto_only else "_" + name + "_cc_proto"

        proto_deps = ["_" + dep + "_only" for dep in deps if dep.find(":") == -1]
        proto_deps += [dep.split(":")[0] + ":" + "_" + dep.split(":")[1] + "_only" for dep in deps if dep.find(":") != -1]
        if well_known_protos:
            proto_deps += well_known_proto_libs()

        native.proto_library(
            name = proto_target,
            srcs = srcs,
            deps = proto_deps,
            **kwargs
        )

        native.cc_proto_library(
            name = cc_proto_target,
            deps = [":" + proto_target],
            **kwargs
        )
        extra_deps.append(":" + cc_proto_target)
        proto_targets.append(proto_target)
    else:
        if not srcs:
            fail("srcs cannot be empty", "srcs")
        proto_targets += srcs

    if not proto_only:
        codegen_grpc_target = "_" + name + "_grpc_codegen"
        generate_cc(
            name = codegen_grpc_target,
            srcs = proto_targets,
            plugin = "@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin",
            well_known_protos = well_known_protos,
            generate_mocks = generate_mocks,
            **kwargs
        )

        native.cc_library(
            name = name,
            srcs = [":" + codegen_grpc_target],
            hdrs = [":" + codegen_grpc_target],
            deps = deps +
                   extra_deps +
                   ["@com_github_grpc_grpc//:grpc++_codegen_proto"],
            **kwargs
        )
