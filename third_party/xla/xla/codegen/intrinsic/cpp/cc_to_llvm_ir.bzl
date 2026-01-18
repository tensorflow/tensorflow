"""
A rule to compile a C++ file to a header containing LLVM IR.
//third_party/tensorflow/compiler/xla/service/cpu/tests
This rule is critical for generating LLVM IR bitcode that is embedded into the XLA compiler.
It uses standard cc_library with clang flags to generate IR, then extracts it.
"""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

visibility(DEFAULT_LOAD_VISIBILITY)

def to_camel_case(s):
    """Converts a snake_case or kebab-case string to CamelCase."""
    return "".join([p.capitalize() for p in s.replace("-", "_").split("_")])

def _extract_object_files_impl(ctx):
    """Implementation of the _extract_object_files rule."""
    dep = ctx.attr.dep
    if CcInfo not in dep:
        fail("Dependency must provide CcInfo")

    cc_info = dep[CcInfo]
    objects = []

    # Determine expected object file name pattern based on source file
    src_file_name = ctx.file.src.basename
    base_name = src_file_name.rpartition(".")[0]

    target_lib_name = ctx.attr.library_name
    match_pattern = "/_objs/{}/".format(target_lib_name)

    # Iterate over linker inputs to find object files
    if cc_info.linking_context:
        for linker_input in cc_info.linking_context.linker_inputs.to_list():
            for library in linker_input.libraries:
                objs = library.pic_objects
                if not objs:
                    objs = library.objects

                for obj in objs:
                    # Filter for objects that look like they come from our source.
                    # We check:
                    # 1. Basename matches source (e.g. tanh.pic.o vs tanh.cc)
                    # 2. Extension is object file
                    # 3. Path contains the library name (to distinguish from dependencies)
                    if obj.basename.startswith(base_name) and \
                       (obj.basename.endswith(".o") or obj.basename.endswith(".obj")) and \
                       match_pattern in obj.short_path:
                        objects.append(obj)

    if not objects:
        pass

    return DefaultInfo(files = depset(objects))

_extract_object_files = rule(
    implementation = _extract_object_files_impl,
    attrs = {
        "dep": attr.label(mandatory = True, providers = [CcInfo]),
        "src": attr.label(mandatory = True, allow_single_file = True),
        "library_name": attr.string(mandatory = True),
    },
)

def cc_ir_header(name, src, deps = [], copts = [], **kwargs):
    """A macro that generates an IR header and wraps it in a cc_library.

    Args:
      name: The name of the generated cc_library.
      src: The C++ source file to compile.
      deps: The C++ dependencies of the source file.
      copts: Additional compiler flags.
      **kwargs: Additional arguments to pass to the generated cc_library.
    """

    # Extract arguments that are not for cc_library
    base_name = kwargs.pop("base_name", name)
    namespace = kwargs.pop("namespace", "llvm_ir")

    common_attrs = {}
    for attr in ["visibility", "testonly"]:
        if attr in kwargs:
            common_attrs[attr] = kwargs[attr]

    compatible_with = None

    # Do a little dance so the line below matches copybara rules.
    # copybara_removed compatible_with = ["//buildenv/target:non_prod"]
    compatible_with = kwargs.get("compatible_with", compatible_with)
    if compatible_with:
        common_attrs["compatible_with"] = compatible_with

    # Define intermediate targets
    lib_name = name + "_lib"
    out_header = name + ".h"

    compile_flags = [
        "-emit-llvm",
        "-O3",
        "-DNDEBUG",
        "-mprefer-vector-width=512",
        "-DEIGEN_VECTORIZE_GENERIC",
        "-fno-builtin",
        "-Wno-psabi",
        "-std=c++17",
    ] + copts

    # Disabled features to avoid instrumentations in the IR
    # AND disable thin archives to ensure we have actual content to extract.
    disabled_features = [
        "thin_lto",
        "cfi",
        "thin_archives",
        "per_object_debug_info",
        "module_maps",
        "fdo_optimize",
        "fdo_instrument",
        "asan",
        "hwasan",
        "msan",
        "tsan",
        "ubsan",
    ]

    # Prefix features with '-'
    features = ["-" + f for f in disabled_features] + kwargs.pop("features", [])
    features.append("cfi_opt_out")

    cc_library(
        name = lib_name,
        srcs = [src],
        deps = deps,
        copts = compile_flags,
        features = features,
        tags = ["manual"],
        **common_attrs
    )

    # Extract the object file (which is bitcode) using CcInfo.
    _extract_object_files(
        name = name + "_extract_bc",
        dep = ":" + lib_name,
        src = src,
        library_name = lib_name,
        tags = ["manual"],
        **common_attrs
    )

    # Generate the header file from the IR (Bitcode).
    # TODO(talts): In the next version, I will update this to store the bitcode in a shared object
    # file so that we aren't using LLVM's text format.
    variable_name = "k{}Ir".format(to_camel_case(base_name))

    ir_to_string_tool = "//xla/codegen/intrinsic/cpp:ir_to_string"

    native.genrule(
        name = name + "_gen_header",
        srcs = [":" + name + "_extract_bc"],
        outs = [out_header],
        tools = [ir_to_string_tool],
        cmd = "$(location {}) $< $@ {} {}".format(ir_to_string_tool, variable_name, namespace),
        tags = ["manual"],
        **common_attrs
    )

    # Exposed library
    cc_library(
        name = name,
        hdrs = [":" + out_header],
        deps = deps,
        **kwargs
    )
