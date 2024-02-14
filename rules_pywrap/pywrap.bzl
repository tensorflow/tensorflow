load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

PywrapInfo = provider(
    fields = {
        "cc_info": "Wrapped CcInfo",
        "private_linker_inputs": "Libraries to link only to individual pywrap libraries, but not in commmon library",
        "owner": "Owner's label",
        "py_stub": "Pybind Python stub used to resolve cross-package references",
        "outer_module_name": "Outer module name for deduping libraries with the same name",
        "cc_only": "True if this PywrapInfo represents cc-only library (no PyIni_)"
    }
)

CollectedPywrapInfo = provider(
    fields = {
        "pywrap_infos": "depset of PywrapInfo providers"
    }
)

# TODO: we need to generate win_def_file, but it should be simple
def pywrap_library(
        name,
        deps,
        win_def_file = None,
        pywrap_count = None,
        extra_deps = ["@pybind11//:pybind11"],
        visibility = None,
        testonly = None,
        compatible_with = None):

    # 0) If pywrap_count is not specified, assume we pass pybind_extension,
    # targets directly, so actual pywrap_count should just be equal to  number
    # of deps.
    actual_pywrap_count = len(deps) if pywrap_count == None else pywrap_count

    # 1) Create common pywrap library. The common library should link in
    # everything except the object file with Python Extension's init function
    # PyInit_<extension_name>.
    #
    pywrap_info_collector_name = "_%s_info_collector" % name

    collected_pywrap_infos(
        name = pywrap_info_collector_name,
        deps = deps,
        pywrap_count = actual_pywrap_count,
    )

    pywrap_common_name = "_%s_pywrap_internal" % name
    _pywrap_split_library(
        name = pywrap_common_name,
        dep = ":%s" % pywrap_info_collector_name,
        testonly = testonly,
        compatible_with = compatible_with,
        pywrap_index = -1,
    )

    common_deps = []

    pywrap_common_cc_binary_name = "%s_internal" % name
    native.cc_binary(
        name = pywrap_common_cc_binary_name,
        deps = [":%s" % pywrap_common_name],
        linkstatic = True,
        linkshared = True,
        testonly = testonly,
        compatible_with = compatible_with,
        win_def_file = win_def_file,
    )

    # The following filegroup/cc_import shenanigans to extract .if.lib from
    # cc_binary should not be needed, but otherwise bazel can't consume
    # cc_binary properly as a dep in downstream cc_binary/cc_test targets.
    # I.e. cc_binary does not work as a dependency downstream, but if wrapped
    # into a cc_import it all of a sudden starts working. I wish bazel team
    # fixed it...
    pywrap_common_if_lib_name = "%s_if_lib" % pywrap_common_name
    native.filegroup(
        name = pywrap_common_if_lib_name,
        srcs = [":%s" % pywrap_common_cc_binary_name],
        output_group = "interface_library",
        testonly = testonly,
        compatible_with = compatible_with,
    )

    pywrap_common_import_name = "%s_pywrap_internal_import" % name
    native.cc_import(
        name = pywrap_common_import_name,
        interface_library = ":%s" % pywrap_common_if_lib_name,
        shared_library = ":%s" % pywrap_common_cc_binary_name,
        testonly = testonly,
        compatible_with = compatible_with,
    )
    common_deps.append(":%s" % pywrap_common_import_name)

    # 2) Create individual super-thin pywrap libraries, which depend on the
    # common one. The individual libraries must link in statically only the
    # object file with Python Extension's init function PyInit_<extension_name>
    #
    shared_objects = []
    common_deps.extend(extra_deps)

    for pywrap_index in range(0, actual_pywrap_count):
        dep_name = "_%s_%s" % (name, pywrap_index)
        shared_object_name = "%s_shared_object" % dep_name
        win_def_name = "%s_win_def" % dep_name
        pywrap_name = "%s_pywrap" % dep_name

        _pywrap_split_library(
            name = pywrap_name,
            dep = ":%s" % pywrap_info_collector_name,
            pywrap_index = pywrap_index,
            testonly = testonly,
            compatible_with = compatible_with,
        )

        _generated_win_def_file(
            name = win_def_name,
            dep = ":%s" % pywrap_info_collector_name,
            pywrap_index = pywrap_index,
            testonly = testonly,
            compatible_with = compatible_with,
        )

        native.cc_binary(
            name = shared_object_name,
            srcs = [],
            deps = [":%s" % pywrap_name] + common_deps,
            linkshared = True,
            linkstatic = True,
            win_def_file = ":%s" % win_def_name,
            testonly = testonly,
            compatible_with = compatible_with,
        )
        shared_objects.append(":%s" % shared_object_name)


    # 3) Construct final binaries with proper names and put them as data
    # attribute in a py_library, which is the final and only public artifact of
    # this macro
    #
    pywrap_binaries_name = "_%s_binaries" % name
    _pywrap_binaries(
        name = pywrap_binaries_name,
        collected_pywraps = ":%s" % pywrap_info_collector_name,
        deps = shared_objects,
        extension = select({
            "@bazel_tools//src/conditions:windows": ".pyd",
            "//conditions:default": ".so",
        }),
        testonly = testonly,
        compatible_with = compatible_with,
     )

    binaries_data = ["%s" % pywrap_binaries_name] + [shared_objects[0]]
    binaries_data.append(":%s" % pywrap_common_cc_binary_name)

    native.py_library(
        name = name,
        srcs = [":%s" % pywrap_info_collector_name],
        data = binaries_data,
        testonly = testonly,
        compatible_with = compatible_with,
        visibility = visibility,
    )

    # For debugging purposes only
    native.filegroup(
        name = "_%s_binaries_all" % name ,
        srcs = binaries_data,
        testonly = testonly,
        compatible_with = compatible_with,
    )

def pywrap_common_library(name, dep):
    native.alias(
        name = name,
        actual = "%s_pywrap_internal_import" % dep,
    )

def _generated_win_def_file_impl(ctx):
    pywrap_infos = ctx.attr.dep[CollectedPywrapInfo].pywrap_infos.to_list()
    pywrap_info = pywrap_infos[ctx.attr.pywrap_index]
    win_def_file_name = "%s.def" % pywrap_info.owner.name
    win_def_file = ctx.actions.declare_file(win_def_file_name)

    if pywrap_info.cc_only:
        command = "echo \"EXPORTS\r\n\">> {win_def_file}"
    else:
        command = "echo \"EXPORTS\r\n  PyInit_{owner}\">> {win_def_file}"

    ctx.actions.run_shell(
        inputs = [],
        command = command.format(
            owner = pywrap_info.owner.name,
            win_def_file = win_def_file.path
        ),
        outputs = [win_def_file],
    )

    return [DefaultInfo(files = depset(direct = [win_def_file]))]

_generated_win_def_file = rule(
    attrs = {
        "dep": attr.label(
            allow_files = False,
            providers = [CollectedPywrapInfo],
        ),
        "pywrap_index": attr.int(mandatory = True),
    },
    implementation = _generated_win_def_file_impl,
)


def _pywrap_split_library_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    dependency_libraries = []
    pywrap_index = ctx.attr.pywrap_index
    pywrap_infos = ctx.attr.dep[CollectedPywrapInfo].pywrap_infos.to_list()
    if pywrap_index >= 0:
        pywrap_infos = [pywrap_infos[pywrap_index]]

    for pywrap_info in pywrap_infos:
        cc_linker_inputs = pywrap_info.cc_info.linking_context.linker_inputs
        # TODO: we should not rely on order of object files in CcInfo
        private_linker_inputs = None
        excluded_linker_inputs = {}

        # pywrap_index >= 0 means we are building small individual _pywrap library
        if pywrap_index >= 0:
            if pywrap_info.cc_only:
                linker_inputs = []
            else:
                linker_inputs = cc_linker_inputs.to_list()[:1]
            if pywrap_info.private_linker_inputs:
                private_linker_inputs = [
                    depset(direct = pywrap_info.private_linker_inputs.keys())
                ]
        # pywrap_index < 0 means we are building big common _pywrap library
        else:
            if pywrap_info.cc_only:
                linker_inputs = cc_linker_inputs.to_list()
            else:
                linker_inputs = cc_linker_inputs.to_list()[1:]
            excluded_linker_inputs = pywrap_info.private_linker_inputs

        for linker_input in linker_inputs:
            if linker_input in excluded_linker_inputs:
                continue
            for lib in linker_input.libraries:
                lib_copy = lib;
                if not lib.alwayslink:
                    lib_copy = cc_common.create_library_to_link(
                        actions = ctx.actions,
                        cc_toolchain = cc_toolchain,
                        feature_configuration = feature_configuration,
                        static_library = lib.static_library,
                        pic_static_library = lib.pic_static_library,
                        interface_library = lib.interface_library,
                        alwayslink = True,
                    )
                dependency_libraries.append(lib_copy)

    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        libraries = depset(direct = dependency_libraries),
    )

    linking_context = cc_common.create_linking_context(
        linker_inputs = depset(
            direct = [linker_input],
            transitive = private_linker_inputs
        ),
    )

    return [CcInfo(linking_context = linking_context)]

_pywrap_split_library = rule(
    attrs = {
        "dep": attr.label(
            allow_files = False,
            providers = [CollectedPywrapInfo],
        ),
        "_cc_toolchain": attr.label(
            default = "@bazel_tools//tools/cpp:current_cc_toolchain",
        ),
        "pywrap_index": attr.int(mandatory = True, default = -1),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
    implementation = _pywrap_split_library_impl,
)

def pybind_extension(
        name,
        deps,
        srcs = [],
        private_deps = [],
        visibility = None,
        win_def_file = None,
        testonly = None,
        compatible_with = None,
        outer_module_name = "",
        additional_exported_symbols = [],
        **kwargs):

    cc_library_name = "_%s_cc_library" % name

    native.cc_library(
        name = cc_library_name,
        deps = deps + private_deps + ["@pybind11//:pybind11"],
        srcs = srcs,
        linkstatic = True,
        alwayslink = True,
        visibility = visibility,
        testonly = testonly,
        compatible_with = compatible_with,
        **kwargs
    )

    if not srcs:
        _cc_only_pywrap_info_wrapper(
            name = name,
            deps = ["%s" % cc_library_name],
            testonly = testonly,
            compatible_with = compatible_with,
            visibility = visibility,
        )
    else:
        _pywrap_info_wrapper(
            name = name,
            deps = ["%s" % cc_library_name],
            private_deps = private_deps,
            outer_module_name = outer_module_name,
            additional_exported_symbols = additional_exported_symbols,
            testonly = testonly,
            compatible_with = compatible_with,
            visibility = visibility,
        )

def _pywrap_info_wrapper_impl(ctx):
    #the attribute is called deps not dep to match aspect's attr_aspects
    if len(ctx.attr.deps) != 1:
        fail("deps attribute must contain exactly one dependency")

    py_stub = ctx.actions.declare_file("%s.py" % ctx.attr.name)
    substitutions = {}
    outer_module_name = ctx.attr.outer_module_name
    if outer_module_name:
        val = 'outer_module_name = "%s."' %  outer_module_name
        substitutions['outer_module_name = "" # template_val'] = val

    additional_exported_symbols = ctx.attr.additional_exported_symbols
    if additional_exported_symbols:
        val = "extra_names = %s # template_val" % additional_exported_symbols
        substitutions["extra_names = [] # template_val"] = val

    ctx.actions.expand_template(
        template = ctx.file.py_stub_src,
        output = py_stub,
        substitutions = substitutions,
    )

    wrapped_dep = ctx.attr.deps[0]

    private_linker_inputs = {}
    for private_dep in ctx.attr.private_deps:
        linker_inputs = private_dep[CcInfo].linking_context.linker_inputs.to_list()
#        private_linker_inputs[linker_inputs] = private_dep.label
        for linker_input in linker_inputs:
            private_linker_inputs[linker_input] = linker_input.owner

    return [
        PyInfo(transitive_sources = depset()),
        PywrapInfo(
            cc_info = wrapped_dep[CcInfo],
            private_linker_inputs = private_linker_inputs,
            owner = ctx.label,
            py_stub = py_stub,
            outer_module_name = outer_module_name,
            cc_only = False,
        ),
    ]

_pywrap_info_wrapper = rule(
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
        "private_deps": attr.label_list(providers = [CcInfo]),
        "outer_module_name": attr.string(mandatory = False, default = ""),
        "py_stub_src": attr.label(
            allow_single_file = True,
            default = Label("//rules_pywrap:pybind_extension.py.tpl")
        ),
        "additional_exported_symbols": attr.string_list(
            mandatory = False,
            default = []
        ),
    },
    implementation = _pywrap_info_wrapper_impl
)

def _cc_only_pywrap_info_wrapper_impl(ctx):
    wrapped_dep = ctx.attr.deps[0]
    return [
        PyInfo(transitive_sources = depset()),
        PywrapInfo(
            cc_info = wrapped_dep[CcInfo],
            private_linker_inputs = {},
            owner = ctx.label,
            py_stub = None,
            outer_module_name = None,
            cc_only = True,
        ),
    ]

_cc_only_pywrap_info_wrapper = rule(
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
    },
    implementation = _cc_only_pywrap_info_wrapper_impl
)

def _pywrap_info_collector_aspect_impl(target, ctx):
    pywrap_infos = []
    transitive_pywrap_infos = []
    if PywrapInfo in target:
        pywrap_infos.append(target[PywrapInfo])

    if hasattr(ctx.rule.attr, "deps"):
        for dep in ctx.rule.attr.deps:
            if CollectedPywrapInfo in dep:
                collected_pywrap_info = dep[CollectedPywrapInfo]
                transitive_pywrap_infos.append(collected_pywrap_info.pywrap_infos)

    return [
        CollectedPywrapInfo(
            pywrap_infos = depset(
                direct = pywrap_infos,
                transitive = transitive_pywrap_infos,
                order = "topological"
            ),
        )
    ]

_pywrap_info_collector_aspect = aspect(
    attr_aspects = ["deps"],
    implementation = _pywrap_info_collector_aspect_impl
)

def _collected_pywrap_infos_impl(ctx):
    pywrap_infos = []
    for dep in ctx.attr.deps:
        if CollectedPywrapInfo in dep:
            pywrap_infos.append(dep[CollectedPywrapInfo].pywrap_infos)

    rv = CollectedPywrapInfo(
        pywrap_infos = depset(
            transitive = pywrap_infos,
            order = "topological"
        )
    )
    pywraps = rv.pywrap_infos.to_list();

    if ctx.attr.pywrap_count != len(pywraps):
        found_pywraps = "\n        ".join([str(pw.owner) for pw in pywraps])
        fail("""
    Number of actual pywrap libraries does not match expected pywrap_count.
    Expected pywrap_count: {expected_pywrap_count}
    Actual pywrap_count: {actual_pywra_count}
    Actual pywrap libraries in the transitive closure of {label}:
        {found_pywraps}
    """.format(expected_pywrap_count = ctx.attr.pywrap_count,
               actual_pywra_count = len(pywraps),
               label = ctx.label,
               found_pywraps = found_pywraps))

    py_stubs = []
    for pywrap in pywraps:
        if pywrap.py_stub:
            py_stubs.append(pywrap.py_stub)

    return [
        DefaultInfo(files = depset(direct = py_stubs)),
        rv,
    ]

collected_pywrap_infos = rule(
    attrs = {
        "deps": attr.label_list(
            aspects = [_pywrap_info_collector_aspect],
            providers = [PyInfo]
        ),
        "pywrap_count": attr.int(mandatory = True, default = 1),
    },

    implementation = _collected_pywrap_infos_impl
)

def _pywrap_binaries_impl(ctx):
    deps = ctx.attr.deps
    dep = ctx.attr.collected_pywraps
    extension = ctx.attr.extension

    pywrap_infos = dep[CollectedPywrapInfo].pywrap_infos.to_list()
    original_binaries = deps

    if len(pywrap_infos) != len(original_binaries):
        fail()

    final_binaries = []
    for i in range(0, len(pywrap_infos)):
        pywrap_info = pywrap_infos[i]
        original_binary = original_binaries[i]
        subfolder = ""
        if pywrap_info.outer_module_name:
            subfolder = pywrap_info.outer_module_name + "/"
        final_binary_name = "%s%s%s" % (subfolder, pywrap_info.owner.name, extension)
        final_binary = ctx.actions.declare_file(final_binary_name)
        original_binary_file = original_binary.files.to_list()[0]
        ctx.actions.run_shell(
            inputs = [original_binary_file],
            command = "cp {original} {final}".format(
                original = original_binary_file.path,
                final = final_binary.path
            ),
            outputs = [final_binary],
        )

#        print("{index} {final}".format(
#            index = i,
#            final = final_binary.path
#            )
#        )

        final_binaries.append(final_binary)

    return [DefaultInfo(files = depset(direct = final_binaries))]


_pywrap_binaries = rule(
    attrs = {
        "deps": attr.label_list(mandatory = True, allow_files = False),
        "collected_pywraps": attr.label(mandatory = True, allow_files = False),
        "extension": attr.string(default = ".so"),
    },

    implementation = _pywrap_binaries_impl
)