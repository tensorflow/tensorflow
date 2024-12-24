load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

PywrapInfo = provider(
    fields = {
        "cc_info": "Wrapped CcInfo",
        "private_deps": "Libraries to link only to individual pywrap libraries, but not in commmon library",
        "owner": "Owner's label",
        "common_lib_packages": "Packages in which to search for common pywrap library",
        "py_stub": "Pybind Python stub used to resolve cross-package references",
        "cc_only": "True if this PywrapInfo represents cc-only library (no PyIni_)",
    },
)

CollectedPywrapInfo = provider(
    fields = {
        "pywrap_infos": "depset of PywrapInfo providers",
    },
)

PywrapFilters = provider(
    fields = {
        "py_cc_linker_inputs": "",
        "cc_linker_inputs": "",
        "pywrap_private_linker_inputs": "",
    },
)

def pywrap_library(
        name,
        deps,
        py_cc_deps_filter = [],
        cc_deps_filter = [],
        linkopts = [],
        py_cc_linkopts = [],
        win_def_file = None,
        py_cc_win_def_file = None,
        pywrap_count = None,
        extra_deps = ["@pybind11//:pybind11"],
        visibility = None,
        testonly = None,
        compatible_with = None):
    # 0) If pywrap_count is not specified, assume we pass pybind_extension,
    # targets directly, so actual pywrap_count should just be equal to  number
    # of deps.
    actual_pywrap_count = len(deps) if pywrap_count == None else pywrap_count

    # 1) Create common libraries cc-only (C API) and py-specific (parts reused
    # by different pywrap libraries but dependin on Python symbols).
    # The common library should link in everything except the object file with
    # Python Extension's init function PyInit_<extension_name>.
    info_collector_name = "_%s_info_collector" % name
    collected_pywrap_infos(
        name = info_collector_name,
        deps = deps,
        pywrap_count = actual_pywrap_count,
    )

    linker_input_filters_name = "_%s_linker_input_filters" % name
    _linker_input_filters(
        name = linker_input_filters_name,
        dep = ":%s" % info_collector_name,
        py_cc_deps_filter = py_cc_deps_filter,
        cc_deps_filter = cc_deps_filter,
    )

    # _internal binary
    common_split_name = "_%s_split" % name
    _pywrap_split_library(
        name = common_split_name,
        mode = "cc_common",
        dep = ":%s" % info_collector_name,
        linker_input_filters = "%s" % linker_input_filters_name,
        testonly = testonly,
        compatible_with = compatible_with,
    )

    common_cc_binary_name = "%s_internal" % name
    common_import_name = _construct_common_binary(
        common_cc_binary_name,
        [":%s" % common_split_name],
        linkopts,
        testonly,
        compatible_with,
        win_def_file,
        None,
    )

    # _py_internal binary
    py_common_split_name = "_%s_py_split" % name
    _pywrap_split_library(
        name = py_common_split_name,
        mode = "py_common",
        dep = ":%s" % info_collector_name,
        linker_input_filters = "%s" % linker_input_filters_name,
        testonly = testonly,
        compatible_with = compatible_with,
    )

    common_py_cc_binary_name = "%s_py_internal" % name
    common_py_import_name = _construct_common_binary(
        common_py_cc_binary_name,
        [
            ":%s" % py_common_split_name,
            ":%s" % common_import_name,
            "@pybind11//:pybind11",
        ],
        py_cc_linkopts,
        testonly,
        compatible_with,
        py_cc_win_def_file,
        None,
    )

    common_deps = extra_deps + [
        ":%s" % common_py_import_name,
        ":%s" % common_import_name,
    ]
    binaries_data = [
        ":%s" % common_cc_binary_name,
        ":%s" % common_py_cc_binary_name,
    ]

    # 2) Create individual super-thin pywrap libraries, which depend on the
    # common one. The individual libraries must link in statically only the
    # object file with Python Extension's init function PyInit_<extension_name>
    #
    shared_objects = []
    for pywrap_index in range(0, actual_pywrap_count):
        dep_name = "_%s_%s" % (name, pywrap_index)
        shared_object_name = "%s_shared_object" % dep_name
        win_def_name = "%s_win_def" % dep_name
        pywrap_name = "%s_pywrap" % dep_name

        _pywrap_split_library(
            name = pywrap_name,
            mode = "pywrap",
            dep = ":%s" % info_collector_name,
            linker_input_filters = "%s" % linker_input_filters_name,
            pywrap_index = pywrap_index,
            testonly = testonly,
            compatible_with = compatible_with,
        )

        _generated_win_def_file(
            name = win_def_name,
            dep = ":%s" % info_collector_name,
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
    pywrap_binaries_name = "%s_internal_binaries" % name
    _pywrap_binaries(
        name = pywrap_binaries_name,
        collected_pywraps = ":%s" % info_collector_name,
        deps = shared_objects,
        extension = select({
            "@bazel_tools//src/conditions:windows": ".pyd",
            "//conditions:default": ".so",
        }),
        wheel_locations_json = ":%s_wheel_locations.json" % pywrap_binaries_name,
        testonly = testonly,
        compatible_with = compatible_with,
    )

    binaries_data.append("%s" % pywrap_binaries_name)
    binaries_data.extend([shared_objects[0]])

    native.py_library(
        name = name,
        srcs = [":%s" % info_collector_name],
        data = binaries_data,
        testonly = testonly,
        compatible_with = compatible_with,
        visibility = visibility,
    )

    # For debugging purposes only
    native.filegroup(
        name = "_%s_all_binaries" % name,
        srcs = binaries_data,
        testonly = testonly,
        compatible_with = compatible_with,
    )

def _construct_common_binary(
        name,
        deps,
        linkopts,
        testonly,
        compatible_with,
        win_def_file,
        local_defines):
    native.cc_binary(
        name = name,
        deps = deps,
        linkstatic = True,
        linkshared = True,
        linkopts = linkopts + select({
            "@bazel_tools//src/conditions:windows": [],
            "//conditions:default": [
                "-Wl,-soname,lib%s.so" % name,
                "-Wl,-rpath='$$ORIGIN'",
            ],
        }),
        testonly = testonly,
        compatible_with = compatible_with,
        win_def_file = win_def_file,
        local_defines = local_defines,
    )

    if_lib_name = "%s_if_lib" % name
    native.filegroup(
        name = if_lib_name,
        srcs = [":%s" % name],
        output_group = "interface_library",
        testonly = testonly,
        compatible_with = compatible_with,
    )

    import_name = "%s_import" % name
    native.cc_import(
        name = import_name,
        shared_library = ":%s" % name,
        interface_library = ":%s" % if_lib_name,
        testonly = testonly,
        compatible_with = compatible_with,
    )

    return import_name

def _pywrap_split_library_impl(ctx):
    pywrap_index = ctx.attr.pywrap_index
    pywrap_infos = ctx.attr.dep[CollectedPywrapInfo].pywrap_infos.to_list()
    split_linker_inputs = []
    private_linker_inputs = []

    mode = ctx.attr.mode
    filters = ctx.attr.linker_input_filters[PywrapFilters]
    py_cc_linker_inputs = filters.py_cc_linker_inputs
    user_link_flags = []

    if mode == "pywrap":
        pw = pywrap_infos[pywrap_index]

        # print("%s matches %s" % (str(pw.owner), ctx.label))
        li = pw.cc_info.linking_context.linker_inputs.to_list()[0]
        user_link_flags.extend(li.user_link_flags)
        if not pw.cc_only:
            split_linker_inputs.append(li)
            private_linker_inputs = [
                depset(direct = filters.pywrap_private_linker_inputs[pywrap_index].keys()),
            ]
    else:
        for i in range(0, len(pywrap_infos)):
            pw = pywrap_infos[i]
            pw_private_linker_inputs = filters.pywrap_private_linker_inputs[i]
            pw_lis = pw.cc_info.linking_context.linker_inputs.to_list()[1:]
            for li in pw_lis:
                if li in pw_private_linker_inputs:
                    continue
                if li in filters.py_cc_linker_inputs:
                    if mode == "py_common":
                        split_linker_inputs.append(li)
                elif mode == "cc_common":
                    split_linker_inputs.append(li)

    dependency_libraries = _construct_dependency_libraries(
        ctx,
        split_linker_inputs,
    )

    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        libraries = depset(direct = dependency_libraries),
        user_link_flags = depset(direct = user_link_flags),
    )

    linking_context = cc_common.create_linking_context(
        linker_inputs = depset(
            direct = [linker_input],
            transitive = private_linker_inputs,
        ),
    )

    return [CcInfo(linking_context = linking_context)]

_pywrap_split_library = rule(
    attrs = {
        "dep": attr.label(
            allow_files = False,
            providers = [CollectedPywrapInfo],
        ),
        # py_deps, meaning C++ deps which depend on Python symbols
        "linker_input_filters": attr.label(
            allow_files = False,
            providers = [PywrapFilters],
            mandatory = True,
        ),
        "pywrap_index": attr.int(mandatory = False, default = -1),
        "mode": attr.string(
            mandatory = True,
            values = ["pywrap", "cc_common", "py_common"],
        ),
        "_cc_toolchain": attr.label(
            default = "@bazel_tools//tools/cpp:current_cc_toolchain",
        ),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
    implementation = _pywrap_split_library_impl,
)

def _construct_dependency_libraries(ctx, split_linker_inputs):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    dependency_libraries = []
    for split_linker_input in split_linker_inputs:
        for lib in split_linker_input.libraries:
            lib_copy = lib
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

    return dependency_libraries

def _linker_input_filters_impl(ctx):
    py_cc_linker_inputs = {}
    for py_cc_dep in ctx.attr.py_cc_deps_filter:
        for li in py_cc_dep[CcInfo].linking_context.linker_inputs.to_list()[:1]:
            py_cc_linker_inputs[li] = li.owner

    cc_linker_inputs = {}
    for cc_dep in ctx.attr.cc_deps_filter:
        for li in cc_dep[CcInfo].linking_context.linker_inputs.to_list()[:1]:
            cc_linker_inputs[li] = li.owner

    pywrap_infos = ctx.attr.dep[CollectedPywrapInfo].pywrap_infos.to_list()
    pywrap_private_linker_inputs = []

    for pw in pywrap_infos:
        private_linker_inputs = {}

        for private_dep in pw.private_deps:
            for priv_li in private_dep[CcInfo].linking_context.linker_inputs.to_list():
                if (priv_li not in py_cc_linker_inputs) and (priv_li not in cc_linker_inputs):
                    private_linker_inputs[priv_li] = priv_li.owner
        pywrap_private_linker_inputs.append(private_linker_inputs)

    return [
        PywrapFilters(
            py_cc_linker_inputs = py_cc_linker_inputs,
            pywrap_private_linker_inputs = pywrap_private_linker_inputs,
        ),
    ]

_linker_input_filters = rule(
    attrs = {
        "dep": attr.label(
            allow_files = False,
            providers = [CollectedPywrapInfo],
        ),
        "py_cc_deps_filter": attr.label_list(
            allow_files = False,
            providers = [CcInfo],
            mandatory = False,
            default = [],
        ),
        "cc_deps_filter": attr.label_list(
            allow_files = False,
            providers = [CcInfo],
            mandatory = False,
            default = [],
        ),
    },
    implementation = _linker_input_filters_impl,
)

def pywrap_common_library(name, dep):
    native.alias(
        name = name,
        actual = "%s_internal_import" % dep,
    )

def pywrap_py_common_library(name, dep):
    native.alias(
        name = name,
        actual = "%s_py_internal_import" % dep,
    )

def pywrap_binaries(name, dep):
    native.filegroup(
        name = name,
        srcs = [
            "%s_internal_binaries_wheel_locations.json" % dep,
            "%s_internal_binaries" % dep,
            "%s_py_internal" % dep,
            "%s_internal" % dep,
        ],
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
            win_def_file = win_def_file.path,
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

def _calculate_rpath(common_lib_package, current_package):
    common_pkg_components = common_lib_package.split("/")
    current_pkg_comonents = current_package.split("/")
    min_len = min(len(common_pkg_components), len(current_pkg_comonents))
    common_prefix_i = 0
    for i in range(0, min_len):
        if common_pkg_components[i] == current_pkg_comonents[i]:
            common_prefix_i = i + 1
        else:
            break

    levels_up = "../" * (len(current_pkg_comonents) - common_prefix_i)
    remaining_pkg = "/".join(common_pkg_components[common_prefix_i:])

    return levels_up + remaining_pkg

def pybind_extension(
        name,
        deps,
        srcs = [],
        private_deps = [],
        common_lib_packages = [],
        visibility = None,
        win_def_file = None,
        testonly = None,
        compatible_with = None,
        additional_exported_symbols = [],
        default_deps = ["@pybind11//:pybind11"],
        linkopts = [],
        **kwargs):
    cc_library_name = "_%s_cc_library" % name

    actual_linkopts = ["-Wl,-rpath,'$$ORIGIN/'"]
    for common_lib_package in common_lib_packages:
        origin_pkg = _calculate_rpath(common_lib_package, native.package_name())
        actual_linkopts.append("-Wl,-rpath,'$$ORIGIN/%s'" % origin_pkg)

    native.cc_library(
        name = cc_library_name,
        deps = deps + private_deps + default_deps,
        srcs = srcs,
        linkstatic = True,
        alwayslink = True,
        visibility = visibility,
        testonly = testonly,
        compatible_with = compatible_with,
        local_defines = ["PROTOBUF_USE_DLLS", "ABSL_CONSUME_DLL"],
        linkopts = linkopts + select({
            "@bazel_tools//src/conditions:windows": [],
            "//conditions:default": actual_linkopts,
        }),
        **kwargs
    )

    if not srcs:
        _cc_only_pywrap_info_wrapper(
            name = name,
            deps = ["%s" % cc_library_name],
            testonly = testonly,
            compatible_with = compatible_with,
            common_lib_packages = common_lib_packages,
            visibility = visibility,
        )
    else:
        _pywrap_info_wrapper(
            name = name,
            deps = ["%s" % cc_library_name],
            private_deps = private_deps,
            common_lib_packages = common_lib_packages,
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

    additional_exported_symbols = ctx.attr.additional_exported_symbols

    py_pkgs = []
    for pkg in ctx.attr.common_lib_packages:
        if pkg:
            py_pkgs.append(pkg.replace("/", ".") + "." + ctx.attr.name)

    if py_pkgs:
        val = "imports_paths = %s # template_val" % py_pkgs
        substitutions["imports_paths = []  # template_val"] = val

    if additional_exported_symbols:
        val = "extra_names = %s # template_val" % additional_exported_symbols
        substitutions["extra_names = []  # template_val"] = val

    ctx.actions.expand_template(
        template = ctx.file.py_stub_src,
        output = py_stub,
        substitutions = substitutions,
    )

    return [
        PyInfo(transitive_sources = depset()),
        PywrapInfo(
            cc_info = ctx.attr.deps[0][CcInfo],
            private_deps = ctx.attr.private_deps,
            owner = ctx.label,
            common_lib_packages = ctx.attr.common_lib_packages,
            py_stub = py_stub,
            cc_only = False,
        ),
    ]

_pywrap_info_wrapper = rule(
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
        "private_deps": attr.label_list(providers = [CcInfo]),
        "common_lib_packages": attr.string_list(default = []),
        "py_stub_src": attr.label(
            allow_single_file = True,
            default = Label("//third_party/py/rules_pywrap:pybind_extension.py.tpl"),
        ),
        "additional_exported_symbols": attr.string_list(
            mandatory = False,
            default = [],
        ),
    },
    implementation = _pywrap_info_wrapper_impl,
)

def _cc_only_pywrap_info_wrapper_impl(ctx):
    wrapped_dep = ctx.attr.deps[0]
    return [
        PyInfo(transitive_sources = depset()),
        PywrapInfo(
            cc_info = wrapped_dep[CcInfo],
            private_deps = [],
            owner = ctx.label,
            common_lib_packages = ctx.attr.common_lib_packages,
            py_stub = None,
            cc_only = True,
        ),
    ]

_cc_only_pywrap_info_wrapper = rule(
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
        "common_lib_packages": attr.string_list(default = []),
    },
    implementation = _cc_only_pywrap_info_wrapper_impl,
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
                order = "topological",
            ),
        ),
    ]

_pywrap_info_collector_aspect = aspect(
    attr_aspects = ["deps"],
    implementation = _pywrap_info_collector_aspect_impl,
)

def _collected_pywrap_infos_impl(ctx):
    pywrap_infos = []
    for dep in ctx.attr.deps:
        if CollectedPywrapInfo in dep:
            pywrap_infos.append(dep[CollectedPywrapInfo].pywrap_infos)

    rv = CollectedPywrapInfo(
        pywrap_infos = depset(
            transitive = pywrap_infos,
            order = "topological",
        ),
    )
    pywraps = rv.pywrap_infos.to_list()

    if ctx.attr.pywrap_count != len(pywraps):
        found_pywraps = "\n        ".join([str(pw.owner) for pw in pywraps])
        fail("""
    Number of actual pywrap libraries does not match expected pywrap_count.
    Expected pywrap_count: {expected_pywrap_count}
    Actual pywrap_count: {actual_pywra_count}
    Actual pywrap libraries in the transitive closure of {label}:
        {found_pywraps}
    """.format(
            expected_pywrap_count = ctx.attr.pywrap_count,
            actual_pywra_count = len(pywraps),
            label = ctx.label,
            found_pywraps = found_pywraps,
        ))

    py_stubs = []
    for pw in pywraps:
        if pw.py_stub:
            py_stubs.append(pw.py_stub)

    return [
        DefaultInfo(files = depset(direct = py_stubs)),
        rv,
    ]

collected_pywrap_infos = rule(
    attrs = {
        "deps": attr.label_list(
            aspects = [_pywrap_info_collector_aspect],
            providers = [PyInfo],
        ),
        "pywrap_count": attr.int(mandatory = True, default = 1),
    },
    implementation = _collected_pywrap_infos_impl,
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
    original_to_final_binaries = [
        "\n\nvvv Shared objects corresondence map, target = {} vvv".format(ctx.label),
    ]
    wheel_locations = {}
    for i in range(0, len(pywrap_infos)):
        pywrap_info = pywrap_infos[i]
        original_binary = original_binaries[i]
        subfolder = ""
        final_binary_name = "%s%s%s" % (subfolder, pywrap_info.owner.name, extension)
        final_binary = ctx.actions.declare_file(final_binary_name)
        original_binary_file = original_binary.files.to_list()[0]
        ctx.actions.run_shell(
            inputs = [original_binary_file],
            command = "cp {original} {final}".format(
                original = original_binary_file.path,
                final = final_binary.path,
            ),
            outputs = [final_binary],
        )

        original_to_final_binaries.append(
            "    '{original}' => '{final}'".format(
                original = original_binary_file.path,
                final = final_binary.path,
            ),
        )

        final_binaries.append(final_binary)

        final_binary_location = ""
        if not pywrap_info.cc_only:
            final_binary_location = "{root}{new_package}/{basename}".format(
                root = final_binary.path.split(final_binary.short_path, 1)[0],
                new_package = pywrap_info.owner.package,
                basename = final_binary.basename,
            )

        wheel_locations[final_binary.path] = final_binary_location
        if pywrap_info.py_stub:
            wheel_locations[pywrap_info.py_stub.path] = ""

    ctx.actions.write(
        output = ctx.outputs.wheel_locations_json,
        content = str(wheel_locations),
    )

    original_to_final_binaries.append(
        "^^^ Shared objects corresondence map^^^\n\n",
    )
    print("\n".join(original_to_final_binaries))

    return [DefaultInfo(files = depset(direct = final_binaries))]

_pywrap_binaries = rule(
    attrs = {
        "deps": attr.label_list(mandatory = True, allow_files = False),
        "collected_pywraps": attr.label(mandatory = True, allow_files = False),
        "extension": attr.string(default = ".so"),
        "wheel_locations_json": attr.output(mandatory = True),
    },
    implementation = _pywrap_binaries_impl,
)

def _stripped_cc_info_impl(ctx):
    filtered_libraries = []

    for dep in ctx.attr.deps:
        cc_info = dep[CcInfo]
        cc_linker_inputs = cc_info.linking_context.linker_inputs
        linker_input = cc_linker_inputs.to_list()[0]

        for lib in linker_input.libraries:
            filtered_libraries.append(lib)

    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        libraries = depset(direct = filtered_libraries),
    )

    linking_context = cc_common.create_linking_context(
        linker_inputs = depset(direct = [linker_input]),
    )

    return [CcInfo(linking_context = linking_context)]

stripped_cc_info = rule(
    attrs = {
        "deps": attr.label_list(
            allow_files = False,
            providers = [CcInfo],
        ),
    },
    implementation = _stripped_cc_info_impl,
)
