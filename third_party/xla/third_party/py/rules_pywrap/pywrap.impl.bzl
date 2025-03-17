load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

PywrapInfo = provider(
    fields = {
        "cc_info": "Wrapped CcInfo",
        "owner": "Owner's label",
        "common_lib_packages": "Packages in which to search for common pywrap library",
        "py_stub": "Pybind Python stub used to resolve cross-package references",
        "cc_only": "True if this PywrapInfo represents cc-only library (no PyIni_)",
        "starlark_only": "",
        "default_runfiles": "",
    },
)

CollectedPywrapInfo = provider(
    fields = {
        "pywrap_infos": "depset of PywrapInfo providers",
    },
)

PywrapFilters = provider(
    fields = {
        "pywrap_lib_filter": "",
        "common_lib_filters": "",
        "dynamic_lib_filter": "",
    },
)

def pywrap_library(
        name,
        deps,
        starlark_only_deps = [],
        pywrap_lib_filter = None,
        pywrap_lib_exclusion_filter = None,
        common_lib_filters = {},
        common_lib_version_scripts = {},
        common_lib_linkopts = {},
        win_def_file = None,
        pywrap_count = None,
        starlark_only_pywrap_count = 0,
        extra_deps = ["@pybind11//:pybind11"],
        visibility = None,
        testonly = None,
        compatible_with = None):
    # 0) If pywrap_count is not specified, assume we pass pybind_extension,
    # targets directly, so actual pywrap_count should just be equal to  number
    # of deps.
    actual_pywrap_count = len(deps) if pywrap_count == None else pywrap_count
    if starlark_only_deps:
        starlark_only_pywrap_count = len(starlark_only_deps)
    actual_deps = deps + starlark_only_deps

    # 1) Create common libraries cc-only (C API) and py-specific (parts reused
    # by different pywrap libraries but dependin on Python symbols).
    # The common library should link in everything except the object file with
    # Python Extension's init function PyInit_<extension_name>.
    info_collector_name = "_%s_info_collector" % name
    collected_pywrap_infos(
        name = info_collector_name,
        deps = actual_deps,
        pywrap_count = actual_pywrap_count,
        starlark_only_pywrap_count = starlark_only_pywrap_count,
    )

    linker_input_filters_name = "_%s_linker_input_filters" % name

    cur_pkg = native.package_name()
    cur_pkg = cur_pkg + "/" if native.package_name() else cur_pkg
    starlark_only_filter_full_name = None
    if starlark_only_pywrap_count > 0:
        starlark_only_filter_full_name = "%s%s__starlark_only_common" % (cur_pkg, name)

    inverse_common_lib_filters = _construct_inverse_common_lib_filters(
        common_lib_filters,
    )

    _linker_input_filters(
        name = linker_input_filters_name,
        dep = ":%s" % info_collector_name,
        pywrap_lib_filter = pywrap_lib_filter,
        pywrap_lib_exclusion_filter = pywrap_lib_exclusion_filter,
        common_lib_filters = inverse_common_lib_filters,
        starlark_only_filter_name = starlark_only_filter_full_name,
    )

    common_deps = []
    starlark_only_common_deps = []
    binaries_data = {}
    starlark_only_binaries_data = {}
    internal_binaries = []

    common_lib_full_names = []
    common_lib_full_names.extend(common_lib_filters.keys())
    common_lib_full_names.append("%s%s_common" % (cur_pkg, name))
    if starlark_only_filter_full_name:
        common_lib_full_names.append(starlark_only_filter_full_name)

    for common_lib_full_name in common_lib_full_names:
        common_lib_pkg, common_lib_name = _get_common_lib_package_and_name(
            common_lib_full_name,
        )
        common_split_name = "_%s_split" % common_lib_name
        _pywrap_common_split_library(
            name = common_split_name,
            dep = ":%s" % info_collector_name,
            common_lib_full_name = common_lib_full_name,
            linker_input_filters = "%s" % linker_input_filters_name,
            testonly = testonly,
            compatible_with = compatible_with,
        )
        ver_script = common_lib_version_scripts.get(common_lib_full_name, None)
        linkopts = common_lib_linkopts.get(common_lib_full_name, [])

        common_cc_binary_name = "%s" % common_lib_name
        common_import_name = _construct_common_binary(
            common_cc_binary_name,
            common_deps + [":%s" % common_split_name],
            linkopts,
            testonly,
            compatible_with,
            win_def_file,
            None,
            binaries_data.values(),
            common_lib_pkg,
            ver_script,
            data = [":%s" % common_split_name],
        )
        actual_binaries_data = binaries_data
        actual_common_deps = common_deps
        if common_lib_full_name == starlark_only_filter_full_name:
            actual_binaries_data = starlark_only_binaries_data
            actual_common_deps = starlark_only_common_deps
        internal_binaries.append(":%s" % common_cc_binary_name)
        actual_binaries_data[":%s" % common_cc_binary_name] = common_lib_pkg
        actual_common_deps.append(":%s" % common_import_name)

    # 2) Create individual super-thin pywrap libraries, which depend on the
    # common one. The individual libraries must link in statically only the
    # object file with Python Extension's init function PyInit_<extension_name>
    #
    shared_objects = []
    for pywrap_index in range(0, actual_pywrap_count + starlark_only_pywrap_count):
        dep_name = "_%s_%s" % (name, pywrap_index)
        shared_object_name = "%s_shared_object" % dep_name
        win_def_name = "%s_win_def" % dep_name
        pywrap_name = "%s_pywrap" % dep_name

        _pywrap_split_library(
            name = pywrap_name,
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

        actual_common_deps = common_deps
        if pywrap_index >= actual_pywrap_count:
            actual_common_deps = common_deps + starlark_only_common_deps

        native.cc_binary(
            name = shared_object_name,
            srcs = [],
            deps = [":%s" % pywrap_name] + actual_common_deps,
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
    pywrap_binaries_name = "%s_common_binaries" % name
    wheel_locations_json_name = ":%s_wheel_locations.json" % pywrap_binaries_name
    _pywrap_binaries(
        name = pywrap_binaries_name,
        collected_pywraps = ":%s" % info_collector_name,
        deps = shared_objects,
        common_binaries = binaries_data,
        starlark_only_common_binaries = starlark_only_binaries_data,
        extension = select({
            "@bazel_tools//src/conditions:windows": ".pyd",
            "//conditions:default": ".so",
        }),
        wheel_locations_json = wheel_locations_json_name,
        testonly = testonly,
        compatible_with = compatible_with,
    )
    internal_binaries.append(":%s" % pywrap_binaries_name)
    internal_binaries.append(wheel_locations_json_name)

    all_binaries_data = list(binaries_data.keys())
    all_binaries_data.extend(starlark_only_binaries_data.keys())
    all_binaries_data.append(":%s" % pywrap_binaries_name)
    all_binaries_data.extend([shared_objects[-1]])

    native.py_library(
        name = name,
        srcs = [":%s" % info_collector_name],
        data = all_binaries_data,
        testonly = testonly,
        compatible_with = compatible_with,
        visibility = visibility,
    )

    native.filegroup(
        name = name + "_all_binaries",
        srcs = internal_binaries,
    )

def _construct_common_binary(
        name,
        deps,
        linkopts,
        testonly,
        compatible_with,
        win_def_file,
        local_defines,
        dependency_common_lib_packages,
        dependent_common_lib_package,
        version_script,
        data):
    actual_linkopts = _construct_linkopt_soname(name) + _construct_linkopt_rpaths(
        dependency_common_lib_packages,
        dependent_common_lib_package,
    ) + _construct_linkopt_version_script(version_script)

    native.cc_binary(
        name = name,
        deps = deps + ([version_script] if version_script else []),
        linkstatic = True,
        linkshared = True,
        linkopts = linkopts + select({
            "@bazel_tools//src/conditions:windows": [],
            "//conditions:default": actual_linkopts,
        }),
        testonly = testonly,
        compatible_with = compatible_with,
        win_def_file = win_def_file,
        local_defines = local_defines,
        #        data = data,
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
        # TODO: put it back to fix Windows
        #        interface_library = ":%s" % if_lib_name,
        testonly = testonly,
        compatible_with = compatible_with,
    )

    cc_lib_name = "%s_cc_library" % name
    native.cc_library(
        name = cc_lib_name,
        deps = [":%s" % import_name],
        testonly = testonly,
        data = data,
    )

    return import_name

def _pywrap_split_library_impl(ctx):
    pywrap_index = ctx.attr.pywrap_index
    pw_list = ctx.attr.dep[CollectedPywrapInfo].pywrap_infos.to_list()
    pw = pw_list[pywrap_index]
    linker_inputs = pw.cc_info.linking_context.linker_inputs.to_list()
    li = linker_inputs[0]
    user_link_flags = li.user_link_flags

    split_linker_inputs = []
    private_linker_inputs = []
    default_runfiles = None
    if not pw.cc_only:
        split_linker_inputs.append(li)
        pywrap_lib_filter = ctx.attr.linker_input_filters[PywrapFilters].pywrap_lib_filter
        private_lis = []
        for li in linker_inputs[1:]:
            if li in pywrap_lib_filter:
                private_lis.append(li)
        private_linker_inputs = [
            depset(direct = private_lis),
        ]

    #        default_runfiles = pw.default_runfiles

    return _construct_split_library_cc_info(
        ctx,
        split_linker_inputs,
        user_link_flags,
        private_linker_inputs,
        default_runfiles,
    )

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
        "_cc_toolchain": attr.label(
            default = "@bazel_tools//tools/cpp:current_cc_toolchain",
        ),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
    implementation = _pywrap_split_library_impl,
)

def _pywrap_common_split_library_impl(ctx):
    pywrap_infos = ctx.attr.dep[CollectedPywrapInfo].pywrap_infos.to_list()
    split_linker_inputs = []

    filters = ctx.attr.linker_input_filters[PywrapFilters]

    libs_to_exclude = {}
    libs_to_include = {}
    include_all_not_excluded = False

    if ctx.attr.common_lib_full_name not in filters.common_lib_filters:
        for common_lib_filter in filters.common_lib_filters.values():
            libs_to_exclude.update(common_lib_filter)
        include_all_not_excluded = True
    else:
        libs_to_include = filters.common_lib_filters[ctx.attr.common_lib_full_name]

    user_link_flags = {}
    dynamic_lib_filter = filters.dynamic_lib_filter
    default_runfiles = ctx.runfiles()
    for pw in pywrap_infos:
        pw_lis = pw.cc_info.linking_context.linker_inputs.to_list()[1:]
        pw_runfiles_merged = False
        for li in pw_lis:
            if li in libs_to_exclude:
                continue
            if include_all_not_excluded or (li in libs_to_include) or li in dynamic_lib_filter:
                split_linker_inputs.append(li)
                for user_link_flag in li.user_link_flags:
                    user_link_flags[user_link_flag] = True
                if not pw_runfiles_merged:
                    default_runfiles = default_runfiles.merge(pw.default_runfiles)
                    pw_runfiles_merged = True

    return _construct_split_library_cc_info(
        ctx,
        split_linker_inputs,
        list(user_link_flags.keys()),
        [],
        default_runfiles,
    )

_pywrap_common_split_library = rule(
    attrs = {
        "dep": attr.label(
            allow_files = False,
            providers = [CollectedPywrapInfo],
        ),
        "common_lib_full_name": attr.string(mandatory = True),
        "linker_input_filters": attr.label(
            allow_files = False,
            providers = [PywrapFilters],
            mandatory = True,
        ),
        "_cc_toolchain": attr.label(
            default = "@bazel_tools//tools/cpp:current_cc_toolchain",
        ),
    },
    fragments = ["cpp"],
    toolchains = use_cpp_toolchain(),
    implementation = _pywrap_common_split_library_impl,
)

def _construct_split_library_cc_info(
        ctx,
        split_linker_inputs,
        user_link_flags,
        private_linker_inputs,
        default_runfiles):
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

    return [
        CcInfo(linking_context = linking_context),
        #        DefaultInfo(files = default_runfiles.files)
        DefaultInfo(runfiles = default_runfiles),
    ]

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
            if not lib.alwayslink and (lib.static_library or lib.pic_static_library):
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
    pywrap_lib_exclusion_filter = {}
    pywrap_lib_filter = {}
    visited_filters = {}

    #
    # pywrap private filter
    #
    if ctx.attr.pywrap_lib_exclusion_filter:
        for li in ctx.attr.pywrap_lib_exclusion_filter[CcInfo].linking_context.linker_inputs.to_list():
            pywrap_lib_exclusion_filter[li] = li.owner

    if ctx.attr.pywrap_lib_filter:
        for li in ctx.attr.pywrap_lib_filter[CcInfo].linking_context.linker_inputs.to_list():
            if li not in pywrap_lib_exclusion_filter:
                pywrap_lib_filter[li] = li.owner

    common_lib_filters = {k: {} for k in ctx.attr.common_lib_filters.values()}

    #
    # common lib filters
    #
    for filter, name in ctx.attr.common_lib_filters.items():
        filter_li = filter[CcInfo].linking_context.linker_inputs.to_list()
        for li in filter_li:
            if li not in visited_filters:
                common_lib_filters[name][li] = li.owner
                visited_filters[li] = li.owner

    #
    # starlark -only filter
    #
    pywrap_infos = ctx.attr.dep[CollectedPywrapInfo].pywrap_infos.to_list()
    starlark_only_filter = {}

    if ctx.attr.starlark_only_filter_name:
        for pw in pywrap_infos:
            if pw.starlark_only:
                for li in pw.cc_info.linking_context.linker_inputs.to_list()[1:]:
                    starlark_only_filter[li] = li.owner

        for pw in pywrap_infos:
            if not pw.starlark_only:
                for li in pw.cc_info.linking_context.linker_inputs.to_list()[1:]:
                    starlark_only_filter.pop(li, None)

        common_lib_filters[ctx.attr.starlark_only_filter_name] = starlark_only_filter

    #
    # dynamic libs filter
    #
    dynamic_lib_filter = {}
    empty_lib_filter = {}
    for pw in pywrap_infos:
        for li in pw.cc_info.linking_context.linker_inputs.to_list()[1:]:
            all_dynamic = None
            for lib in li.libraries:
                if lib.static_library or lib.pic_static_library or not lib.dynamic_library:
                    all_dynamic = False
                    break
                elif all_dynamic == None:
                    all_dynamic = True
            if all_dynamic:
                dynamic_lib_filter[li] = li.owner

    return [
        PywrapFilters(
            pywrap_lib_filter = pywrap_lib_filter,
            common_lib_filters = common_lib_filters,
            dynamic_lib_filter = dynamic_lib_filter,
        ),
    ]

_linker_input_filters = rule(
    attrs = {
        "dep": attr.label(
            allow_files = False,
            providers = [CollectedPywrapInfo],
        ),
        "pywrap_lib_filter": attr.label(
            allow_files = False,
            providers = [CcInfo],
            mandatory = False,
        ),
        "pywrap_lib_exclusion_filter": attr.label(
            allow_files = False,
            providers = [CcInfo],
            mandatory = False,
        ),
        "common_lib_filters": attr.label_keyed_string_dict(
            allow_files = False,
            providers = [CcInfo],
            mandatory = False,
            default = {},
        ),
        "starlark_only_filter_name": attr.string(mandatory = False),
    },
    implementation = _linker_input_filters_impl,
)

def pywrap_common_library(name, dep, filter_name = None):
    native.alias(
        name = name,
        actual = "%s_cc_library" % (filter_name if filter_name else dep + "_common"),
    )

def pywrap_binaries(name, dep, **kwargs):
    native.alias(
        name = name,
        actual = "%s_all_binaries" % dep,
        **kwargs
    )
    native.alias(
        name = name + ".json",
        actual = "%s_common_binaries_wheel_locations.json" % dep,
        **kwargs
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

def pybind_extension(
        name,
        deps,
        srcs = [],
        common_lib_packages = [],
        visibility = None,
        win_def_file = None,
        testonly = None,
        compatible_with = None,
        additional_exported_symbols = [],
        default_deps = ["@pybind11//:pybind11"],
        linkopts = [],
        starlark_only = False,
        **kwargs):
    cc_library_name = "_%s_cc_library" % name
    native.cc_library(
        name = cc_library_name,
        deps = deps + default_deps,
        srcs = srcs,
        linkstatic = True,
        alwayslink = True,
        visibility = visibility,
        testonly = testonly,
        compatible_with = compatible_with,
        local_defines = ["PROTOBUF_USE_DLLS", "ABSL_CONSUME_DLL"],
        linkopts = linkopts + select({
            "@bazel_tools//src/conditions:windows": [],
            "//conditions:default": _construct_linkopt_rpaths(
                common_lib_packages + [native.package_name()],
                native.package_name(),
            ),
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
            common_lib_packages = common_lib_packages,
            additional_exported_symbols = additional_exported_symbols,
            starlark_only = starlark_only,
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

    default_runfiles = ctx.runfiles().merge(
        ctx.attr.deps[0][DefaultInfo].default_runfiles,
    )

    return [
        PyInfo(transitive_sources = depset()),
        PywrapInfo(
            cc_info = ctx.attr.deps[0][CcInfo],
            default_runfiles = default_runfiles,
            owner = ctx.label,
            common_lib_packages = ctx.attr.common_lib_packages,
            py_stub = py_stub,
            cc_only = False,
            starlark_only = ctx.attr.starlark_only,
        ),
    ]

_pywrap_info_wrapper = rule(
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
        "common_lib_packages": attr.string_list(default = []),
        "py_stub_src": attr.label(
            allow_single_file = True,
            default = Label("//third_party/py/rules_pywrap:pybind_extension.py.tpl"),
        ),
        "additional_exported_symbols": attr.string_list(
            mandatory = False,
            default = [],
        ),
        "starlark_only": attr.bool(mandatory = False, default = False),
    },
    implementation = _pywrap_info_wrapper_impl,
)

def _cc_only_pywrap_info_wrapper_impl(ctx):
    wrapped_dep = ctx.attr.deps[0]
    default_runfiles = ctx.runfiles().merge(
        ctx.attr.deps[0][DefaultInfo].default_runfiles,
    )

    return [
        PyInfo(transitive_sources = depset()),
        PywrapInfo(
            cc_info = wrapped_dep[CcInfo],
            owner = ctx.label,
            default_runfiles = default_runfiles,
            common_lib_packages = ctx.attr.common_lib_packages,
            py_stub = None,
            cc_only = True,
            starlark_only = False,
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
    pywrap_depsets = []
    for dep in ctx.attr.deps:
        if CollectedPywrapInfo in dep:
            pywrap_depsets.append(dep[CollectedPywrapInfo].pywrap_infos)

    all_pywraps = CollectedPywrapInfo(
        pywrap_infos = depset(
            transitive = pywrap_depsets,
            order = "topological",
        ),
    )

    pywraps = []
    sl_only_pywraps = []
    py_stubs = []

    for pw in all_pywraps.pywrap_infos.to_list():
        if pw.starlark_only:
            sl_only_pywraps.append(pw)
        else:
            pywraps.append(pw)
        if pw.py_stub:
            py_stubs.append(pw.py_stub)

    pw_count = ctx.attr.pywrap_count
    sl_pw_count = ctx.attr.starlark_only_pywrap_count

    if pw_count != len(pywraps) or sl_pw_count != len(sl_only_pywraps):
        found_pws = "\n        ".join([str(pw.owner) for pw in pywraps])
        found_sl_pws = "\n        ".join([str(pw.owner) for pw in sl_only_pywraps])
        fail("""
    Number of actual pywrap libraries does not match expected pywrap_count.
    Expected regular pywrap_count: {expected_pywrap_count}
    Actual regular pywrap_count: {actual_pywrap_count}
    Expected starlark-only pywrap_count: {expected_starlark_only_pywrap_count}
    Actual starlark-only pywrap_count: {starlark_only_pywrap_count}
    Actual regualar pywrap libraries in the transitive closure of {label}:
        {found_pws}
    Actual starlark-only pywrap libraries in the transitive closure of {label}:
        {found_sl_pws}
    """.format(
            expected_pywrap_count = pw_count,
            expected_starlark_only_pywrap_count = sl_pw_count,
            actual_pywrap_count = len(pywraps),
            starlark_only_pywrap_count = len(sl_only_pywraps),
            label = ctx.label,
            found_pws = found_pws,
            found_sl_pws = found_sl_pws,
        ))

    categorized_pywraps = CollectedPywrapInfo(
        pywrap_infos = depset(
            direct = pywraps,
            transitive = [depset(sl_only_pywraps)],
            order = "topological",
        ),
    )

    return [
        DefaultInfo(files = depset(direct = py_stubs)),
        categorized_pywraps,
    ]

collected_pywrap_infos = rule(
    attrs = {
        "deps": attr.label_list(
            aspects = [_pywrap_info_collector_aspect],
            providers = [PyInfo],
        ),
        "pywrap_count": attr.int(mandatory = True, default = 1),
        "starlark_only_pywrap_count": attr.int(mandatory = True, default = 0),
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
            "    '{original}' => '{final}'{starlark_only}".format(
                original = original_binary_file.path,
                final = final_binary.path,
                starlark_only = " (excluded from wheel)" if pywrap_info.starlark_only else "",
            ),
        )

        final_binaries.append(final_binary)

        final_binary_location = ""
        if not pywrap_info.cc_only and not pywrap_info.starlark_only:
            final_binary_location = _construct_final_binary_location(
                final_binary,
                pywrap_info.owner.package,
            )

        wheel_locations[final_binary.path] = final_binary_location
        if pywrap_info.py_stub:
            wheel_locations[pywrap_info.py_stub.path] = ""

    for common_binary, common_binary_pkg in ctx.attr.common_binaries.items():
        final_binary = common_binary.files.to_list()[0]
        final_binary_location = _construct_final_binary_location(
            final_binary,
            common_binary_pkg,
        )
        original_to_final_binaries.append(
            "    common lib => '{}'".format(
                final_binary.path,
            ),
        )
        wheel_locations[final_binary.path] = final_binary_location
    for starlark_only_common_binary in ctx.attr.starlark_only_common_binaries:
        final_binary = starlark_only_common_binary.files.to_list()[0]
        original_to_final_binaries.append(
            "    common lib => '{}' (excluded from wheel)".format(
                final_binary.path,
            ),
        )
        wheel_locations[final_binary.path] = ""

    ctx.actions.write(
        output = ctx.outputs.wheel_locations_json,
        content = str(wheel_locations),
    )

    original_to_final_binaries.append(
        "^^^ Shared objects corresondence map^^^\n\n",
    )
    print("\n".join(original_to_final_binaries))

    return [DefaultInfo(files = depset(direct = final_binaries))]

def _construct_final_binary_location(final_binary, new_package):
    return "{root}{new_package}/{basename}".format(
        root = final_binary.path.split(final_binary.short_path, 1)[0],
        new_package = new_package,
        basename = final_binary.basename,
    )

_pywrap_binaries = rule(
    attrs = {
        "deps": attr.label_list(mandatory = True, allow_files = False),
        "common_binaries": attr.label_keyed_string_dict(
            allow_files = False,
            mandatory = True,
        ),
        "starlark_only_common_binaries": attr.label_keyed_string_dict(
            allow_files = False,
            mandatory = True,
        ),
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

def _get_common_lib_package_and_name(common_lib_full_name):
    if "/" in common_lib_full_name:
        return common_lib_full_name.rsplit("/", 1)
    return "", common_lib_full_name

def _construct_inverse_common_lib_filters(common_lib_filters):
    inverse_common_lib_filters = {}
    for common_lib_k, common_lib_v in common_lib_filters.items():
        new_common_lib_k = common_lib_v
        if type(common_lib_v) == type([]):
            new_common_lib_k = "_%s_common_lib_filter" % common_lib_k.rsplit("/", 1)[-1]
            native.cc_library(
                name = new_common_lib_k,
                deps = common_lib_v,
            )

        inverse_common_lib_filters[new_common_lib_k] = common_lib_k
    return inverse_common_lib_filters

def _construct_linkopt_soname(name):
    soname = name.rsplit("/", 1)[1] if "/" in name else name
    soname = soname if name.startswith("lib") else ("lib%s" % soname)
    if ".so" not in name:
        soname += ".so"
    return ["-Wl,-soname,%s" % soname]

def _construct_linkopt_rpaths(dependency_lib_packages, dependent_lib_package):
    linkopts = {}
    for dependency_lib_package in dependency_lib_packages:
        origin_pkg = _construct_rpath(dependency_lib_package, dependent_lib_package)
        linkopts["-rpath,'$$ORIGIN/%s'" % origin_pkg] = True
    return ["-Wl," + ",".join(linkopts.keys())] if linkopts else []

def _construct_rpath(dependency_lib_package, dependent_lib_package):
    dependency_pkg_components = dependency_lib_package.split("/")
    dependent_pkg_comonents = dependent_lib_package.split("/")
    min_len = min(len(dependency_pkg_components), len(dependent_pkg_comonents))
    common_prefix_i = 0
    for i in range(0, min_len):
        if dependency_pkg_components[i] == dependent_pkg_comonents[i]:
            common_prefix_i = i + 1
        else:
            break

    levels_up = "../" * (len(dependent_pkg_comonents) - common_prefix_i)
    remaining_pkg = "/".join(dependency_pkg_components[common_prefix_i:])

    return levels_up + remaining_pkg

def _construct_linkopt_version_script(version_script):
    if not version_script:
        return []
    return ["-Wl,--version-script,$(location {})".format(version_script)]
