load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@rules_python//python:py_library.bzl", "py_library")

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

ObjectFiles = provider(
    fields = {
        "objects": "",
    },
)

PywrapFilters = provider(
    fields = {
        "pywrap_lib_filter": "",
        "common_lib_filters": "",
        "dynamic_lib_filter": "",
    },
)

_SELECT_TYPE = type(select({"//conditions:default": []}))
_LIST_TYPE = type([])

def pywrap_library(
        name,
        deps,
        starlark_only_deps = [],
        # Makes no sense for wrapped PyInit_ case, and should not be used
        pywrap_lib_filter = None,
        pywrap_lib_exclusion_filter = None,
        common_lib_filters = {},
        common_lib_versions = {},
        common_lib_version_scripts = {},
        common_lib_def_files_or_filters = {},
        common_lib_linkopts = {},
        enable_common_lib_starlark_only_filter = True,
        pywrap_count = None,
        starlark_only_pywrap_count = 0,
        extra_deps = ["@pybind11//:pybind11"],
        visibility = None,
        testonly = None,
        compatible_with = None):
    """A macro which does final linking of multiple C++ Python extensions tha share common parts.

    A macro which builds a set of C++ Python extensions from all of the python_extension targets
    found in deps. Note, the extensions do not have to be direct dependencies of this macro and may
    be anywhere in the transitive closure of deps.

    The python_extension macro does compilation of the C++ code of the extensions and preserves just
    enough metadata necessary for the linking of the final artifacts (.so for Linux and Mac, and
    .pyd for Windows) which is done by this macro.

    Such separation of compilation and linking workflows allows constructing of multiple C++
    extensions together in a controllable manner, making them immune to various forms of ODR
    violations, unnecessary code duplication, artifact size bloating and allows maintaining uniform
    bazel surface of the rules regardless of the underlying OS.

    E.g., for the following pywrap_library target:

    pywrap_library(
        name = "my_pywrap",
        deps = [
            ":extension_a",
            ":extension_b",
        ],
    )

    This macro will build Python C++ extension binary artifacts and will create a single public
    py_library target with the name matching this macro; the py_library target will have the
    following binaries as its data dependencies:

    - extension_a.so (extension_a.pyd on Windows)
    - extension_b.so (extension_b.pyd on Windows)
    - my_pywrap_common.so (my_pywrap_common.dylib on Mac, or my_pywrap_common.dll on Windows)

    The common my_pywrap_common.so artifact will contain all the common parts among extension_*.so
    artifacts, while each of them will depend on it.

    To use the built extensions in a py_test or a py_binary simply add my_pywrap target as a
    dependency.

    Many of the arguments to this macro are for advanced use only and designed to allow fine-tuning
    of the final common artifacts structure. Such fine-tuning is needed mainly to support backward
    comptability with previous pybind_extension implementations, and should be strictly discouraged
    in new code.

    Args:
        name: Name of the py_library target to be created, this is the only public target and the
            one to depend on in downstream targets, such as py_test or py_binary.
        deps: List of pybind_extension targets to be built by this macro together into a cohesive
            set of binary artifacts.
        starlark_only_deps: For advanced use only.
        pywrap_lib_filter: For advanced use only.
        pywrap_lib_exclusion_filter: For advanced use only.
        common_lib_filters: For advanced use only.
        common_lib_versions: For advanced use only.
        common_lib_version_scripts: A map of versions scripts to control visibility of the symbols
            exposed by common artifacts. The keys are the names of the common artifacts
            (e.g.{name}_common), and the values are the labels of the version script files.
        common_lib_def_files_or_filters: Similar to common_lib_version_scripts argument, but is
            Windows-specific; accepts either direct .def files or .json symbol filter files (the
            syntax of .json filter file mimics .lds format for version scripts on Linux); filtering
            is necessary to deal with 2^16 exported symbols limit for a single .dll on Winodows
            platform.
        common_lib_linkopts: Linkopts for common artifacts. The Linkopts for each individual
            extension must be provided directly to python_extension targets instead.
        enable_common_lib_starlark_only_filter: For advanced use only.
        pywrap_count: Number of python_extension artifacts in the transitive closure of deps; if
            the number python_extensions found in deps does not match this number an error will be
            thrown. This parameter is necesary for technical reasons. If not provided it will be
            assumed to be equal to the size of deps, meaning deps contains each and every
            python_extension target directly and nothing else.
        starlark_only_pywrap_count: For advanced use only.
        extra_deps: Extra dependencies to be added to the common library.
        visibility: The visibility argument of the resultant py_library target.
        testonly: The testonly argument of the resultant py_library target.
        compatible_with: The compatible_with of the py_library target.
    """

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
        enable_common_lib_starlark_only_filter = enable_common_lib_starlark_only_filter,
    )

    common_deps = [] + extra_deps
    starlark_only_common_deps = []
    binaries_data = {}
    starlark_only_binaries_data = {}
    win_binaries_data = {}
    win_starlark_only_binaries_data = {}
    internal_binaries = []
    win_internal_binaries = []

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
            collect_objects = select({
                "@bazel_tools//src/conditions:windows": True,
                "//conditions:default": False,
            }),
        )

        win_def_name = "_%s_def" % common_lib_name
        def_file_or_filter = common_lib_def_files_or_filters.get(
            common_lib_full_name,
            None,
        )
        generated_common_win_def_file(
            name = win_def_name,
            dep = ":%s" % common_split_name,
            filter = def_file_or_filter,
        )

        linkopts = common_lib_linkopts.get(common_lib_full_name, [])
        ver_script = common_lib_version_scripts.get(common_lib_full_name, None)
        common_cc_binary_name = "%s" % common_lib_name

        common_import_name, win_import_library_name = _construct_common_binary(
            common_cc_binary_name,
            common_deps + [":%s" % common_split_name],
            linkopts,
            testonly,
            compatible_with,
            ":%s" % win_def_name,
            None,
            binaries_data.values(),
            common_lib_pkg,
            ver_script,
            [":%s" % common_split_name],
            common_lib_versions.get(common_lib_full_name, ""),
        )
        actual_binaries_data = binaries_data
        actual_common_deps = common_deps
        actual_win_binaries_data = win_binaries_data
        if common_lib_full_name == starlark_only_filter_full_name:
            actual_binaries_data = starlark_only_binaries_data
            actual_common_deps = starlark_only_common_deps
            actual_win_binaries_data = win_starlark_only_binaries_data
        internal_binaries.append(":%s" % common_cc_binary_name)
        win_internal_binaries.append(":%s" % win_import_library_name)
        actual_binaries_data[":%s" % common_cc_binary_name] = common_lib_pkg
        actual_win_binaries_data[":%s" % win_import_library_name] = common_lib_pkg
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

    win_binaries_data.update(binaries_data)
    win_starlark_only_binaries_data.update(starlark_only_binaries_data)

    _pywrap_binaries(
        name = pywrap_binaries_name,
        collected_pywraps = ":%s" % info_collector_name,
        deps = shared_objects,
        common_binaries = select({
            "@bazel_tools//src/conditions:windows": win_binaries_data,
            "//conditions:default": binaries_data,
        }),
        starlark_only_common_binaries = select({
            "@bazel_tools//src/conditions:windows": win_starlark_only_binaries_data,
            "//conditions:default": starlark_only_binaries_data,
        }),
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

    py_library(
        name = name,
        srcs = [":%s" % info_collector_name],
        data = all_binaries_data,
        testonly = testonly,
        compatible_with = compatible_with,
        visibility = visibility,
    )

    native.filegroup(
        name = name + "_all_binaries",
        srcs = select({
            "@bazel_tools//src/conditions:windows": internal_binaries + win_internal_binaries,
            "//conditions:default": internal_binaries,
        }),
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
        data,
        version = ""):
    version_str = ".{}".format(version) if version else version
    linux_binary_name = "lib{}.so{}".format(name, version_str)
    win_binary_name = "{}{}.dll".format(name, version_str)
    darwin_binary_name = "lib{}{}.dylib".format(name, version_str)

    actual_version_script = None
    if version_script:
        actual_version_script = "{}_version_script".format(name)
        native.alias(
            name = actual_version_script,
            actual = version_script,
        )
        actual_version_script = ":{}".format(actual_version_script)

    linux_linkopts = _construct_linkopt_soname(
        linux_binary_name,
        False,
    ) + _construct_linkopt_rpaths(
        dependency_common_lib_packages,
        dependent_common_lib_package,
        False,
    ) + _construct_linkopt_version_script(actual_version_script, False)

    native.cc_binary(
        name = linux_binary_name,
        deps = deps + ([actual_version_script] if actual_version_script else []),
        linkstatic = True,
        linkshared = True,
        linkopts = linkopts + select({
            "@bazel_tools//src/conditions:windows": [],
            "@bazel_tools//src/conditions:darwin": [],
            "//conditions:default": linux_linkopts,
        }),
        testonly = testonly,
        compatible_with = compatible_with,
        local_defines = local_defines,
    )

    native.cc_binary(
        name = win_binary_name,
        deps = deps,
        linkstatic = True,
        linkshared = True,
        linkopts = linkopts + select({
            "@bazel_tools//src/conditions:windows": [],
            "@bazel_tools//src/conditions:darwin": [],
            "//conditions:default": [],
        }),
        testonly = testonly,
        compatible_with = compatible_with,
        win_def_file = win_def_file,
        local_defines = local_defines,
    )

    darwin_linkopts = _construct_linkopt_soname(
        darwin_binary_name,
        True,
    ) + _construct_linkopt_rpaths(
        dependency_common_lib_packages,
        dependent_common_lib_package,
        True,
    ) + _construct_linkopt_version_script(actual_version_script, True)

    native.cc_binary(
        name = darwin_binary_name,
        deps = deps + ([actual_version_script] if actual_version_script else []),
        linkstatic = True,
        linkshared = True,
        linkopts = linkopts + select({
            "@bazel_tools//src/conditions:windows": [],
            "@bazel_tools//src/conditions:darwin": darwin_linkopts,
            "//conditions:default": [],
        }),
        testonly = testonly,
        compatible_with = compatible_with,
        local_defines = local_defines,
    )

    if_lib_name = "{}{}_if_lib".format(name, version_str)
    native.filegroup(
        name = if_lib_name,
        srcs = [":%s" % win_binary_name],
        output_group = "interface_library",
        testonly = testonly,
        compatible_with = compatible_with,
    )

    native.alias(
        name = name,
        actual = select({
            "@bazel_tools//src/conditions:windows": ":%s" % win_binary_name,
            "@bazel_tools//src/conditions:darwin": ":%s" % darwin_binary_name,
            "//conditions:default": ":%s" % linux_binary_name,
        }),
    )

    import_name = "%s_import" % name

    native.cc_import(
        name = import_name,
        shared_library = "%s" % name,
        interface_library = select({
            "@bazel_tools//src/conditions:windows": ":%s" % if_lib_name,
            "//conditions:default": None,
        }),
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

    return cc_lib_name, if_lib_name

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
        ctx.attr.collect_objects,
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
        "collect_objects": attr.bool(default = False, mandatory = False),
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

    user_link_flags = []
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
                user_link_flags.extend(li.user_link_flags)
                if not pw_runfiles_merged:
                    default_runfiles = default_runfiles.merge(pw.default_runfiles)
                    pw_runfiles_merged = True

    return _construct_split_library_cc_info(
        ctx,
        split_linker_inputs,
        user_link_flags,
        [],
        default_runfiles,
        ctx.attr.collect_objects,
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
        "collect_objects": attr.bool(default = False, mandatory = False),
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
        default_runfiles,
        collect_objects):
    dependency_libraries, objects = _construct_dependency_libraries(
        ctx,
        split_linker_inputs,
        collect_objects,
    )

    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        libraries = depset(direct = dependency_libraries),
        user_link_flags = user_link_flags,
    )

    linking_context = cc_common.create_linking_context(
        linker_inputs = depset(
            direct = [linker_input],
            transitive = private_linker_inputs,
        ),
    )

    return [
        CcInfo(linking_context = linking_context),
        ObjectFiles(objects = depset(direct = objects)),
        DefaultInfo(runfiles = default_runfiles),
    ]

def _construct_dependency_libraries(ctx, split_linker_inputs, collect_objects):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    dependency_libraries = []
    objects = []
    for split_linker_input in split_linker_inputs:
        for lib in split_linker_input.libraries:
            lib_copy = lib
            if lib.static_library or lib.pic_static_library:
                if collect_objects:
                    objects.extend(lib.objects)
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

    return dependency_libraries, objects

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
        if ctx.attr.enable_common_lib_starlark_only_filter:
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
        "enable_common_lib_starlark_only_filter": attr.bool(
            mandatory = False,
            default = True,
        ),
    },
    implementation = _linker_input_filters_impl,
)

def pywrap_common_library(name, dep, filter_name = None, **kwargs):
    native.alias(
        name = name,
        actual = "%s_cc_library" % (filter_name if filter_name else dep + "_common"),
        **kwargs
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
        mnemonic = "PywrapWinDefFile",
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

def python_extension(
        name,
        deps,
        srcs = [],
        common_lib_packages = [],
        visibility = None,
        win_def_file = None,
        testonly = None,
        compatible_with = None,
        additional_exported_symbols = [],
        default_deps = [],
        linkopts = [],
        starlark_only = False,
        local_defines = [],
        wrap_py_init = None,
        **kwargs):
    """A macro responsible for creating each individual Python C++ extension

    This macro consists of consists of two parts:a cc_library compiling the extension and a custom
    rule which preserves enough information for pywrap_library to be able to do its job of linking
    multiple extensions together.

    Different python_extension tarets may depend on each other, depend on or be depended on by
    any number of py_library targets. To use python_extension in py_test or py_binary, do not depend
    on it directly, instead create a pywrap_library target, which should depend on all
    python_extensions needed in your test or binary, and then depend on pywrap_library itself. This
    is necessary because the actual construction of binary artifacts happens in pywrap_library.

    Args:
        name: The name of the extension, it must match the name of actual Python extension module;
            the package of the module will correspond to the bazel package of the target.
        deps: The C++ dependencies of the extension.
        srcs: The C++ sources of the extension.
        common_lib_packages: The list of packages for all the pywrap_library targets this
            python_extension is supposed to be used in. This argument exists for technical reasons.
            If you are getting NoModuleFoundError for this extension's module while running your
            code that depends on a pywrap_library (which in its turn depends on this extension),
            most likely you need to add the name of the problematic pywrap_library package in this
            list.
        visibility: The visibility of the extension target.
        win_def_file: The win_def_file of the extension.
        testonly: The testonly argument of the extension.
        compatible_with: The compatible_with of the extension.
        additional_exported_symbols: For advanced use only.
        default_deps: The default dependencies of the extension.
        linkopts: The linkopts of the extension.
        starlark_only: For advanced use only.
        local_defines: The local defines of the extension.
        wrap_py_init: Whether to wrap the PyInit_* function, making the extension artifact
            super-thin, containin only one PyInit_{name} function, with the rest of the logic being
            linked in the common artifact. Use this if you want to expose as little symbols as
            possible from common artifacts. It also may be very handy for Windows development, as
            linking multiple dynamic libraries together is much harder on Windows.
        **kwargs: Additional arguments to pass to the cc_library.
    """

    # For backward compatibility that I don't want to mess with
    _ignore = [additional_exported_symbols]

    if not srcs:
        wrap_py_init = False

    cc_library_deps = deps + default_deps
    wrapped_cc_library_name = "_%s__wrapped__cc_library" % name

    # If no wrapping is requested, this target will simply remain unused and
    # never compiled
    native.cc_library(
        name = wrapped_cc_library_name,
        deps = cc_library_deps,
        srcs = srcs,
        linkstatic = True,
        alwayslink = True,
        visibility = visibility,
        testonly = testonly,
        compatible_with = compatible_with,
        linkopts = linkopts,
        local_defines = local_defines + _if_wrapped_py_init(
            wrap_py_init,
            ["PyInit_{}=Wrapped_PyInit_{}".format(name, name)],
        ),
        **kwargs
    )

    cc_library_name = "_%s_cc_library" % name
    native.cc_library(
        name = cc_library_name,
        deps = _if_wrapped_py_init(
            wrap_py_init,
            [":{}".format(wrapped_cc_library_name)],
            cc_library_deps,
            cc_library_name,
            "deps",
        ),
        srcs = _if_wrapped_py_init(
            wrap_py_init,
            [Label(":wrapped_py_init.cc")],
            srcs,
            cc_library_name,
            "srcs",
        ),
        linkstatic = True,
        alwayslink = True,
        visibility = visibility,
        testonly = testonly,
        compatible_with = compatible_with,
        local_defines = local_defines + _if_wrapped_py_init(
            wrap_py_init,
            ["WRAPPED_PY_MODULE_NAME={}".format(name)],
        ),
        linkopts = linkopts + select({
            "@bazel_tools//src/conditions:windows": [],
            "@bazel_tools//src/conditions:darwin": _construct_linkopt_rpaths(
                common_lib_packages + [native.package_name()],
                native.package_name(),
                True,
            ),
            "//conditions:default": _construct_linkopt_rpaths(
                common_lib_packages + [native.package_name()],
                native.package_name(),
                False,
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
            starlark_only = starlark_only,
            testonly = testonly,
            compatible_with = compatible_with,
            visibility = visibility,
        )

def _if_wrapped_py_init(wrap_py_init, if_true = [], if_false = [], dep_name = "", dep_type = ""):
    if wrap_py_init == None:
        return select({
            Label(":config_wrap_py_init"): _wrap_cc_select(dep_name, dep_type, if_true),
            "//conditions:default": _wrap_cc_select(dep_name, dep_type, if_false),
        })

    return if_true if wrap_py_init else if_false

def _wrap_cc_select(name, dep_type, deps):
    if type(deps) == _SELECT_TYPE:
        wrapping_select_target = "_{}_{}".format(name, dep_type)
        if dep_type == "deps":
            native.cc_library(
                name = wrapping_select_target,
                deps = deps,
            )
        else:
            native.filegroup(
                name = wrapping_select_target,
                srcs = deps,
            )

        return [":{}".format(wrapping_select_target)]
    else:
        return deps

# For backward compatibility with the old name
def pybind_extension(name, default_deps = None, **kwargs):
    """Wrapper around pybind_extension that specifies default dependency on pybind11. 

    Note that python_extension works with nanobind as well.

    Args:
        name: Same as in python_extension.
        default_deps: The default dependencies of the extension, if not specified, the default
            dependency on pybind11 will be added.
        **kwargs: Additional arguments to pass to the python_extension.
    """

    actual_default_deps = ["@pybind11//:pybind11"]
    if default_deps != None:
        actual_default_deps = default_deps
    python_extension(
        name = name,
        default_deps = actual_default_deps,
        **kwargs
    )

def _pywrap_info_wrapper_impl(ctx):
    #the attribute is called deps not dep to match aspect's attr_aspects
    if len(ctx.attr.deps) != 1:
        fail("deps attribute must contain exactly one dependency")

    py_stub = ctx.actions.declare_file("%s.py" % ctx.attr.name)
    substitutions = {}
    py_pkgs = []
    for pkg in ctx.attr.common_lib_packages:
        if pkg:
            py_pkgs.append(pkg.replace("/", ".") + "." + ctx.attr.name)

    if py_pkgs:
        val = "imports_paths = %s # template_val" % py_pkgs
        substitutions["imports_paths = []  # template_val"] = val

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
            default = Label(":pybind_extension.py.tpl"),
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
        "\n\nvvv Shared objects correspondence map, target = {} vvv".format(ctx.label),
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
            mnemonic = "PywrapBinaryRename",
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
        "^^^ Shared objects correspondence map^^^\n\n",
    )
    # print("\n".join(original_to_final_binaries))

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
        if type(common_lib_v) == _LIST_TYPE or type(common_lib_v) == _SELECT_TYPE:
            new_common_lib_k = "_%s_common_lib_filter" % common_lib_k.rsplit("/", 1)[-1]
            native.cc_library(
                name = new_common_lib_k,
                deps = common_lib_v,
            )

        inverse_common_lib_filters[new_common_lib_k] = common_lib_k
    return inverse_common_lib_filters

def _construct_linkopt_soname(name, darwin):
    soname = name.rsplit("/", 1)[1] if "/" in name else name
    soname = soname if name.startswith("lib") else ("lib{}".format(soname))
    extension = ".so"
    arg_name = "-soname"
    if darwin:
        extension = ".dylib"
        arg_name = "-install_name"
        soname = "@rpath/" + soname
    if extension not in name:
        soname += extension
    return ["-Wl,{},{}".format(arg_name, soname)]

def _construct_linkopt_rpaths(dependency_lib_packages, dependent_lib_package, darwin):
    linkopts = {}
    origin = "@loader_path" if darwin else "$$ORIGIN"
    for dependency_lib_package in dependency_lib_packages:
        origin_pkg = _construct_rpath(dependency_lib_package, dependent_lib_package)
        linkopts["-rpath,'{}/{}'".format(origin, origin_pkg)] = True
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

def _construct_linkopt_version_script(version_script, darwin):
    if not version_script:
        return []
    arg_name = "-exported_symbols_list" if darwin else "--version-script"
    return ["-Wl,{},$(location {})".format(arg_name, version_script)]

def _generated_common_win_def_file_impl(ctx):
    win_raw_def_file_name = "%s.gen.def" % ctx.attr.name
    if ctx.attr.filter:
        if ctx.file.filter.extension != "json":
            return [DefaultInfo(files = depset(direct = [ctx.file.filter]))]
        win_raw_def_file_name = "%s.raw.gen.def" % ctx.attr.name
    win_raw_def_file = ctx.actions.declare_file(win_raw_def_file_name)

    args = ctx.actions.args()
    args.add(win_raw_def_file)
    args.add("")
    obj_files_args = ctx.actions.args()
    obj_files_args.add_all(ctx.attr.dep[ObjectFiles].objects)
    obj_files_args.use_param_file("@%s", use_always = True)
    obj_files_args.set_param_file_format("multiline")

    ctx.actions.run(
        inputs = ctx.attr.dep[ObjectFiles].objects,
        tools = [ctx.executable.parser],
        executable = ctx.executable.parser,
        arguments = [args, obj_files_args],
        outputs = [win_raw_def_file],
        mnemonic = "WinDefFileParse",
    )

    win_def_file = win_raw_def_file
    if ctx.attr.filter:
        win_def_file_name = "%s.gen.def" % ctx.attr.name
        win_def_file = ctx.actions.declare_file(win_def_file_name)

        filter_args = ctx.actions.args()
        filter_args.add("--def-file", win_raw_def_file)
        filter_args.add("--def-file-filter", ctx.file.filter)
        filter_args.add("--filtered-def-file", win_def_file)

        ctx.actions.run(
            inputs = [win_raw_def_file, ctx.file.filter],
            tools = [ctx.executable.filter_tool],
            executable = ctx.executable.filter_tool,
            arguments = [filter_args],
            outputs = [win_def_file],
            mnemonic = "WinDefFileFilter",
        )

    return [DefaultInfo(files = depset(direct = [win_def_file]))]

generated_common_win_def_file = rule(
    attrs = {
        "dep": attr.label(
            providers = [ObjectFiles],
            mandatory = True,
        ),
        "filter": attr.label(
            allow_single_file = True,
            mandatory = False,
        ),
        "parser": attr.label(
            allow_single_file = True,
            default = Label("@bazel_tools//tools/def_parser:def_parser"),
            executable = True,
            cfg = "host",
        ),
        "filter_tool": attr.label(
            default = Label(":def_file_filter_tool"),
            executable = True,
            cfg = "host",
        ),
    },
    implementation = _generated_common_win_def_file_impl,
)
