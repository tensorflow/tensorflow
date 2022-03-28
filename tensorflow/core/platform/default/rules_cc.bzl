"""Forwards rules_cc's relevant rules + cc_shared_library under a common file

We may change the implementation at any moment or even delete this file. Do not
rely on this. It requires bazel >1.2  and passing the flag
--experimental_cc_shared_library.
"""

# TODO(rostam): Delete this module after the release of Bazel built-in cc_shared_library.

load(
    "@rules_cc//cc:defs.bzl",
    _cc_binary = "cc_binary",
    _cc_import = "cc_import",
    _cc_library = "cc_library",
    _cc_test = "cc_test",
)
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load(
    "@rules_cc//examples:experimental_cc_shared_library.bzl",
    "CcSharedLibraryInfo",
    "CcSharedLibraryPermissionsInfo",
)

cc_binary = _cc_binary
cc_import = _cc_import
cc_library = _cc_library
cc_test = _cc_test

def find_cc_toolchain(ctx):
    """Returns the current `CcToolchainInfo`.

    Args:
      ctx: The rule context for which to find a toolchain.
    Returns:
      A CcToolchainInfo.
    """

    # Check the incompatible flag for toolchain resolution.
    if hasattr(cc_common, "is_cc_toolchain_resolution_enabled_do_not_use") and cc_common.is_cc_toolchain_resolution_enabled_do_not_use(ctx = ctx):
        if not "@bazel_tools//tools/cpp:toolchain_type" in ctx.toolchains:  # copybara-use-repo-external-label
            fail("In order to use find_cc_toolchain, your rule has to depend on C++ toolchain. See find_cc_toolchain.bzl docs for details.")
        toolchain_info = ctx.toolchains["@bazel_tools//tools/cpp:toolchain_type"]  # copybara-use-repo-external-label
        if hasattr(toolchain_info, "cc_provider_in_toolchain") and hasattr(toolchain_info, "cc"):
            return toolchain_info.cc
        return toolchain_info

    # Fall back to the legacy implicit attribute lookup.
    if hasattr(ctx.attr, "_cc_toolchain"):
        return ctx.attr._cc_toolchain[cc_common.CcToolchainInfo]

    # We didn't find anything.
    fail("In order to use find_cc_toolchain, your rule has to depend on C++ toolchain. See find_cc_toolchain.bzl docs for details.")

# TODO(#5200): Add export_define to library_to_link and cc_library

# Add this as a tag to any target that can be linked by more than one
# cc_shared_library because it doesn't have static initializers or anything
# else that may cause issues when being linked more than once. This should be
# used sparingly after making sure it's safe to use.
LINKABLE_MORE_THAN_ONCE = "LINKABLE_MORE_THAN_ONCE"

GraphNodeInfo = provider(
    "Nodes in the graph of shared libraries.",
    fields = {
        "children": "Other GraphNodeInfo from dependencies of this target",
        "label": "Label of the target visited",
        "linkable_more_than_once": "Linkable into more than a single cc_shared_library",
    },
)

def _separate_static_and_dynamic_link_libraries(
        direct_children,
        can_be_linked_dynamically,
        preloaded_deps_direct_labels):
    node = None
    all_children = list(direct_children)
    link_statically_labels = {}
    link_dynamically_labels = {}

    seen_labels = {}

    # Horrible I know. Perhaps Starlark team gives me a way to prune a tree.
    for i in range(2147483647):
        if i == len(all_children):
            break

        node = all_children[i]
        node_label = str(node.label)

        if node_label in seen_labels:
            continue
        seen_labels[node_label] = True

        if node_label in can_be_linked_dynamically:
            link_dynamically_labels[node_label] = True
        elif node_label not in preloaded_deps_direct_labels:
            link_statically_labels[node_label] = node.linkable_more_than_once
            all_children.extend(node.children)

    return (link_statically_labels, link_dynamically_labels)

def _create_linker_context(ctx, linker_inputs):
    return cc_common.create_linking_context(
        linker_inputs = depset(linker_inputs, order = "topological"),
    )

def _merge_cc_shared_library_infos(ctx):
    dynamic_deps = []
    transitive_dynamic_deps = []
    for dep in ctx.attr.dynamic_deps:
        if dep[CcSharedLibraryInfo].preloaded_deps != None:
            fail("{} can only be a direct dependency of a " +
                 " cc_binary because it has " +
                 "preloaded_deps".format(str(dep.label)))
        dynamic_dep_entry = (
            dep[CcSharedLibraryInfo].exports,
            dep[CcSharedLibraryInfo].linker_input,
            dep[CcSharedLibraryInfo].link_once_static_libs,
        )
        dynamic_deps.append(dynamic_dep_entry)
        transitive_dynamic_deps.append(dep[CcSharedLibraryInfo].dynamic_deps)

    return depset(direct = dynamic_deps, transitive = transitive_dynamic_deps)

def _build_exports_map_from_only_dynamic_deps(merged_shared_library_infos):
    exports_map = {}
    for entry in merged_shared_library_infos.to_list():
        exports = entry[0]
        linker_input = entry[1]
        for export in exports:
            if export in exports_map:
                fail("Two shared libraries in dependencies export the same symbols. Both " +
                     exports_map[export].libraries[0].dynamic_library.short_path +
                     " and " + linker_input.libraries[0].dynamic_library.short_path +
                     " export " + export)
            exports_map[export] = linker_input
    return exports_map

def _build_link_once_static_libs_map(merged_shared_library_infos):
    link_once_static_libs_map = {}
    for entry in merged_shared_library_infos.to_list():
        link_once_static_libs = entry[2]
        linker_input = entry[1]
        for static_lib in link_once_static_libs:
            if static_lib in link_once_static_libs_map:
                fail("Two shared libraries in dependencies link the same " +
                     " library statically. Both " + link_once_static_libs_map[static_lib] +
                     " and " + str(linker_input.owner) +
                     " link statically" + static_lib)
            link_once_static_libs_map[static_lib] = str(linker_input.owner)
    return link_once_static_libs_map

def _wrap_static_library_with_alwayslink(ctx, feature_configuration, cc_toolchain, linker_input):
    new_libraries_to_link = []
    for old_library_to_link in linker_input.libraries:
        # TODO(#5200): This will lose the object files from a library to link.
        # Not too bad for the prototype but as soon as the library_to_link
        # constructor has object parameters this should be changed.
        if old_library_to_link.static_library == None and old_library_to_link.pic_static_library == None:
            new_libraries_to_link.append(old_library_to_link)
            continue

        new_library_to_link = cc_common.create_library_to_link(
            actions = ctx.actions,
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            static_library = old_library_to_link.static_library,
            pic_static_library = old_library_to_link.pic_static_library,
            alwayslink = True,
        )
        new_libraries_to_link.append(new_library_to_link)

    return cc_common.create_linker_input(
        owner = linker_input.owner,
        libraries = depset(direct = new_libraries_to_link),
        user_link_flags = depset(direct = linker_input.user_link_flags),
        additional_inputs = depset(direct = linker_input.additional_inputs),
    )

def _check_if_target_under_path(value, pattern):
    if pattern.workspace_name != value.workspace_name:
        return False
    if pattern.name == "__pkg__":
        return pattern.package == value.package
    if pattern.name == "__subpackages__":
        return _same_package_or_above(pattern, value)

    return pattern.package == value.package and pattern.name == value.name

def _check_if_target_can_be_exported(target, current_label, permissions):
    if permissions == None:
        return True

    if (target.workspace_name != current_label.workspace_name or
        _same_package_or_above(current_label, target)):
        return True

    matched_by_target = False
    for permission in permissions:
        for permission_target in permission[CcSharedLibraryPermissionsInfo].targets:
            if _check_if_target_under_path(target, permission_target):
                return True

    return False

def _check_if_target_should_be_exported_without_filter(target, current_label, permissions):
    return _check_if_target_should_be_exported_with_filter(target, current_label, None, permissions)

def _check_if_target_should_be_exported_with_filter(target, current_label, exports_filter, permissions):
    should_be_exported = False
    if exports_filter == None:
        should_be_exported = True
    else:
        for export_filter in exports_filter:
            export_filter_label = current_label.relative(export_filter)
            if _check_if_target_under_path(target, export_filter_label):
                should_be_exported = True
                break

    if should_be_exported:
        if _check_if_target_can_be_exported(target, current_label, permissions):
            return True
        else:
            matched_by_filter_text = ""
            if exports_filter:
                matched_by_filter_text = " (matched by filter) "
            fail(str(target) + matched_by_filter_text +
                 " cannot be exported from " + str(current_label) +
                 " because it's not in the same package/subpackage and the library " +
                 "doesn't have the necessary permissions. Use cc_shared_library_permissions.")

    return False

def _filter_inputs(
        ctx,
        feature_configuration,
        cc_toolchain,
        transitive_exports,
        preloaded_deps_direct_labels,
        link_once_static_libs_map):
    linker_inputs = []
    link_once_static_libs = []

    graph_structure_aspect_nodes = []
    dependency_linker_inputs = []
    direct_exports = {}
    for export in ctx.attr.roots:
        direct_exports[str(export.label)] = True
        dependency_linker_inputs.extend(export[CcInfo].linking_context.linker_inputs.to_list())
        graph_structure_aspect_nodes.append(export[GraphNodeInfo])

    can_be_linked_dynamically = {}
    for linker_input in dependency_linker_inputs:
        owner = str(linker_input.owner)
        if owner in transitive_exports:
            can_be_linked_dynamically[owner] = True

    (link_statically_labels, link_dynamically_labels) = _separate_static_and_dynamic_link_libraries(
        graph_structure_aspect_nodes,
        can_be_linked_dynamically,
        preloaded_deps_direct_labels,
    )

    exports = {}
    owners_seen = {}
    for linker_input in dependency_linker_inputs:
        owner = str(linker_input.owner)
        if owner in owners_seen:
            continue
        owners_seen[owner] = True
        if owner in link_dynamically_labels:
            dynamic_linker_input = transitive_exports[owner]
            linker_inputs.append(dynamic_linker_input)
        elif owner in link_statically_labels:
            if owner in link_once_static_libs_map:
                fail(owner + " is already linked statically in " +
                     link_once_static_libs_map[owner] + " but not exported")

            if owner in direct_exports:
                wrapped_library = _wrap_static_library_with_alwayslink(
                    ctx,
                    feature_configuration,
                    cc_toolchain,
                    linker_input,
                )

                if not link_statically_labels[owner]:
                    link_once_static_libs.append(owner)
                linker_inputs.append(wrapped_library)
            else:
                can_be_linked_statically = False

                for static_dep_path in ctx.attr.static_deps:
                    static_dep_path_label = ctx.label.relative(static_dep_path)
                    if _check_if_target_under_path(linker_input.owner, static_dep_path_label):
                        can_be_linked_statically = True
                        break

                if _check_if_target_should_be_exported_with_filter(
                    linker_input.owner,
                    ctx.label,
                    ctx.attr.exports_filter,
                    _get_permissions(ctx),
                ):
                    exports[owner] = True
                    can_be_linked_statically = True

                if can_be_linked_statically:
                    if not link_statically_labels[owner]:
                        link_once_static_libs.append(owner)
                    linker_inputs.append(linker_input)
                else:
                    fail("We can't link " +
                         str(owner) + " either statically or dynamically")

    return (exports, linker_inputs, link_once_static_libs)

def _same_package_or_above(label_a, label_b):
    if label_a.workspace_name != label_b.workspace_name:
        return False
    package_a_tokenized = label_a.package.split("/")
    package_b_tokenized = label_b.package.split("/")
    if len(package_b_tokenized) < len(package_a_tokenized):
        return False

    if package_a_tokenized[0] != "":
        for i in range(len(package_a_tokenized)):
            if package_a_tokenized[i] != package_b_tokenized[i]:
                return False

    return True

def _get_permissions(ctx):
    if ctx.attr._enable_permissions_check[BuildSettingInfo].value:
        return ctx.attr.permissions
    return None

def _cc_shared_library_impl(ctx):
    cc_common.check_experimental_cc_shared_library()
    cc_toolchain = find_cc_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    merged_cc_shared_library_info = _merge_cc_shared_library_infos(ctx)
    exports_map = _build_exports_map_from_only_dynamic_deps(merged_cc_shared_library_info)
    for export in ctx.attr.roots:
        if str(export.label) in exports_map:
            fail("Trying to export a library already exported by a different shared library: " +
                 str(export.label))

        _check_if_target_should_be_exported_without_filter(export.label, ctx.label, _get_permissions(ctx))

    preloaded_deps_direct_labels = {}
    preloaded_dep_merged_cc_info = None
    if len(ctx.attr.preloaded_deps) != 0:
        preloaded_deps_cc_infos = []
        for preloaded_dep in ctx.attr.preloaded_deps:
            preloaded_deps_direct_labels[str(preloaded_dep.label)] = True
            preloaded_deps_cc_infos.append(preloaded_dep[CcInfo])

        preloaded_dep_merged_cc_info = cc_common.merge_cc_infos(cc_infos = preloaded_deps_cc_infos)

    link_once_static_libs_map = _build_link_once_static_libs_map(merged_cc_shared_library_info)

    (exports, linker_inputs, link_once_static_libs) = _filter_inputs(
        ctx,
        feature_configuration,
        cc_toolchain,
        exports_map,
        preloaded_deps_direct_labels,
        link_once_static_libs_map,
    )

    linking_context = _create_linker_context(ctx, linker_inputs)

    user_link_flags = []
    for user_link_flag in ctx.attr.user_link_flags:
        user_link_flags.append(ctx.expand_location(user_link_flag, targets = ctx.attr.additional_linker_inputs))

    debug_files = []
    if ctx.attr._experimental_debug[BuildSettingInfo].value:
        exports_debug_file = ctx.actions.declare_file(ctx.label.name + "_exports.txt")
        ctx.actions.write(content = "\n".join(exports.keys()), output = exports_debug_file)

        link_once_static_libs_debug_file = ctx.actions.declare_file(ctx.label.name + "_link_once_static_libs.txt")
        ctx.actions.write(content = "\n".join(link_once_static_libs), output = link_once_static_libs_debug_file)

        debug_files.append(exports_debug_file)
        debug_files.append(link_once_static_libs_debug_file)

    linking_outputs = cc_common.link(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        linking_contexts = [linking_context],
        user_link_flags = user_link_flags,
        additional_inputs = ctx.files.additional_linker_inputs + debug_files,
        name = ctx.label.name,
        output_type = "dynamic_library",
    )

    # TODO(rostam): This will go away once Tensorflow is using Bazel head
    # ---------- BEGIN ----------
    output_files = []
    if ctx.attr.shared_lib_name:
        actual_file = linking_outputs.library_to_link.resolved_symlink_dynamic_library
        if actual_file == None:
            actual_file = linking_outputs.library_to_link.dynamic_library
        output_file = ctx.actions.declare_file(ctx.attr.shared_lib_name)
        ctx.actions.run_shell(
            outputs = [output_file],
            inputs = [actual_file],
            command = "cp %s %s" % (actual_file.path, output_file.path),
            mnemonic = "TemporaryHackCopyingDynamicLibrary",
        )
        output_files = [output_file]

    # ---------- END ----------

    resolved_symlink_dynamic_library = []
    if linking_outputs.library_to_link.resolved_symlink_dynamic_library != None:
        resolved_symlink_dynamic_library = [linking_outputs.library_to_link.resolved_symlink_dynamic_library]
    runfiles_shared = resolved_symlink_dynamic_library + output_files
    if linking_outputs.library_to_link.dynamic_library != None:
        runfiles_shared.append(linking_outputs.library_to_link.dynamic_library)
    runfiles = ctx.runfiles(
        files = runfiles_shared,
        collect_data = True,
    )
    for dep in ctx.attr.dynamic_deps:
        runfiles = runfiles.merge(dep[DefaultInfo].data_runfiles)

    for export in ctx.attr.roots:
        exports[str(export.label)] = True

    if not ctx.attr._incompatible_link_once[BuildSettingInfo].value:
        link_once_static_libs = []

    return [
        DefaultInfo(
            files = depset(
                resolved_symlink_dynamic_library + debug_files + output_files,
            ),
            runfiles = runfiles,
        ),
        OutputGroupInfo(
            custom_name_shared_library = depset(output_files),
        ),
        CcSharedLibraryInfo(
            dynamic_deps = merged_cc_shared_library_info,
            exports = exports.keys(),
            link_once_static_libs = link_once_static_libs,
            linker_input = cc_common.create_linker_input(
                owner = ctx.label,
                libraries = depset([linking_outputs.library_to_link]),
            ),
            preloaded_deps = preloaded_dep_merged_cc_info,
        ),
    ]

def _graph_structure_aspect_impl(target, ctx):
    children = []

    if hasattr(ctx.rule.attr, "deps"):
        for dep in ctx.rule.attr.deps:
            if GraphNodeInfo in dep:
                children.append(dep[GraphNodeInfo])

    # TODO(bazel-team): Add flag to Bazel that can toggle the initialization of
    # linkable_more_than_once.
    linkable_more_than_once = False
    if hasattr(ctx.rule.attr, "tags"):
        for tag in ctx.rule.attr.tags:
            if tag == LINKABLE_MORE_THAN_ONCE:
                linkable_more_than_once = True

    return [GraphNodeInfo(
        label = ctx.label,
        children = children,
        linkable_more_than_once = linkable_more_than_once,
    )]

def _cc_shared_library_permissions_impl(ctx):
    targets = []
    for target_filter in ctx.attr.targets:
        target_filter_label = ctx.label.relative(target_filter)
        if not _check_if_target_under_path(target_filter_label, ctx.label.relative(":__subpackages__")):
            fail("A cc_shared_library_permissions rule can only list " +
                 "targets that are in the same package or a sub-package")
        targets.append(target_filter_label)

    return [CcSharedLibraryPermissionsInfo(
        targets = targets,
    )]

graph_structure_aspect = aspect(
    attr_aspects = ["*"],
    implementation = _graph_structure_aspect_impl,
)

cc_shared_library_permissions = rule(
    implementation = _cc_shared_library_permissions_impl,
    attrs = {
        "targets": attr.string_list(),
    },
)

cc_shared_library = rule(
    implementation = _cc_shared_library_impl,
    attrs = {
        "additional_linker_inputs": attr.label_list(allow_files = True),
        "shared_lib_name": attr.string(),
        "dynamic_deps": attr.label_list(providers = [CcSharedLibraryInfo]),
        "exports_filter": attr.string_list(),
        "permissions": attr.label_list(providers = [CcSharedLibraryPermissionsInfo]),
        "preloaded_deps": attr.label_list(providers = [CcInfo]),
        "roots": attr.label_list(providers = [CcInfo], aspects = [graph_structure_aspect]),
        "static_deps": attr.string_list(),
        "user_link_flags": attr.string_list(),
        "data": attr.label_list(allow_files = True),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
        "_enable_permissions_check": attr.label(default = "//tensorflow/core/platform/default:enable_permissions_check"),
        "_experimental_debug": attr.label(default = "//tensorflow/core/platform/default:experimental_debug"),
        "_incompatible_link_once": attr.label(default = "//tensorflow/core/platform/default:incompatible_link_once"),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],  # copybara-use-repo-external-label
    fragments = ["cpp"],
)
