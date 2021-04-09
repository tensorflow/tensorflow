"""BUILD extensions for MLIR table generation."""

TdInfo = provider(
    "Holds tablegen files and the dependencies and include paths necessary to" +
    " build them.",
    fields = {
        "transitive_sources": "td files transitively used by this rule.",
        "transitive_includes": (
            "include arguments to add to the final tablegen invocation. These" +
            " are the absolute directory paths that will be added with '-I'."
        ),
    },
)

# For now we allow anything that provides DefaultInfo to just forward its files.
# In particular, this allows filegroups to be used. This is mostly to ease
# transition. In the future, the TdInfo provider will be required.
# TODO(gcmn): Switch to enforcing TdInfo provider.
def _get_dep_transitive_srcs(dep):
    """Extract TdInfo.transitive_sources, falling back to DefaultInfo.files."""
    if TdInfo in dep:
        return dep[TdInfo].transitive_sources
    return dep[DefaultInfo].files

def _get_dep_transitive_includes(dep):
    """Extract TdInfo.transitive_includes, falling back to an empty depset()."""
    if TdInfo in dep:
        return dep[TdInfo].transitive_includes
    return depset()

def _get_transitive_srcs(srcs, deps):
    """Obtain the source files for a target and its transitive dependencies.

    Args:
      srcs: a list of source files
      deps: a list of targets that are direct dependencies
    Returns:
      a collection of the transitive sources
    """
    return depset(
        direct = srcs,
        transitive = [_get_dep_transitive_srcs(dep) for dep in deps],
    )

def _get_transitive_includes(includes, deps):
    """Obtain the includes paths for a target and its transitive dependencies.

    Args:
      includes: a list of include paths
      deps: a list of targets that are direct dependencies
    Returns:
      a collection of the transitive include paths
    """
    return depset(
        direct = includes,
        transitive = [_get_dep_transitive_includes(dep) for dep in deps],
    )

def _prefix_roots(ctx, includes):
    """Map the given includes to be relative to all root directories.

    This will expand them to be relative to all the root directories available
    in the execution environment for ctx.run (bin and genfiles in addition to
    the normal source root)
    """
    prefixed_includes = []
    for include in includes:
        prefixed_includes.append(include)
        prefixed_includes.append(ctx.genfiles_dir.path + "/" + include)
        prefixed_includes.append(ctx.bin_dir.path + "/" + include)
    return prefixed_includes

def _resolve_includes(ctx, includes):
    """Resolves include paths to paths relative to the execution root.

    Relative paths are interpreted as relative to the current label's package.
    Absolute paths are interpreted as relative to the current label's workspace
    root."""
    package = ctx.label.package
    workspace_root = ctx.label.workspace_root
    workspace_root = workspace_root if workspace_root else "."
    resolved_includes = []
    for include in includes:
        if not include.startswith("/"):
            include = "/" + package + "/" + include
        include = workspace_root + include
        resolved_includes.extend(_prefix_roots(ctx, [include]))
    return resolved_includes

def _td_library_impl(ctx):
    trans_srcs = _get_transitive_srcs(ctx.files.srcs, ctx.attr.deps)
    trans_includes = _get_transitive_includes(
        _resolve_includes(ctx, ctx.attr.includes),
        ctx.attr.deps,
    )
    return [
        DefaultInfo(files = trans_srcs),
        TdInfo(
            transitive_sources = trans_srcs,
            transitive_includes = trans_includes,
        ),
    ]

td_library = rule(
    _td_library_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "includes": attr.string_list(
            doc = "Include paths to be added to the final tablegen tool" +
                  " invocation. Relative paths are interpreted as relative to" +
                  " the current label's package. Absolute paths are" +
                  " interpreted as relative to the current label's workspace",
        ),
        # TODO(gcmn): limit to TdInfo providers.
        "deps": attr.label_list(
            doc = "Dependencies providing tablegen source files and include" +
                  " paths.",
        ),
    },
)

def _gentbl_rule_impl(ctx):
    td_file = ctx.file.td_file

    trans_srcs = _get_transitive_srcs(
        ctx.files.td_srcs + [td_file],
        ctx.attr.deps,
    )

    # Note that we have two types of includes here. The deprecated ones expanded
    # only by "_prefix_roots" are already relative to the execution root, i.e.
    # may contain an `external/<workspace_name>` prefix if the current workspace
    # is not the main workspace (where workspace_name is something configured
    # per-project and therefore generally not known). Note that dirname also
    # already includes this prefix. The new style of includes have it prepended
    # automatically by `_resolve_includes` to avoid BUILD files having to depend
    # on project specific configurations and Bazel implementation details.
    trans_includes = _get_transitive_includes(
        _resolve_includes(ctx, ctx.attr.includes + ["/"]) +
        _prefix_roots(ctx, ctx.attr.td_includes + [td_file.dirname]),
        ctx.attr.deps,
    )

    args = ctx.actions.args()
    args.add_all(ctx.attr.opts)
    args.add(td_file)
    args.add_all(trans_includes, before_each = "-I")

    args.add("-o", ctx.outputs.out.path)

    ctx.actions.run(
        outputs = [ctx.outputs.out],
        inputs = trans_srcs,
        executable = ctx.executable.tblgen,
        arguments = [args],
    )

    return [DefaultInfo()]

gentbl_rule = rule(
    _gentbl_rule_impl,
    doc = "Generates tabular code from a table definition file.",
    # Match genrule behavior
    output_to_genfiles = True,
    attrs = {
        "tblgen": attr.label(
            doc = "The tablegen executable with which to generate `out`.",
            executable = True,
            cfg = "exec",
        ),
        "td_file": attr.label(
            doc = "The tablegen file to run through `tblgen`.",
            allow_single_file = True,
            mandatory = True,
        ),
        "td_srcs": attr.label_list(
            doc = "Additional tablegen files included by `td_file`. It is not" +
                  " necessary to list td_file here (though not an error).",
            allow_files = True,
        ),
        # TODO(gcmn): limit to TdInfo providers.
        "deps": attr.label_list(
            doc = "Dependencies providing tablegen source files and include" +
                  " paths.",
        ),
        "out": attr.output(
            doc = "The output file for the tablegen invocation.",
            mandatory = True,
        ),
        "opts": attr.string_list(
            doc = "Additional command line options to add to the tablegen" +
                  " invocation. For include arguments, prefer to use" +
                  " `includes`.",
        ),
        "includes": attr.string_list(
            doc = "Include paths to be added to the final tablegen tool" +
                  " invocation. Relative paths are interpreted as relative to" +
                  " the current label's package. Absolute paths are" +
                  " interpreted as relative to the current label's workspace." +
                  " Includes are applied from all roots available in the" +
                  " execution environment (source, genfiles, and bin" +
                  " directories). The execution roots themselves and the " +
                  " directory of td_file are always added.",
        ),
        "td_includes": attr.string_list(
            doc = "Include paths to add to the tablegen invocation. Paths are" +
                  " interpreted as relative to the current label's workspace" +
                  " root and applied from all roots available in the" +
                  " execution environment (source, genfiles, and bin" +
                  " directories). Deprecated. Use `includes` instead.",
        ),
    },
)

# TODO(gcmn): Figure out how to reduce duplication with _gentbl_rule_impl
def _gentbl_test_impl(ctx):
    td_file = ctx.file.td_file

    trans_srcs = _get_transitive_srcs(
        ctx.files.td_srcs + [td_file],
        ctx.attr.deps,
    )

    # Note that we have two types of includes here. The deprecated ones expanded
    # only by "_prefix_roots" are already relative to the execution root, i.e.
    # may contain an `external/<workspace_name>` prefix if the current workspace
    # is not the main workspace (where workspace_name is something configured
    # per-project and therefore generally not known). Note that dirname also
    # already includes this prefix. The new style of includes have it prepended
    # automatically by `_resolve_includes` to avoid BUILD files having to depend
    # on project specific configurations and Bazel implementation details.
    trans_includes = _get_transitive_includes(
        _resolve_includes(ctx, ctx.attr.includes + ["/"]) +
        _prefix_roots(ctx, ctx.attr.td_includes + [td_file.dirname]),
        ctx.attr.deps,
    )

    test_args = [ctx.executable.tblgen.short_path]
    test_args.extend(ctx.attr.opts)
    test_args.append(td_file.path)
    test_args.extend(["-I " + include for include in trans_includes.to_list()])

    test_args.extend(["-o", "/dev/null"])

    ctx.actions.write(
        ctx.outputs.executable,
        content = " ".join(test_args),
        is_executable = True,
    )

    return [DefaultInfo(
        runfiles = ctx.runfiles(
            [ctx.executable.tblgen],
            transitive_files = trans_srcs,
        ),
    )]

gentbl_test = rule(
    _gentbl_test_impl,
    test = True,
    doc = "A shell test that tests the given tablegen invocation. Note" +
          " that unlike gentbl_rule, this builds and invokes `tblgen` in the" +
          " target configuration. Takes all the same arguments as gentbl_rule" +
          " except for `out` (as it does not generate any output)",
    # Match genrule behavior
    output_to_genfiles = True,
    attrs = {
        "tblgen": attr.label(
            doc = "The tablegen executable run in the shell command. Note" +
                  " that this is built in the target configuration.",
            executable = True,
            cfg = "target",
        ),
        "td_file": attr.label(
            doc = "See gentbl_rule.td_file",
            allow_single_file = True,
            mandatory = True,
        ),
        "td_srcs": attr.label_list(
            doc = "See gentbl_rule.td_srcs",
            allow_files = True,
        ),
        "deps": attr.label_list(doc = "See gentbl_rule.deps"),
        "opts": attr.string_list(doc = "See gentbl_rule.opts"),
        "includes": attr.string_list(doc = "See gentbl_rule.includes"),
        "td_includes": attr.string_list(doc = "See gentbl_rule.td_includes"),
    },
)

def gentbl(
        name,
        tblgen,
        td_file,
        tbl_outs,
        td_srcs = [],
        td_includes = [],
        includes = [],
        td_relative_includes = [],
        deps = [],
        strip_include_prefix = None,
        test = False,
        **kwargs):
    """Create multiple tablegen generated files using the same tool and input.

    All generated outputs are bundled in a cc_library rule.

    Args:
      name: The name of the generated cc_library rule for use in dependencies.
      tblgen: The binary used to produce the output.
      td_file: The primary table definitions file.
      tbl_outs: A list of tuples (opts, out), where each opts is a string of
        options passed to tblgen, and the out is the corresponding output file
        produced.
      td_srcs: See gentbl_rule.td_srcs
      includes: See gentbl_rule.includes
      td_includes: See gentbl_rule.td_includes
      td_relative_includes: An alias for "includes". Deprecated. Use includes
        instead.
      deps: See gentbl_rule.deps
      strip_include_prefix: attribute to pass through to cc_library.
      test: whether to create a shell test that invokes the tool too.
      **kwargs: Extra keyword arguments to pass to all generated rules.
    """

    # TODO(gcmn): Update callers to td_library and explicit includes and
    # drop this hardcoded include.
    hardcoded_includes = [
        "external/llvm-project/mlir/include",
    ]

    for (opts_string, out) in tbl_outs:
        # TODO(gcmn): The API of opts as single string is preserved for backward
        # compatibility. Change to taking a sequence.
        opts = opts_string.split(" ") if opts_string else []

        # Filter out empty options
        opts = [opt for opt in opts if opt]

        first_opt = opts[0] if opts else ""
        rule_suffix = "_{}_{}".format(
            first_opt.replace("-", "_").replace("=", "_"),
            str(hash(opts_string)),
        )
        gentbl_name = "%s_%s_genrule" % (name, rule_suffix)
        gentbl_rule(
            name = gentbl_name,
            td_file = td_file,
            tblgen = tblgen,
            opts = opts,
            td_srcs = td_srcs,
            deps = deps,
            includes = includes + td_relative_includes,
            td_includes = td_includes + hardcoded_includes,
            out = out,
            **kwargs
        )
        if test:
            # Also run the generator in the target configuration as a test. This
            # means it gets run with asserts and sanitizers and such when they
            # are enabled and is counted in coverage.
            gentbl_test(
                name = "%s_test" % (gentbl_name,),
                td_file = td_file,
                tblgen = tblgen,
                opts = opts,
                td_srcs = td_srcs,
                deps = deps,
                includes = includes + td_relative_includes,
                td_includes = td_includes + hardcoded_includes,
                # Shell files not executable on Windows.
                # TODO(gcmn): Support windows.
                tags = ["no_windows"],
                **kwargs
            )

    # List of opts that do not generate cc files.
    skip_opts = ["-gen-op-doc"]
    hdrs = [f for (opts, f) in tbl_outs if opts not in skip_opts]
    native.cc_library(
        name = name,
        # strip_include_prefix does not apply to textual_hdrs.
        # https://github.com/bazelbuild/bazel/issues/12424
        hdrs = hdrs if strip_include_prefix else [],
        strip_include_prefix = strip_include_prefix,
        textual_hdrs = hdrs,
        **kwargs
    )
