"""Helper rules for writing LIT tests."""

load("@bazel_skylib//lib:paths.bzl", "paths")

def enforce_glob(files, **kwargs):
    """A utility to enforce that a list matches a glob expression.

    Note that the comparison is done in an order-independent fashion.

    Args:
        files: a list that is expected to contain the same files as the
            specified glob expression.
        **kwargs: keyword arguments forwarded to the glob.

    Returns:
        files. The input argument unchanged
    """
    glob_result = native.glob(**kwargs)

    # glob returns a sorted list.
    if sorted(files) != sorted(glob_result):
        missing = [k for k in glob_result if k not in files]
        extra = [k for k in files if k not in glob_result]
        expected_formatted = "\n".join(['"{}",'.format(file) for file in glob_result])
        fail(("Error in enforce_glob." +
              "\nExpected {}." +
              "\nGot {}." +
              "\nMissing {}." +
              "\nExtra {}" +
              "\nPaste this into the first enforce_glob argument:" +
              "\n{}").format(
            glob_result,
            files,
            missing,
            extra,
            expected_formatted,
        ))
    return files

def lit_test_suite(
        name,
        srcs,
        cfg,
        tools = None,
        args = None,
        data = None,
        visibility = None,
        env = None,
        timeout = None,
        default_tags = None,
        tags_override = None,
        **kwargs):
    """Creates one lit test per source file and a test suite that bundles them.

    Args:
      name: string. the name of the generated test suite.
      srcs: label_list. The files which contain the lit tests.
      cfg: label. The lit config file. It must list the file extension of
        the files in `srcs` in config.suffixes and must be in a parent directory
        of `srcs`.
      tools: label list. Tools invoked in the lit RUN lines. These binaries will
        be symlinked into a directory which is on the path. They must therefore
        have unique basenames.
      args: string list. Additional arguments to pass to lit. Note that the test
        file, `-v`, and a `--path` argument for the directory to which `tools`
        are symlinked are added automatically.
      data: label list. Additional data dependencies of the test. Note that
        targets in `cfg` and `tools`, as well as their data dependencies, are
        added automatically.
      visibility: visibility of the generated test targets and test suite.
      env: string_dict. Environment variables available during test execution.
        See the common Bazel test attribute.
      timeout: timeout argument passed to the individual tests.
      default_tags: string list. Tags applied to all tests.
      tags_override: string_dict. Tags applied in addition to only select tests.
      **kwargs: additional keyword arguments to pass to all generated rules.

    See https://llvm.org/docs/CommandGuide/lit.html for details on lit
    """
    # If there are kwargs that need to be passed to only some of the generated
    # rules, they should be extracted into separate named arguments.

    args = args or []
    data = data or []
    tools = tools or []
    default_tags = default_tags or []
    tags_override = tags_override or {}

    tests = []
    for test_file in srcs:
        # It's generally good practice to prefix any generated names with the
        # macro name, but it's also nice to have the test name just match the
        # file name.
        test_name = "%s.test" % (test_file)
        tests.append(test_name)
        lit_test(
            name = test_name,
            test_file = test_file,
            cfg = cfg,
            tools = tools,
            args = args,
            data = data,
            visibility = visibility,
            env = env,
            timeout = timeout,
            tags = default_tags + tags_override.get(test_file, []),
            **kwargs
        )

    native.test_suite(
        name = name,
        tests = tests,
        **kwargs
    )

def lit_test(
        name,
        test_file,
        cfg,
        tools = None,
        args = None,
        data = None,
        visibility = None,
        env = None,
        timeout = None,
        **kwargs):
    """Runs a single test file with LLVM's lit tool.

    Args:
      name: string. the name of the generated test target.
      test_file: label. The file on which to run lit.
      cfg: label. The lit config file. It must list the file extension of
        `test_file` in config.suffixes and must be in a parent directory of
        `test_file`.
      tools: label list. Tools invoked in the lit RUN lines. These binaries will
        be symlinked into a directory which is on the path. They must therefore
        have unique basenames.
      args: string list. Additional arguments to pass to lit. Note that the test
        file, `-v`, and a `--path` argument for the directory to which `tools`
        are symlinked are added automatically.
      data: label list. Additional data dependencies of the test. Note that
        targets in `cfg` and `tools`, as well as their data dependencies, are
        added automatically.
      visibility: visibility of the generated test target.
      env: string_dict. Environment variables available during test execution.
        See the common Bazel test attribute.
      timeout: bazel test timeout string, as per common bazel definitions.
      **kwargs: additional keyword arguments to pass to all generated rules.

    See https://llvm.org/docs/CommandGuide/lit.html for details on lit
    """
    args = args or []
    data = data or []
    tools = tools or []
    env = env or {}

    tools_on_path_target_name = "_{}_tools_on_path".format(name)

    llvm_symbolizer = "@llvm-project//llvm:llvm-symbolizer"
    if llvm_symbolizer not in tools:
        tools.append(llvm_symbolizer)

    filecheck_env_var = "FILECHECK_OPTS"
    if filecheck_env_var not in env:
        env[filecheck_env_var] = "--enable-var-scope"

    bin_dir = paths.join(
        native.package_name(),
        tools_on_path_target_name,
        "lit_bin",
    )

    _tools_on_path(
        name = tools_on_path_target_name,
        testonly = True,
        srcs = tools,
        bin_dir = bin_dir,
        visibility = ["//visibility:private"],
        **kwargs
    )
    lit_name = "//third_party/py/lit:lit"

    # copybara:comment_begin(oss-only)
    lit_name = "lit_custom_" + name
    native.py_binary(
        name = lit_name,
        main = "@llvm-project//llvm:utils/lit/lit.py",
        srcs = ["@llvm-project//llvm:utils/lit/lit.py"],
        testonly = True,
        deps = [
            "@llvm-project//llvm:lit_lib",
            "@pypi_lit//:pkg",
        ],
    )

    # copybara:comment_end
    native_test(
        name = name,
        src = lit_name,
        args = [
            "-a",
            "--path",
            bin_dir,
            "$(location {})".format(test_file),
        ] + args,
        data = [
            lit_name,
            test_file,

            # TODO(cheshire): Config is not passed properly when it's not
            # called lit.cfg.py
            cfg,
            tools_on_path_target_name,
        ] + data + ["@pypi_lit//:pkg"],
        visibility = visibility,
        env = env,
        timeout = timeout,
        **kwargs
    )

def _shared_impl(ctx):
    out = ctx.attr.out
    if not out:
        out = ctx.attr.name
    output = ctx.actions.declare_file(out)
    ctx.actions.symlink(
        target_file = ctx.executable.src,
        output = output,
        is_executable = True,
    )

    runfiles = ctx.runfiles(files = ctx.files.data)

    # For Bazel 4.x support. Drop when Bazel 4.x is no longer supported
    to_merge = ([d[DefaultInfo].default_runfiles for d in ctx.attr.data] +
                [ctx.attr.src[DefaultInfo].default_runfiles])
    if hasattr(runfiles, "merge_all"):
        runfiles = runfiles.merge_all(to_merge)
    else:
        for m in to_merge:
            runfiles = runfiles.merge(m)
    return DefaultInfo(
        executable = output,
        files = depset([output]),
        runfiles = runfiles,
    )

def _native_test_impl(ctx):
    default_info = _shared_impl(ctx)
    return [default_info, testing.TestEnvironment(ctx.attr.env)]

def _tools_on_path_impl(ctx):
    runfiles = ctx.runfiles()

    # For Bazel 4.x support. Drop when Bazel 4.x is no longer supported
    to_merge = [d[DefaultInfo].default_runfiles for d in ctx.attr.srcs]
    if hasattr(runfiles, "merge_all"):
        runfiles = runfiles.merge_all(to_merge)
    else:
        for m in to_merge:
            runfiles = runfiles.merge(m)

    runfiles_symlinks = {}

    for src in ctx.attr.srcs:
        exe = src[DefaultInfo].files_to_run.executable
        if not exe:
            fail("All targets used as tools by lit tests must have exactly one" +
                 " executable, but {} has none".format(src))
        bin_path = paths.join(ctx.attr.bin_dir, exe.basename)
        if bin_path in runfiles_symlinks:
            fail("All tools used by lit tests must have unique basenames, as" +
                 " they are added to the path." +
                 " {} and {} conflict".format(runfiles_symlinks[bin_path], exe))
        runfiles_symlinks[bin_path] = exe

    return [
        DefaultInfo(runfiles = ctx.runfiles(
            symlinks = runfiles_symlinks,
        ).merge(runfiles)),
    ]

_tools_on_path = rule(
    _tools_on_path_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True, mandatory = True),
        "bin_dir": attr.string(mandatory = True),
    },
    doc = "Symlinks srcs into a single lit_bin directory. All basenames must be unique.",
)

# We have to manually set "env" on the test rule because the builtin one is only
# available in native rules. See
# https://docs.bazel.build/versions/main/be/common-definitions.html#test.env
_TEST_ATTRS = {
    "src": attr.label(
        executable = True,
        allow_files = True,
        mandatory = True,
        cfg = "target",
    ),
    "data": attr.label_list(allow_files = True),
    # "out" is attr.string instead of attr.output, so that it is select()'able.
    "out": attr.string(),
    "env": attr.string_dict(
        doc = "Mirrors the common env attribute that otherwise is" +
              " only available on native rules. See" +
              " https://docs.bazel.build/versions/main/be/common-definitions.html#test.env",
    ),
}

native_test = rule(
    implementation = _native_test_impl,
    attrs = _TEST_ATTRS,
    test = True,
)
