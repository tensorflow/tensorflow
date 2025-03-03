"""Defines the cc_binary_disable_onednn build rule to disable oneDNN."""

load(
    "//xla/tsl:package_groups.bzl",
    "DEFAULT_LOAD_VISIBILITY",
    "LEGACY_TSL_FRAMEWORK_CONTRACTION_BUILD_DEFS_USERS",
)

visibility(DEFAULT_LOAD_VISIBILITY + LEGACY_TSL_FRAMEWORK_CONTRACTION_BUILD_DEFS_USERS)

def _disable_onednn_transition_impl(settings, attr):
    _ignore = (settings, attr)  # @unused
    return {"//xla/tsl/framework/contraction:disable_onednn_contraction_kernel": True}

_disable_onednn_transition = transition(
    implementation = _disable_onednn_transition_impl,
    inputs = [],
    outputs = ["//xla/tsl/framework/contraction:disable_onednn_contraction_kernel"],
)

def _disable_onednn_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name)

    # Put the cc binary in the expected location.
    ctx.actions.run_shell(
        inputs = [ctx.executable.cc_binary],
        outputs = [out],
        command = "cp %s %s" % (ctx.executable.cc_binary.path, out.path),
    )

    wrapped_defaultinfo = ctx.attr.cc_binary[0][DefaultInfo]
    runfiles = ctx.runfiles(files = [out])
    wrapped_default_runfiles = wrapped_defaultinfo.default_runfiles.files.to_list()

    # Remove the wrapped cc_binary from the runfiles
    if ctx.executable.cc_binary in wrapped_default_runfiles:
        wrapped_default_runfiles.remove(ctx.executable.cc_binary)

    return [
        DefaultInfo(
            executable = out,
            files = depset([out]),
            # Merge the wrapped executable's data into runfiles
            runfiles = runfiles.merge(ctx.runfiles(files = wrapped_default_runfiles)),
        ),
    ]

# This rule sets the flag value to disable oneDNN when compiling the referenced
# cc_binary.
_cc_binary_disable_onednn = rule(
    implementation = _disable_onednn_impl,
    attrs = {
        "cc_binary": attr.label(
            cfg = _disable_onednn_transition,
            allow_single_file = True,
            mandatory = True,
            executable = True,
        ),
        # Deps is unused, but some other rules assume all targets have a "deps" attribute
        # (such as scaffolding_registration_test)
        "deps": attr.label_list(
            default = [],
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
    # Marking this executable means it works with "$ bazel run".
    executable = True,
)

def cc_binary_disable_onednn(name, visibility = [], **kwargs):
    """A wrapper of cc_binary that disables the oneDNN contraction library.

    Using cc_binary_disable_onednn is equivalent to adding the build flag:
    "--//third_party/tensorflow/compiler/xla/tsl/framework/contraction:onednn_contraction_kernel=disable"

    Args:
      name: The name of the resulting binary.
      visibility: The visibility of the resulting binary.
      **kwargs: All other args are passed to the wrapped cc_binary.
    """

    wrapped_binary_name = "%s_wrapped_binary" % name

    # When users reference ":${name}" they will actually reference the output
    # of this transition rule, instead of the wrapped binary. This causes the
    # build system to apply our transition when evaluating the build graph.
    _cc_binary_disable_onednn(
        name = name,
        cc_binary = ":%s" % wrapped_binary_name,
        visibility = visibility,
    )
    native.cc_binary(
        name = wrapped_binary_name,
        visibility = visibility,
        **kwargs
    )
