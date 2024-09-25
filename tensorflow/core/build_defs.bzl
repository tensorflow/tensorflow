"""Defines the build rules to disable non-core TF libraries."""

load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")

def _tf_core_transition_impl(settings, attr):
    _ignore = (settings, attr)  # @unused
    return {
        "@local_xla//xla/tsl/framework/contraction:disable_onednn_contraction_kernel": True,
        "//tensorflow/compiler/mlir/python:disable_mlir": True,
    }

_tf_core_transition = transition(
    implementation = _tf_core_transition_impl,
    inputs = [],
    outputs = [
        "@local_xla//xla/tsl/framework/contraction:disable_onednn_contraction_kernel",
        "//tensorflow/compiler/mlir/python:disable_mlir",
    ],
)

def _py_binary_tf_core_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name)

    # Put the py binary in the expected location.
    ctx.actions.run_shell(
        inputs = [ctx.executable.py_binary],
        outputs = [out],
        command = "cp %s %s" % (ctx.executable.py_binary.path, out.path),
    )

    wrapped_defaultinfo = ctx.attr.py_binary[0][DefaultInfo]
    runfiles = ctx.runfiles(files = [out])
    wrapped_default_runfiles = wrapped_defaultinfo.default_runfiles.files.to_list()

    # Remove the wrapped py_binary from the runfiles
    if ctx.executable.py_binary in wrapped_default_runfiles:
        wrapped_default_runfiles.remove(ctx.executable.py_binary)

    return [
        DefaultInfo(
            executable = out,
            files = depset([out]),
            # Merge the wrapped executable's data into runfiles
            runfiles = runfiles.merge(ctx.runfiles(files = wrapped_default_runfiles)),
        ),
    ]

# This rule sets the flag values to disable non-core TF libraries when compiling the referenced
# py_binary.
_py_binary_tf_core = rule(
    implementation = _py_binary_tf_core_impl,
    attrs = {
        "py_binary": attr.label(
            cfg = _tf_core_transition,
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

def py_binary_tf_core(name, visibility = [], **kwargs):
    """A wrapper of py_binary that disables non-core TF libraries.

    Args:
      name: The name of the resulting binary.
      visibility: The visibility of the resulting binary.
      **kwargs: All other args are passed to the wrapped py_binary.
    """

    wrapped_binary_name = "%s_wrapped_binary" % name

    # When users reference ":${name}" they will actually reference the output
    # of this transition rule, instead of the wrapped binary. This causes the
    # build system to apply our transition when evaluating the build graph.
    _py_binary_tf_core(
        name = name,
        py_binary = ":%s" % wrapped_binary_name,
        visibility = visibility,
    )
    py_binary(
        name = wrapped_binary_name,
        visibility = visibility,
        **kwargs
    )
