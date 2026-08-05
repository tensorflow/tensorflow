"""repository_rule for KDNN (Kunpeng Deep Neural Network) library.

This file is INTENTIONALLY separate from build_defs.bzl. `repository_rule`
is a Bazel primitive that only makes sense in WORKSPACE / MODULE.bazel
evaluation contexts; loading it from a `.bzl` that is itself loaded by
BUILD files (as build_defs.bzl is, via tensorflow/tensorflow.bzl) is
legal but conflates two namespaces. Keeping the repository rule in its
own file means:

  * `tensorflow/tensorflow.bzl` does not see `repository_rule` at all,
    which matches the modern pattern of separating BUILD-loadable
    macros from WORKSPACE-loadable repository rules.
  * Operators adding this PR to their fork only need to load one
    additional file in WORKSPACE.

Usage from WORKSPACE:

    load("//third_party/KDNN:repository.bzl", "kdnn_repository")

    kdnn_repository(
        name = "kdnn",
        build_file = "//third_party/KDNN:BUILD",
    )

The `_kdnn_autoconf_impl` below mirrors the pattern of TF's
`mkl_repository`: it consumes the `KDNN_ROOT` environment variable to
locate the operator-supplied KAIL BoostKit install, and fails loudly
otherwise (the KDNN source is not auto-downloaded because the upstream
distribution is gated behind openEuler's package model).
"""

_KDNN_ROOT = "KDNN_ROOT"

def _kdnn_autoconf_impl(repository_ctx):
    """Implementation of the kdnn_repository rule.

    Mirrors the mkl_repository pattern: if KDNN_ROOT is set, use the
    header + lib from that directory; otherwise fail with a clear error
    instructing the user to obtain KDNN from openEuler BoostKit.
    """
    if _KDNN_ROOT in repository_ctx.os.environ:
        root = repository_ctx.os.environ[_KDNN_ROOT]
        repository_ctx.symlink(root + "/include", "include")
        repository_ctx.symlink(root + "/lib", "lib")
        repository_ctx.symlink(
            repository_ctx.attr.build_file,
            "BUILD",
        )
    else:
        # We do NOT auto-download KDNN: the upstream is gated behind
        # openEuler's package distribution model and is not an
        # open-source Apache repo we can clone. Fail with an actionable
        # error message.
        fail(
            "KDNN_ROOT is not set. To enable KDNN, install the KAIL " +
            "BoostKit from openEuler and set KDNN_ROOT to the install " +
            "directory. See third_party/KDNN/README.md for details.",
        )

kdnn_repository = repository_rule(
    implementation = _kdnn_autoconf_impl,
    environ = [_KDNN_ROOT],
    attrs = {
        "build_file": attr.label(),
    },
)
