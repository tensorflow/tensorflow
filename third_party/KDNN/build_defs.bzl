"""Starlark macros for KDNN (Kunpeng Deep Neural Network) library.

KDNN is the third-party library developed by Huawei as part of openEuler's
KAIL BoostKit, targeting Kunpeng 920 (ARMv8) CPUs.

if_enable_kdnn is a `select` that is true IFF:
  - --define=enable_kdnn=true was passed, AND
  - the target platform is Linux aarch64 (Kunpeng class).

kdnn_deps returns the link-time dependency on libkdnn.so.
kdnn_repository is the analog of mkl_repository: it locates the libkdnn
install via the KDNN_ROOT environment variable.
"""

def if_enable_kdnn(if_true, if_false = []):
    """Shorthand to select() if KDNN is enabled at build time.

    Mirrors the style of `if_zendnn` in tensorflow/tensorflow.bzl but
    *additionally* gates on aarch64. KDNN on x86 is meaningless (the
    library is Kunpeng 920 only).

    Args:
      if_true: list of attrs / deps to include when KDNN is enabled.
      if_false: list to include otherwise. Defaults to [].

    Returns:
      A select() that evaluates to if_true only when both:
        * --define=enable_kdnn=true is set, AND
        * the platform is Linux aarch64.
      Otherwise evaluates to if_false.
    """
    return select({
        Label("//third_party/KDNN:enable_kdnn"): if_true,
        "//conditions:default": if_false,
    })

def kdnn_deps():
    """Returns the link-time dependency on libkdnn.

    Returns:
      A select() that resolves to ["@kdnn//:kdnn"] when KDNN is enabled
      and an empty list otherwise. Use as a `deps = [...]` entry.
    """
    return select({
        Label("//third_party/KDNN:enable_kdnn"): ["@kdnn//:kdnn"],
        "//conditions:default": [],
    })

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
