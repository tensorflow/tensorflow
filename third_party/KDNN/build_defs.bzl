"""Starlark macros for KDNN (Kunpeng Deep Neural Network) library.

KDNN is the third-party library developed by Huawei as part of openEuler's
KAIL BoostKit, targeting Kunpeng 920 (ARMv8) CPUs.

if_enable_kdnn is a `select` that is true IFF:
  - --define=enable_kdnn=true was passed, AND
  - the target platform is Linux aarch64 (Kunpeng class).

kdnn_deps returns the link-time dependency on libkdnn.so via the
@kdnn repository (declared in WORKSPACE via kdnn_repository, which
lives in the companion file repository.bzl so that loading this
file from a BUILD graph does not transitively pull in
`repository_rule`).

The kdnn_repository rule itself is in third_party/KDNN/repository.bzl
and is loaded only from WORKSPACE / MODULE.bazel. This split mirrors
the modern Bazel pattern of keeping BUILD-loadable macros separate
from WORKSPACE-loadable repository rules.
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

    Resolves to ["@kdnn//:kdnn"] when KDNN is enabled AND the
    `@kdnn` repository has been declared in WORKSPACE via
    `kdnn_repository` (from third_party/KDNN/repository.bzl). When
    KDNN is not enabled, returns an empty list.

    If the operator has not configured KDNN_ROOT and called
    `kdnn_repository` in WORKSPACE, this macro will fail at
    build-rule analysis time on the `@kdnn//:kdnn` label — the same
    behaviour as `mkl_deps()` on an unconfigured TF build.

    Returns:
      A select() that resolves to ["@kdnn//:kdnn"] when KDNN is
      enabled and an empty list otherwise. Use as a `deps = [...]`
      entry.
    """
    return select({
        Label("//third_party/KDNN:enable_kdnn"): ["@kdnn//:kdnn"],
        "//conditions:default": [],
    })
