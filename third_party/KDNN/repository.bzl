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
locate the operator-supplied KAIL BoostKit install. When KDNN_ROOT is
unset (the default for upstream CI), it produces a no-op @kdnn repo
so that @kdnn//:kdnn resolves to an empty cc_library and default
builds are unaffected. The KDNN source is not auto-downloaded because
the upstream distribution is gated behind openEuler's package model.
"""

_KDNN_ROOT = "KDNN_ROOT"

def _kdnn_autoconf_impl(repository_ctx):
    """Implementation of the kdnn_repository rule.

    Three modes:
      1. KDNN_ROOT unset: write a no-op @kdnn repo (empty kdnn.h + a
         BUILD file with an empty cc_library) so that labels like
         @kdnn//:kdnn still resolve. This keeps default TF builds
         (no --define=enable_kdnn=true) green on hosts that don't
         have libkdnn.so installed, and keeps the WORKSPACE-level
         `maybe(kdnn_repository, ...)` wrapper a true no-op.
      2. KDNN_ROOT set but the path doesn't exist: same as mode 1,
         but emit a warning so an operator who intended to enable
         KDNN notices. We deliberately do NOT fail() here — a
         misconfigured env var shouldn't break unrelated builds.
      3. KDNN_ROOT set and the path exists: symlink the operator's
         include/ and lib/ into the repo and use the supplied
         BUILD file (the BUILD at //third_party/KDNN:BUILD declares
         a `kdnn` cc_library that depends on @kdnn//:kdnn_header,
         which in turn points at kdnn.h).

    The mode 1 + mode 2 behavior is what the rest of the tree relies
    on: `kdnn_deps()` resolves to ["@kdnn//:kdnn"] only when the
    `enable_kdnn` config_setting matches (which requires both
    --define=enable_kdnn=true AND aarch64), and the corresponding
    kdnn kernels live behind `#ifdef KERNEL_KDNN`. So even when
    @kdnn is a no-op stub, default builds see only an empty
    cc_library and the kernels compile to nothing.
    """
    kdnn_root = repository_ctx.os.environ.get(_KDNN_ROOT, "")
    if kdnn_root and repository_ctx.path(kdnn_root).exists:
        repository_ctx.symlink(kdnn_root + "/include", "include")
        repository_ctx.symlink(kdnn_root + "/lib", "lib")
        repository_ctx.symlink(
            repository_ctx.attr.build_file,
            "BUILD",
        )
        return

    # KDNN_ROOT unset, or set to a path that doesn't resolve. Provide
    # an empty @kdnn repo so that @kdnn//:kdnn resolves to a no-op
    # cc_library and unrelated build targets don't break.
    if kdnn_root:
        # Don't silently swallow a misconfiguration: print to the
        # build's progress stream so the operator sees it in CI logs.
        print(
            "WARNING: KDNN_ROOT is set to '%s' but that path does " %
            kdnn_root +
            "not exist. Falling back to a no-op @kdnn repo. KDNN " +
            "kernels will not be linked; the rest of TF will build " +
            "normally. See third_party/KDNN/README.md for setup.",
        )
    repository_ctx.file(
        "BUILD",
        'licenses(["restricted"])\n' +
        'package(default_visibility = ["//visibility:public"])\n' +
        'cc_library(name = "kdnn", hdrs = [], srcs = [])\n',
    )
    repository_ctx.file("kdnn.h", "// no-op stub: KDNN_ROOT unset\n")

kdnn_repository = repository_rule(
    implementation = _kdnn_autoconf_impl,
    environ = [_KDNN_ROOT],
    attrs = {
        "build_file": attr.label(),
    },
)
