"""Starlark macros for MKL and oneDNN.

if_mkl is a conditional to check if we are building with MKL.
if_mkl_ml is a conditional to check if we are building with MKL-ML.
if_mkl_ml_only is a conditional to check for MKL-ML-only (no MKL-DNN) mode.
if_mkl_lnx_x64 is a conditional to check for MKL
if_enable_mkl is a conditional to check if building with MKL and MKL is enabled.
if_mkldnn_openmp checks if we are building x86 backend with OpenMP.
if_mkldnn_aarch64_acl checks if we are building with Arm Compute Library.
if_mkldnn_aarch64_acl_openmp checks if we are building ACL with OpenMP.

mkl_repository is a repository rule for creating MKL repository rule that can
be pointed to either a local folder, or download it from the internet.
mkl_repository depends on the following environment variables:
  * `TF_MKL_ROOT`: The root folder where a copy of libmkl is located.
"""

_TF_MKL_ROOT = "TF_MKL_ROOT"

def if_mkl(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with oneDNN.

      OneDNN gets built if we are building on platforms that support oneDNN
      (x86 linux/windows) or if specifcially configured to use oneDNN.

    Args:
      if_true: expression to evaluate if building with oneDNN.
      if_false: expression to evaluate if building without oneDNN.

    Returns:
      a select evaluating to either if_true or if_false as appropriate.

    TODO(intel-tf):
      the first "if_true" line is kept because non-x86 platforms (e.g., ARM)
      may need it. It may be deleted in future with refactoring.
    """
    return select({
        "@local_xla//xla/tsl/mkl:build_with_mkl_aarch64": if_true,
        "@local_xla//xla/tsl:linux_x86_64": if_true,
        "@local_xla//xla/tsl:windows": if_true,
        "//conditions:default": if_false,
    })

def if_mkl_ml(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with MKL-ML.

    Args:
      if_true: expression to evaluate if building with MKL-ML.
      if_false: expression to evaluate if building without MKL-ML
        (i.e. without MKL at all, or with MKL-DNN only).

    Returns:
      a select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        "@local_xla//xla/tsl/mkl:build_with_mkl_opensource": if_false,
        "@local_xla//xla/tsl/mkl:build_with_mkl": if_true,
        "//conditions:default": if_false,
    })

def if_mkl_lnx_x64(if_true, if_false = []):
    """Shorthand to select() if building with MKL and the target is Linux x86-64.

    Args:
      if_true: expression to evaluate if building with MKL is enabled and the
        target platform is Linux x86-64.
      if_false: expression to evaluate if building without MKL or for a
        different platform.

    Returns:
      a select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        "@local_xla//xla/tsl/mkl:build_with_mkl_lnx_x64": if_true,
        "//conditions:default": if_false,
    })

def if_enable_mkl(if_true, if_false = []):
    """Shorthand to select() if we are building with MKL and MKL is enabled.

    This is only effective when built with MKL.

    Args:
      if_true: expression to evaluate if building with MKL and MKL is enabled
      if_false: expression to evaluate if building without MKL or MKL is not enabled.

    Returns:
      A select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        "@local_xla//xla/tsl/mkl:enable_mkl": if_true,
        "//conditions:default": if_false,
    })

def mkl_deps():
    """Returns the correct set of oneDNN library dependencies.

      Shorthand for select() to pull in the correct set of oneDNN library deps
      depending on the platform. x86 Linux/Windows with or without --config=mkl
      will always build with oneDNN library.

    Returns:
      a select evaluating to a list of library dependencies, suitable for
      inclusion in the deps attribute of rules.
    """
    return select({
        "@local_xla//xla/tsl/mkl:build_with_mkl_aarch64": ["@mkl_dnn_acl_compatible//:mkl_dnn_acl"],
        "@local_xla//xla/tsl:linux_x86_64": ["@onednn//:mkl_dnn"],
        "@local_xla//xla/tsl:windows": ["@onednn//:mkl_dnn"],
        "//conditions:default": [],
    })

def onednn_v3_define():
    """Returns a define to build with oneDNN v3.x if it is enabled. 

    Returns:
      A define to build with oneDNN v3.x for Linux x86 and Windows x86 builds.
      An empty list of all other cases (include ARM builds).
    """
    return select({
        "@local_xla//xla/tsl/mkl:build_with_mkl_aarch64": ["-DENABLE_ONEDNN_V3"],
        "@local_xla//xla/tsl:linux_x86_64": ["-DENABLE_ONEDNN_V3"],
        "@local_xla//xla/tsl:windows": ["-DENABLE_ONEDNN_V3"],
        "//conditions:default": [],
    })

def if_mkldnn_openmp(if_true, if_false = []):
    """Returns `if_true` if OpenMP is used with oneDNN.

    Shorthand for select()'ing on whether we're building with
    oneDNN open source library only with openmp

    Returns a select statement which evaluates to if_true if we're building
    with oneDNN open source library only with OpenMP. Otherwise, the
    select statement evaluates to if_false.

    """
    return select({
        "@local_xla//xla/tsl/mkl:build_with_mkldnn_openmp": if_true,
        "//conditions:default": if_false,
    })

def if_mkldnn_aarch64_acl(if_true, if_false = []):
    return select({
        "@local_xla//xla/tsl/mkl:build_with_mkl_aarch64": if_true,
        "//conditions:default": if_false,
    })

def if_mkldnn_aarch64_acl_openmp(if_true, if_false = []):
    return select({
        "@local_xla//xla/tsl/mkl:build_with_mkl_aarch64_openmp": if_true,
        "//conditions:default": if_false,
    })

def _enable_local_mkl(repository_ctx):
    return _TF_MKL_ROOT in repository_ctx.os.environ

def _mkl_autoconf_impl(repository_ctx):
    """Implementation of the local_mkl_autoconf repository rule."""

    if _enable_local_mkl(repository_ctx):
        # Symlink lib and include local folders.
        mkl_root = repository_ctx.os.environ[_TF_MKL_ROOT]
        mkl_lib_path = "%s/lib" % mkl_root
        repository_ctx.symlink(mkl_lib_path, "lib")
        mkl_include_path = "%s/include" % mkl_root
        repository_ctx.symlink(mkl_include_path, "include")
        mkl_license_path = "%s/license.txt" % mkl_root
        repository_ctx.symlink(mkl_license_path, "license.txt")
    else:
        # setup remote mkl repository.
        repository_ctx.download_and_extract(
            repository_ctx.attr.urls,
            sha256 = repository_ctx.attr.sha256,
            stripPrefix = repository_ctx.attr.strip_prefix,
        )

    # Also setup BUILD file.
    repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD")

mkl_repository = repository_rule(
    implementation = _mkl_autoconf_impl,
    environ = [
        _TF_MKL_ROOT,
    ],
    attrs = {
        "build_file": attr.label(),
        "urls": attr.string_list(default = []),
        "sha256": attr.string(default = ""),
        "strip_prefix": attr.string(default = ""),
    },
)
