# -*- Python -*-
"""Skylark macros for MKL.
if_mkl is a conditional to check if MKL is enabled or not.

mkl_repository is a repository rule for creating MKL repository rule that can
be pointed to either a local folder, or download it from the internet.
mkl_repository depends on the following environment variables:
  * `TF_MKL_ROOT`: The root folder where a copy of libmkl is located.
"""


_TF_MKL_ROOT = "TF_MKL_ROOT"


def if_mkl(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with MKL.

    Returns a select statement which evaluates to if_true if we're building
    with MKL enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        str(Label("//third_party/mkl:using_mkl")): if_true,
        "//conditions:default": if_false
    })

def if_mkl_lnx_x64(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with MKL.

    Returns a select statement which evaluates to if_true if we're building
    with MKL enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        str(Label("//third_party/mkl:using_mkl_lnx_x64")): if_true,
        "//conditions:default": if_false
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
        sha256=repository_ctx.attr.sha256,
        stripPrefix=repository_ctx.attr.strip_prefix,
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
