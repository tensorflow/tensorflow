# -*- Python -*-
"""SYCL autoconfiguration.
`sycl_configure` depends on the following environment variables:

  * HOST_CXX_COMPILER:  The host C++ compiler
  * HOST_C_COMPILER:    The host C compiler
  * COMPUTECPP_TOOLKIT_PATH: The path to the ComputeCpp toolkit.
"""

_HOST_CXX_COMPILER = "HOST_CXX_COMPILER"
_HOST_C_COMPILER= "HOST_C_COMPILER"
_COMPUTECPP_TOOLKIT_PATH = "COMPUTECPP_TOOLKIT_PATH"

def auto_configure_fail(msg):
  """Output failure message when auto configuration fails."""
  red = "\033[0;31m"
  no_color = "\033[0m"
  fail("\n%sAuto-Configuration Error:%s %s\n" % (red, no_color, msg))
# END cc_configure common functions (see TODO above).

def find_c(repository_ctx):
  """Find host C compiler."""
  c_name = ""
  if _HOST_C_COMPILER in repository_ctx.os.environ:
    c_name = repository_ctx.os.environ[_HOST_C_COMPILER].strip()
  if c_name.startswith("/"):
    return c_name
  c = repository_ctx.which(c_name)
  if c == None:
    fail("Cannot find C compiler, please correct your path.")
  return c

def find_cc(repository_ctx):
  """Find host C++ compiler."""
  cc_name = ""
  if _HOST_CXX_COMPILER in repository_ctx.os.environ:
    cc_name = repository_ctx.os.environ[_HOST_CXX_COMPILER].strip()
  if cc_name.startswith("/"):
    return cc_name
  cc = repository_ctx.which(cc_name)
  if cc == None:
    fail("Cannot find C++ compiler, please correct your path.")
  return cc

def find_computecpp_root(repository_ctx):
  """Find ComputeCpp compiler."""
  sycl_name = ""
  if _COMPUTECPP_TOOLKIT_PATH in repository_ctx.os.environ:
    sycl_name = repository_ctx.os.environ[_COMPUTECPP_TOOLKIT_PATH].strip()
  if sycl_name.startswith("/"):
    return sycl_name
  fail( "Cannot find SYCL compiler, please correct your path")

def _check_lib(repository_ctx, toolkit_path, lib):
  """Checks if lib exists under sycl_toolkit_path or fail if it doesn't.

  Args:
    repository_ctx: The repository context.
    toolkit_path: The toolkit directory containing the libraries.
    ib: The library to look for under toolkit_path.
  """
  lib_path = toolkit_path + "/" + lib
  if not repository_ctx.path(lib_path).exists:
    auto_configure_fail("Cannot find %s" % lib_path)

def _check_dir(repository_ctx, directory):
  """Checks whether the directory exists and fail if it does not.

  Args:
    repository_ctx: The repository context.
    directory: The directory to check the existence of.
  """
  if not repository_ctx.path(directory).exists:
    auto_configure_fail("Cannot find dir: %s" % directory)

def _symlink_dir(repository_ctx, src_dir, dest_dir):
  """Symlinks all the files in a directory.

  Args:
    repository_ctx: The repository context.
    src_dir: The source directory.
    dest_dir: The destination directory to create the symlinks in.
  """
  files = repository_ctx.path(src_dir).readdir()
  for src_file in files:
    repository_ctx.symlink(src_file, dest_dir + "/" + src_file.basename)

def _tpl(repository_ctx, tpl, substitutions={}, out=None):
  if not out:
    out = tpl.replace(":", "/")
  repository_ctx.template(
      out,
      Label("//third_party/sycl/%s.tpl" % tpl),
      substitutions)

def _file(repository_ctx, label):
  repository_ctx.template(
      label.replace(":", "/"),
      Label("//third_party/sycl/%s.tpl" % label),
      {})

def _sycl_autoconf_imp(repository_ctx):
  """Implementation of the sycl_autoconf rule."""

  # copy template files
  _file(repository_ctx, "sycl:build_defs.bzl")
  _file(repository_ctx, "sycl:BUILD")
  _file(repository_ctx, "sycl:platform.bzl")
  _file(repository_ctx, "crosstool:BUILD")
  _tpl(repository_ctx, "crosstool:computecpp",
  {
    "%{host_cxx_compiler}" : find_cc(repository_ctx),
    "%{host_c_compiler}" : find_c(repository_ctx),
  })

  computecpp_root = find_computecpp_root(repository_ctx);
  _check_dir(repository_ctx, computecpp_root)

  _tpl(repository_ctx, "crosstool:CROSSTOOL",
  {
    "%{computecpp_toolkit_path}" : computecpp_root,
  })

  # symlink libraries
  _check_lib(repository_ctx, computecpp_root+"/lib", "libComputeCpp.so" )
  _symlink_dir(repository_ctx, computecpp_root + "/lib", "sycl/lib")
  _symlink_dir(repository_ctx, computecpp_root + "/include", "sycl/include")
  _symlink_dir(repository_ctx, computecpp_root + "/bin", "sycl/bin")

sycl_configure = repository_rule(
  implementation = _sycl_autoconf_imp,
  local = True,
)
"""Detects and configures the SYCL toolchain.

Add the following to your WORKSPACE FILE:

```python
sycl_configure(name = "local_config_sycl")
```

Args:
  name: A unique name for this workspace rule.
"""
