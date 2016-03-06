# First part of ./configure
def _find_python(ctx):
  if "PYTHON_BIN_PATH" in ctx.os.environ:
    return ctx.os.environ["PYTHON_BIN_PATH"]
  else:
    python = ctx.which("python")
    if not python:
      fail("Impossible to find python, please " +
           "set the PYTHON_BIN_PATH environment")
    return str(python)

# From python_config.sh
def _get_python_info(ctx, python, code, errMsg):
  result = ctx.execute(
    [
      python,
      "-c",
      "from __future__ import print_function; " + code
    ])
  stdout = result.stdout.strip()
  if stdout == "":
    fail(errMsg)
  return stdout

def _setup_python(ctx):
  python = _find_python(ctx)
  py_version = _get_python_info(
    ctx, python,
    "import sys; print(sys.version_info[0]);",
    "Problem getting python version.  Is %s the correct python binary?" % python)
  py_include = _get_python_info(
    ctx, python,
    "from distutils import sysconfig; print(sysconfig.get_python_inc())",
    "Problem getting python include path.  Is distutils installed?")
  py_lib = _get_python_info(
    ctx, python,
    "from distutils import sysconfig; print(sysconfig.get_python_lib())",
    "Problem getting python lib path.  Is distutils installed?")
  numpy_include = _get_python_info(
    ctx, python,
    "import numpy; print(numpy.get_include());",
    "Problem getting numpy include path.  Is numpy installed?")

  ctx.symlink(py_include, "util/python/python_include")
  ctx.symlink(py_lib, "util/python/python_lib")
  ctx.symlink(numpy_include, "third_party/py/numpy/numpy_include")
  ctx.symlink(python, "util/python/python")
  ctx.file("util/python/BUILD", """
licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "python_headers",
    hdrs = glob([
        "python_include/**/*.h",
    ]),
    data = ["python"],
    includes = ["python_include"],
)
""")
  ctx.file("third_party/py/numpy/BUILD", """
licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob([
        "numpy_include/**/*.h",
    ]),
    includes = [
        "numpy_include",
    ],
)
""")
  ctx.file("util/python/version.bzl", "PY_VERSION = 'PY%s'" % py_version.strip())

tf_py_configure_rule = repository_rule(_setup_python)

def tf_py_configure():
  tf_py_configure_rule(name = "local_tf_py_config")
