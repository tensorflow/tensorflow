# -*- Python -*-
"""Repository rule for Python autoconfiguration.

`python_configure` depends on the following environment variables:

  * `NUMPY_INCLUDE_PATH`: Location of Numpy libraries.
  * `PYTHON_BIN_PATH`: location of python binary.
  * `PYTHON_INCLUDE_PATH`: Location of python binaries.
"""

_NUMPY_INCLUDE_PATH = "NUMPY_INCLUDE_PATH"
_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"
_PYTHON_INCLUDE_PATH = "PYTHON_INCLUDE_PATH"


def _tpl(repository_ctx, tpl, substitutions={}, out=None):
  if not out:
    out = tpl
  repository_ctx.template(
      out,
      Label("//third_party/py:%s.tpl" % tpl),
      substitutions)


def _python_configure_warning(msg):
  """Output warning message during auto configuration."""
  yellow = "\033[1;33m"
  no_color = "\033[0m"
  print("\n%sPython Configuration Warning:%s %s\n" % (yellow, no_color, msg))


def _python_configure_fail(msg):
  """Output failure message when auto configuration fails."""
  red = "\033[0;31m"
  no_color = "\033[0m"
  fail("\n%sPython Configuration Error:%s %s\n" % (red, no_color, msg))


def _get_env_var(repository_ctx, name, default = None, enable_warning = True):
  """Find an environment variable in system path."""
  if name in repository_ctx.os.environ:
    return repository_ctx.os.environ[name]
  if default != None:
    if enable_warning:
      _python_configure_warning(
          "'%s' environment variable is not set, using '%s' as default" % (name, default))
    return default
  _python_configure_fail("'%s' environment variable is not set" % name)


def _is_windows(repository_ctx):
  """Returns true if the host operating system is windows."""
  os_name = repository_ctx.os.name.lower()
  if os_name.find("windows") != -1:
    return True
  return False


def _symlink_genrule_for_dir(repository_ctx, src_dir, dest_dir, genrule_name):
  """returns a genrule to symlink all files in a directory."""
  # Get the list of files under this directory
  find_result = None
  if _is_windows(repository_ctx):
    find_result = repository_ctx.execute([
        "dir", src_dir, "/b", "/s", "/a-d",
    ])
  else:
    find_result = repository_ctx.execute([
        "find", src_dir, "-follow", "-type", "f",
    ])
  # Create a list with the src_dir stripped to use for outputs.
  dest_files = find_result.stdout.replace(src_dir, '').splitlines()
  src_files = find_result.stdout.splitlines()
  command = []
  command_windows = []
  outs = []
  outs_windows = []
  for i in range(len(dest_files)):
    if dest_files[i] != "":
      command.append('ln -s ' + src_files[i] + ' $(@D)/' +
                     dest_dir + dest_files[i])
      # ln -sf is actually implemented as copying in msys since creating
      # symbolic links is privileged on Windows. But copying is too slow, so
      # invoke mklink to create junctions on Windows.
      command_windows.append('mklink /J ' + src_files[i] + ' $(@D)/' +
                             dest_dir + dest_files[i])
      outs.append('      "' + dest_dir + dest_files[i] + '",')
      outs_windows.append('      "' + dest_dir + '_windows' +
                          dest_files[i] + '",')
  genrule = _genrule(src_dir, genrule_name, ' && '.join(command),
                     '\n'.join(outs))
  genrule_windows = _genrule(src_dir, genrule_name + '_windows',
                             "cmd /c \"" + ' && '.join(command_windows) + "\"",
                             '\n'.join(outs_windows))
  return genrule + '\n' + genrule_windows


def _genrule(src_dir, genrule_name, command, outs):
  """Returns a string with a genrule.

  Genrule executes the given command and produces the given outputs.
  """
  return (
      'genrule(\n' +
      '    name = "' +
      genrule_name + '",\n' +
      '    outs = [\n' +
      outs +
      '    ],\n' +
      '    cmd = """\n' +
      command +
      '    """,\n' +
      ')\n'
  )


def _check_python_bin(repository_ctx, python_bin):
  """Checks the python bin path."""
  cmd =  '[[ -x "%s" ]] && [[ ! -d "%s" ]]' % (python_bin, python_bin)
  result = repository_ctx.execute(["bash", "-c", cmd])
  if result.return_code == 1:
    _python_configure_fail(
        "PYTHON_BIN_PATH is not executable.  Is it the python binary?")


def _get_python_include(repository_ctx, python_bin):
  """Gets the python include path."""
  result = repository_ctx.execute([python_bin, "-c",
                                   'from __future__ import print_function;' +
                                   'from distutils import sysconfig;' +
                                   'print(sysconfig.get_python_inc())'])
  if result == "":
    _python_configure_fail(
        "Problem getting python include path.  Is distutils installed?")
  return result.stdout.splitlines()[0]


def _get_numpy_include(repository_ctx, python_bin):
  """Gets the numpy include path."""
  result = repository_ctx.execute([python_bin, "-c",
                                   'from __future__ import print_function;' +
                                   'import numpy;' +
                                   ' print(numpy.get_include());'])
  if result == "":
    _python_configure_fail(
        "Problem getting numpy include path.  Is numpy installed?")
  return result.stdout.splitlines()[0]


def _create_python_repository(repository_ctx):
  """Creates the repository containing files set up to build with Python."""
  python_include = None
  numpy_include = None
  # If local checks were requested, the python and numpy include will be auto
  # detected on the host config (using _PYTHON_BIN_PATH).
  if repository_ctx.attr.local_checks:
    python_bin = _get_env_var(repository_ctx, _PYTHON_BIN_PATH)
    _check_python_bin(repository_ctx, python_bin)
    python_include = _get_python_include(repository_ctx, python_bin)
    numpy_include = _get_numpy_include(repository_ctx, python_bin) + '/numpy'
  else:
    # Otherwise, we assume user provides all paths (via ENV or attrs)
    python_include = _get_env_var(repository_ctx, _PYTHON_INCLUDE_PATH,
                                  repository_ctx.attr.python_include)
    numpy_include = _get_env_var(repository_ctx, _NUMPY_INCLUDE_PATH,
                                 repository_ctx.attr.numpy_include) + '/numpy'

  python_include_rule = _symlink_genrule_for_dir(
      repository_ctx, python_include, 'python_include', 'python_include')
  numpy_include_rule = _symlink_genrule_for_dir(
      repository_ctx, numpy_include, 'numpy_include/numpy', 'numpy_include')
  _tpl(repository_ctx, "BUILD", {
      "%{PYTHON_INCLUDE_GENRULE}": python_include_rule,
      "%{NUMPY_INCLUDE_GENRULE}": numpy_include_rule,
  })


def _python_autoconf_impl(repository_ctx):
  """Implementation of the python_autoconf repository rule."""
  _create_python_repository(repository_ctx)


python_configure = repository_rule(
    implementation = _python_autoconf_impl,
    attrs = {
        "local_checks": attr.bool(mandatory = False, default = True),
        "python_include": attr.string(mandatory = False),
        "numpy_include": attr.string(mandatory = False),
    },
    environ = [
        _PYTHON_BIN_PATH,
        _PYTHON_INCLUDE_PATH,
        _NUMPY_INCLUDE_PATH,
    ],
)
"""Detects and configures the local Python.

Add the following to your WORKSPACE FILE:

```python
python_configure(name = "local_config_python")
```

Args:
  name: A unique name for this workspace rule.
"""
