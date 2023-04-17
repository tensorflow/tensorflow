"""Repository rule for system library autoconfiguration.

`syslibs_configure` depends on the following environment variables:

  * `TF_SYSTEM_LIBS`: list of third party dependencies that should use
    the system version instead
"""

_TF_SYSTEM_LIBS = "TF_SYSTEM_LIBS"

VALID_LIBS = [
    "absl_py",
    "astor_archive",
    "astunparse_archive",
    "boringssl",
    "com_github_googlecloudplatform_google_cloud_cpp",
    "com_github_grpc_grpc",
    "com_google_absl",
    "com_google_protobuf",
    "com_googlesource_code_re2",
    "curl",
    "cython",
    "dill_archive",
    "double_conversion",
    "flatbuffers",
    "functools32_archive",
    "gast_archive",
    "gif",
    "hwloc",
    "icu",
    "jsoncpp_git",
    "libjpeg_turbo",
    "nasm",
    "nsync",
    "opt_einsum_archive",
    "org_sqlite",
    "pasta",
    "png",
    "pybind11",
    "six_archive",
    "snappy",
    "tblib_archive",
    "termcolor_archive",
    "typing_extensions_archive",
    "wrapt",
    "zlib",
]

def auto_configure_fail(msg):
    """Output failure message when syslibs configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sSystem Library Configuration Error:%s %s\n" % (red, no_color, msg))

def _is_windows(repository_ctx):
    """Returns true if the host operating system is windows."""
    os_name = repository_ctx.os.name.lower()
    if os_name.find("windows") != -1:
        return True
    return False

def _enable_syslibs(repository_ctx):
    s = repository_ctx.os.environ.get(_TF_SYSTEM_LIBS, "").strip()
    if not _is_windows(repository_ctx) and s != None and s != "":
        return True
    return False

def _get_system_lib_list(repository_ctx):
    """Gets the list of deps that should use the system lib.

    Args:
      repository_ctx: The repository context.

    Returns:
      A string version of a python list
    """
    if _TF_SYSTEM_LIBS not in repository_ctx.os.environ:
        return []

    libenv = repository_ctx.os.environ[_TF_SYSTEM_LIBS].strip()
    libs = []

    for lib in list(libenv.split(",")):
        lib = lib.strip()
        if lib == "":
            continue
        if lib not in VALID_LIBS:
            auto_configure_fail("Invalid system lib set: %s" % lib)
            return []
        libs.append(lib)

    return libs

def _format_system_lib_list(repository_ctx):
    """Formats the list of deps that should use the system lib.

    Args:
      repository_ctx: The repository context.

    Returns:
      A list of the names of deps that should use the system lib.
    """
    libs = _get_system_lib_list(repository_ctx)
    ret = ""
    for lib in libs:
        ret += "'%s',\n" % lib

    return ret

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl.replace(":", "")
    repository_ctx.template(
        out,
        Label("//third_party/systemlibs%s.tpl" % tpl),
        substitutions,
        False,
    )

def _create_dummy_repository(repository_ctx):
    """Creates the dummy repository to build with all bundled libraries."""

    _tpl(repository_ctx, ":BUILD")
    _tpl(
        repository_ctx,
        ":build_defs.bzl",
        {
            "%{syslibs_enabled}": "False",
            "%{syslibs_list}": "",
        },
    )

def _create_local_repository(repository_ctx):
    """Creates the repository to build with system libraries."""

    _tpl(repository_ctx, ":BUILD")
    _tpl(
        repository_ctx,
        ":build_defs.bzl",
        {
            "%{syslibs_enabled}": "True",
            "%{syslibs_list}": _format_system_lib_list(repository_ctx),
        },
    )

def _syslibs_autoconf_impl(repository_ctx):
    """Implementation of the syslibs_configure repository rule."""
    if not _enable_syslibs(repository_ctx):
        _create_dummy_repository(repository_ctx)
    else:
        _create_local_repository(repository_ctx)

syslibs_configure = repository_rule(
    implementation = _syslibs_autoconf_impl,
    environ = [
        _TF_SYSTEM_LIBS,
    ],
)

"""Configures the build to link to system libraries
instead of using bundled versions.

Add the following to your WORKSPACE FILE:

```python
syslibs_configure(name = "local_config_syslibs")
```

Args:
  name: A unique name for this workspace rule.
"""
