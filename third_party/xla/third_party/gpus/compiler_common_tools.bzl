"""Common compiler functions. """

load(
    "//third_party/remote_config:common.bzl",
    "err_out",
    "raw_exec",
    "realpath",
)

def to_list_of_strings(elements):
    """Convert the list of ["a", "b", "c"] into '"a", "b", "c"'.

    This is to be used to put a list of strings into the bzl file templates
    so it gets interpreted as list of strings in Starlark.

    Args:
      elements: list of string elements

    Returns:
      single string of elements wrapped in quotes separated by a comma."""
    quoted_strings = ["\"" + element + "\"" for element in elements]
    return ", ".join(quoted_strings)

_INC_DIR_MARKER_BEGIN = "#include <...>"

# OSX add " (framework directory)" at the end of line, strip it.
_OSX_FRAMEWORK_SUFFIX = " (framework directory)"
_OSX_FRAMEWORK_SUFFIX_LEN = len(_OSX_FRAMEWORK_SUFFIX)

# TODO(dzc): Once these functions have been factored out of Bazel's
# cc_configure.bzl, load them from @bazel_tools instead.
def _cxx_inc_convert(path):
    """Convert path returned by cc -E xc++ in a complete path."""
    path = path.strip()
    if path.endswith(_OSX_FRAMEWORK_SUFFIX):
        path = path[:-_OSX_FRAMEWORK_SUFFIX_LEN].strip()
    return path

def _normalize_include_path(repository_ctx, path):
    """Normalizes include paths before writing them to the crosstool.

      If path points inside the 'crosstool' folder of the repository, a relative
      path is returned.
      If path points outside the 'crosstool' folder, an absolute path is returned.
    """
    path = str(repository_ctx.path(path))
    crosstool_folder = str(repository_ctx.path(".").get_child("crosstool"))

    if path.startswith(crosstool_folder):
        # We drop the path to "$REPO/crosstool" and a trailing path separator.
        return path[len(crosstool_folder) + 1:]
    return path

def _is_compiler_option_supported(repository_ctx, cc, option):
    """Checks that `option` is supported by the C compiler. Doesn't %-escape the option."""
    result = repository_ctx.execute([
        cc,
        option,
        "-o",
        "/dev/null",
        "-c",
        str(repository_ctx.path("tools/cpp/empty.cc")),
    ])
    return result.stderr.find(option) == -1

def _get_cxx_inc_directories_impl(repository_ctx, cc, lang_is_cpp, tf_sys_root):
    """Compute the list of default C or C++ include directories."""
    if lang_is_cpp:
        lang = "c++"
    else:
        lang = "c"
    sysroot = []
    if tf_sys_root:
        sysroot += ["--sysroot", tf_sys_root]
    result = raw_exec(repository_ctx, [cc, "-E", "-x" + lang, "-", "-v"] +
                                      sysroot)
    stderr = err_out(result)
    index1 = stderr.find(_INC_DIR_MARKER_BEGIN)
    if index1 == -1:
        return []
    index1 = stderr.find("\n", index1)
    if index1 == -1:
        return []
    index2 = stderr.rfind("\n ")
    if index2 == -1 or index2 < index1:
        return []
    index2 = stderr.find("\n", index2 + 1)
    if index2 == -1:
        inc_dirs = stderr[index1 + 1:]
    else:
        inc_dirs = stderr[index1 + 1:index2].strip()

    print_resource_dir_supported = _is_compiler_option_supported(
        repository_ctx,
        cc,
        "-print-resource-dir",
    )

    if print_resource_dir_supported:
        resource_dir = repository_ctx.execute(
            [cc, "-print-resource-dir"],
        ).stdout.strip() + "/share"
        inc_dirs += "\n" + resource_dir

    compiler_includes = [
        _normalize_include_path(repository_ctx, _cxx_inc_convert(p))
        for p in inc_dirs.split("\n")
    ]

    # The compiler might be on a symlink, e.g. /symlink -> /opt/gcc
    # The above keeps only the resolved paths to the default includes (e.g. /opt/gcc/include/c++/11)
    # but Bazel might encounter either (usually reported by the compiler)
    # especially when a compiler wrapper (e.g. ccache) is used.
    # So we need to also include paths where symlinks are not resolved.

    # Try to find real path to CC installation to "see through" compiler wrappers
    # GCC has the path to g++
    index1 = result.stderr.find("COLLECT_GCC=")
    if index1 != -1:
        index1 = result.stderr.find("=", index1)
        index2 = result.stderr.find("\n", index1)
        cc_topdir = repository_ctx.path(result.stderr[index1 + 1:index2]).dirname.dirname
    else:
        # Clang has the directory
        index1 = result.stderr.find("InstalledDir: ")
        if index1 != -1:
            index1 = result.stderr.find(" ", index1)
            index2 = result.stderr.find("\n", index1)
            cc_topdir = repository_ctx.path(result.stderr[index1 + 1:index2]).dirname
        else:
            # Fallback to the CC path
            cc_topdir = repository_ctx.path(cc).dirname.dirname

    # We now have the compiler installation prefix, e.g. /symlink/gcc
    # And the resolved installation prefix, e.g. /opt/gcc
    cc_topdir_resolved = str(realpath(repository_ctx, cc_topdir)).strip()
    cc_topdir = str(cc_topdir).strip()

    # If there is (any!) symlink involved we add paths where the unresolved installation prefix is kept.
    # e.g. [/opt/gcc/include/c++/11, /opt/gcc/lib/gcc/x86_64-linux-gnu/11/include, /other/path]
    # adds [/symlink/include/c++/11, /symlink/lib/gcc/x86_64-linux-gnu/11/include]
    if cc_topdir_resolved != cc_topdir:
        unresolved_compiler_includes = [
            cc_topdir + inc[len(cc_topdir_resolved):]
            for inc in compiler_includes
            if inc.startswith(cc_topdir_resolved)
        ]
        compiler_includes = compiler_includes + unresolved_compiler_includes
    return compiler_includes

def get_cxx_inc_directories(repository_ctx, cc, tf_sys_root):
    """Compute the list of default C and C++ include directories."""

    # For some reason `clang -xc` sometimes returns include paths that are
    # different from the ones from `clang -xc++`. (Symlink and a dir)
    # So we run the compiler with both `-xc` and `-xc++` and merge resulting lists
    includes_cpp = _get_cxx_inc_directories_impl(
        repository_ctx,
        cc,
        True,
        tf_sys_root,
    )
    includes_c = _get_cxx_inc_directories_impl(
        repository_ctx,
        cc,
        False,
        tf_sys_root,
    )

    return includes_cpp + [
        inc
        for inc in includes_c
        if inc not in includes_cpp
    ]
