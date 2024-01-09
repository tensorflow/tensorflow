"""Repository rule for aarch64 docker container
"""

load(
    "//third_party/remote_config:common.bzl",
    "err_out",
    "get_host_environ",
    "raw_exec",
    "which",
)

_GCC_HOST_COMPILER_PATH = "GCC_HOST_COMPILER_PATH"
_GCC_HOST_COMPILER_PREFIX = "GCC_HOST_COMPILER_PREFIX"
_TF_SYSROOT = "TF_SYSROOT"
_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"

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

def verify_build_defines(params):
    """Verify all variables that crosstool/BUILD.tpl expects are substituted.

    Args:
      params: dict of variables that will be passed to the BUILD.tpl template.
    """
    missing = []
    for param in [
        "cxx_builtin_include_directories",
        "extra_no_canonical_prefixes_flags",
        "host_compiler_path",
        "host_compiler_prefix",
        "host_compiler_warnings",
        "linker_bin_path",
        "compiler_deps",
        "unfiltered_compile_flags",
    ]:
        if ("%{" + param + "}") not in params:
            missing.append(param)

    if missing:
        auto_configure_fail(
            "BUILD.tpl template is missing these variables: " +
            str(missing) +
            ".\nWe only got: " +
            str(params) +
            ".",
        )

# TODO(dzc): Once these functions have been factored out of Bazel's
# cc_configure.bzl, load them from @bazel_tools instead.
# BEGIN cc_configure common functions.
def find_cc(repository_ctx):
    """Find the C++ compiler."""
    if _use_clang(repository_ctx):
        target_cc_name = "clang"
        cc_path_envvar = "CLANG_COMPILER_PATH"
    else:
        target_cc_name = "gcc"
        cc_path_envvar = _GCC_HOST_COMPILER_PATH
    cc_name = target_cc_name

    cc_name_from_env = get_host_environ(repository_ctx, cc_path_envvar)
    if cc_name_from_env:
        cc_name = cc_name_from_env
    if cc_name.startswith("/"):
        # Absolute path, maybe we should make this supported by our which function.
        return cc_name
    cc = which(repository_ctx, cc_name)
    if cc == None:
        fail(("Cannot find {}, either correct your path or set the {}" +
              " environment variable").format(target_cc_name, cc_path_envvar))
    return cc

_INC_DIR_MARKER_BEGIN = "#include <...>"

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

def _get_cxx_inc_directories_impl(repository_ctx, cc, lang_is_cpp, tf_sysroot):
    """Compute the list of default C or C++ include directories."""
    if lang_is_cpp:
        lang = "c++"
    else:
        lang = "c"
    sysroot = []
    if tf_sysroot:
        sysroot += ["--sysroot", tf_sysroot]
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

    return [
        _normalize_include_path(repository_ctx, p.strip())
        for p in inc_dirs.split("\n")
    ]

def get_cxx_inc_directories(repository_ctx, cc, tf_sysroot):
    """Compute the list of default C and C++ include directories."""

    # For some reason `clang -xc` sometimes returns include paths that are
    # different from the ones from `clang -xc++`. (Symlink and a dir)
    # So we run the compiler with both `-xc` and `-xc++` and merge resulting lists
    includes_cpp = _get_cxx_inc_directories_impl(
        repository_ctx,
        cc,
        True,
        tf_sysroot,
    )
    includes_c = _get_cxx_inc_directories_impl(
        repository_ctx,
        cc,
        False,
        tf_sysroot,
    )

    return includes_cpp + [
        inc
        for inc in includes_c
        if inc not in includes_cpp
    ]

def auto_configure_fail(msg):
    """Output failure message when aarch64 gcc configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sAARCH64 gcc Configuration Error:%s %s\n" % (red, no_color, msg))

# END cc_configure common functions (see TODO above).

def _use_clang(repository_ctx):
    return get_host_environ(repository_ctx, "CC_TOOLCHAIN_NAME") == "linux_llvm_aarch64"

def _tf_sysroot(repository_ctx):
    return get_host_environ(repository_ctx, _TF_SYSROOT, "")

def _tpl_path(repository_ctx, filename):
    return repository_ctx.path(Label("//tensorflow/tools/toolchains/cpus/aarch64/%s.tpl" % filename))

def _create_local_aarch64_repository(repository_ctx):
    """Creates the repository containing files set up to build with gcc."""

    # Resolve all labels before doing any real work. Resolving causes the
    # function to be restarted with all previous state being lost. This
    # can easily lead to a O(n^2) runtime in the number of labels.
    # See https://github.com/tensorflow/tensorflow/commit/62bd3534525a036f07d9851b3199d68212904778
    tpl_paths = {filename: _tpl_path(repository_ctx, filename) for filename in [
        "crosstool:BUILD",
        "crosstool:cc_toolchain_config.bzl",
    ]}

    tf_sysroot = _tf_sysroot(repository_ctx)

    cc = find_cc(repository_ctx)
    cc_fullpath = cc

    host_compiler_includes = get_cxx_inc_directories(
        repository_ctx,
        cc_fullpath,
        tf_sysroot,
    )

    aarch64_defines = {}
    aarch64_defines["%{builtin_sysroot}"] = tf_sysroot
    aarch64_defines["%{compiler}"] = "gcc"

    host_compiler_prefix = get_host_environ(repository_ctx, _GCC_HOST_COMPILER_PREFIX)
    if not host_compiler_prefix:
        host_compiler_prefix = "/usr/bin"

    aarch64_defines["%{host_compiler_prefix}"] = host_compiler_prefix

    aarch64_defines["%{linker_bin_path}"] = host_compiler_prefix
    aarch64_defines["%{host_compiler_path}"] = str(cc)
    aarch64_defines["%{host_compiler_warnings}"] = ""
    aarch64_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(host_compiler_includes)
    aarch64_defines["%{compiler_deps}"] = ":aarch64_gcc_pieces"
    aarch64_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-fno-canonical-system-headers\""
    aarch64_defines["%{unfiltered_compile_flags}"] = ""

    verify_build_defines(aarch64_defines)

    # Only expand template variables in the BUILD file
    repository_ctx.template(
        "crosstool/BUILD",
        tpl_paths["crosstool:BUILD"],
        aarch64_defines,
    )

    # No templating of cc_toolchain_config - use attributes and templatize the
    # BUILD file.
    repository_ctx.template(
        "crosstool/cc_toolchain_config.bzl",
        tpl_paths["crosstool:cc_toolchain_config.bzl"],
        {},
    )

def _create_local_aarch64_clang_repository(repository_ctx):
    """Creates the repository containing files set up to build with clang."""

    # Resolve all labels before doing any real work. Resolving causes the
    # function to be restarted with all previous state being lost. This
    # can easily lead to a O(n^2) runtime in the number of labels.
    # See https://github.com/tensorflow/tensorflow/commit/62bd3534525a036f07d9851b3199d68212904778
    tpl_paths = {filename: _tpl_path(repository_ctx, filename) for filename in [
        "crosstool:BUILD",
        "crosstool:cc_toolchain_config.bzl",
    ]}

    tf_sysroot = _tf_sysroot(repository_ctx)

    cc = find_cc(repository_ctx)
    cc_fullpath = cc

    host_compiler_includes = get_cxx_inc_directories(
        repository_ctx,
        cc_fullpath,
        tf_sysroot,
    )

    aarch64_clang_defines = {}
    aarch64_clang_defines["%{builtin_sysroot}"] = tf_sysroot
    aarch64_clang_defines["%{compiler}"] = "clang"

    host_compiler_prefix = get_host_environ(repository_ctx, _GCC_HOST_COMPILER_PREFIX)
    if not host_compiler_prefix:
        host_compiler_prefix = "/usr/bin"

    aarch64_clang_defines["%{host_compiler_prefix}"] = host_compiler_prefix

    aarch64_clang_defines["%{linker_bin_path}"] = host_compiler_prefix
    aarch64_clang_defines["%{host_compiler_path}"] = str(cc)
    aarch64_clang_defines["%{host_compiler_warnings}"] = """
        # Some parts of the codebase set -Werror and hit this warning, so
        # switch it off for now.
        "-Wno-invalid-partial-specialization"
    """
    aarch64_clang_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(host_compiler_includes)
    aarch64_clang_defines["%{compiler_deps}"] = ":empty"
    aarch64_clang_defines["%{extra_no_canonical_prefixes_flags}"] = ""
    aarch64_clang_defines["%{unfiltered_compile_flags}"] = ""

    verify_build_defines(aarch64_clang_defines)

    # Only expand template variables in the BUILD file
    repository_ctx.template(
        "crosstool/BUILD",
        tpl_paths["crosstool:BUILD"],
        aarch64_clang_defines,
    )

    # No templating of cc_toolchain_config - use attributes and templatize the
    # BUILD file.
    repository_ctx.template(
        "crosstool/cc_toolchain_config.bzl",
        tpl_paths["crosstool:cc_toolchain_config.bzl"],
        {},
    )

def _clang_autoconf_impl(repository_ctx):
    if _use_clang(repository_ctx):
        _create_local_aarch64_clang_repository(repository_ctx)
    else:
        _create_local_aarch64_repository(repository_ctx)

_ENVIRONS = [
    "CC_TOOLCHAIN_NAME",
    "CLANG_COMPILER_PATH",
    _GCC_HOST_COMPILER_PATH,
    _GCC_HOST_COMPILER_PREFIX,
    _PYTHON_BIN_PATH,
    "TMP",
    "TMPDIR",
]

remote_aarch64_configure = repository_rule(
    implementation = _clang_autoconf_impl,
    environ = _ENVIRONS,
    remotable = True,
    attrs = {
        "environ": attr.string_dict(),
    },
)
