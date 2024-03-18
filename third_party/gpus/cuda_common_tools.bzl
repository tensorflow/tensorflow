"""Common compilator and CUDA configuration functions. """

load(
    "@bazel_tools//tools/cpp:lib_cc_configure.bzl",
    "escape_string",
    "get_env_var",
)
load(
    "@bazel_tools//tools/cpp:windows_cc_configure.bzl",
    "find_msvc_tool",
    "find_vc_path",
    "setup_vc_env_vars",
)
load(
    "//third_party/remote_config:common.bzl",
    "err_out",
    "get_host_environ",
    "is_windows",
    "raw_exec",
    "realpath",
    "which",
)

GCC_HOST_COMPILER_PATH = "GCC_HOST_COMPILER_PATH"
GCC_HOST_COMPILER_PREFIX = "GCC_HOST_COMPILER_PREFIX"
CLANG_CUDA_COMPILER_PATH = "CLANG_CUDA_COMPILER_PATH"
TF_SYSROOT = "TF_SYSROOT"
CUDA_TOOLKIT_PATH = "CUDA_TOOLKIT_PATH"
TF_CUDA_VERSION = "TF_CUDA_VERSION"
TF_CUDNN_VERSION = "TF_CUDNN_VERSION"
CUDNN_INSTALL_PATH = "CUDNN_INSTALL_PATH"
TF_CUDA_COMPUTE_CAPABILITIES = "TF_CUDA_COMPUTE_CAPABILITIES"
TF_CUDA_CONFIG_REPO = "TF_CUDA_CONFIG_REPO"
TF_DOWNLOAD_CLANG = "TF_DOWNLOAD_CLANG"
PYTHON_BIN_PATH = "PYTHON_BIN_PATH"
TF_NEED_CUDA = "TF_NEED_CUDA"
TF_CUDA_CLANG = "TF_CUDA_CLANG"
TF_NVCC_CLANG = "TF_NVCC_CLANG"

# For @bazel_tools//tools/cpp:windows_cc_configure.bzl
MSVC_ENVVARS = [
    "BAZEL_VC",
    "BAZEL_VC_FULL_VERSION",
    "BAZEL_VS",
    "BAZEL_WINSDK_FULL_VERSION",
    "VS90COMNTOOLS",
    "VS100COMNTOOLS",
    "VS110COMNTOOLS",
    "VS120COMNTOOLS",
    "VS140COMNTOOLS",
    "VS150COMNTOOLS",
    "VS160COMNTOOLS",
]

# General purpose functions.

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

def flag_enabled(repository_ctx, flag_name):
    return get_host_environ(repository_ctx, flag_name) == "1"

def use_cuda_clang(repository_ctx):
    # Returns the flag if we need to use clang both for C++ and Cuda.
    return flag_enabled(repository_ctx, TF_CUDA_CLANG)

def use_nvcc_and_clang(repository_ctx):
    # Returns the flag if we need to use clang for C++ and NVCC for Cuda.
    return flag_enabled(repository_ctx, TF_NVCC_CLANG)

def enable_cuda(repository_ctx):
    """Returns whether to build with CUDA support."""
    return int(get_host_environ(repository_ctx, TF_NEED_CUDA, False))

def tf_sysroot(repository_ctx):
    return get_host_environ(repository_ctx, TF_SYSROOT, "")

# CUDA-specific functions.

def py_tmpl_dict(d):
    return {"%{cuda_config}": str(d)}

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
        "msvc_cl_path",
        "msvc_env_include",
        "msvc_env_lib",
        "msvc_env_path",
        "msvc_env_tmp",
        "msvc_lib_path",
        "msvc_link_path",
        "msvc_ml_path",
        "unfiltered_compile_flags",
        "win_compiler_deps",
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

def compute_capabilities(repository_ctx):
    """Returns a list of strings representing cuda compute capabilities.

    Args:
      repository_ctx: the repo rule's context.

    Returns:
      list of cuda architectures to compile for. 'compute_xy' refers to
      both PTX and SASS, 'sm_xy' refers to SASS only.
    """
    capabilities = get_host_environ(
        repository_ctx,
        TF_CUDA_COMPUTE_CAPABILITIES,
        "compute_35,compute_52",
    ).split(",")

    # Map old 'x.y' capabilities to 'compute_xy'.
    if len(capabilities) > 0 and all([len(x.split(".")) == 2 for x in capabilities]):
        # If all capabilities are in 'x.y' format, only include PTX for the
        # highest capability.
        cc_list = sorted([x.replace(".", "") for x in capabilities])
        capabilities = ["sm_%s" % x for x in cc_list[:-1]] + ["compute_%s" % cc_list[-1]]
    for i, capability in enumerate(capabilities):
        parts = capability.split(".")
        if len(parts) != 2:
            continue
        capabilities[i] = "compute_%s%s" % (parts[0], parts[1])

    # Make list unique
    capabilities = dict(zip(capabilities, capabilities)).keys()

    # Validate capabilities.
    for capability in capabilities:
        if not capability.startswith(("compute_", "sm_")):
            auto_configure_fail("Invalid compute capability: %s" % capability)
        for prefix in ["compute_", "sm_"]:
            if not capability.startswith(prefix):
                continue
            if len(capability) == len(prefix) + 2 and capability[-2:].isdigit():
                continue
            if len(capability) == len(prefix) + 3 and capability.endswith("90a"):
                continue
            auto_configure_fail("Invalid compute capability: %s" % capability)

    return capabilities

def lib_name(base_name, cpu_value, version = None, static = False):
    """Constructs the platform-specific name of a library.

    Args:
    base_name: The name of the library, such as "cudart"
    cpu_value: The name of the host operating system.
    version: The version of the library.
    static: True the library is static or False if it is a shared object.

    Returns:
    The platform-specific name of the library.
    """
    version = "" if not version else "." + version
    if cpu_value in ("Linux", "FreeBSD"):
        if static:
            return "lib%s.a" % base_name
        return "lib%s.so%s" % (base_name, version)
    elif cpu_value == "Windows":
        return "%s.lib" % base_name
    elif cpu_value == "Darwin":
        if static:
            return "lib%s.a" % base_name
        return "lib%s%s.dylib" % (base_name, version)
    else:
        auto_configure_fail("Invalid cpu_value: %s" % cpu_value)

def cudart_static_linkopt(cpu_value):
    """Returns additional platform-specific linkopts for cudart."""
    return "" if cpu_value == "Darwin" else "\"-lrt\","

def compute_cuda_extra_copts(repository_ctx, compute_capabilities):
    copts = ["--no-cuda-include-ptx=all"] if use_cuda_clang(repository_ctx) else []
    for capability in compute_capabilities:
        if capability.startswith("compute_"):
            capability = capability.replace("compute_", "sm_")
            copts.append("--cuda-include-ptx=%s" % capability)
        copts.append("--cuda-gpu-arch=%s" % capability)

    return str(copts)

# Compiler-specific functions.

# TODO(dzc): Once these functions have been factored out of Bazel's
# cc_configure.bzl, load them from @bazel_tools instead.
# BEGIN cc_configure common functions.
def find_cc(repository_ctx, use_cuda_clang):
    """Find the C++ compiler."""
    if is_windows(repository_ctx):
        return _get_msvc_compiler(repository_ctx)

    if use_cuda_clang:
        target_cc_name = "clang"
        cc_path_envvar = CLANG_CUDA_COMPILER_PATH
        if flag_enabled(repository_ctx, TF_DOWNLOAD_CLANG):
            return "extra_tools/bin/clang"
    else:
        target_cc_name = "gcc"
        cc_path_envvar = GCC_HOST_COMPILER_PATH
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

def get_nvcc_tmp_dir_for_windows(repository_ctx):
    """Return the Windows tmp directory for nvcc to generate intermediate source files."""
    escaped_tmp_dir = escape_string(
        get_env_var(repository_ctx, "TMP", "C:\\Windows\\Temp").replace(
            "\\",
            "\\\\",
        ),
    )
    return escaped_tmp_dir + "\\\\nvcc_inter_files_tmp_dir"

def _get_msvc_compiler(repository_ctx):
    vc_path = find_vc_path(repository_ctx)
    return find_msvc_tool(repository_ctx, vc_path, "cl.exe").replace("\\", "/")

def get_win_cuda_defines(repository_ctx):
    """Return CROSSTOOL defines for Windows"""

    # If we are not on Windows, return fake vaules for Windows specific fields.
    # This ensures the CROSSTOOL file parser is happy.
    if not is_windows(repository_ctx):
        return {
            "%{msvc_env_tmp}": "msvc_not_used",
            "%{msvc_env_path}": "msvc_not_used",
            "%{msvc_env_include}": "msvc_not_used",
            "%{msvc_env_lib}": "msvc_not_used",
            "%{msvc_cl_path}": "msvc_not_used",
            "%{msvc_ml_path}": "msvc_not_used",
            "%{msvc_link_path}": "msvc_not_used",
            "%{msvc_lib_path}": "msvc_not_used",
        }

    vc_path = find_vc_path(repository_ctx)
    if not vc_path:
        auto_configure_fail(
            "Visual C++ build tools not found on your machine." +
            "Please check your installation following https://docs.bazel.build/versions/master/windows.html#using",
        )
        return {}

    env = setup_vc_env_vars(repository_ctx, vc_path)
    escaped_paths = escape_string(env["PATH"])
    escaped_include_paths = escape_string(env["INCLUDE"])
    escaped_lib_paths = escape_string(env["LIB"])
    escaped_tmp_dir = escape_string(
        get_env_var(repository_ctx, "TMP", "C:\\Windows\\Temp").replace(
            "\\",
            "\\\\",
        ),
    )

    msvc_cl_path = "windows/msvc_wrapper_for_nvcc.bat"
    msvc_ml_path = find_msvc_tool(repository_ctx, vc_path, "ml64.exe").replace(
        "\\",
        "/",
    )
    msvc_link_path = find_msvc_tool(repository_ctx, vc_path, "link.exe").replace(
        "\\",
        "/",
    )
    msvc_lib_path = find_msvc_tool(repository_ctx, vc_path, "lib.exe").replace(
        "\\",
        "/",
    )

    # nvcc will generate some temporary source files under %{nvcc_tmp_dir}
    # The generated files are guaranteed to have unique name, so they can share
    # the same tmp directory
    escaped_cxx_include_directories = [
        get_nvcc_tmp_dir_for_windows(repository_ctx),
        "C:\\\\botcode\\\\w",
    ]
    for path in escaped_include_paths.split(";"):
        if path:
            escaped_cxx_include_directories.append(path)

    return {
        "%{msvc_env_tmp}": escaped_tmp_dir,
        "%{msvc_env_path}": escaped_paths,
        "%{msvc_env_include}": escaped_include_paths,
        "%{msvc_env_lib}": escaped_lib_paths,
        "%{msvc_cl_path}": msvc_cl_path,
        "%{msvc_ml_path}": msvc_ml_path,
        "%{msvc_link_path}": msvc_link_path,
        "%{msvc_lib_path}": msvc_lib_path,
        "%{cxx_builtin_include_directories}": to_list_of_strings(
            escaped_cxx_include_directories,
        ),
    }

_INC_DIR_MARKER_BEGIN = "#include <...>"

# OSX add " (framework directory)" at the end of line, strip it.
_OSX_FRAMEWORK_SUFFIX = " (framework directory)"
_OSX_FRAMEWORK_SUFFIX_LEN = len(_OSX_FRAMEWORK_SUFFIX)

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

def auto_configure_fail(msg):
    """Output failure message when cuda configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sCuda Configuration Error:%s %s\n" % (red, no_color, msg))

# END cc_configure common functions (see TODO above).
