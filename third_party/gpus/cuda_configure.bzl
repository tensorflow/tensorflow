"""Repository rule for CUDA autoconfiguration.

`cuda_configure` depends on the following environment variables:

  * `TF_NEED_CUDA`: Whether to enable building with CUDA.
  * `GCC_HOST_COMPILER_PATH`: The GCC host compiler path
  * `TF_CUDA_CLANG`: Whether to use clang as a cuda compiler.
  * `CLANG_CUDA_COMPILER_PATH`: The clang compiler path that will be used for
    both host and device code compilation if TF_CUDA_CLANG is 1.
  * `TF_SYSROOT`: The sysroot to use when compiling.
  * `TF_DOWNLOAD_CLANG`: Whether to download a recent release of clang
    compiler and use it to build tensorflow. When this option is set
    CLANG_CUDA_COMPILER_PATH is ignored.
  * `TF_CUDA_PATHS`: The base paths to look for CUDA and cuDNN. Default is
    `/usr/local/cuda,usr/`.
  * `CUDA_TOOLKIT_PATH` (deprecated): The path to the CUDA toolkit. Default is
    `/usr/local/cuda`.
  * `TF_CUDA_VERSION`: The version of the CUDA toolkit. If this is blank, then
    use the system default.
  * `TF_CUDNN_VERSION`: The version of the cuDNN library.
  * `CUDNN_INSTALL_PATH` (deprecated): The path to the cuDNN library. Default is
    `/usr/local/cuda`.
  * `TF_CUDA_COMPUTE_CAPABILITIES`: The CUDA compute capabilities. Default is
    `3.5,5.2`.
  * `PYTHON_BIN_PATH`: The python binary path
"""

load("//third_party/clang_toolchain:download_clang.bzl", "download_clang")
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

_GCC_HOST_COMPILER_PATH = "GCC_HOST_COMPILER_PATH"
_GCC_HOST_COMPILER_PREFIX = "GCC_HOST_COMPILER_PREFIX"
_CLANG_CUDA_COMPILER_PATH = "CLANG_CUDA_COMPILER_PATH"
_TF_SYSROOT = "TF_SYSROOT"
_CUDA_TOOLKIT_PATH = "CUDA_TOOLKIT_PATH"
_TF_CUDA_VERSION = "TF_CUDA_VERSION"
_TF_CUDNN_VERSION = "TF_CUDNN_VERSION"
_CUDNN_INSTALL_PATH = "CUDNN_INSTALL_PATH"
_TF_CUDA_COMPUTE_CAPABILITIES = "TF_CUDA_COMPUTE_CAPABILITIES"
_TF_CUDA_CONFIG_REPO = "TF_CUDA_CONFIG_REPO"
_TF_DOWNLOAD_CLANG = "TF_DOWNLOAD_CLANG"
_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"

_DEFAULT_CUDA_COMPUTE_CAPABILITIES = ["3.5", "5.2"]

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

def _get_python_bin(repository_ctx):
    """Gets the python bin path."""
    python_bin = repository_ctx.os.environ.get(_PYTHON_BIN_PATH)
    if python_bin != None:
        return python_bin
    python_bin_name = "python.exe" if _is_windows(repository_ctx) else "python"
    python_bin_path = repository_ctx.which(python_bin_name)
    if python_bin_path != None:
        return str(python_bin_path)
    auto_configure_fail(
        "Cannot find python in PATH, please make sure " +
        "python is installed and add its directory in PATH, or --define " +
        "%s='/something/else'.\nPATH=%s" % (
            _PYTHON_BIN_PATH,
            repository_ctx.os.environ.get("PATH", ""),
        ),
    )

def _get_nvcc_tmp_dir_for_windows(repository_ctx):
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

def _get_win_cuda_defines(repository_ctx):
    """Return CROSSTOOL defines for Windows"""

    # If we are not on Windows, return fake vaules for Windows specific fields.
    # This ensures the CROSSTOOL file parser is happy.
    if not _is_windows(repository_ctx):
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

    msvc_cl_path = _get_python_bin(repository_ctx)
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
        _get_nvcc_tmp_dir_for_windows(repository_ctx),
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

# TODO(dzc): Once these functions have been factored out of Bazel's
# cc_configure.bzl, load them from @bazel_tools instead.
# BEGIN cc_configure common functions.
def find_cc(repository_ctx):
    """Find the C++ compiler."""
    if _is_windows(repository_ctx):
        return _get_msvc_compiler(repository_ctx)

    if _use_cuda_clang(repository_ctx):
        target_cc_name = "clang"
        cc_path_envvar = _CLANG_CUDA_COMPILER_PATH
        if _flag_enabled(repository_ctx, _TF_DOWNLOAD_CLANG):
            return "extra_tools/bin/clang"
    else:
        target_cc_name = "gcc"
        cc_path_envvar = _GCC_HOST_COMPILER_PATH
    cc_name = target_cc_name

    if cc_path_envvar in repository_ctx.os.environ:
        cc_name_from_env = repository_ctx.os.environ[cc_path_envvar].strip()
        if cc_name_from_env:
            cc_name = cc_name_from_env
    if cc_name.startswith("/"):
        # Absolute path, maybe we should make this supported by our which function.
        return cc_name
    cc = repository_ctx.which(cc_name)
    if cc == None:
        fail(("Cannot find {}, either correct your path or set the {}" +
              " environment variable").format(target_cc_name, cc_path_envvar))
    return cc

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

def _get_cxx_inc_directories_impl(repository_ctx, cc, lang_is_cpp, tf_sysroot):
    """Compute the list of default C or C++ include directories."""
    if lang_is_cpp:
        lang = "c++"
    else:
        lang = "c"
    sysroot = []
    if tf_sysroot:
        sysroot += ["--sysroot", tf_sysroot]
    result = repository_ctx.execute([cc, "-E", "-x" + lang, "-", "-v"] +
                                    sysroot)
    index1 = result.stderr.find(_INC_DIR_MARKER_BEGIN)
    if index1 == -1:
        return []
    index1 = result.stderr.find("\n", index1)
    if index1 == -1:
        return []
    index2 = result.stderr.rfind("\n ")
    if index2 == -1 or index2 < index1:
        return []
    index2 = result.stderr.find("\n", index2 + 1)
    if index2 == -1:
        inc_dirs = result.stderr[index1 + 1:]
    else:
        inc_dirs = result.stderr[index1 + 1:index2].strip()

    return [
        _normalize_include_path(repository_ctx, _cxx_inc_convert(p))
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
    """Output failure message when cuda configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sCuda Configuration Error:%s %s\n" % (red, no_color, msg))

# END cc_configure common functions (see TODO above).

def _cuda_include_path(repository_ctx, cuda_config):
    """Generates the Starlark string with cuda include directories.

      Args:
        repository_ctx: The repository context.
        cc: The path to the gcc host compiler.

      Returns:
        A list of the gcc host compiler include directories.
      """
    nvcc_path = repository_ctx.path("%s/bin/nvcc%s" % (
        cuda_config.cuda_toolkit_path,
        ".exe" if cuda_config.cpu_value == "Windows" else "",
    ))
    result = repository_ctx.execute([
        nvcc_path,
        "-v",
        "/dev/null",
        "-o",
        "/dev/null",
    ])
    target_dir = ""
    for one_line in result.stderr.splitlines():
        if one_line.startswith("#$ _TARGET_DIR_="):
            target_dir = (
                cuda_config.cuda_toolkit_path + "/" + one_line.replace(
                    "#$ _TARGET_DIR_=",
                    "",
                ) + "/include"
            )
    inc_entries = []
    if target_dir != "":
        inc_entries.append(target_dir)
    inc_entries.append(cuda_config.cuda_toolkit_path + "/include")
    return inc_entries

def enable_cuda(repository_ctx):
    """Returns whether to build with CUDA support."""
    return int(repository_ctx.os.environ.get("TF_NEED_CUDA", False))

def matches_version(environ_version, detected_version):
    """Checks whether the user-specified version matches the detected version.

      This function performs a weak matching so that if the user specifies only
      the
      major or major and minor versions, the versions are still considered
      matching
      if the version parts match. To illustrate:

          environ_version  detected_version  result
          -----------------------------------------
          5.1.3            5.1.3             True
          5.1              5.1.3             True
          5                5.1               True
          5.1.3            5.1               False
          5.2.3            5.1.3             False

      Args:
        environ_version: The version specified by the user via environment
          variables.
        detected_version: The version autodetected from the CUDA installation on
          the system.
      Returns: True if user-specified version matches detected version and False
        otherwise.
    """
    environ_version_parts = environ_version.split(".")
    detected_version_parts = detected_version.split(".")
    if len(detected_version_parts) < len(environ_version_parts):
        return False
    for i, part in enumerate(detected_version_parts):
        if i >= len(environ_version_parts):
            break
        if part != environ_version_parts[i]:
            return False
    return True

_NVCC_VERSION_PREFIX = "Cuda compilation tools, release "

_DEFINE_CUDNN_MAJOR = "#define CUDNN_MAJOR"

def compute_capabilities(repository_ctx):
    """Returns a list of strings representing cuda compute capabilities."""
    if _TF_CUDA_COMPUTE_CAPABILITIES not in repository_ctx.os.environ:
        return _DEFAULT_CUDA_COMPUTE_CAPABILITIES
    capabilities_str = repository_ctx.os.environ[_TF_CUDA_COMPUTE_CAPABILITIES]
    capabilities = capabilities_str.split(",")
    for capability in capabilities:
        # Workaround for Skylark's lack of support for regex. This check should
        # be equivalent to checking:
        #     if re.match("[0-9]+.[0-9]+", capability) == None:
        parts = capability.split(".")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            auto_configure_fail("Invalid compute capability: %s" % capability)
    return capabilities

def get_cpu_value(repository_ctx):
    """Returns the name of the host operating system.

      Args:
        repository_ctx: The repository context.

      Returns:
        A string containing the name of the host operating system.
      """
    os_name = repository_ctx.os.name.lower()
    if os_name.startswith("mac os"):
        return "Darwin"
    if os_name.find("windows") != -1:
        return "Windows"
    result = repository_ctx.execute(["uname", "-s"])
    return result.stdout.strip()

def _is_windows(repository_ctx):
    """Returns true if the host operating system is windows."""
    return repository_ctx.os.name.lower().find("windows") >= 0

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

def find_lib(repository_ctx, paths, check_soname = True):
    """
      Finds a library among a list of potential paths.

      Args:
        paths: List of paths to inspect.

      Returns:
        Returns the first path in paths that exist.
    """
    objdump = repository_ctx.which("objdump")
    mismatches = []
    for path in [repository_ctx.path(path) for path in paths]:
        if not path.exists:
            continue
        if check_soname and objdump != None and not _is_windows(repository_ctx):
            output = repository_ctx.execute([objdump, "-p", str(path)]).stdout
            output = [line for line in output.splitlines() if "SONAME" in line]
            sonames = [line.strip().split(" ")[-1] for line in output]
            if not any([soname == path.basename for soname in sonames]):
                mismatches.append(str(path))
                continue
        return path
    if mismatches:
        auto_configure_fail(
            "None of the libraries match their SONAME: " + ", ".join(mismatches),
        )
    auto_configure_fail("No library found under: " + ", ".join(paths))

def _find_cuda_lib(
        lib,
        repository_ctx,
        cpu_value,
        basedir,
        version,
        static = False):
    """Finds the given CUDA or cuDNN library on the system.

      Args:
        lib: The name of the library, such as "cudart"
        repository_ctx: The repository context.
        cpu_value: The name of the host operating system.
        basedir: The install directory of CUDA or cuDNN.
        version: The version of the library.
        static: True if static library, False if shared object.

      Returns:
        Returns the path to the library.
      """
    file_name = lib_name(lib, cpu_value, version, static)
    return find_lib(
        repository_ctx,
        ["%s/%s" % (basedir, file_name)],
        check_soname = version and not static,
    )

def _find_libs(repository_ctx, cuda_config):
    """Returns the CUDA and cuDNN libraries on the system.

      Args:
        repository_ctx: The repository context.
        cuda_config: The CUDA config as returned by _get_cuda_config

      Returns:
        Map of library names to structs of filename and path.
      """
    cpu_value = cuda_config.cpu_value
    stub_dir = "" if _is_windows(repository_ctx) else "/stubs"
    return {
        "cuda": _find_cuda_lib(
            "cuda",
            repository_ctx,
            cpu_value,
            cuda_config.config["cuda_library_dir"] + stub_dir,
            None,
        ),
        "cudart": _find_cuda_lib(
            "cudart",
            repository_ctx,
            cpu_value,
            cuda_config.config["cuda_library_dir"],
            cuda_config.cuda_version,
        ),
        "cudart_static": _find_cuda_lib(
            "cudart_static",
            repository_ctx,
            cpu_value,
            cuda_config.config["cuda_library_dir"],
            cuda_config.cuda_version,
            static = True,
        ),
        "cublas": _find_cuda_lib(
            "cublas",
            repository_ctx,
            cpu_value,
            cuda_config.config["cublas_library_dir"],
            cuda_config.cuda_lib_version,
        ),
        "cusolver": _find_cuda_lib(
            "cusolver",
            repository_ctx,
            cpu_value,
            cuda_config.config["cuda_library_dir"],
            cuda_config.cuda_lib_version,
        ),
        "curand": _find_cuda_lib(
            "curand",
            repository_ctx,
            cpu_value,
            cuda_config.config["cuda_library_dir"],
            cuda_config.cuda_lib_version,
        ),
        "cufft": _find_cuda_lib(
            "cufft",
            repository_ctx,
            cpu_value,
            cuda_config.config["cuda_library_dir"],
            cuda_config.cuda_lib_version,
        ),
        "cudnn": _find_cuda_lib(
            "cudnn",
            repository_ctx,
            cpu_value,
            cuda_config.config["cudnn_library_dir"],
            cuda_config.cudnn_version,
        ),
        "cupti": _find_cuda_lib(
            "cupti",
            repository_ctx,
            cpu_value,
            cuda_config.config["cupti_library_dir"],
            cuda_config.cuda_version,
        ),
        "cusparse": _find_cuda_lib(
            "cusparse",
            repository_ctx,
            cpu_value,
            cuda_config.config["cuda_library_dir"],
            cuda_config.cuda_lib_version,
        ),
    }

def _cudart_static_linkopt(cpu_value):
    """Returns additional platform-specific linkopts for cudart."""
    return "" if cpu_value == "Darwin" else "\"-lrt\","

# TODO(csigg): Only call once instead of from here, tensorrt_configure.bzl,
# and nccl_configure.bzl.
def find_cuda_config(repository_ctx, cuda_libraries):
    """Returns CUDA config dictionary from running find_cuda_config.py"""
    exec_result = repository_ctx.execute([
        _get_python_bin(repository_ctx),
        repository_ctx.path(Label("@org_tensorflow//third_party/gpus:find_cuda_config.py")),
    ] + cuda_libraries)
    if exec_result.return_code:
        auto_configure_fail("Failed to run find_cuda_config.py: %s" % exec_result.stderr)

    # Parse the dict from stdout.
    return dict([tuple(x.split(": ")) for x in exec_result.stdout.splitlines()])

def _get_cuda_config(repository_ctx):
    """Detects and returns information about the CUDA installation on the system.

      Args:
        repository_ctx: The repository context.

      Returns:
        A struct containing the following fields:
          cuda_toolkit_path: The CUDA toolkit installation directory.
          cudnn_install_basedir: The cuDNN installation directory.
          cuda_version: The version of CUDA on the system.
          cudnn_version: The version of cuDNN on the system.
          compute_capabilities: A list of the system's CUDA compute capabilities.
          cpu_value: The name of the host operating system.
      """
    config = find_cuda_config(repository_ctx, ["cuda", "cudnn"])
    cpu_value = get_cpu_value(repository_ctx)
    toolkit_path = config["cuda_toolkit_path"]

    is_windows = _is_windows(repository_ctx)
    cuda_version = config["cuda_version"].split(".")
    cuda_major = cuda_version[0]
    cuda_minor = cuda_version[1]

    cuda_version = ("64_%s%s" if is_windows else "%s.%s") % (cuda_major, cuda_minor)
    cudnn_version = ("64_%s" if is_windows else "%s") % config["cudnn_version"]

    # cuda_lib_version is for libraries like cuBLAS, cuFFT, cuSOLVER, etc.
    # It changed from 'x.y' to just 'x' in CUDA 10.1.
    if (int(cuda_major), int(cuda_minor)) >= (10, 1):
        cuda_lib_version = ("64_%s" if is_windows else "%s") % cuda_major
    else:
        cuda_lib_version = cuda_version

    return struct(
        cuda_toolkit_path = toolkit_path,
        cuda_version = cuda_version,
        cudnn_version = cudnn_version,
        cuda_lib_version = cuda_lib_version,
        compute_capabilities = compute_capabilities(repository_ctx),
        cpu_value = cpu_value,
        config = config,
    )

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl.replace(":", "/")
    repository_ctx.template(
        out,
        Label("//third_party/gpus/%s.tpl" % tpl),
        substitutions,
    )

def _file(repository_ctx, label):
    repository_ctx.template(
        label.replace(":", "/"),
        Label("//third_party/gpus/%s.tpl" % label),
        {},
    )

_DUMMY_CROSSTOOL_BZL_FILE = """
def error_gpu_disabled():
  fail("ERROR: Building with --config=cuda but TensorFlow is not configured " +
       "to build with GPU support. Please re-run ./configure and enter 'Y' " +
       "at the prompt to build with GPU support.")

  native.genrule(
      name = "error_gen_crosstool",
      outs = ["CROSSTOOL"],
      cmd = "echo 'Should not be run.' && exit 1",
  )

  native.filegroup(
      name = "crosstool",
      srcs = [":CROSSTOOL"],
      output_licenses = ["unencumbered"],
  )
"""

_DUMMY_CROSSTOOL_BUILD_FILE = """
load("//crosstool:error_gpu_disabled.bzl", "error_gpu_disabled")

error_gpu_disabled()
"""

def _create_dummy_repository(repository_ctx):
    cpu_value = get_cpu_value(repository_ctx)

    # Set up BUILD file for cuda/.
    _tpl(
        repository_ctx,
        "cuda:build_defs.bzl",
        {
            "%{cuda_is_configured}": "False",
            "%{cuda_extra_copts}": "[]",
        },
    )
    _tpl(
        repository_ctx,
        "cuda:BUILD",
        {
            "%{cuda_driver_lib}": lib_name("cuda", cpu_value),
            "%{cudart_static_lib}": lib_name(
                "cudart_static",
                cpu_value,
                static = True,
            ),
            "%{cudart_static_linkopt}": _cudart_static_linkopt(cpu_value),
            "%{cudart_lib}": lib_name("cudart", cpu_value),
            "%{cublas_lib}": lib_name("cublas", cpu_value),
            "%{cusolver_lib}": lib_name("cusolver", cpu_value),
            "%{cudnn_lib}": lib_name("cudnn", cpu_value),
            "%{cufft_lib}": lib_name("cufft", cpu_value),
            "%{curand_lib}": lib_name("curand", cpu_value),
            "%{cupti_lib}": lib_name("cupti", cpu_value),
            "%{cusparse_lib}": lib_name("cusparse", cpu_value),
            "%{copy_rules}": """
filegroup(name="cuda-include")
filegroup(name="cublas-include")
filegroup(name="cudnn-include")
""",
        },
    )

    # Create dummy files for the CUDA toolkit since they are still required by
    # tensorflow/core/platform/default/build_config:cuda.
    repository_ctx.file("cuda/cuda/include/cuda.h")
    repository_ctx.file("cuda/cuda/include/cublas.h")
    repository_ctx.file("cuda/cuda/include/cudnn.h")
    repository_ctx.file("cuda/cuda/extras/CUPTI/include/cupti.h")
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cuda", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cudart", cpu_value))
    repository_ctx.file(
        "cuda/cuda/lib/%s" % lib_name("cudart_static", cpu_value),
    )
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cublas", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cusolver", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cudnn", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("curand", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cufft", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cupti", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cusparse", cpu_value))

    # Set up cuda_config.h, which is used by
    # tensorflow/stream_executor/dso_loader.cc.
    _tpl(
        repository_ctx,
        "cuda:cuda_config.h",
        {
            "%{cuda_version}": "",
            "%{cuda_lib_version}": "",
            "%{cudnn_version}": "",
            "%{cuda_compute_capabilities}": ",".join([
                "CudaVersion(\"%s\")" % c
                for c in _DEFAULT_CUDA_COMPUTE_CAPABILITIES
            ]),
            "%{cuda_toolkit_path}": "",
        },
        "cuda/cuda/cuda_config.h",
    )

    # If cuda_configure is not configured to build with GPU support, and the user
    # attempts to build with --config=cuda, add a dummy build rule to intercept
    # this and fail with an actionable error message.
    repository_ctx.file(
        "crosstool/error_gpu_disabled.bzl",
        _DUMMY_CROSSTOOL_BZL_FILE,
    )
    repository_ctx.file("crosstool/BUILD", _DUMMY_CROSSTOOL_BUILD_FILE)

def _execute(
        repository_ctx,
        cmdline,
        error_msg = None,
        error_details = None,
        empty_stdout_fine = False):
    """Executes an arbitrary shell command.

      Args:
        repository_ctx: the repository_ctx object
        cmdline: list of strings, the command to execute
        error_msg: string, a summary of the error if the command fails
        error_details: string, details about the error or steps to fix it
        empty_stdout_fine: bool, if True, an empty stdout result is fine,
          otherwise it's an error
      Return: the result of repository_ctx.execute(cmdline)
    """
    result = repository_ctx.execute(cmdline)
    if result.stderr or not (empty_stdout_fine or result.stdout):
        auto_configure_fail(
            "\n".join([
                error_msg.strip() if error_msg else "Repository command failed",
                result.stderr.strip(),
                error_details if error_details else "",
            ]),
        )
    return result

def _norm_path(path):
    """Returns a path with '/' and remove the trailing slash."""
    path = path.replace("\\", "/")
    if path[-1] == "/":
        path = path[:-1]
    return path

def make_copy_files_rule(repository_ctx, name, srcs, outs):
    """Returns a rule to copy a set of files."""
    cmds = []

    # Copy files.
    for src, out in zip(srcs, outs):
        cmds.append('cp -f "%s" "$(location %s)"' % (src, out))
    outs = [('        "%s",' % out) for out in outs]
    return """genrule(
    name = "%s",
    outs = [
%s
    ],
    cmd = \"""%s \""",
)""" % (name, "\n".join(outs), " && \\\n".join(cmds))

def make_copy_dir_rule(repository_ctx, name, src_dir, out_dir):
    """Returns a rule to recursively copy a directory."""
    src_dir = _norm_path(src_dir)
    out_dir = _norm_path(out_dir)
    outs = _read_dir(repository_ctx, src_dir)
    outs = [('        "%s",' % out.replace(src_dir, out_dir)) for out in outs]

    # '@D' already contains the relative path for a single file, see
    # http://docs.bazel.build/versions/master/be/make-variables.html#predefined_genrule_variables
    out_dir = "$(@D)/%s" % out_dir if len(outs) > 1 else "$(@D)"
    return """genrule(
    name = "%s",
    outs = [
%s
    ],
    cmd = \"""cp -rLf "%s/." "%s/" \""",
)""" % (name, "\n".join(outs), src_dir, out_dir)

def _read_dir(repository_ctx, src_dir):
    """Returns a string with all files in a directory.

      Finds all files inside a directory, traversing subfolders and following
      symlinks. The returned string contains the full path of all files
      separated by line breaks.
      """
    if _is_windows(repository_ctx):
        src_dir = src_dir.replace("/", "\\")
        find_result = _execute(
            repository_ctx,
            ["cmd.exe", "/c", "dir", src_dir, "/b", "/s", "/a-d"],
            empty_stdout_fine = True,
        )

        # src_files will be used in genrule.outs where the paths must
        # use forward slashes.
        result = find_result.stdout.replace("\\", "/")
    else:
        find_result = _execute(
            repository_ctx,
            ["find", src_dir, "-follow", "-type", "f"],
            empty_stdout_fine = True,
        )
        result = find_result.stdout
    return sorted(result.splitlines())

def _flag_enabled(repository_ctx, flag_name):
    if flag_name in repository_ctx.os.environ:
        value = repository_ctx.os.environ[flag_name].strip()
        return value == "1"
    return False

def _use_cuda_clang(repository_ctx):
    return _flag_enabled(repository_ctx, "TF_CUDA_CLANG")

def _tf_sysroot(repository_ctx):
    if _TF_SYSROOT in repository_ctx.os.environ:
        return repository_ctx.os.environ[_TF_SYSROOT]
    return ""

def _compute_cuda_extra_copts(repository_ctx, compute_capabilities):
    capability_flags = [
        "--cuda-gpu-arch=sm_" + cap.replace(".", "")
        for cap in compute_capabilities
    ]

    # Capabilities are handled in the "crosstool_wrapper_driver_is_not_gcc" for nvcc
    # TODO(csigg): Make this consistent with cuda clang and pass unconditionally.
    return "if_cuda_clang(%s)" % str(capability_flags)

def _create_local_cuda_repository(repository_ctx):
    """Creates the repository containing files set up to build with CUDA."""
    cuda_config = _get_cuda_config(repository_ctx)

    cuda_include_path = cuda_config.config["cuda_include_dir"]
    cublas_include_path = cuda_config.config["cublas_include_dir"]
    cudnn_header_dir = cuda_config.config["cudnn_include_dir"]
    cupti_header_dir = cuda_config.config["cupti_include_dir"]
    nvvm_libdevice_dir = cuda_config.config["nvvm_library_dir"]

    # Create genrule to copy files from the installed CUDA toolkit into execroot.
    copy_rules = [
        make_copy_dir_rule(
            repository_ctx,
            name = "cuda-include",
            src_dir = cuda_include_path,
            out_dir = "cuda/include",
        ),
        make_copy_dir_rule(
            repository_ctx,
            name = "cuda-nvvm",
            src_dir = nvvm_libdevice_dir,
            out_dir = "cuda/nvvm/libdevice",
        ),
        make_copy_dir_rule(
            repository_ctx,
            name = "cuda-extras",
            src_dir = cupti_header_dir,
            out_dir = "cuda/extras/CUPTI/include",
        ),
    ]

    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "cublas-include",
        srcs = [
            cublas_include_path + "/cublas.h",
            cublas_include_path + "/cublas_v2.h",
            cublas_include_path + "/cublas_api.h",
        ],
        outs = [
            "cublas/include/cublas.h",
            "cublas/include/cublas_v2.h",
            "cublas/include/cublas_api.h",
        ],
    ))

    cuda_libs = _find_libs(repository_ctx, cuda_config)
    cuda_lib_srcs = []
    cuda_lib_outs = []
    for path in cuda_libs.values():
        cuda_lib_srcs.append(str(path))
        cuda_lib_outs.append("cuda/lib/" + path.basename)
    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "cuda-lib",
        srcs = cuda_lib_srcs,
        outs = cuda_lib_outs,
    ))

    # copy files mentioned in third_party/nccl/build_defs.bzl.tpl
    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "cuda-bin",
        srcs = [
            cuda_config.cuda_toolkit_path + "/bin/" + "crt/link.stub",
            cuda_config.cuda_toolkit_path + "/bin/" + "nvlink",
            cuda_config.cuda_toolkit_path + "/bin/" + "fatbinary",
            cuda_config.cuda_toolkit_path + "/bin/" + "bin2c",
        ],
        outs = [
            "cuda/bin/" + "crt/link.stub",
            "cuda/bin/" + "nvlink",
            "cuda/bin/" + "fatbinary",
            "cuda/bin/" + "bin2c",
        ],
    ))

    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "cudnn-include",
        srcs = [cudnn_header_dir + "/cudnn.h"],
        outs = ["cudnn/include/cudnn.h"],
    ))

    # Set up BUILD file for cuda/
    _tpl(
        repository_ctx,
        "cuda:build_defs.bzl",
        {
            "%{cuda_is_configured}": "True",
            "%{cuda_extra_copts}": _compute_cuda_extra_copts(
                repository_ctx,
                cuda_config.compute_capabilities,
            ),
        },
    )
    _tpl(
        repository_ctx,
        "cuda:BUILD.windows" if _is_windows(repository_ctx) else "cuda:BUILD",
        {
            "%{cuda_driver_lib}": cuda_libs["cuda"].basename,
            "%{cudart_static_lib}": cuda_libs["cudart_static"].basename,
            "%{cudart_static_linkopt}": _cudart_static_linkopt(cuda_config.cpu_value),
            "%{cudart_lib}": cuda_libs["cudart"].basename,
            "%{cublas_lib}": cuda_libs["cublas"].basename,
            "%{cusolver_lib}": cuda_libs["cusolver"].basename,
            "%{cudnn_lib}": cuda_libs["cudnn"].basename,
            "%{cufft_lib}": cuda_libs["cufft"].basename,
            "%{curand_lib}": cuda_libs["curand"].basename,
            "%{cupti_lib}": cuda_libs["cupti"].basename,
            "%{cusparse_lib}": cuda_libs["cusparse"].basename,
            "%{copy_rules}": "\n".join(copy_rules),
        },
        "cuda/BUILD",
    )

    is_cuda_clang = _use_cuda_clang(repository_ctx)
    tf_sysroot = _tf_sysroot(repository_ctx)

    should_download_clang = is_cuda_clang and _flag_enabled(
        repository_ctx,
        _TF_DOWNLOAD_CLANG,
    )
    if should_download_clang:
        download_clang(repository_ctx, "crosstool/extra_tools")

    # Set up crosstool/
    cc = find_cc(repository_ctx)
    cc_fullpath = cc if not should_download_clang else "crosstool/" + cc

    host_compiler_includes = get_cxx_inc_directories(
        repository_ctx,
        cc_fullpath,
        tf_sysroot,
    )
    cuda_defines = {}
    cuda_defines["%{builtin_sysroot}"] = tf_sysroot
    cuda_defines["%{cuda_toolkit_path}"] = ""
    if is_cuda_clang:
        cuda_defines["%{cuda_toolkit_path}"] = cuda_config.config["cuda_toolkit_path"]

    host_compiler_prefix = "/usr/bin"
    if _GCC_HOST_COMPILER_PREFIX in repository_ctx.os.environ:
        host_compiler_prefix = repository_ctx.os.environ[_GCC_HOST_COMPILER_PREFIX].strip()
    cuda_defines["%{host_compiler_prefix}"] = host_compiler_prefix

    # Bazel sets '-B/usr/bin' flag to workaround build errors on RHEL (see
    # https://github.com/bazelbuild/bazel/issues/760).
    # However, this stops our custom clang toolchain from picking the provided
    # LLD linker, so we're only adding '-B/usr/bin' when using non-downloaded
    # toolchain.
    # TODO: when bazel stops adding '-B/usr/bin' by default, remove this
    #       flag from the CROSSTOOL completely (see
    #       https://github.com/bazelbuild/bazel/issues/5634)
    if should_download_clang:
        cuda_defines["%{linker_bin_path}"] = ""
    else:
        cuda_defines["%{linker_bin_path}"] = host_compiler_prefix

    cuda_defines["%{extra_no_canonical_prefixes_flags}"] = ""
    cuda_defines["%{unfiltered_compile_flags}"] = ""
    if is_cuda_clang:
        cuda_defines["%{host_compiler_path}"] = str(cc)
        cuda_defines["%{host_compiler_warnings}"] = """
        # Some parts of the codebase set -Werror and hit this warning, so
        # switch it off for now.
        "-Wno-invalid-partial-specialization"
    """
        cuda_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(host_compiler_includes)
        cuda_defines["%{compiler_deps}"] = ":empty"
        cuda_defines["%{win_compiler_deps}"] = ":empty"
        repository_ctx.file(
            "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
            "",
        )
        repository_ctx.file("crosstool/windows/msvc_wrapper_for_nvcc.py", "")
    else:
        cuda_defines["%{host_compiler_path}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"
        cuda_defines["%{host_compiler_warnings}"] = ""

        # nvcc has the system include paths built in and will automatically
        # search them; we cannot work around that, so we add the relevant cuda
        # system paths to the allowed compiler specific include paths.
        cuda_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(
            host_compiler_includes + _cuda_include_path(
                repository_ctx,
                cuda_config,
            ) + [cupti_header_dir, cudnn_header_dir],
        )

        # For gcc, do not canonicalize system header paths; some versions of gcc
        # pick the shortest possible path for system includes when creating the
        # .d file - given that includes that are prefixed with "../" multiple
        # time quickly grow longer than the root of the tree, this can lead to
        # bazel's header check failing.
        cuda_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-fno-canonical-system-headers\""

        nvcc_path = str(
            repository_ctx.path("%s/nvcc%s" % (
                cuda_config.config["cuda_binary_dir"],
                ".exe" if _is_windows(repository_ctx) else "",
            )),
        )
        cuda_defines["%{compiler_deps}"] = ":crosstool_wrapper_driver_is_not_gcc"
        cuda_defines["%{win_compiler_deps}"] = ":windows_msvc_wrapper_files"

        wrapper_defines = {
            "%{cpu_compiler}": str(cc),
            "%{cuda_version}": cuda_config.cuda_version,
            "%{nvcc_path}": nvcc_path,
            "%{gcc_host_compiler_path}": str(cc),
            "%{cuda_compute_capabilities}": ", ".join(
                ["\"%s\"" % c for c in cuda_config.compute_capabilities],
            ),
            "%{nvcc_tmp_dir}": _get_nvcc_tmp_dir_for_windows(repository_ctx),
        }
        _tpl(
            repository_ctx,
            "crosstool:clang/bin/crosstool_wrapper_driver_is_not_gcc",
            wrapper_defines,
        )
        _tpl(
            repository_ctx,
            "crosstool:windows/msvc_wrapper_for_nvcc.py",
            wrapper_defines,
        )

    cuda_defines.update(_get_win_cuda_defines(repository_ctx))

    verify_build_defines(cuda_defines)

    # Only expand template variables in the BUILD file
    _tpl(repository_ctx, "crosstool:BUILD", cuda_defines)

    # No templating of cc_toolchain_config - use attributes and templatize the
    # BUILD file.
    _file(repository_ctx, "crosstool:cc_toolchain_config.bzl")

    # Set up cuda_config.h, which is used by
    # tensorflow/stream_executor/dso_loader.cc.
    _tpl(
        repository_ctx,
        "cuda:cuda_config.h",
        {
            "%{cuda_version}": cuda_config.cuda_version,
            "%{cuda_lib_version}": cuda_config.cuda_lib_version,
            "%{cudnn_version}": cuda_config.cudnn_version,
            "%{cuda_compute_capabilities}": ", ".join([
                "CudaVersion(\"%s\")" % c
                for c in cuda_config.compute_capabilities
            ]),
            "%{cuda_toolkit_path}": cuda_config.cuda_toolkit_path,
        },
        "cuda/cuda/cuda_config.h",
    )

def _create_remote_cuda_repository(repository_ctx, remote_config_repo):
    """Creates pointers to a remotely configured repo set up to build with CUDA."""
    _tpl(
        repository_ctx,
        "cuda:build_defs.bzl",
        {
            "%{cuda_is_configured}": "True",
            "%{cuda_extra_copts}": _compute_cuda_extra_copts(
                repository_ctx,
                compute_capabilities(repository_ctx),
            ),
        },
    )
    repository_ctx.template(
        "cuda/BUILD",
        Label(remote_config_repo + "/cuda:BUILD"),
        {},
    )
    repository_ctx.template(
        "cuda/build_defs.bzl",
        Label(remote_config_repo + "/cuda:build_defs.bzl"),
        {},
    )
    repository_ctx.template(
        "cuda/cuda/cuda_config.h",
        Label(remote_config_repo + "/cuda:cuda/cuda_config.h"),
        {},
    )

def _cuda_autoconf_impl(repository_ctx):
    """Implementation of the cuda_autoconf repository rule."""
    if not enable_cuda(repository_ctx):
        _create_dummy_repository(repository_ctx)
    elif _TF_CUDA_CONFIG_REPO in repository_ctx.os.environ:
        if (_TF_CUDA_VERSION not in repository_ctx.os.environ or
            _TF_CUDNN_VERSION not in repository_ctx.os.environ):
            auto_configure_fail("%s and %s must also be set if %s is specified" %
                                (_TF_CUDA_VERSION, _TF_CUDNN_VERSION, _TF_CUDA_CONFIG_REPO))
        _create_remote_cuda_repository(
            repository_ctx,
            repository_ctx.os.environ[_TF_CUDA_CONFIG_REPO],
        )
    else:
        _create_local_cuda_repository(repository_ctx)

cuda_configure = repository_rule(
    implementation = _cuda_autoconf_impl,
    environ = [
        _GCC_HOST_COMPILER_PATH,
        _GCC_HOST_COMPILER_PREFIX,
        _CLANG_CUDA_COMPILER_PATH,
        "TF_NEED_CUDA",
        "TF_CUDA_CLANG",
        _TF_DOWNLOAD_CLANG,
        _CUDA_TOOLKIT_PATH,
        _CUDNN_INSTALL_PATH,
        _TF_CUDA_VERSION,
        _TF_CUDNN_VERSION,
        _TF_CUDA_COMPUTE_CAPABILITIES,
        _TF_CUDA_CONFIG_REPO,
        "NVVMIR_LIBRARY_DIR",
        _PYTHON_BIN_PATH,
        "TMP",
        "TMPDIR",
        "TF_CUDA_PATHS",
    ],
)

"""Detects and configures the local CUDA toolchain.

Add the following to your WORKSPACE FILE:

```python
cuda_configure(name = "local_config_cuda")
```

Args:
  name: A unique name for this workspace rule.
"""
