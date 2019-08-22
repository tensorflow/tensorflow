# -*- Python -*-
"""Repository rule for ROCm autoconfiguration.

`rocm_configure` depends on the following environment variables:

  * `TF_NEED_ROCM`: Whether to enable building with ROCm.
  * `GCC_HOST_COMPILER_PATH`: The GCC host compiler path
  * `ROCM_TOOLKIT_PATH`: The path to the ROCm toolkit. Default is
    `/opt/rocm`.
  * `TF_ROCM_VERSION`: The version of the ROCm toolkit. If this is blank, then
    use the system default.
  * `TF_MIOPEN_VERSION`: The version of the MIOpen library.
  * `TF_ROCM_AMDGPU_TARGETS`: The AMDGPU targets. Default is
    `gfx803,gfx900`.
"""

load(
    ":cuda_configure.bzl",
    "make_copy_dir_rule",
    "make_copy_files_rule",
    "to_list_of_strings",
    "verify_build_defines",
)

_GCC_HOST_COMPILER_PATH = "GCC_HOST_COMPILER_PATH"
_GCC_HOST_COMPILER_PREFIX = "GCC_HOST_COMPILER_PREFIX"
_ROCM_TOOLKIT_PATH = "ROCM_TOOLKIT_PATH"
_TF_ROCM_VERSION = "TF_ROCM_VERSION"
_TF_MIOPEN_VERSION = "TF_MIOPEN_VERSION"
_TF_ROCM_AMDGPU_TARGETS = "TF_ROCM_AMDGPU_TARGETS"
_TF_ROCM_CONFIG_REPO = "TF_ROCM_CONFIG_REPO"

_DEFAULT_ROCM_VERSION = ""
_DEFAULT_MIOPEN_VERSION = ""
_DEFAULT_ROCM_TOOLKIT_PATH = "/opt/rocm"
_DEFAULT_ROCM_AMDGPU_TARGETS = ["gfx803", "gfx900"]

def _get_win_rocm_defines(repository_ctx):
    """Return CROSSTOOL defines for Windows"""

    # Return fake vaules for Windows specific fields.
    # This ensures the CROSSTOOL file parser is happy.
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

def find_cc(repository_ctx):
    """Find the C++ compiler."""

    # Return a dummy value for GCC detection here to avoid error
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

def _cxx_inc_convert(path):
    """Convert path returned by cc -E xc++ in a complete path."""
    path = path.strip()
    return path

def _get_cxx_inc_directories_impl(repository_ctx, cc, lang_is_cpp):
    """Compute the list of default C or C++ include directories."""
    if lang_is_cpp:
        lang = "c++"
    else:
        lang = "c"

    # TODO: We pass -no-canonical-prefixes here to match the compiler flags,
    #       but in rocm_clang CROSSTOOL file that is a `feature` and we should
    #       handle the case when it's disabled and no flag is passed
    result = repository_ctx.execute([
        cc,
        "-no-canonical-prefixes",
        "-E",
        "-x" + lang,
        "-",
        "-v",
    ])
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
        str(repository_ctx.path(_cxx_inc_convert(p)))
        for p in inc_dirs.split("\n")
    ]

def get_cxx_inc_directories(repository_ctx, cc):
    """Compute the list of default C and C++ include directories."""

    # For some reason `clang -xc` sometimes returns include paths that are
    # different from the ones from `clang -xc++`. (Symlink and a dir)
    # So we run the compiler with both `-xc` and `-xc++` and merge resulting lists
    includes_cpp = _get_cxx_inc_directories_impl(repository_ctx, cc, True)
    includes_c = _get_cxx_inc_directories_impl(repository_ctx, cc, False)

    includes_cpp_set = depset(includes_cpp)
    return includes_cpp + [
        inc
        for inc in includes_c
        if inc not in includes_cpp_set.to_list()
    ]

def auto_configure_fail(msg):
    """Output failure message when rocm configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sROCm Configuration Error:%s %s\n" % (red, no_color, msg))

# END cc_configure common functions (see TODO above).

def _host_compiler_includes(repository_ctx, cc):
    """Computed the list of gcc include directories.

    Args:
      repository_ctx: The repository context.
      cc: The path to the gcc host compiler.

    Returns:
      A list of gcc include directories.
    """
    inc_dirs = get_cxx_inc_directories(repository_ctx, cc)

    # Add numpy headers
    inc_dirs.append("/usr/lib/python2.7/dist-packages/numpy/core/include")

    return inc_dirs

def _rocm_include_path(repository_ctx, rocm_config):
    """Generates the cxx_builtin_include_directory entries for rocm inc dirs.

    Args:
      repository_ctx: The repository context.
      rocm_config: The path to the gcc host compiler.

    Returns:
      A string containing the Starlark string for each of the gcc
      host compiler include directories, which can be added to the CROSSTOOL
      file.
    """
    inc_dirs = []

    # general ROCm include path
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/include")

    # Add HSA headers
    inc_dirs.append("/opt/rocm/hsa/include")

    # Add HIP headers
    inc_dirs.append("/opt/rocm/include/hip")
    inc_dirs.append("/opt/rocm/include/hip/hcc_detail")
    inc_dirs.append("/opt/rocm/hip/include")

    # Add HIP-Clang headers
    inc_dirs.append("/opt/rocm/llvm/lib/clang/8.0/include")
    inc_dirs.append("/opt/rocm/llvm/lib/clang/9.0.0/include")
    inc_dirs.append("/opt/rocm/llvm/lib/clang/10.0.0/include")

    # Add rocrand and hiprand headers
    inc_dirs.append("/opt/rocm/rocrand/include")
    inc_dirs.append("/opt/rocm/hiprand/include")

    # Add rocfft headers
    inc_dirs.append("/opt/rocm/rocfft/include")

    # Add rocBLAS headers
    inc_dirs.append("/opt/rocm/rocblas/include")

    # Add MIOpen headers
    inc_dirs.append("/opt/rocm/miopen/include")

    # Add hcc headers
    inc_dirs.append("/opt/rocm/hcc/include")
    inc_dirs.append("/opt/rocm/hcc/compiler/lib/clang/7.0.0/include/")
    inc_dirs.append("/opt/rocm/hcc/lib/clang/7.0.0/include")

    # Newer hcc builds use/are based off of clang 8.0.0.
    inc_dirs.append("/opt/rocm/hcc/compiler/lib/clang/8.0.0/include/")
    inc_dirs.append("/opt/rocm/hcc/lib/clang/8.0.0/include")

    # Support hcc based off clang 9.0.0, included in ROCm2.2
    inc_dirs.append("/opt/rocm/hcc/compiler/lib/clang/9.0.0/include/")
    inc_dirs.append("/opt/rocm/hcc/lib/clang/9.0.0/include")

    # Support hcc based off clang 10.0.0, included in ROCm2.8
    inc_dirs.append("/opt/rocm/hcc/compiler/lib/clang/10.0.0/include/")
    inc_dirs.append("/opt/rocm/hcc/lib/clang/10.0.0/include")

    return inc_dirs

def _enable_rocm(repository_ctx):
    if "TF_NEED_ROCM" in repository_ctx.os.environ:
        enable_rocm = repository_ctx.os.environ["TF_NEED_ROCM"].strip()
        return enable_rocm == "1"
    return False

def _rocm_toolkit_path(repository_ctx):
    """Finds the rocm toolkit directory.

    Args:
      repository_ctx: The repository context.

    Returns:
      A speculative real path of the rocm toolkit install directory.
    """
    rocm_toolkit_path = _DEFAULT_ROCM_TOOLKIT_PATH
    if _ROCM_TOOLKIT_PATH in repository_ctx.os.environ:
        rocm_toolkit_path = repository_ctx.os.environ[_ROCM_TOOLKIT_PATH].strip()
    if not repository_ctx.path(rocm_toolkit_path).exists:
        auto_configure_fail("Cannot find rocm toolkit path.")
    return str(repository_ctx.path(rocm_toolkit_path).realpath)

def _amdgpu_targets(repository_ctx):
    """Returns a list of strings representing AMDGPU targets."""
    if _TF_ROCM_AMDGPU_TARGETS not in repository_ctx.os.environ:
        return _DEFAULT_ROCM_AMDGPU_TARGETS
    amdgpu_targets_str = repository_ctx.os.environ[_TF_ROCM_AMDGPU_TARGETS]
    amdgpu_targets = amdgpu_targets_str.split(",")
    for amdgpu_target in amdgpu_targets:
        if amdgpu_target[:3] != "gfx" or not amdgpu_target[3:].isdigit():
            auto_configure_fail("Invalid AMDGPU target: %s" % amdgpu_target)
    return amdgpu_targets

def _hipcc_env(repository_ctx):
    """Returns the environment variable string for hipcc.

    Args:
        repository_ctx: The repository context.

    Returns:
        A string containing environment variables for hipcc.
    """
    hipcc_env = ""
    for name in [
        "HIP_CLANG_PATH",
        "DEVICE_LIB_PATH",
        "HIP_VDI_HOME",
        "HIPCC_VERBOSE",
        "HIPCC_COMPILE_FLAGS_APPEND",
        "HIPPCC_LINK_FLAGS_APPEND",
        "HCC_AMDGPU_TARGET",
        "HIP_PLATFORM",
    ]:
        if name in repository_ctx.os.environ:
            hipcc_env = (hipcc_env + " " + name + "=\"" +
                         repository_ctx.os.environ[name].strip() + "\";")
    return hipcc_env.strip()

def _hipcc_is_hipclang(repository_ctx):
    """Returns if hipcc is based on hip-clang toolchain.

    Args:
        repository_ctx: The repository context.

    Returns:
        A string "True" if hipcc is based on hip-clang toolchain.
        The functions returns "False" if not (ie: based on HIP/HCC toolchain).
    """

    #  check user-defined hip-clang environment variables
    for name in ["HIP_CLANG_PATH", "HIP_VDI_HOME"]:
        if name in repository_ctx.os.environ:
            return "True"

    # grep for "HIP_COMPILER=clang" in /opt/rocm/hip/lib/.hipInfo
    grep_result = _execute(
        repository_ctx,
        ["grep", "HIP_COMPILER=clang", "/opt/rocm/hip/lib/.hipInfo"],
        empty_stdout_fine = True,
    )
    result = grep_result.stdout
    if result == "HIP_COMPILER=clang":
        return "True"
    return "False"

def _if_hipcc_is_hipclang(repository_ctx, if_true, if_false = []):
    """
    Returns either the if_true or if_false arg based on whether hipcc
    is based on the hip-clang toolchain

    Args :
        repository_ctx: The repository context.
        if_true : value to return if hipcc is hip-clang based
        if_false : value to return if hipcc is not hip-clang based
                   (optional, defaults to empty list)

    Returns :
        either the if_true arg or the of_False arg
    """
    if _hipcc_is_hipclang(repository_ctx) == "True":
        return if_true
    return if_false

def _crosstool_verbose(repository_ctx):
    """Returns the environment variable value CROSSTOOL_VERBOSE.

    Args:
        repository_ctx: The repository context.

    Returns:
        A string containing value of environment variable CROSSTOOL_VERBOSE.
    """
    name = "CROSSTOOL_VERBOSE"
    if name in repository_ctx.os.environ:
        return repository_ctx.os.environ[name].strip()
    return "0"

def _cpu_value(repository_ctx):
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

def _lib_name(lib, cpu_value, version = "", static = False):
    """Constructs the platform-specific name of a library.

    Args:
      lib: The name of the library, such as "hip"
      cpu_value: The name of the host operating system.
      version: The version of the library.
      static: True the library is static or False if it is a shared object.

    Returns:
      The platform-specific name of the library.
    """
    if cpu_value in ("Linux", "FreeBSD"):
        if static:
            return "lib%s.a" % lib
        else:
            if version:
                version = ".%s" % version
            return "lib%s.so%s" % (lib, version)
    elif cpu_value == "Windows":
        return "%s.lib" % lib
    elif cpu_value == "Darwin":
        if static:
            return "lib%s.a" % lib
        elif version:
            version = ".%s" % version
        return "lib%s%s.dylib" % (lib, version)
    else:
        auto_configure_fail("Invalid cpu_value: %s" % cpu_value)

def _find_rocm_lib(
        lib,
        repository_ctx,
        cpu_value,
        basedir,
        version = "",
        static = False):
    """Finds the given ROCm libraries on the system.

    Args:
      lib: The name of the library, such as "hip"
      repository_ctx: The repository context.
      cpu_value: The name of the host operating system.
      basedir: The install directory of ROCm.
      version: The version of the library.
      static: True if static library, False if shared object.

    Returns:
      Returns a struct with the following fields:
        file_name: The basename of the library found on the system.
        path: The full path to the library.
    """
    file_name = _lib_name(lib, cpu_value, version, static)
    if cpu_value == "Linux":
        path = repository_ctx.path("%s/lib64/%s" % (basedir, file_name))
        if path.exists:
            return struct(file_name = file_name, path = str(path.realpath))
        path = repository_ctx.path("%s/lib64/stubs/%s" % (basedir, file_name))
        if path.exists:
            return struct(file_name = file_name, path = str(path.realpath))
        path = repository_ctx.path(
            "%s/lib/x86_64-linux-gnu/%s" % (basedir, file_name),
        )
        if path.exists:
            return struct(file_name = file_name, path = str(path.realpath))

    path = repository_ctx.path("%s/lib/%s" % (basedir, file_name))
    if path.exists:
        return struct(file_name = file_name, path = str(path.realpath))
    path = repository_ctx.path("%s/%s" % (basedir, file_name))
    if path.exists:
        return struct(file_name = file_name, path = str(path.realpath))

    auto_configure_fail("Cannot find rocm library %s" % file_name)

def _find_libs(repository_ctx, rocm_config):
    """Returns the ROCm libraries on the system.

    Args:
      repository_ctx: The repository context.
      rocm_config: The ROCm config as returned by _get_rocm_config

    Returns:
      Map of library names to structs of filename and path as returned by
      _find_rocm_lib.
    """
    cpu_value = rocm_config.cpu_value
    return {
        "hip": _find_rocm_lib(
            "hip_hcc",
            repository_ctx,
            cpu_value,
            rocm_config.rocm_toolkit_path,
        ),
        "rocblas": _find_rocm_lib(
            "rocblas",
            repository_ctx,
            cpu_value,
            rocm_config.rocm_toolkit_path + "/rocblas",
        ),
        "rocfft": _find_rocm_lib(
            "rocfft",
            repository_ctx,
            cpu_value,
            rocm_config.rocm_toolkit_path + "/rocfft",
        ),
        "hiprand": _find_rocm_lib(
            "hiprand",
            repository_ctx,
            cpu_value,
            rocm_config.rocm_toolkit_path + "/hiprand",
        ),
        "miopen": _find_rocm_lib(
            "MIOpen",
            repository_ctx,
            cpu_value,
            rocm_config.rocm_toolkit_path + "/miopen",
        ),
    }

def _get_rocm_config(repository_ctx):
    """Detects and returns information about the ROCm installation on the system.

    Args:
      repository_ctx: The repository context.

    Returns:
      A struct containing the following fields:
        rocm_toolkit_path: The ROCm toolkit installation directory.
        amdgpu_targets: A list of the system's AMDGPU targets.
        cpu_value: The name of the host operating system.
    """
    cpu_value = _cpu_value(repository_ctx)
    rocm_toolkit_path = _rocm_toolkit_path(repository_ctx)
    return struct(
        rocm_toolkit_path = rocm_toolkit_path,
        amdgpu_targets = _amdgpu_targets(repository_ctx),
        cpu_value = cpu_value,
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
  fail("ERROR: Building with --config=rocm but TensorFlow is not configured " +
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
    cpu_value = _cpu_value(repository_ctx)

    # Set up BUILD file for rocm/.
    _tpl(
        repository_ctx,
        "rocm:build_defs.bzl",
        {
            "%{rocm_is_configured}": "False",
            "%{rocm_extra_copts}": "[]",
        },
    )
    _tpl(
        repository_ctx,
        "rocm:BUILD",
        {
            "%{hip_lib}": _lib_name("hip", cpu_value),
            "%{rocblas_lib}": _lib_name("rocblas", cpu_value),
            "%{miopen_lib}": _lib_name("miopen", cpu_value),
            "%{rocfft_lib}": _lib_name("rocfft", cpu_value),
            "%{hiprand_lib}": _lib_name("hiprand", cpu_value),
            "%{copy_rules}": "",
            "%{rocm_headers}": "",
        },
    )

    # Create dummy files for the ROCm toolkit since they are still required by
    # tensorflow/core/platform/default/build_config:rocm.
    repository_ctx.file("rocm/hip/include/hip/hip_runtime.h", "")

    # Set up rocm_config.h, which is used by
    # tensorflow/stream_executor/dso_loader.cc.
    _tpl(
        repository_ctx,
        "rocm:rocm_config.h",
        {
            "%{rocm_toolkit_path}": _DEFAULT_ROCM_TOOLKIT_PATH,
        },
        "rocm/rocm/rocm_config.h",
    )

    # If rocm_configure is not configured to build with GPU support, and the user
    # attempts to build with --config=rocm, add a dummy build rule to intercept
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
      empty_stdout_fine: bool, if True, an empty stdout result is fine, otherwise
        it's an error
    Return:
      the result of repository_ctx.execute(cmdline)
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

def _genrule(src_dir, genrule_name, command, outs):
    """Returns a string with a genrule.

    Genrule executes the given command and produces the given outputs.
    """
    return (
        "genrule(\n" +
        '    name = "' +
        genrule_name + '",\n' +
        "    outs = [\n" +
        outs +
        "\n    ],\n" +
        '    cmd = """\n' +
        command +
        '\n   """,\n' +
        ")\n"
    )

def _read_dir(repository_ctx, src_dir):
    """Returns a string with all files in a directory.

    Finds all files inside a directory, traversing subfolders and following
    symlinks. The returned string contains the full path of all files
    separated by line breaks.
    """
    find_result = _execute(
        repository_ctx,
        ["find", src_dir, "-follow", "-type", "f"],
        empty_stdout_fine = True,
    )
    result = find_result.stdout
    return result

def _compute_rocm_extra_copts(repository_ctx, amdgpu_targets):
    if False:
        amdgpu_target_flags = ["--amdgpu-target=" +
                               amdgpu_target for amdgpu_target in amdgpu_targets]
    else:
        # AMDGPU targets are handled in the "crosstool_wrapper_driver_is_not_gcc"
        amdgpu_target_flags = []
    return str(amdgpu_target_flags)

def _create_local_rocm_repository(repository_ctx):
    """Creates the repository containing files set up to build with ROCm."""
    rocm_config = _get_rocm_config(repository_ctx)

    # Copy header and library files to execroot.
    # rocm_toolkit_path
    rocm_toolkit_path = rocm_config.rocm_toolkit_path
    copy_rules = [
        make_copy_dir_rule(
            repository_ctx,
            name = "rocm-include",
            src_dir = rocm_toolkit_path + "/include",
            out_dir = "rocm/include",
        ),
        make_copy_dir_rule(
            repository_ctx,
            name = "rocfft-include",
            src_dir = rocm_toolkit_path + "/rocfft/include",
            out_dir = "rocm/include/rocfft",
        ),
        make_copy_dir_rule(
            repository_ctx,
            name = "rocblas-include",
            src_dir = rocm_toolkit_path + "/rocblas/include",
            out_dir = "rocm/include/rocblas",
        ),
        make_copy_dir_rule(
            repository_ctx,
            name = "miopen-include",
            src_dir = rocm_toolkit_path + "/miopen/include",
            out_dir = "rocm/include/miopen",
        ),
    ]

    rocm_libs = _find_libs(repository_ctx, rocm_config)
    rocm_lib_srcs = []
    rocm_lib_outs = []
    for lib in rocm_libs.values():
        rocm_lib_srcs.append(lib.path)
        rocm_lib_outs.append("rocm/lib/" + lib.file_name)
    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "rocm-lib",
        srcs = rocm_lib_srcs,
        outs = rocm_lib_outs,
    ))

    # Set up BUILD file for rocm/
    _tpl(
        repository_ctx,
        "rocm:build_defs.bzl",
        {
            "%{rocm_is_configured}": "True",
            "%{rocm_extra_copts}": _compute_rocm_extra_copts(
                repository_ctx,
                rocm_config.amdgpu_targets,
            ),
        },
    )
    _tpl(
        repository_ctx,
        "rocm:BUILD",
        {
            "%{hip_lib}": rocm_libs["hip"].file_name,
            "%{rocblas_lib}": rocm_libs["rocblas"].file_name,
            "%{rocfft_lib}": rocm_libs["rocfft"].file_name,
            "%{hiprand_lib}": rocm_libs["hiprand"].file_name,
            "%{miopen_lib}": rocm_libs["miopen"].file_name,
            "%{copy_rules}": "\n".join(copy_rules),
            "%{rocm_headers}": ('":rocm-include",\n' +
                                '":rocfft-include",\n' +
                                '":rocblas-include",\n' +
                                '":miopen-include",'),
        },
    )

    # Set up crosstool/
    cc = find_cc(repository_ctx)

    host_compiler_includes = get_cxx_inc_directories(repository_ctx, cc)

    host_compiler_prefix = "/usr/bin"
    if _GCC_HOST_COMPILER_PREFIX in repository_ctx.os.environ:
        host_compiler_prefix = repository_ctx.os.environ[_GCC_HOST_COMPILER_PREFIX].strip()

    rocm_defines = {}

    rocm_defines["%{host_compiler_prefix}"] = host_compiler_prefix

    rocm_defines["%{linker_bin_path}"] = "/opt/rocm/hcc/compiler/bin"

    # For gcc, do not canonicalize system header paths; some versions of gcc
    # pick the shortest possible path for system includes when creating the
    # .d file - given that includes that are prefixed with "../" multiple
    # time quickly grow longer than the root of the tree, this can lead to
    # bazel's header check failing.
    rocm_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-fno-canonical-system-headers\""

    rocm_defines["%{unfiltered_compile_flags}"] = to_list_of_strings([
        "-DTENSORFLOW_USE_ROCM=1",
        "-D__HIP_PLATFORM_HCC__",
        "-DEIGEN_USE_HIP",
    ] + _if_hipcc_is_hipclang(repository_ctx, [
        #
        # define "TENSORFLOW_COMPILER_IS_HIP_CLANG" when we are using clang
        # based hipcc to compile/build tensorflow
        #
        # Note that this #define should not be used to check whether or not
        # tensorflow is being built with ROCm support
        # (only TENSORFLOW_USE_ROCM should be used for that purpose)
        #
        "-DTENSORFLOW_COMPILER_IS_HIP_CLANG=1",
    ]))

    rocm_defines["%{host_compiler_path}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"

    # # Enable a few more warnings that aren't part of -Wall.
    # compiler_flag: "-Wunused-but-set-parameter"

    # # But disable some that are problematic.
    # compiler_flag: "-Wno-free-nonheap-object" # has false positives

    rocm_defines["%{host_compiler_warnings}"] = to_list_of_strings(["-Wunused-but-set-parameter", "-Wno-free-nonheap-object"])

    rocm_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(host_compiler_includes +
                                                                            _rocm_include_path(repository_ctx, rocm_config))

    rocm_defines["%{linker_files}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"

    rocm_defines["%{win_linker_files}"] = ":empty"

    # Add the dummy defines for windows...requried to pass the "verify_build_defines" check
    rocm_defines.update(_get_win_rocm_defines(repository_ctx))

    verify_build_defines(rocm_defines)

    # Only expand template variables in the BUILD file
    _tpl(repository_ctx, "crosstool:BUILD", rocm_defines)

    # No templating of cc_toolchain_config - use attributes and templatize the
    # BUILD file.
    _tpl(
        repository_ctx,
        "crosstool:hipcc_cc_toolchain_config.bzl",
        out = "crosstool/cc_toolchain_config.bzl",
    )

    _tpl(
        repository_ctx,
        "crosstool:clang/bin/crosstool_wrapper_driver_rocm",
        {
            "%{cpu_compiler}": str(cc),
            "%{hipcc_path}": "/opt/rocm/bin/hipcc",
            "%{hipcc_env}": _hipcc_env(repository_ctx),
            "%{hipcc_is_hipclang}": _hipcc_is_hipclang(repository_ctx),
            "%{rocr_runtime_path}": "/opt/rocm/lib",
            "%{rocr_runtime_library}": "hsa-runtime64",
            "%{hip_runtime_path}": "/opt/rocm/hip/lib",
            "%{hip_runtime_library}": "hip_hcc",
            "%{hcc_runtime_path}": "/opt/rocm/hcc/lib",
            "%{hcc_runtime_library}": "mcwamp",
            "%{crosstool_verbose}": _crosstool_verbose(repository_ctx),
            "%{gcc_host_compiler_path}": str(cc),
            "%{rocm_amdgpu_targets}": ",".join(
                ["\"%s\"" % c for c in rocm_config.amdgpu_targets],
            ),
        },
        out = "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
    )

    # Set up rocm_config.h, which is used by
    # tensorflow/stream_executor/dso_loader.cc.
    _tpl(
        repository_ctx,
        "rocm:rocm_config.h",
        {
            "%{rocm_amdgpu_targets}": ",".join(
                ["\"%s\"" % c for c in rocm_config.amdgpu_targets],
            ),
            "%{rocm_toolkit_path}": rocm_config.rocm_toolkit_path,
        },
        "rocm/rocm/rocm_config.h",
    )

def _create_remote_rocm_repository(repository_ctx, remote_config_repo):
    """Creates pointers to a remotely configured repo set up to build with ROCm."""
    _tpl(
        repository_ctx,
        "rocm:build_defs.bzl",
        {
            "%{rocm_is_configured}": "True",
            "%{rocm_extra_copts}": _compute_rocm_extra_copts(
                repository_ctx,
                [],  #_compute_capabilities(repository_ctx)
            ),
        },
    )
    repository_ctx.template(
        "rocm/BUILD",
        Label(remote_config_repo + "/rocm:BUILD"),
        {},
    )
    repository_ctx.template(
        "rocm/build_defs.bzl",
        Label(remote_config_repo + "/rocm:build_defs.bzl"),
        {},
    )
    repository_ctx.template(
        "rocm/rocm/rocm_config.h",
        Label(remote_config_repo + "/rocm:rocm/rocm_config.h"),
        {},
    )

def _rocm_autoconf_impl(repository_ctx):
    """Implementation of the rocm_autoconf repository rule."""
    if not _enable_rocm(repository_ctx):
        _create_dummy_repository(repository_ctx)
    elif _TF_ROCM_CONFIG_REPO in repository_ctx.os.environ:
        _create_remote_rocm_repository(
            repository_ctx,
            repository_ctx.os.environ[_TF_ROCM_CONFIG_REPO],
        )
    else:
        _create_local_rocm_repository(repository_ctx)

rocm_configure = repository_rule(
    implementation = _rocm_autoconf_impl,
    environ = [
        _GCC_HOST_COMPILER_PATH,
        _GCC_HOST_COMPILER_PREFIX,
        "TF_NEED_ROCM",
        _ROCM_TOOLKIT_PATH,
        _TF_ROCM_VERSION,
        _TF_MIOPEN_VERSION,
        _TF_ROCM_AMDGPU_TARGETS,
        _TF_ROCM_CONFIG_REPO,
    ],
)

"""Detects and configures the local ROCm toolchain.

Add the following to your WORKSPACE FILE:

```python
rocm_configure(name = "local_config_rocm")
```

Args:
  name: A unique name for this workspace rule.
"""
