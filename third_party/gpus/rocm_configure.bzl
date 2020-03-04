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
)
load(
    "//third_party/remote_config:common.bzl",
    "config_repo_label",
    "err_out",
    "execute",
    "files_exist",
    "get_bash_bin",
    "get_cpu_value",
    "get_host_environ",
    "raw_exec",
    "realpath",
    "which",
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

def verify_build_defines(params):
    """Verify all variables that crosstool/BUILD.rocm.tpl expects are substituted.

    Args:
      params: dict of variables that will be passed to the BUILD.tpl template.
    """
    missing = []
    for param in [
        "cxx_builtin_include_directories",
        "extra_no_canonical_prefixes_flags",
        "host_compiler_path",
        "host_compiler_prefix",
        "linker_bin_path",
        "unfiltered_compile_flags",
    ]:
        if ("%{" + param + "}") not in params:
            missing.append(param)

    if missing:
        auto_configure_fail(
            "BUILD.rocm.tpl template is missing these variables: " +
            str(missing) +
            ".\nWe only got: " +
            str(params) +
            ".",
        )

def find_cc(repository_ctx):
    """Find the C++ compiler."""

    # Return a dummy value for GCC detection here to avoid error
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
    result = raw_exec(repository_ctx, [
        cc,
        "-no-canonical-prefixes",
        "-E",
        "-x" + lang,
        "-",
        "-v",
    ])
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

def auto_configure_warning(msg):
    """Output warning message during auto configuration."""
    yellow = "\033[1;33m"
    no_color = "\033[0m"
    print("\n%sAuto-Configuration Warning:%s %s\n" % (yellow, no_color, msg))

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
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hsa/include")

    # Add HIP headers
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/include/hip")
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/include/hip/hcc_detail")
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hip/include")

    # Add HIP-Clang headers
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/llvm/lib/clang/8.0/include")
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/llvm/lib/clang/9.0.0/include")
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/llvm/lib/clang/10.0.0/include")
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/llvm/lib/clang/11.0.0/include")

    # Add rocrand and hiprand headers
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/rocrand/include")
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hiprand/include")

    # Add rocfft headers
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/rocfft/include")

    # Add rocBLAS headers
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/rocblas/include")

    # Add MIOpen headers
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/miopen/include")

    # Add RCCL headers
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/rccl/include")

    # Add hcc headers
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hcc/include")
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hcc/compiler/lib/clang/7.0.0/include/")
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hcc/lib/clang/7.0.0/include")

    # Newer hcc builds use/are based off of clang 8.0.0.
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hcc/compiler/lib/clang/8.0.0/include/")
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hcc/lib/clang/8.0.0/include")

    # Support hcc based off clang 9.0.0, included in ROCm2.2
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hcc/compiler/lib/clang/9.0.0/include/")
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hcc/lib/clang/9.0.0/include")

    # Support hcc based off clang 10.0.0, included in ROCm2.8
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hcc/compiler/lib/clang/10.0.0/include/")
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hcc/lib/clang/10.0.0/include")

    # Support hcc based off clang 11.0.0, included in ROCm3.1
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hcc/compiler/lib/clang/11.0.0/include/")
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hcc/lib/clang/11.0.0/include")

    return inc_dirs

def _enable_rocm(repository_ctx):
    enable_rocm = get_host_environ(repository_ctx, "TF_NEED_ROCM")
    if enable_rocm == "1":
        if get_cpu_value(repository_ctx) != "Linux":
            auto_configure_warning("ROCm configure is only supported on Linux")
            return False
        return True
    return False

def _rocm_toolkit_path(repository_ctx, bash_bin):
    """Finds the rocm toolkit directory.

    Args:
      repository_ctx: The repository context.

    Returns:
      A speculative real path of the rocm toolkit install directory.
    """
    rocm_toolkit_path = get_host_environ(repository_ctx, _ROCM_TOOLKIT_PATH, _DEFAULT_ROCM_TOOLKIT_PATH)
    if files_exist(repository_ctx, [rocm_toolkit_path], bash_bin) != [True]:
        auto_configure_fail("Cannot find rocm toolkit path.")
    return realpath(repository_ctx, rocm_toolkit_path, bash_bin)

def _amdgpu_targets(repository_ctx):
    """Returns a list of strings representing AMDGPU targets."""
    amdgpu_targets_str = get_host_environ(repository_ctx, _TF_ROCM_AMDGPU_TARGETS)
    if not amdgpu_targets_str:
        return _DEFAULT_ROCM_AMDGPU_TARGETS
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
        env_value = get_host_environ(repository_ctx, name)
        if env_value:
            hipcc_env = (hipcc_env + " " + name + "=\"" + env_value + "\";")
    return hipcc_env.strip()

def _hipcc_is_hipclang(repository_ctx, rocm_config, bash_bin):
    """Returns if hipcc is based on hip-clang toolchain.

    Args:
        repository_ctx: The repository context.
        rocm_config: The path to the hip compiler.
        bash_bin: the path to the bash interpreter

    Returns:
        A string "True" if hipcc is based on hip-clang toolchain.
        The functions returns "False" if not (ie: based on HIP/HCC toolchain).
    """

    #  check user-defined hip-clang environment variables
    for name in ["HIP_CLANG_PATH", "HIP_VDI_HOME"]:
        if get_host_environ(repository_ctx, name):
            return "True"

    # grep for "HIP_COMPILER=clang" in /opt/rocm/hip/lib/.hipInfo
    cmd = "grep HIP_COMPILER=clang %s/hip/lib/.hipInfo || true" % rocm_config.rocm_toolkit_path
    grep_result = execute(repository_ctx, [bash_bin, "-c", cmd], empty_stdout_fine = True)
    result = grep_result.stdout.strip()
    if result == "HIP_COMPILER=clang":
        return "True"
    return "False"

def _if_hipcc_is_hipclang(repository_ctx, rocm_config, bash_bin, if_true, if_false = []):
    """
    Returns either the if_true or if_false arg based on whether hipcc
    is based on the hip-clang toolchain

    Args :
        repository_ctx: The repository context.
        rocm_config: The path to the hip compiler.
        if_true : value to return if hipcc is hip-clang based
        if_false : value to return if hipcc is not hip-clang based
                   (optional, defaults to empty list)

    Returns :
        either the if_true arg or the of_False arg
    """
    if _hipcc_is_hipclang(repository_ctx, rocm_config, bash_bin) == "True":
        return if_true
    return if_false

def _crosstool_verbose(repository_ctx):
    """Returns the environment variable value CROSSTOOL_VERBOSE.

    Args:
        repository_ctx: The repository context.

    Returns:
        A string containing value of environment variable CROSSTOOL_VERBOSE.
    """
    return get_host_environ(repository_ctx, "CROSSTOOL_VERBOSE", "0")

def _lib_name(lib, version = "", static = False):
    """Constructs the name of a library on Linux.

    Args:
      lib: The name of the library, such as "hip"
      version: The version of the library.
      static: True the library is static or False if it is a shared object.

    Returns:
      The platform-specific name of the library.
    """
    if static:
        return "lib%s.a" % lib
    else:
        if version:
            version = ".%s" % version
        return "lib%s.so%s" % (lib, version)

def _rocm_lib_paths(repository_ctx, lib, basedir):
    file_name = _lib_name(lib, version = "", static = False)
    return [
        repository_ctx.path("%s/lib64/%s" % (basedir, file_name)),
        repository_ctx.path("%s/lib64/stubs/%s" % (basedir, file_name)),
        repository_ctx.path("%s/lib/x86_64-linux-gnu/%s" % (basedir, file_name)),
        repository_ctx.path("%s/lib/%s" % (basedir, file_name)),
        repository_ctx.path("%s/%s" % (basedir, file_name)),
    ]

def _batch_files_exist(repository_ctx, libs_paths, bash_bin):
    all_paths = []
    for _, lib_paths in libs_paths:
        for lib_path in lib_paths:
            all_paths.append(lib_path)
    return files_exist(repository_ctx, all_paths, bash_bin)

def _select_rocm_lib_paths(repository_ctx, libs_paths, bash_bin):
    test_results = _batch_files_exist(repository_ctx, libs_paths, bash_bin)

    libs = {}
    i = 0
    for name, lib_paths in libs_paths:
        selected_path = None
        for path in lib_paths:
            if test_results[i] and selected_path == None:
                # For each lib select the first path that exists.
                selected_path = path
            i = i + 1
        if selected_path == None:
            auto_configure_fail("Cannot find rocm library %s" % name)

        libs[name] = struct(file_name = selected_path.basename, path = realpath(repository_ctx, selected_path, bash_bin))

    return libs

def _find_libs(repository_ctx, rocm_config, bash_bin):
    """Returns the ROCm libraries on the system.

    Args:
      repository_ctx: The repository context.
      rocm_config: The ROCm config as returned by _get_rocm_config
      bash_bin: the path to the bash interpreter

    Returns:
      Map of library names to structs of filename and path
    """

    libs_paths = [
        (name, _rocm_lib_paths(repository_ctx, name, path))
        for name, path in [
            ("hip_hcc", rocm_config.rocm_toolkit_path),
            ("rocblas", rocm_config.rocm_toolkit_path + "/rocblas"),
            ("rocfft", rocm_config.rocm_toolkit_path + "/rocfft"),
            ("hiprand", rocm_config.rocm_toolkit_path + "/hiprand"),
            ("MIOpen", rocm_config.rocm_toolkit_path + "/miopen"),
            ("rccl", rocm_config.rocm_toolkit_path + "/rccl"),
            ("hipsparse", rocm_config.rocm_toolkit_path + "/hipsparse"),
        ]
    ]

    return _select_rocm_lib_paths(repository_ctx, libs_paths, bash_bin)

def _get_rocm_config(repository_ctx, bash_bin):
    """Detects and returns information about the ROCm installation on the system.

    Args:
      repository_ctx: The repository context.
      bash_bin: the path to the path interpreter

    Returns:
      A struct containing the following fields:
        rocm_toolkit_path: The ROCm toolkit installation directory.
        amdgpu_targets: A list of the system's AMDGPU targets.
    """
    rocm_toolkit_path = _rocm_toolkit_path(repository_ctx, bash_bin)
    return struct(
        rocm_toolkit_path = rocm_toolkit_path,
        amdgpu_targets = _amdgpu_targets(repository_ctx),
    )

def _tpl_path(repository_ctx, labelname):
    return repository_ctx.path(Label("//third_party/gpus/%s.tpl" % labelname))

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl.replace(":", "/")
    repository_ctx.template(
        out,
        _tpl_path(repository_ctx, tpl),
        substitutions,
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
            "%{hip_lib}": _lib_name("hip"),
            "%{rocblas_lib}": _lib_name("rocblas"),
            "%{miopen_lib}": _lib_name("miopen"),
            "%{rccl_lib}": _lib_name("rccl"),
            "%{rocfft_lib}": _lib_name("rocfft"),
            "%{hiprand_lib}": _lib_name("hiprand"),
            "%{hipsparse_lib}": _lib_name("hipsparse"),
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

    tpl_paths = {labelname: _tpl_path(repository_ctx, labelname) for labelname in [
        "rocm:build_defs.bzl",
        "rocm:BUILD",
        "crosstool:BUILD.rocm",
        "crosstool:hipcc_cc_toolchain_config.bzl",
        "crosstool:clang/bin/crosstool_wrapper_driver_rocm",
        "rocm:rocm_config.h",
    ]}

    bash_bin = get_bash_bin(repository_ctx)
    rocm_config = _get_rocm_config(repository_ctx, bash_bin)

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
        make_copy_dir_rule(
            repository_ctx,
            name = "rccl-include",
            src_dir = rocm_toolkit_path + "/rccl/include",
            out_dir = "rocm/include/rccl",
        ),
        make_copy_dir_rule(
            repository_ctx,
            name = "hipsparse-include",
            src_dir = rocm_toolkit_path + "/hipsparse/include",
            out_dir = "rocm/include/hipsparse",
        ),
    ]

    rocm_libs = _find_libs(repository_ctx, rocm_config, bash_bin)
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
    repository_ctx.template(
        "rocm/build_defs.bzl",
        tpl_paths["rocm:build_defs.bzl"],
        {
            "%{rocm_is_configured}": "True",
            "%{rocm_extra_copts}": _compute_rocm_extra_copts(
                repository_ctx,
                rocm_config.amdgpu_targets,
            ),
        },
    )
    repository_ctx.template(
        "rocm/BUILD",
        tpl_paths["rocm:BUILD"],
        {
            "%{hip_lib}": rocm_libs["hip_hcc"].file_name,
            "%{rocblas_lib}": rocm_libs["rocblas"].file_name,
            "%{rocfft_lib}": rocm_libs["rocfft"].file_name,
            "%{hiprand_lib}": rocm_libs["hiprand"].file_name,
            "%{miopen_lib}": rocm_libs["MIOpen"].file_name,
            "%{rccl_lib}": rocm_libs["rccl"].file_name,
            "%{hipsparse_lib}": rocm_libs["hipsparse"].file_name,
            "%{copy_rules}": "\n".join(copy_rules),
            "%{rocm_headers}": ('":rocm-include",\n' +
                                '":rocfft-include",\n' +
                                '":rocblas-include",\n' +
                                '":miopen-include",\n' +
                                '":rccl-include",\n' +
                                '":hipsparse-include",'),
        },
    )

    # Set up crosstool/

    cc = find_cc(repository_ctx)

    host_compiler_includes = get_cxx_inc_directories(repository_ctx, cc)

    host_compiler_prefix = get_host_environ(repository_ctx, _GCC_HOST_COMPILER_PREFIX, "/usr/bin")

    rocm_defines = {}

    rocm_defines["%{host_compiler_prefix}"] = host_compiler_prefix

    rocm_defines["%{linker_bin_path}"] = rocm_config.rocm_toolkit_path + "/hcc/compiler/bin"

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
    ] + _if_hipcc_is_hipclang(repository_ctx, rocm_config, bash_bin, [
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

    rocm_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(host_compiler_includes +
                                                                            _rocm_include_path(repository_ctx, rocm_config))

    verify_build_defines(rocm_defines)

    # Only expand template variables in the BUILD file
    repository_ctx.template(
        "crosstool/BUILD",
        tpl_paths["crosstool:BUILD.rocm"],
        rocm_defines,
    )

    # No templating of cc_toolchain_config - use attributes and templatize the
    # BUILD file.
    repository_ctx.template(
        "crosstool/cc_toolchain_config.bzl",
        tpl_paths["crosstool:hipcc_cc_toolchain_config.bzl"],
    )

    repository_ctx.template(
        "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
        tpl_paths["crosstool:clang/bin/crosstool_wrapper_driver_rocm"],
        {
            "%{cpu_compiler}": str(cc),
            "%{hipcc_path}": rocm_config.rocm_toolkit_path + "/bin/hipcc",
            "%{hipcc_env}": _hipcc_env(repository_ctx),
            "%{hipcc_is_hipclang}": _hipcc_is_hipclang(repository_ctx, rocm_config, bash_bin),
            "%{rocr_runtime_path}": rocm_config.rocm_toolkit_path + "/lib",
            "%{rocr_runtime_library}": "hsa-runtime64",
            "%{hip_runtime_path}": rocm_config.rocm_toolkit_path + "/hip/lib",
            "%{hip_runtime_library}": "hip_hcc",
            "%{hcc_runtime_path}": rocm_config.rocm_toolkit_path + "/hcc/lib",
            "%{hcc_runtime_library}": "mcwamp",
            "%{crosstool_verbose}": _crosstool_verbose(repository_ctx),
            "%{gcc_host_compiler_path}": str(cc),
            "%{rocm_amdgpu_targets}": ",".join(
                ["\"%s\"" % c for c in rocm_config.amdgpu_targets],
            ),
        },
    )

    # Set up rocm_config.h, which is used by
    # tensorflow/stream_executor/dso_loader.cc.
    repository_ctx.template(
        "rocm/rocm/rocm_config.h",
        tpl_paths["rocm:rocm_config.h"],
        {
            "%{rocm_amdgpu_targets}": ",".join(
                ["\"%s\"" % c for c in rocm_config.amdgpu_targets],
            ),
            "%{rocm_toolkit_path}": rocm_config.rocm_toolkit_path,
        },
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
        config_repo_label(remote_config_repo, "rocm:BUILD"),
        {},
    )
    repository_ctx.template(
        "rocm/build_defs.bzl",
        config_repo_label(remote_config_repo, "rocm:build_defs.bzl"),
        {},
    )
    repository_ctx.template(
        "rocm/rocm/rocm_config.h",
        config_repo_label(remote_config_repo, "rocm:rocm/rocm_config.h"),
        {},
    )
    repository_ctx.template(
        "crosstool/BUILD",
        config_repo_label(remote_config_repo, "crosstool:BUILD"),
        {},
    )
    repository_ctx.template(
        "crosstool/cc_toolchain_config.bzl",
        config_repo_label(remote_config_repo, "crosstool:cc_toolchain_config.bzl"),
        {},
    )
    repository_ctx.template(
        "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
        config_repo_label(remote_config_repo, "crosstool:clang/bin/crosstool_wrapper_driver_is_not_gcc"),
        {},
    )

def _rocm_autoconf_impl(repository_ctx):
    """Implementation of the rocm_autoconf repository rule."""
    if not _enable_rocm(repository_ctx):
        _create_dummy_repository(repository_ctx)
    elif get_host_environ(repository_ctx, _TF_ROCM_CONFIG_REPO) != None:
        _create_remote_rocm_repository(
            repository_ctx,
            get_host_environ(repository_ctx, _TF_ROCM_CONFIG_REPO),
        )
    else:
        _create_local_rocm_repository(repository_ctx)

_ENVIRONS = [
    _GCC_HOST_COMPILER_PATH,
    _GCC_HOST_COMPILER_PREFIX,
    "TF_NEED_ROCM",
    _ROCM_TOOLKIT_PATH,
    _TF_ROCM_VERSION,
    _TF_MIOPEN_VERSION,
    _TF_ROCM_AMDGPU_TARGETS,
]

remote_rocm_configure = repository_rule(
    implementation = _create_local_rocm_repository,
    environ = _ENVIRONS,
    remotable = True,
    attrs = {
        "environ": attr.string_dict(),
    },
)

rocm_configure = repository_rule(
    implementation = _rocm_autoconf_impl,
    environ = _ENVIRONS + [_TF_ROCM_CONFIG_REPO],
)
"""Detects and configures the local ROCm toolchain.

Add the following to your WORKSPACE FILE:

```python
rocm_configure(name = "local_config_rocm")
```

Args:
  name: A unique name for this workspace rule.
"""
