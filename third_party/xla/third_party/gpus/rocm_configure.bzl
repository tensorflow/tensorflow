"""Repository rule for ROCm autoconfiguration.

`rocm_configure` depends on the following environment variables:

  * `TF_NEED_ROCM`: Whether to enable building with ROCm.
  * `GCC_HOST_COMPILER_PATH`: The GCC host compiler path.
  * `TF_ROCM_CLANG`: Whether to use clang for C++ and HIPCC for ROCm compilation.
  * `TF_SYSROOT`: The sysroot to use when compiling.
  * `CLANG_COMPILER_PATH`: The clang compiler path that will be used for
    host code compilation if TF_ROCM_CLANG is 1.
  * `ROCM_PATH`: The path to the ROCm toolkit. Default is `/opt/rocm`.
  * `TF_ROCM_AMDGPU_TARGETS`: The AMDGPU targets.
"""

load(
    "//third_party/gpus/rocm:rocm_redist.bzl",
    "rocm_redist",
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
    "get_python_bin",
    "raw_exec",
    "realpath",
    "relative_to",
    "which",
)
load(
    ":compiler_common_tools.bzl",
    "to_list_of_strings",
)
load(
    ":cuda_configure.bzl",
    "enable_cuda",
)
load(
    ":sycl_configure.bzl",
    "enable_sycl",
)

_GCC_HOST_COMPILER_PATH = "GCC_HOST_COMPILER_PATH"
_GCC_HOST_COMPILER_PREFIX = "GCC_HOST_COMPILER_PREFIX"
_CLANG_COMPILER_PATH = "CLANG_COMPILER_PATH"
_TF_SYSROOT = "TF_SYSROOT"
_ROCM_TOOLKIT_PATH = "ROCM_PATH"
_TF_ROCM_AMDGPU_TARGETS = "TF_ROCM_AMDGPU_TARGETS"
_TF_ROCM_CONFIG_REPO = "TF_ROCM_CONFIG_REPO"
_DISTRIBUTION_PATH = "rocm/rocm_dist"
_OS = "OS"
_ROCM_VERSION = "ROCM_VERSION"

_DEFAULT_ROCM_TOOLKIT_PATH = "/opt/rocm"

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

def find_cc(repository_ctx, use_rocm_clang):
    """Find the C++ compiler."""

    if use_rocm_clang:
        target_cc_name = "clang"
        cc_path_envvar = _CLANG_COMPILER_PATH
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

def _cxx_inc_convert(path):
    """Convert path returned by cc -E xc++ in a complete path."""
    path = path.strip()
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

    # TODO: We pass -no-canonical-prefixes here to match the compiler flags,
    #       but in rocm_clang CROSSTOOL file that is a `feature` and we should
    #       handle the case when it's disabled and no flag is passed
    result = raw_exec(repository_ctx, [
        cc,
        "-E",
        "-x" + lang,
        "-",
        "-v",
    ] + sysroot)
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

def _rocm_include_path(repository_ctx, rocm_config, bash_bin):
    """Generates the entries for rocm inc dirs based on rocm_config.

    Args:
      repository_ctx: The repository context.
      rocm_config: The path to the gcc host compiler.
      bash_bin: path to the bash interpreter.

    Returns:
      A string containing the Starlark string for each of the hipcc
      compiler include directories, which can be added to the CROSSTOOL
      file.
    """
    inc_dirs = []

    # Add HIP-Clang headers (relative to rocm root)
    rocm_path = repository_ctx.path(rocm_config.rocm_toolkit_path)
    clang_path = rocm_path.get_child("llvm/bin/clang")
    resource_dir_result = execute(repository_ctx, [str(clang_path), "-print-resource-dir"])

    if resource_dir_result.return_code:
        auto_configure_fail("Failed to run hipcc -print-resource-dir: %s" % err_out(resource_dir_result))

    resource_dir_abs = resource_dir_result.stdout.strip()

    resource_dir_rel = relative_to(repository_ctx, str(rocm_path.realpath), resource_dir_abs, bash_bin)

    resource_dir = str(rocm_path.get_child(resource_dir_rel))

    inc_dirs.append(resource_dir + "/include")
    inc_dirs.append(resource_dir + "/share")

    return inc_dirs

def _enable_rocm(repository_ctx):
    enable_rocm = get_host_environ(repository_ctx, "TF_NEED_ROCM")
    if enable_rocm == "1":
        if get_cpu_value(repository_ctx) != "Linux":
            auto_configure_warning("ROCm configure is only supported on Linux")
            return False
        return True
    return False

def _amdgpu_targets(repository_ctx, rocm_toolkit_path, bash_bin):
    """Returns a list of strings representing AMDGPU targets."""
    amdgpu_targets_str = get_host_environ(repository_ctx, _TF_ROCM_AMDGPU_TARGETS)
    if not amdgpu_targets_str:
        cmd = "%s/bin/rocm_agent_enumerator" % rocm_toolkit_path
        result = execute(repository_ctx, [bash_bin, "-c", cmd])
        targets = [target for target in result.stdout.strip().split("\n") if target != "gfx000"]
        targets = {x: None for x in targets}
        targets = list(targets.keys())
        amdgpu_targets_str = ",".join(targets)
    amdgpu_targets = amdgpu_targets_str.split(",")
    for amdgpu_target in amdgpu_targets:
        if amdgpu_target[:3] != "gfx":
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
    for row in libs_paths:
        lib_paths = row[1]
        for lib_path in lib_paths:
            all_paths.append(lib_path)
    return files_exist(repository_ctx, all_paths, bash_bin)

def _select_rocm_lib_paths(repository_ctx, libs_paths, bash_bin):
    test_results = _batch_files_exist(repository_ctx, libs_paths, bash_bin)

    libs = {}
    i = 0
    for row in libs_paths:
        name = row[0]
        lib_paths = row[1]
        optional = (len(row) > 2 and row[2] == True)
        selected_path = None
        for path in lib_paths:
            if test_results[i] and selected_path == None:
                # For each lib select the first path that exists.
                selected_path = path
            i = i + 1
        if selected_path == None:
            if optional:
                libs[name] = None
                continue
            else:
                auto_configure_fail("Cannot find rocm library %s" % name)

        libs[name] = struct(file_name = selected_path.basename, path = realpath(repository_ctx, selected_path, bash_bin))

    return libs

def _find_libs(repository_ctx, rocm_config, miopen_path, rccl_path, bash_bin):
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
            ("amdhip64", rocm_config.rocm_toolkit_path),
            ("rocblas", rocm_config.rocm_toolkit_path),
            ("hiprand", rocm_config.rocm_toolkit_path),
            ("MIOpen", miopen_path),
            ("rccl", rccl_path),
            ("hipsparse", rocm_config.rocm_toolkit_path),
            ("roctracer64", rocm_config.rocm_toolkit_path),
            ("rocsolver", rocm_config.rocm_toolkit_path),
        ]
    ]
    if int(rocm_config.rocm_version_number) >= 40500:
        libs_paths.append(("hipsolver", _rocm_lib_paths(repository_ctx, "hipsolver", rocm_config.rocm_toolkit_path)))
        libs_paths.append(("hipblas", _rocm_lib_paths(repository_ctx, "hipblas", rocm_config.rocm_toolkit_path)))

    # hipblaslt may be absent even in versions of ROCm where it exists
    # (it is not installed by default in some containers). Autodetect.
    libs_paths.append(("hipblaslt", _rocm_lib_paths(repository_ctx, "hipblaslt", rocm_config.rocm_toolkit_path), True))
    return _select_rocm_lib_paths(repository_ctx, libs_paths, bash_bin)

def find_rocm_config(repository_ctx, rocm_path):
    """Returns ROCm config dictionary from running find_rocm_config.py"""
    python_bin = get_python_bin(repository_ctx)
    exec_result = execute(repository_ctx, [python_bin, repository_ctx.attr._find_rocm_config], env_vars = {"ROCM_PATH": rocm_path})
    if exec_result.return_code:
        auto_configure_fail("Failed to run find_rocm_config.py: %s" % err_out(exec_result))

    # Parse the dict from stdout.
    return dict([tuple(x.split(": ")) for x in exec_result.stdout.splitlines()])

def _get_rocm_config(repository_ctx, bash_bin, rocm_path, install_path):
    """Detects and returns information about the ROCm installation on the system.

    Args:
      repository_ctx: The repository context.
      bash_bin: the path to the path interpreter

    Returns:
      A struct containing the following fields:
        rocm_toolkit_path: The ROCm toolkit installation directory.
        amdgpu_targets: A list of the system's AMDGPU targets.
        rocm_version_number: The version of ROCm on the system.
        miopen_version_number: The version of MIOpen on the system.
        hipruntime_version_number: The version of HIP Runtime on the system.
    """
    config = find_rocm_config(repository_ctx, rocm_path)
    rocm_toolkit_path = config["rocm_toolkit_path"]
    rocm_version_number = config["rocm_version_number"]
    miopen_version_number = config["miopen_version_number"]
    hipruntime_version_number = config["hipruntime_version_number"]
    return struct(
        amdgpu_targets = _amdgpu_targets(repository_ctx, rocm_toolkit_path, bash_bin),
        rocm_toolkit_path = rocm_toolkit_path,
        rocm_version_number = rocm_version_number,
        miopen_version_number = miopen_version_number,
        hipruntime_version_number = hipruntime_version_number,
        install_path = install_path,
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
            "%{gpu_is_configured}": "if_true" if enable_cuda(repository_ctx) or enable_sycl(repository_ctx) else "if_false",
            "%{cuda_or_rocm}": "if_true" if enable_cuda(repository_ctx) else "if_false",
            "%{rocm_extra_copts}": "[]",
            "%{rocm_gpu_architectures}": "[]",
            "%{rocm_version_number}": "0",
            "%{rocm_hipblaslt}": "False",
        },
    )
    _tpl(
        repository_ctx,
        "rocm:BUILD",
        {
            "%{hip_lib}": _lib_name("hip"),
            "%{rocblas_lib}": _lib_name("rocblas"),
            "%{hipblas_lib}": _lib_name("hipblas"),
            "%{miopen_lib}": _lib_name("miopen"),
            "%{rccl_lib}": _lib_name("rccl"),
            "%{hiprand_lib}": _lib_name("hiprand"),
            "%{hipsparse_lib}": _lib_name("hipsparse"),
            "%{roctracer_lib}": _lib_name("roctracer64"),
            "%{rocsolver_lib}": _lib_name("rocsolver"),
            "%{hipsolver_lib}": _lib_name("hipsolver"),
            "%{hipblaslt_lib}": _lib_name("hipblaslt"),
            "%{rocm_headers}": "",
        },
    )

    # Create dummy files for the ROCm toolkit since they are still required by
    # tensorflow/compiler/xla/stream_executor/rocm:rocm_rpath
    repository_ctx.file("rocm/hip/include/hip/hip_runtime.h", "")

    # Set up rocm_config.h, which is used by
    # tensorflow/compiler/xla/stream_executor/dso_loader.cc.
    _tpl(
        repository_ctx,
        "rocm:rocm_config.h",
        {
            "%{rocm_toolkit_path}": _DEFAULT_ROCM_TOOLKIT_PATH,
            "%{hipblaslt_flag}": "0",
        },
        "rocm/rocm_config/rocm_config.h",
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

def _flag_enabled(repository_ctx, flag_name):
    return get_host_environ(repository_ctx, flag_name) == "1"

def _use_rocm_clang(repository_ctx):
    # Returns the flag if we need to use clang for the host.
    return _flag_enabled(repository_ctx, "TF_ROCM_CLANG")

def _tf_sysroot(repository_ctx):
    return get_host_environ(repository_ctx, _TF_SYSROOT, "")

def _compute_rocm_extra_copts(repository_ctx, amdgpu_targets):
    amdgpu_target_flags = ["--offload-arch=" +
                           amdgpu_target for amdgpu_target in amdgpu_targets]
    return str(amdgpu_target_flags)

def _get_file_name(url):
    last_slash_index = url.rfind("/")
    return url[last_slash_index + 1:]

def _download_package(repository_ctx, archive):
    file_name = _get_file_name(archive.url)
    tmp_dir = "tmp"
    repository_ctx.file(tmp_dir + "/.idx")  # create tmp dir

    repository_ctx.report_progress("Downloading and extracting {}, expected hash is {}".format(archive.url, archive.sha256))  # buildifier: disable=print
    repository_ctx.download_and_extract(
        url = archive.url,
        output = tmp_dir if archive.url.endswith(".deb") else _DISTRIBUTION_PATH,
        sha256 = archive.sha256,
    )

    all_files = repository_ctx.path(tmp_dir).readdir()

    matched_files = [f for f in all_files if _get_file_name(str(f)).startswith("data.")]
    for f in matched_files:
        repository_ctx.extract(f, _DISTRIBUTION_PATH)

    repository_ctx.delete(tmp_dir)
    repository_ctx.delete(file_name)

def _remove_root_dir(path, root_dir):
    if path.startswith(root_dir + "/"):
        return path[len(root_dir) + 1:]
    return path

def _setup_rocm_distro_dir(repository_ctx):
    """Sets up the rocm hermetic installation directory to be used in hermetic build"""
    bash_bin = get_bash_bin(repository_ctx)
    os = repository_ctx.os.environ.get(_OS)
    rocm_version = repository_ctx.os.environ.get(_ROCM_VERSION)
    if os and rocm_version:
        redist = rocm_redist[os][rocm_version]
        repository_ctx.file("rocm/.index")
        for archive in redist["archives"]:
            _download_package(repository_ctx, archive)
        return _get_rocm_config(repository_ctx, bash_bin, "{}/{}".format(_DISTRIBUTION_PATH, redist["rocm_root"]), "/{}".format(redist["rocm_root"]))
    else:
        rocm_path = repository_ctx.os.environ.get(_ROCM_TOOLKIT_PATH, _DEFAULT_ROCM_TOOLKIT_PATH)
        repository_ctx.report_progress("Using local rocm installation {}".format(rocm_path))  # buildifier: disable=print
        repository_ctx.symlink(rocm_path, _DISTRIBUTION_PATH)
        return _get_rocm_config(repository_ctx, bash_bin, _DISTRIBUTION_PATH, _DEFAULT_ROCM_TOOLKIT_PATH)

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

    rocm_config = _setup_rocm_distro_dir(repository_ctx)
    rocm_version_number = int(rocm_config.rocm_version_number)

    # For ROCm 5.2 and above, find MIOpen and RCCL in the main rocm lib path
    miopen_path = rocm_config.rocm_toolkit_path + "/miopen" if rocm_version_number < 50200 else rocm_config.rocm_toolkit_path
    rccl_path = rocm_config.rocm_toolkit_path + "/rccl" if rocm_version_number < 50200 else rocm_config.rocm_toolkit_path

    # Copy header and library files to execroot.
    # rocm_toolkit_path
    rocm_toolkit_path = _remove_root_dir(rocm_config.rocm_toolkit_path, "rocm")

    bash_bin = get_bash_bin(repository_ctx)
    rocm_libs = _find_libs(repository_ctx, rocm_config, miopen_path, rccl_path, bash_bin)
    rocm_lib_srcs = []
    rocm_lib_outs = []
    for lib in rocm_libs.values():
        if lib:
            rocm_lib_srcs.append(lib.path)
            rocm_lib_outs.append("rocm/lib/" + lib.file_name)

    clang_offload_bundler_path = rocm_toolkit_path + "/llvm/bin/clang-offload-bundler"

    have_hipblaslt = "1" if rocm_libs["hipblaslt"] != None else "0"

    # Set up BUILD file for rocm/
    repository_ctx.template(
        "rocm/build_defs.bzl",
        tpl_paths["rocm:build_defs.bzl"],
        {
            "%{rocm_is_configured}": "True",
            "%{gpu_is_configured}": "if_true",
            "%{cuda_or_rocm}": "if_true",
            "%{rocm_extra_copts}": _compute_rocm_extra_copts(
                repository_ctx,
                rocm_config.amdgpu_targets,
            ),
            "%{rocm_gpu_architectures}": str(rocm_config.amdgpu_targets),
            "%{rocm_version_number}": str(rocm_version_number),
            "%{rocm_hipblaslt}": "True" if rocm_libs["hipblaslt"] != None else "False",
        },
    )

    repository_dict = {
        "%{rocm_root}": rocm_toolkit_path,
        "%{rocm_toolkit_path}": str(repository_ctx.path(rocm_config.rocm_toolkit_path)),
    }

    is_rocm_clang = _use_rocm_clang(repository_ctx)
    tf_sysroot = _tf_sysroot(repository_ctx)

    if rocm_libs["hipblaslt"] != None:
        repository_dict["%{hipblaslt_lib}"] = rocm_libs["hipblaslt"].file_name

    if rocm_version_number >= 40500:
        repository_dict["%{hipsolver_lib}"] = rocm_libs["hipsolver"].file_name
        repository_dict["%{hipblas_lib}"] = rocm_libs["hipblas"].file_name

    repository_ctx.template(
        "rocm/BUILD",
        tpl_paths["rocm:BUILD"],
        repository_dict,
    )

    # Set up crosstool/
    cc = find_cc(repository_ctx, is_rocm_clang)
    host_compiler_includes = get_cxx_inc_directories(
        repository_ctx,
        cc,
        tf_sysroot,
    )

    # host_compiler_includes = get_cxx_inc_directories(repository_ctx, cc)

    rocm_defines = {}
    rocm_defines["%{builtin_sysroot}"] = tf_sysroot
    rocm_defines["%{compiler}"] = "unknown"
    if is_rocm_clang:
        rocm_defines["%{compiler}"] = "clang"
    host_compiler_prefix = get_host_environ(repository_ctx, _GCC_HOST_COMPILER_PREFIX, "/usr/bin")
    rocm_defines["%{host_compiler_prefix}"] = host_compiler_prefix
    rocm_defines["%{linker_bin_path}"] = rocm_config.rocm_toolkit_path + host_compiler_prefix
    rocm_defines["%{extra_no_canonical_prefixes_flags}"] = ""
    rocm_defines["%{unfiltered_compile_flags}"] = ""
    rocm_defines["%{rocm_hipcc_files}"] = "[]"

    if is_rocm_clang:
        rocm_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-no-canonical-prefixes\""
    else:
        # For gcc, do not canonicalize system header paths; some versions of gcc
        # pick the shortest possible path for system includes when creating the
        # .d file - given that includes that are prefixed with "../" multiple
        # time quickly grow longer than the root of the tree, this can lead to
        # bazel's header check failing.
        rocm_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-fno-canonical-system-headers\""

    rocm_defines["%{unfiltered_compile_flags}"] = to_list_of_strings([
        "-DTENSORFLOW_USE_ROCM=1",
        "-D__HIP_PLATFORM_AMD__",
        "-DEIGEN_USE_HIP",
        "-DUSE_ROCM",
    ])

    rocm_defines["%{host_compiler_path}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"

    rocm_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(
        host_compiler_includes + _rocm_include_path(repository_ctx, rocm_config, bash_bin),
    )

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
        rocm_defines,
    )

    repository_ctx.template(
        "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
        tpl_paths["crosstool:clang/bin/crosstool_wrapper_driver_rocm"],
        {
            "%{cpu_compiler}": str(cc),
            "%{compiler_is_clang}": "True" if is_rocm_clang else "False",
            "%{hipcc_path}": str(repository_ctx.path(rocm_config.rocm_toolkit_path + "/bin/hipcc")),
            "%{hipcc_env}": _hipcc_env(repository_ctx),
            "%{rocm_path}": str(repository_ctx.path(rocm_config.rocm_toolkit_path)),
            "%{rocr_runtime_path}": str(repository_ctx.path(rocm_config.rocm_toolkit_path + "/lib")),
            "%{rocr_runtime_library}": "hsa-runtime64",
            "%{hip_runtime_path}": str(repository_ctx.path(rocm_config.rocm_toolkit_path + "/lib")),
            "%{hip_runtime_library}": "amdhip64",
            "%{crosstool_verbose}": _crosstool_verbose(repository_ctx),
            "%{gcc_host_compiler_path}": str(cc),
        },
    )

    # Set up rocm_config.h, which is used by
    # tensorflow/compiler/xla/stream_executor/dso_loader.cc.
    repository_ctx.template(
        "rocm/rocm_config/rocm_config.h",
        tpl_paths["rocm:rocm_config.h"],
        {
            "%{rocm_amdgpu_targets}": ",".join(
                ["\"%s\"" % c for c in rocm_config.amdgpu_targets],
            ),
            "%{rocm_toolkit_path}": rocm_config.install_path,
            "%{rocm_version_number}": rocm_config.rocm_version_number,
            "%{miopen_version_number}": rocm_config.miopen_version_number,
            "%{hipruntime_version_number}": rocm_config.hipruntime_version_number,
            "%{hipblaslt_flag}": have_hipblaslt,
            "%{hip_soversion_number}": "6" if int(rocm_config.rocm_version_number) >= 60000 else "5",
            "%{rocblas_soversion_number}": "4" if int(rocm_config.rocm_version_number) >= 60000 else "3",
        },
    )

    # Set up rocm_config.h, which is used by
    # tensorflow/compiler/xla/stream_executor/dso_loader.cc.
    repository_ctx.template(
        "rocm/rocm_config_hermetic/rocm_config.h",
        tpl_paths["rocm:rocm_config.h"],
        {
            "%{rocm_amdgpu_targets}": ",".join(
                ["\"%s\"" % c for c in rocm_config.amdgpu_targets],
            ),
            "%{rocm_toolkit_path}": str(repository_ctx.path(rocm_config.rocm_toolkit_path)),
            "%{rocm_version_number}": rocm_config.rocm_version_number,
            "%{miopen_version_number}": rocm_config.miopen_version_number,
            "%{hipruntime_version_number}": rocm_config.hipruntime_version_number,
            "%{hipblaslt_flag}": have_hipblaslt,
            "%{hip_soversion_number}": "6" if int(rocm_config.rocm_version_number) >= 60000 else "5",
            "%{rocblas_soversion_number}": "4" if int(rocm_config.rocm_version_number) >= 60000 else "3",
        },
    )

def _create_remote_rocm_repository(repository_ctx, remote_config_repo):
    """Creates pointers to a remotely configured repo set up to build with ROCm."""
    _tpl(
        repository_ctx,
        "rocm:build_defs.bzl",
        {
            "%{rocm_is_configured}": "True",
            "%{gpu_is_configured}": "if_true",
            "%{cuda_or_rocm}": "if_true",
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
    "TF_ROCM_CLANG",
    "TF_NEED_CUDA",  # Needed by the `if_gpu_is_configured` macro
    _ROCM_TOOLKIT_PATH,
    _TF_ROCM_AMDGPU_TARGETS,
    _OS,
    _ROCM_VERSION,
]

remote_rocm_configure = repository_rule(
    implementation = _create_local_rocm_repository,
    environ = _ENVIRONS,
    remotable = True,
    attrs = {
        "environ": attr.string_dict(),
        "_find_rocm_config": attr.label(
            default = Label("@local_xla//third_party/gpus:find_rocm_config.py"),
        ),
    },
)

rocm_configure = repository_rule(
    implementation = _rocm_autoconf_impl,
    environ = _ENVIRONS + [_TF_ROCM_CONFIG_REPO],
    attrs = {
        "_find_rocm_config": attr.label(
            default = Label("@local_xla//third_party/gpus:find_rocm_config.py"),
        ),
    },
)
"""Detects and configures the local ROCm toolchain.

Add the following to your WORKSPACE FILE:

```python
rocm_configure(name = "local_config_rocm")
```

Args:
  name: A unique name for this workspace rule.
"""
