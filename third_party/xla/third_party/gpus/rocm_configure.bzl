"""Repository rule for ROCm autoconfiguration.

`rocm_configure` depends on the following environment variables:

  * `TF_NEED_ROCM`: Whether to enable building with ROCm.
  * `TF_ROCM_CLANG`: Whether to use clang for C++ and HIPCC for ROCm compilation.
  * `TF_SYSROOT`: The sysroot to use when compiling.
  * `CLANG_COMPILER_PATH`: The clang compiler path that will be used for
    host code compilation if TF_ROCM_CLANG is 1.
  * `TF_ROCM_AMDGPU_TARGETS`: The AMDGPU targets.
  * `TF_ROCM_RBE_DOCKER_IMAGE`: Docker image to be used in rbe worker to execute the action
  * `TF_ROCM_RBE_SINGLE_GPU_POOL`: The name of the rbe pool used to execute single gpu tests
  * `TF_ROCM_RBE_MULTI_GPU_POOL`: The name of the rbe pool used to execute multi gpu tests
"""

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
    "realpath",
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

_TF_ROCM_AMDGPU_TARGETS = "TF_ROCM_AMDGPU_TARGETS"
_TF_ROCM_CONFIG_REPO = "TF_ROCM_CONFIG_REPO"
_DISTRIBUTION_PATH = "rocm/rocm_dist"
_TMPDIR = "TMPDIR"

_TF_ROCM_RBE_DOCKER_IMAGE = "TF_ROCM_RBE_DOCKER_IMAGE"
_TF_ROCM_RBE_POOL = "TF_ROCM_RBE_POOL"
_TF_ROCM_RBE_SINGLE_GPU_POOL = "TF_ROCM_RBE_SINGLE_GPU_POOL"
_TF_ROCM_RBE_MULTI_GPU_POOL = "TF_ROCM_RBE_MULTI_GPU_POOL"
_DEFAULT_TF_ROCM_RBE_POOL = "default"
_DEFAULT_TF_ROCM_RBE_SINGLE_GPU_POOL = "linux_x64_gpu"
_DEFAULT_TF_ROCM_RBE_MULTI_GPU_POOL = "linux_x64_multigpu"

# rocm/tensorflow-build:latest-jammy-python3.11-rocm7.0.2
_DEFAULT_TF_ROCM_RBE_DOCKER_IMAGE = "rocm/tensorflow-build@sha256:a2672ff2510b369b4a5f034272a518dc93c2e492894e3befaeef19649632ccaa"

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

def verify_build_defines(params):
    """Verify all variables that crosstool/BUILD.rocm.tpl expects are substituted.

    Args:
      params: dict of variables that will be passed to the BUILD.tpl template.
    """
    missing = []
    for param in [
        "host_compiler_path",
        "linker_bin_path",
        "unfiltered_compile_flags",
    ]:
        if ("%{" + param + "}") not in params:
            missing.append(param)

    if missing:
        auto_configure_fail(
            "Missing template parameters: %s" % missing,
        )

def _rocm_include_path(repository_ctx, rocm_config, bash_bin):
    """Generates the entries for rocm inc dirs based on rocm_config.

    Args:
      repository_ctx: The repository context.
      rocm_config: The ROCm config struct.
      bash_bin: path to the bash interpreter.

    Returns:
      A list of the ROCm compiler include directories.
    """
    inc_dirs = []

    # Add HIP-Clang headers (relative to rocm root)
    inc_dirs.append(str(repository_ctx.path(rocm_config.rocm_toolkit_path)) + "/include")
    inc_dirs.append(str(repository_ctx.path(rocm_config.rocm_toolkit_path)) + "/lib/llvm/lib/clang/18/include")

    return inc_dirs

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
        "HIPCC_LINK_FLAGS_APPEND",
    ]:
        if get_host_environ(repository_ctx, name):
            hipcc_env = (hipcc_env + " " + name + "=" +
                         get_host_environ(repository_ctx, name))

    return hipcc_env.strip()

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
    amdgpu_targets = [amdgpu for amdgpu in amdgpu_targets_str.split(",") if amdgpu]
    for amdgpu_target in amdgpu_targets:
        if amdgpu_target[:3] != "gfx":
            auto_configure_fail("Invalid AMDGPU target: %s" % amdgpu_target)
    return amdgpu_targets

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
        repository_ctx.path("%s/lib/%s.0" % (basedir, file_name)),  # hipblaslt has this pattern
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

        libs[name] = struct(
            file_name = selected_path.basename,
            path = realpath(repository_ctx, selected_path, bash_bin),
        )

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
    repo_path = str(repository_ctx.path(rocm_config.rocm_toolkit_path))
    libs_paths = [
        (name, _rocm_lib_paths(repository_ctx, name, path))
        for name, path in [
            ("amdhip64", repo_path),
            ("rocblas", repo_path),
            ("hiprand", repo_path),
            ("MIOpen", repo_path),
            ("rccl", repo_path),
            ("hipsparse", repo_path),
            ("roctracer64", repo_path),
            ("rocsolver", repo_path),
            ("rocsolver", repo_path),
            ("hipsolver", repo_path),
            ("hipfft", repo_path),
            ("rocrand", repo_path),
            ("hipblas", repo_path),
            ("hipblaslt", repo_path),
            ("rocprofiler-sdk", repo_path),
        ]
    ]

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
            "%{single_gpu_rbe_pool}": repository_ctx.os.environ.get(_TF_ROCM_RBE_SINGLE_GPU_POOL, _DEFAULT_TF_ROCM_RBE_SINGLE_GPU_POOL),
            "%{multi_gpu_rbe_pool}": repository_ctx.os.environ.get(_TF_ROCM_RBE_MULTI_GPU_POOL, _DEFAULT_TF_ROCM_RBE_MULTI_GPU_POOL),
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
            "%{rocm_toolkit_path}": "/opt/rocm",
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

def _compute_rocm_extra_copts(repository_ctx, amdgpu_targets):
    amdgpu_target_flags = ["--offload-arch=" +
                           amdgpu_target for amdgpu_target in amdgpu_targets]
    return str(amdgpu_target_flags)

def _remove_root_dir(path, root_dir):
    if path.startswith(root_dir + "/"):
        return path[len(root_dir) + 1:]
    return path

def _setup_rocm_distro_dir(repository_ctx):
    """Sets up the rocm hermetic installation directory to be used in hermetic build"""
    bash_bin = get_bash_bin(repository_ctx)

    # Use ROCm dist directory from hipcc_configure repository
    rocm_dist_label = repository_ctx.attr.rocm_dist
    if not rocm_dist_label:
        fail("rocm_dist attribute is required. " +
             "Set it to @config_rocm_hipcc//rocm:rocm_dist")

    # Directly get the path to rocm_dist directory (exported via exports_files)
    hipcc_rocm_path = repository_ctx.path(rocm_dist_label)
    repository_ctx.report_progress("Using ROCm from: {}".format(hipcc_rocm_path))
    repository_ctx.symlink(hipcc_rocm_path, _DISTRIBUTION_PATH)
    return _get_rocm_config(repository_ctx, bash_bin, _DISTRIBUTION_PATH, str(hipcc_rocm_path))

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

    # Copy header and library files to execroot.
    # rocm_toolkit_path
    rocm_toolkit_path = _remove_root_dir(rocm_config.rocm_toolkit_path, "rocm")

    bash_bin = get_bash_bin(repository_ctx)
    rocm_libs = _find_libs(repository_ctx, rocm_config, bash_bin)
    rocm_lib_srcs = []
    rocm_lib_outs = []
    for lib in rocm_libs.values():
        if lib:
            rocm_lib_srcs.append(lib.path)
            rocm_lib_outs.append("rocm/lib/" + lib.file_name)

    clang_offload_bundler_path = rocm_toolkit_path + "/llvm/bin/clang-offload-bundler"

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
            "%{single_gpu_rbe_pool}": repository_ctx.os.environ.get(_TF_ROCM_RBE_SINGLE_GPU_POOL, _DEFAULT_TF_ROCM_RBE_SINGLE_GPU_POOL),
            "%{multi_gpu_rbe_pool}": repository_ctx.os.environ.get(_TF_ROCM_RBE_MULTI_GPU_POOL, _DEFAULT_TF_ROCM_RBE_MULTI_GPU_POOL),
            "%{rocm_gpu_architectures}": str(rocm_config.amdgpu_targets),
            "%{rocm_version_number}": str(rocm_version_number),
            "%{rocm_hipblaslt}": "True",
        },
    )

    repository_dict = {
        "%{rocm_root}": rocm_toolkit_path,
        "%{rocm_toolkit_path}": str(repository_ctx.path(rocm_config.rocm_toolkit_path)),
        "%{rocm_rbe_docker_image}": repository_ctx.os.environ.get(_TF_ROCM_RBE_DOCKER_IMAGE, _DEFAULT_TF_ROCM_RBE_DOCKER_IMAGE),
        "%{rocm_rbe_pool}": repository_ctx.os.environ.get(_TF_ROCM_RBE_POOL, _DEFAULT_TF_ROCM_RBE_POOL),
        "%{rocm_repo_name}": repository_ctx.name,
    }

    repository_ctx.template(
        "rocm/BUILD",
        tpl_paths["rocm:BUILD"],
        repository_dict,
    )

    # Set up crosstool/
    rocm_defines = {}
    rocm_defines["%{linker_bin_path}"] = rocm_config.rocm_toolkit_path + "/usr/bin"

    rocm_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(
        _rocm_include_path(repository_ctx, rocm_config, bash_bin),
    )

    rocm_defines["%{unfiltered_compile_flags}"] = to_list_of_strings([
        "-DTENSORFLOW_USE_ROCM=1",
        "-D__HIP_PLATFORM_AMD__",
        "-DEIGEN_USE_HIP",
        "-DUSE_ROCM",
    ])

    # Use wrapper as the host compiler path for the toolchain
    rocm_defines["%{host_compiler_path}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"

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
            "%{rocm_root}": "external/local_config_rocm/" + str(rocm_config.rocm_toolkit_path),
            "%{hipcc_env}": _hipcc_env(repository_ctx),
            "%{rocr_runtime_library}": "hsa-runtime64",
            "%{crosstool_verbose}": "0",
            "%{tmpdir}": get_host_environ(
                repository_ctx,
                _TMPDIR,
                "",
            ),
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
            "%{hipblaslt_flag}": "1",
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
            "%{hipblaslt_flag}": "1",
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
    "TF_NEED_ROCM",
    "TF_NEED_CUDA",  # Needed by the `if_gpu_is_configured` macro
    _TF_ROCM_AMDGPU_TARGETS,
    _TF_ROCM_RBE_DOCKER_IMAGE,
    _TF_ROCM_RBE_POOL,
    _TF_ROCM_RBE_SINGLE_GPU_POOL,
    _TF_ROCM_RBE_MULTI_GPU_POOL,
]

remote_rocm_configure = repository_rule(
    implementation = _create_local_rocm_repository,
    environ = _ENVIRONS,
    remotable = True,
    attrs = {
        "environ": attr.string_dict(),
        "rocm_dist": attr.label(
            doc = "Label to the rocm_dist directory from hipcc_configure " +
                  "(e.g. @config_rocm_hipcc//rocm:rocm_dist).",
        ),
        "_find_rocm_config": attr.label(
            default = Label("//third_party/gpus:find_rocm_config.py"),
        ),
    },
)

rocm_configure = repository_rule(
    implementation = _rocm_autoconf_impl,
    environ = _ENVIRONS + [_TF_ROCM_CONFIG_REPO],
    attrs = {
        "rocm_dist": attr.label(
            default = Label("@config_rocm_hipcc//rocm:rocm_dist"),
            doc = "Label to the rocm_dist directory from hipcc_configure " +
                  "(e.g. @config_rocm_hipcc//rocm:rocm_dist).",
        ),
        "_find_rocm_config": attr.label(
            default = Label("//third_party/gpus:find_rocm_config.py"),
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
