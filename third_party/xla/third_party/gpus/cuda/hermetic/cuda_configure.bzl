"""Repository rule for hermetic CUDA configuration.

`cuda_configure` depends on the following environment variables:

  * `TF_NEED_CUDA`: Whether to enable building with CUDA toolchain.
  * `USE_CUDA_REDISTRIBUTIONS`: Whether to use CUDA redistributions, but not
    the CUDA toolchain. This can be used to preserve the cache between GPU and
    CPU builds.
  * `TF_NVCC_CLANG` (deprecated): Whether to use clang for C++ and NVCC for Cuda
    compilation.
  * `CUDA_NVCC`: Whether to use NVCC for Cuda compilation.
  * `CLANG_CUDA_COMPILER_PATH`: The clang compiler path that will be used for
    both host and device code compilation.
  * `CC`: The compiler path that will be used for both host and device code
    compilation if `CLANG_CUDA_COMPILER_PATH` is not set.
  * `TF_SYSROOT`: The sysroot to use when compiling.
  * `HERMETIC_CUDA_VERSION`: The version of the CUDA toolkit. If not specified,
    the version will be determined by the `TF_CUDA_VERSION`.
  * `HERMETIC_CUDA_COMPUTE_CAPABILITIES`: The CUDA compute capabilities. Default 
    is `3.5,5.2`. If not specified, the value will be determined by the
    `TF_CUDA_COMPUTE_CAPABILITIES`.
  * `TMPDIR`: specifies the directory to use for temporary files. This
    environment variable is used by GCC compiler.
"""

load("@cuda_cccl//:version.bzl", _cccl_version = "VERSION")
load("@cuda_cublas//:version.bzl", _cublas_version = "VERSION")
load("@cuda_cudart//:version.bzl", _cudart_version = "VERSION")
load("@cuda_cudnn//:version.bzl", _cudnn_version = "VERSION")
load("@cuda_cufft//:version.bzl", _cufft_version = "VERSION")
load("@cuda_cupti//:version.bzl", _cupti_version = "VERSION")
load("@cuda_curand//:version.bzl", _curand_version = "VERSION")
load("@cuda_cusolver//:version.bzl", _cusolver_version = "VERSION")
load("@cuda_cusparse//:version.bzl", _cusparse_version = "VERSION")
load("@cuda_nvcc//:version.bzl", _nvcc_version = "VERSION")
load("@cuda_nvdisasm//:version.bzl", _nvdisasm_version = "VERSION")
load("@cuda_nvjitlink//:version.bzl", _nvjitlink_version = "VERSION")
load("@cuda_nvml//:version.bzl", _nvml_version = "VERSION")
load("@cuda_nvtx//:version.bzl", _nvtx_version = "VERSION")
load(
    "//third_party/gpus:compiler_common_tools.bzl",
    "get_cxx_inc_directories",
    "to_list_of_strings",
)
load(
    "//third_party/gpus/cuda/hermetic:cuda_redist_versions.bzl",
    "PTX_VERSION_DICT",
)
load(
    "//third_party/remote_config:common.bzl",
    "execute",
    "get_bash_bin",
    "get_cpu_value",
    "get_host_environ",
    "realpath",
    "which",
)

def _find_cc(repository_ctx):
    """Find the C++ compiler."""
    cc_name = "clang"

    cc_name_from_env = get_host_environ(
        repository_ctx,
        _CLANG_CUDA_COMPILER_PATH,
    ) or get_host_environ(repository_ctx, _CC)
    if cc_name_from_env:
        cc_name = cc_name_from_env
    cc = which(repository_ctx, cc_name, allow_failure = True)
    if not cc:
        # Use print instead of fail because fail interrupts execution,
        # but tensorflow needs an empty toolchain when the compiler is not found.
        print(("Cannot find {}, either correct your path," +
               " or set the CLANG_CUDA_COMPILER_PATH or CC" +
               " environment variables").format(cc_name))  # buildifier: disable=print
    return None if not cc else realpath(repository_ctx, cc)

def _auto_configure_fail(msg):
    """Output failure message when cuda configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sCuda Configuration Error:%s %s\n" % (red, no_color, msg))

def _verify_build_defines(params):
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
        _auto_configure_fail(
            "BUILD.tpl template is missing these variables: " +
            str(missing) +
            ".\nWe only got: " +
            str(params) +
            ".",
        )

def get_cuda_version(repository_ctx):
    return (get_host_environ(repository_ctx, HERMETIC_CUDA_VERSION) or
            get_host_environ(repository_ctx, TF_CUDA_VERSION))

def _is_clang(cc):
    return "clang" in cc

# Function works only in pair with non-hermetic toolchain
def _get_clang_major_version(repository_ctx, cc):
    """Gets the major version of the clang at `cc`"""
    cmd = "echo __clang_major__ | \"%s\" -E -P -" % cc
    result = execute(
        repository_ctx,
        [get_bash_bin(repository_ctx), "-c", cmd],
    )
    return result.stdout.strip()

# Function works only in pair with non-hermetic toolchain
def _get_cpu_compiler(repository_ctx):
    if _use_hermetic_toolchains(repository_ctx):
        return "clang"

    cc = _find_cc(repository_ctx)
    if not _is_clang(cc):
        # We support builds by clang only
        return "clang"

    return "clang {}".format(_get_clang_major_version(repository_ctx, cc))

def _is_linux_x86_64(repository_ctx):
    return repository_ctx.os.arch == "amd64" and repository_ctx.os.name == "linux"

def _use_hermetic_toolchains(repository_ctx):
    return _flag_enabled(repository_ctx, USE_HERMETIC_CC_TOOLCHAIN)

def enable_cuda(repository_ctx):
    """Returns whether to build with CUDA support."""
    return int(get_host_environ(repository_ctx, TF_NEED_CUDA, False))

def use_cuda_redistributions(repository_ctx):
    """Returns whether to use CUDA redistributions."""
    return (int(get_host_environ(repository_ctx, USE_CUDA_REDISTRIBUTIONS, False)) and
            not int(get_host_environ(repository_ctx, _TF_NEED_ROCM, False)))

def _flag_enabled(repository_ctx, flag_name):
    return get_host_environ(repository_ctx, flag_name) == "1"

def _use_nvcc_and_clang(repository_ctx):
    # Returns the flag if we need to use clang for C++ and NVCC for Cuda.
    return _flag_enabled(repository_ctx, _TF_NVCC_CLANG)

def _use_nvcc_for_cuda(repository_ctx):
    # Returns the flag if we need to use NVCC for Cuda.
    return _flag_enabled(repository_ctx, _CUDA_NVCC)

def _tf_sysroot(repository_ctx):
    tf_sys_root = get_host_environ(repository_ctx, _TF_SYSROOT, "")
    if repository_ctx.path(tf_sys_root).exists:
        return tf_sys_root
    return ""

def _py_tmpl_dict(d):
    return {"%{cuda_config}": str(d)}

def _cudart_static_linkopt(cpu_value):
    """Returns additional platform-specific linkopts for cudart."""
    return "\"\"," if cpu_value == "Darwin" else "\"-lrt\","

def _compute_capabilities(repository_ctx):
    """Returns a list of strings representing cuda compute capabilities.

    Args:
      repository_ctx: the repo rule's context.

    Returns:
      list of cuda architectures to compile for. 'compute_xy' refers to
      both PTX and SASS, 'sm_xy' refers to SASS only.
    """
    capabilities = (get_host_environ(
                        repository_ctx,
                        _HERMETIC_CUDA_COMPUTE_CAPABILITIES,
                    ) or
                    get_host_environ(
                        repository_ctx,
                        _TF_CUDA_COMPUTE_CAPABILITIES,
                    ))
    capabilities = (capabilities or "compute_35,compute_52").split(",")

    # Map old 'x.y' capabilities to 'compute_xy'.
    if len(capabilities) > 0 and all([len(x.split(".")) == 2 for x in capabilities]):
        # If all capabilities are in 'x.y' format, only include PTX for the
        # highest capability.
        cc_list = sorted([x.replace(".", "") for x in capabilities])
        capabilities = [
            "sm_%s" % x
            for x in cc_list[:-1]
        ] + ["compute_%s" % cc_list[-1]]
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
            _auto_configure_fail("Invalid compute capability: %s" % capability)
        for prefix in ["compute_", "sm_"]:
            if not capability.startswith(prefix):
                continue
            version = capability[len(prefix):]

            # Allow PTX accelerated features: sm_90a, sm_100a, etc.
            if version.endswith("a"):
                version = version[:-1]
            if version.isdigit() and len(version) in (2, 3):
                continue
            _auto_configure_fail("Invalid compute capability: %s" % capability)

    return capabilities

def _ptx_version_to_int(version):
    major, minor = version.split(".")
    return int(major + minor)

def _get_cuda_ptx_version(cuda_version, clang_version):
    cuda_version = ".".join(cuda_version.split(".")[:2])
    clang_dict = PTX_VERSION_DICT["clang"]
    cuda_dict = PTX_VERSION_DICT["cuda"]
    if clang_version not in clang_dict:
        _auto_configure_fail(
            ("The supported Clang versions are {supported_versions}. Please" +
             " add max PTX version supported by Clang major version={version}.")
                .format(
                supported_versions = clang_dict.keys(),
                version = clang_version,
            ),
        )
    if cuda_version not in cuda_dict:
        _auto_configure_fail(
            ("The supported CUDA versions are {supported_versions}. Please" +
             " add max PTX version supported by CUDA version={version}.")
                .format(
                supported_versions = cuda_dict.keys(),
                version = cuda_version,
            ),
        )
    ptx_version = min(
        _ptx_version_to_int(clang_dict[clang_version]),
        _ptx_version_to_int(cuda_dict[cuda_version]),
    )

    return ptx_version

def _create_cuda_copts_list(compute_capabilities):
    copts = []

    for capability in compute_capabilities:
        if capability.startswith("compute_"):
            capability = capability.replace("compute_", "sm_")
            copts.append("--cuda-include-ptx=%s" % capability)
        copts.append("--cuda-gpu-arch=%s" % capability)
    return copts

# Function works only in pair with non-hermetic toolchain
def _create_cuda_ptx_copts_list(repository_ctx, cuda_version):
    copts = []

    if _use_hermetic_toolchains(repository_ctx):
        return copts

    cc = _find_cc(repository_ctx)
    if not _is_clang(cc):
        return copts

    ptx_version = _get_cuda_ptx_version(
        cuda_version,
        _get_clang_major_version(repository_ctx, cc),
    )

    copts.append("--no-cuda-include-ptx=all")
    copts.append(
        "--cuda-feature=+ptx{version}".format(version = ptx_version),
    )

    return copts

def _get_cuda_config(repository_ctx):
    """Detects and returns information about the CUDA installation on the system.

      Args:
        repository_ctx: The repository context.

      Returns:
        A struct containing the following fields:
          cuda_version: The version of CUDA on the system.
          cudart_version: The CUDA runtime version on the system.
          cudnn_version: The version of cuDNN on the system.
          compute_capabilities: A list of the system's CUDA compute capabilities.
          cpu_value: The name of the host operating system.
      """

    return struct(
        cuda_version = get_cuda_version(repository_ctx),
        cupti_version = _cupti_version,
        cudart_version = _cudart_version,
        cublas_version = _cublas_version,
        cusolver_version = _cusolver_version,
        curand_version = _curand_version,
        cufft_version = _cufft_version,
        cusparse_version = _cusparse_version,
        cudnn_version = _cudnn_version,
        cccl_version = _cccl_version,
        nvcc_version = _nvcc_version,
        nvdisasm_version = _nvdisasm_version,
        nvjitlink_version = _nvjitlink_version,
        nvml_version = _nvml_version,
        nvtx_version = _nvtx_version,
        compute_capabilities = _compute_capabilities(repository_ctx),
        cpu_value = get_cpu_value(repository_ctx),
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

def _cuda_include_paths(repository_ctx):
    return ["%s/include" % repository_ctx.path(f).dirname for f in [
        repository_ctx.attr.cccl_version,
        repository_ctx.attr.cublas_version,
        repository_ctx.attr.cudart_version,
        repository_ctx.attr.cudnn_version,
        repository_ctx.attr.cufft_version,
        repository_ctx.attr.cupti_version,
        repository_ctx.attr.curand_version,
        repository_ctx.attr.cusolver_version,
        repository_ctx.attr.cusparse_version,
        repository_ctx.attr.nvcc_version,
        repository_ctx.attr.nvjitlink_version,
        repository_ctx.attr.nvml_version,
        repository_ctx.attr.nvtx_version,
    ]]

def _create_dummy_toolchains_repository(repository_ctx):
    repository_ctx.file(
        "crosstool/error_gpu_disabled.bzl",
        _DUMMY_CROSSTOOL_BZL_FILE,
    )
    repository_ctx.file("crosstool/BUILD", _DUMMY_CROSSTOOL_BUILD_FILE)

def _create_local_toolchains_repository(repository_ctx):
    cc = _find_cc(repository_ctx)
    if not cc:
        return False

    is_nvcc_and_clang = _use_nvcc_and_clang(repository_ctx)
    is_nvcc_for_cuda = _use_nvcc_for_cuda(repository_ctx)
    tf_sysroot = _tf_sysroot(repository_ctx)

    host_compiler_includes = get_cxx_inc_directories(
        repository_ctx,
        cc,
        tf_sysroot,
    )

    cuda_defines = {}

    # We do not support hermetic CUDA on Windows.
    # This ensures the CROSSTOOL file parser is happy.
    cuda_defines.update({
        "%{msvc_env_tmp}": "msvc_not_used",
        "%{msvc_env_path}": "msvc_not_used",
        "%{msvc_env_include}": "msvc_not_used",
        "%{msvc_env_lib}": "msvc_not_used",
        "%{msvc_cl_path}": "msvc_not_used",
        "%{msvc_ml_path}": "msvc_not_used",
        "%{msvc_link_path}": "msvc_not_used",
        "%{msvc_lib_path}": "msvc_not_used",
        "%{win_compiler_deps}": ":empty",
    })

    cuda_defines["%{builtin_sysroot}"] = tf_sysroot
    is_clang_compiler = _is_clang(cc)
    if not enable_cuda(repository_ctx):
        cuda_defines["%{cuda_toolkit_path}"] = ""
        cuda_defines["%{cuda_nvcc_files}"] = "[]"
    else:
        if is_clang_compiler:
            cuda_defines["%{cuda_toolkit_path}"] = repository_ctx.attr.nvcc_binary.workspace_root
        else:
            cuda_defines["%{cuda_toolkit_path}"] = ""
        cuda_defines["%{cuda_nvcc_files}"] = "if_cuda([\"@{nvcc_archive}//:bin\", \"@{nvcc_archive}//:nvvm\"])".format(
            nvcc_archive = repository_ctx.attr.nvcc_binary.repo_name,
        )
    if is_clang_compiler:
        cuda_defines["%{compiler}"] = "clang"
        cuda_defines["%{extra_no_canonical_prefixes_flags}"] = ""
        cuda_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(
            host_compiler_includes,
        )
    else:
        cuda_defines["%{compiler}"] = "unknown"
        cuda_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-fno-canonical-system-headers\""
        cuda_includes = []
        if enable_cuda(repository_ctx):
            cuda_includes = _cuda_include_paths(repository_ctx)
        cuda_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(
            host_compiler_includes + cuda_includes,
        )
    cuda_defines["%{host_compiler_prefix}"] = "/usr/bin"
    cuda_defines["%{linker_bin_path}"] = ""
    cuda_defines["%{extra_no_canonical_prefixes_flags}"] = ""
    cuda_defines["%{unfiltered_compile_flags}"] = ""

    if not (is_nvcc_and_clang or is_nvcc_for_cuda):
        cuda_defines["%{host_compiler_path}"] = str(cc)
        cuda_defines["%{host_compiler_warnings}"] = """
          # Some parts of the codebase set -Werror and hit this warning, so
          # switch it off for now.
          "-Wno-invalid-partial-specialization"
      """
        cuda_defines["%{compiler_deps}"] = ":cuda_nvcc_files"

        _create_dummy_nvcc_wrapper(repository_ctx)
    else:
        cuda_defines["%{host_compiler_path}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"
        cuda_defines["%{host_compiler_warnings}"] = ""
        cuda_defines["%{compiler_deps}"] = ":crosstool_wrapper_driver_is_not_gcc"

        _create_nvcc_wrapper(repository_ctx, cc)

    _verify_build_defines(cuda_defines)

    # Only expand template variables in the BUILD file
    repository_ctx.template(
        "crosstool/BUILD",
        repository_ctx.attr.crosstool_build_tpl,
        cuda_defines,
    )

    # No templating of cc_toolchain_config - use attributes and templatize the
    # BUILD file.
    repository_ctx.template(
        "crosstool/cc_toolchain_config.bzl",
        repository_ctx.attr.cc_toolchain_config_tpl,
        {},
    )
    return True

def _create_dummy_nvcc_wrapper(repository_ctx):
    repository_ctx.file(
        "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
        "",
    )

def _create_nvcc_wrapper(repository_ctx, cc):
    cuda_config = _get_cuda_config(repository_ctx)

    nvcc_relative_path = "%s/%s" % (
        repository_ctx.attr.nvcc_binary.workspace_root,
        repository_ctx.attr.nvcc_binary.name,
    )

    wrapper_defines = {
        "%{cpu_compiler}": str(cc),
        "%{cuda_version}": cuda_config.cuda_version,
        "%{nvcc_path}": nvcc_relative_path,
        "%{host_compiler_path}": str(cc),
        "%{use_clang_compiler}": str(True),
        "%{tmpdir}": get_host_environ(
            repository_ctx,
            _TMPDIR,
            "",
        ),
    }

    repository_ctx.template(
        "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
        repository_ctx.attr.crosstool_wrapper_driver_is_not_gcc_tpl,
        wrapper_defines,
    )

def _create_dummy_cuda_repository(repository_ctx):
    cpu_value = get_cpu_value(repository_ctx)

    # Set up BUILD file for cuda/.
    repository_ctx.template(
        "cuda/build_defs.bzl",
        repository_ctx.attr.build_defs_tpl,
        {
            "%{cuda_is_configured}": "False",
            "%{cuda_extra_copts}": "[]",
            "%{cuda_gpu_architectures}": "[]",
            "%{cuda_version}": "0.0",
            "%{use_hermetic_cc_toolchain}": str(_use_hermetic_toolchains(repository_ctx)),
        },
    )

    repository_ctx.template(
        "cuda/BUILD",
        repository_ctx.attr.cuda_build_tpl,
        {
            "%{cudart_static_linkopt}": _cudart_static_linkopt(cpu_value),
        },
    )

    # Set up cuda_config.h, which is used by
    # tensorflow/compiler/xla/stream_executor/dso_loader.cc.
    if use_cuda_redistributions(repository_ctx):
        cuda_config = _get_cuda_config(repository_ctx)
        repository_ctx.template(
            "cuda/cuda/cuda_config.h",
            repository_ctx.attr.cuda_config_tpl,
            {
                "%{cuda_version}": cuda_config.cudart_version,
                "%{cudart_version}": cuda_config.cudart_version,
                "%{cupti_version}": cuda_config.cupti_version,
                "%{cublas_version}": cuda_config.cublas_version,
                "%{cusolver_version}": cuda_config.cusolver_version,
                "%{curand_version}": cuda_config.curand_version,
                "%{cufft_version}": cuda_config.cufft_version,
                "%{cusparse_version}": cuda_config.cusparse_version,
                "%{cudnn_version}": cuda_config.cudnn_version,
                "%{cuda_toolkit_path}": "",
                "%{cuda_compute_capabilities}": "",
            },
        )
    else:
        repository_ctx.template(
            "cuda/cuda/cuda_config.h",
            repository_ctx.attr.cuda_config_tpl,
            {
                "%{cuda_version}": "",
                "%{cudart_version}": "",
                "%{cupti_version}": "",
                "%{cublas_version}": "",
                "%{cusolver_version}": "",
                "%{curand_version}": "",
                "%{cufft_version}": "",
                "%{cusparse_version}": "",
                "%{cudnn_version}": "",
                "%{cuda_toolkit_path}": "",
                "%{cuda_compute_capabilities}": "",
            },
        )

    # Set up cuda_config.py, which is used by gen_build_info to provide
    # static build environment info to the API
    repository_ctx.template(
        "cuda/cuda/cuda_config.py",
        repository_ctx.attr.cuda_config_py_tpl,
        _py_tmpl_dict({}),
    )

def _create_local_cuda_repository(repository_ctx):
    """Creates the repository containing files set up to build with CUDA."""
    cuda_config = _get_cuda_config(repository_ctx)

    # Set up BUILD file for cuda/
    repository_ctx.template(
        "cuda/build_defs.bzl",
        repository_ctx.attr.build_defs_tpl,
        {
            "%{cuda_is_configured}": "True",
            "%{cuda_extra_copts}": str(
                _create_cuda_ptx_copts_list(repository_ctx, cuda_config.cuda_version) +
                _create_cuda_copts_list(cuda_config.compute_capabilities),
            ),
            "%{cuda_gpu_architectures}": str(cuda_config.compute_capabilities),
            "%{cuda_version}": cuda_config.cuda_version,
            "%{use_hermetic_cc_toolchain}": str(_use_hermetic_toolchains(repository_ctx)),
        },
    )

    repository_ctx.template(
        "cuda/BUILD",
        repository_ctx.attr.cuda_build_tpl,
        {
            "%{cudart_static_linkopt}": _cudart_static_linkopt(
                cuda_config.cpu_value,
            ),
        },
    )

    # Set up cuda_config.h, which is used by
    # tensorflow/compiler/xla/stream_executor/dso_loader.cc.
    repository_ctx.template(
        "cuda/cuda/cuda_config.h",
        repository_ctx.attr.cuda_config_tpl,
        {
            "%{cuda_version}": cuda_config.cuda_version,
            "%{cudart_version}": cuda_config.cudart_version,
            "%{cupti_version}": cuda_config.cupti_version,
            "%{cublas_version}": cuda_config.cublas_version,
            "%{cusolver_version}": cuda_config.cusolver_version,
            "%{curand_version}": cuda_config.curand_version,
            "%{cufft_version}": cuda_config.cufft_version,
            "%{cusparse_version}": cuda_config.cusparse_version,
            "%{cudnn_version}": cuda_config.cudnn_version,
            "%{cuda_toolkit_path}": "",
            "%{cuda_compute_capabilities}": ", ".join([
                cc.split("_")[1]
                for cc in cuda_config.compute_capabilities
            ]),
        },
    )

    # Set up cuda_config.py, which is used by gen_build_info to provide
    # static build environment info to the API
    repository_ctx.template(
        "cuda/cuda/cuda_config.py",
        repository_ctx.attr.cuda_config_py_tpl,
        _py_tmpl_dict({
            "cuda_version": cuda_config.cuda_version,
            "cudnn_version": cuda_config.cudnn_version,
            "cuda_compute_capabilities": cuda_config.compute_capabilities,
            "cpu_compiler": _get_cpu_compiler(repository_ctx),
        }),
    )

def _create_cuda_repository(repository_ctx):
    if enable_cuda(repository_ctx):
        _create_local_cuda_repository(repository_ctx)
    else:
        _create_dummy_cuda_repository(repository_ctx)

def _create_toolchains_repository(repository_ctx):
    created = False
    if enable_cuda(repository_ctx) or _is_linux_x86_64(repository_ctx):
        created = _create_local_toolchains_repository(repository_ctx)

    if not created:
        _create_dummy_toolchains_repository(repository_ctx)

def _cuda_configure_impl(repository_ctx):
    """Implementation of the cuda_configure repository rule."""
    build_file = repository_ctx.attr.local_config_cuda_build_file

    _create_cuda_repository(repository_ctx)
    if not _use_hermetic_toolchains(repository_ctx):
        _create_toolchains_repository(repository_ctx)

    repository_ctx.symlink(build_file, "BUILD")

_CC = "CC"
_CLANG_CUDA_COMPILER_PATH = "CLANG_CUDA_COMPILER_PATH"
_HERMETIC_CUDA_COMPUTE_CAPABILITIES = "HERMETIC_CUDA_COMPUTE_CAPABILITIES"
_TF_CUDA_COMPUTE_CAPABILITIES = "TF_CUDA_COMPUTE_CAPABILITIES"
HERMETIC_CUDA_VERSION = "HERMETIC_CUDA_VERSION"
USE_HERMETIC_CC_TOOLCHAIN = "USE_HERMETIC_CC_TOOLCHAIN"
TF_CUDA_VERSION = "TF_CUDA_VERSION"
TF_NEED_CUDA = "TF_NEED_CUDA"
_TF_NEED_ROCM = "TF_NEED_ROCM"
USE_CUDA_REDISTRIBUTIONS = "USE_CUDA_REDISTRIBUTIONS"
_TF_NVCC_CLANG = "TF_NVCC_CLANG"
_CUDA_NVCC = "CUDA_NVCC"
_TF_SYSROOT = "TF_SYSROOT"
_TMPDIR = "TMPDIR"

_ENVIRONS = [
    _CC,
    _CLANG_CUDA_COMPILER_PATH,
    TF_NEED_CUDA,
    _TF_NEED_ROCM,
    _TF_NVCC_CLANG,
    _CUDA_NVCC,
    TF_CUDA_VERSION,
    HERMETIC_CUDA_VERSION,
    _TF_CUDA_COMPUTE_CAPABILITIES,
    _HERMETIC_CUDA_COMPUTE_CAPABILITIES,
    _TF_SYSROOT,
    "TMP",
    _TMPDIR,
    "LOCAL_CUDA_PATH",
    "LOCAL_CUDNN_PATH",
    USE_CUDA_REDISTRIBUTIONS,
]

cuda_configure = repository_rule(
    implementation = _cuda_configure_impl,
    environ = _ENVIRONS,
    attrs = {
        "environ": attr.string_dict(),
        "cccl_version": attr.label(default = Label("@cuda_cccl//:version.bzl")),
        "cublas_version": attr.label(default = Label("@cuda_cublas//:version.bzl")),
        "cudart_version": attr.label(default = Label("@cuda_cudart//:version.bzl")),
        "cudnn_version": attr.label(default = Label("@cuda_cudnn//:version.bzl")),
        "cufft_version": attr.label(default = Label("@cuda_cufft//:version.bzl")),
        "cupti_version": attr.label(default = Label("@cuda_cupti//:version.bzl")),
        "curand_version": attr.label(default = Label("@cuda_curand//:version.bzl")),
        "cusolver_version": attr.label(default = Label("@cuda_cusolver//:version.bzl")),
        "cusparse_version": attr.label(default = Label("@cuda_cusparse//:version.bzl")),
        "nvcc_binary": attr.label(default = Label("@cuda_nvcc//:bin/nvcc")),
        "nvcc_version": attr.label(default = Label("@cuda_nvcc//:version.bzl")),
        "nvjitlink_version": attr.label(default = Label("@cuda_nvjitlink//:version.bzl")),
        "nvml_version": attr.label(default = Label("@cuda_nvml//:version.bzl")),
        "nvtx_version": attr.label(default = Label("@cuda_nvtx//:version.bzl")),
        "local_config_cuda_build_file": attr.label(default = Label("//third_party/gpus:local_config_cuda.BUILD")),
        "build_defs_tpl": attr.label(default = Label("//third_party/gpus/cuda:build_defs.bzl.tpl")),
        "cuda_build_tpl": attr.label(default = Label("//third_party/gpus/cuda/hermetic:BUILD.tpl")),
        "cuda_config_tpl": attr.label(default = Label("//third_party/gpus/cuda:cuda_config.h.tpl")),
        "cuda_config_py_tpl": attr.label(default = Label("//third_party/gpus/cuda:cuda_config.py.tpl")),
        "crosstool_wrapper_driver_is_not_gcc_tpl": attr.label(default = Label("//third_party/gpus/crosstool:clang/bin/crosstool_wrapper_driver_is_not_gcc.tpl")),
        "crosstool_build_tpl": attr.label(default = Label("//third_party/gpus/crosstool:BUILD.tpl")),
        "cc_toolchain_config_tpl": attr.label(default = Label("//third_party/gpus/crosstool:cc_toolchain_config.bzl.tpl")),
    },
)

"""Detects and configures the hermetic CUDA toolchain.

Add the following to your WORKSPACE file:

```python
cuda_configure(name = "local_config_cuda")
```

Args:
  name: A unique name for this workspace rule.
"""  # buildifier: disable=no-effect
