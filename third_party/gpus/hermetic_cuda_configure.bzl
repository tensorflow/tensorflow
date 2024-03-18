"""Repository rule for hermetic CUDA autoconfiguration.

`hermetic_cuda_configure` depends on the following environment variables:

  * `TF_NEED_CUDA`: Whether to enable building with CUDA.
  * `TF_NVCC_CLANG`: Whether to use clang for C++ and NVCC for Cuda compilation.
  * `CLANG_CUDA_COMPILER_PATH`: The clang compiler path that will be used for
    both host and device code compilation.
  * `TF_SYSROOT`: The sysroot to use when compiling.
  * `TF_DOWNLOAD_CLANG`: Whether to download a recent release of clang
    compiler and use it to build tensorflow. When this option is set
    CLANG_CUDA_COMPILER_PATH is ignored.
  * `TF_CUDA_VERSION`: The version of the CUDA toolkit (mandatory).
  * `TF_CUDA_COMPUTE_CAPABILITIES`: The CUDA compute capabilities. Default is
    `3.5,5.2`.
  * `PYTHON_BIN_PATH`: The python binary path
"""

load("//third_party/clang_toolchain:download_clang.bzl", "download_clang")
load(
    "//third_party/remote_config:common.bzl",
    "get_cpu_value",
    "get_host_environ",
    "get_python_bin",
    "is_windows",
)
load(
    ":cuda_common_tools.bzl",
    "CLANG_CUDA_COMPILER_PATH",
    "MSVC_ENVVARS",
    "PYTHON_BIN_PATH",
    "TF_CUDA_COMPUTE_CAPABILITIES",
    "TF_CUDA_VERSION",
    "TF_DOWNLOAD_CLANG",
    "TF_NEED_CUDA",
    "TF_NVCC_CLANG",
    "compute_capabilities",
    "compute_cuda_extra_copts",
    "cudart_static_linkopt",
    "enable_cuda",
    "find_cc",
    "flag_enabled",
    "get_cxx_inc_directories",
    "get_nvcc_tmp_dir_for_windows",
    "get_win_cuda_defines",
    "lib_name",
    "py_tmpl_dict",
    "tf_sysroot",
    "to_list_of_strings",
    "use_nvcc_and_clang",
    "verify_build_defines",
)

def get_cuda_version(repository_ctx):
    return get_host_environ(repository_ctx, TF_CUDA_VERSION)

def _get_absolute_archive_path(repository_ctx, archive_workspace):
    workspace_path = str(repository_ctx.path(archive_workspace))
    ind = workspace_path.find("WORKSPACE")
    return workspace_path[:ind - 1]

def _get_library_version(repository_ctx, archive_workspace):
    workspace_path = _get_absolute_archive_path(repository_ctx, archive_workspace)
    return repository_ctx.read(workspace_path + "/version.txt")

def get_cuda_components_dirs_and_versions(repository_ctx):
    return {
        "cublas_version": _get_library_version(repository_ctx, repository_ctx.attr.cublas_archive_workspace),
        "nvcc_binary_dir": _get_absolute_archive_path(repository_ctx, repository_ctx.attr.nvcc_archive_workspace) + "/bin",
        "cuda_version": get_cuda_version(repository_ctx),
        "cudnn_version": _get_library_version(repository_ctx, repository_ctx.attr.cudnn_archive_workspace),
        "cufft_version": _get_library_version(repository_ctx, repository_ctx.attr.cufft_archive_workspace),
        "curand_version": _get_library_version(repository_ctx, repository_ctx.attr.curand_archive_workspace),
        "cusolver_version": _get_library_version(repository_ctx, repository_ctx.attr.cusolver_archive_workspace),
        "cusparse_version": _get_library_version(repository_ctx, repository_ctx.attr.cusparse_archive_workspace),
    }

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
    config = get_cuda_components_dirs_and_versions(repository_ctx)
    cpu_value = get_cpu_value(repository_ctx)

    cuda_version = config["cuda_version"].split(".")
    cuda_major = cuda_version[0]
    cuda_minor = cuda_version[1]

    cuda_version = "%s.%s" % (cuda_major, cuda_minor)
    cudnn_version = config["cudnn_version"]

    # The libcudart soname in CUDA 11.x is versioned as 11.0 for backward compatibility.
    if int(cuda_major) == 11:
        cudart_version = "11.0"
        cupti_version = cuda_version
    else:
        cudart_version = cuda_major
        cupti_version = cudart_version
    cublas_version = config["cublas_version"].split(".")[0]
    cusolver_version = config["cusolver_version"].split(".")[0]
    curand_version = config["curand_version"].split(".")[0]
    cufft_version = config["cufft_version"].split(".")[0]
    cusparse_version = config["cusparse_version"].split(".")[0]

    return struct(
        cuda_version = cuda_version,
        cupti_version = cupti_version,
        cuda_version_major = cuda_major,
        cudart_version = cudart_version,
        cublas_version = cublas_version,
        cusolver_version = cusolver_version,
        curand_version = curand_version,
        cufft_version = cufft_version,
        cusparse_version = cusparse_version,
        cudnn_version = cudnn_version,
        compute_capabilities = compute_capabilities(repository_ctx),
        cpu_value = cpu_value,
        config = config,
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
    repository_ctx.template(
        "cuda/build_defs.bzl",
        repository_ctx.attr.build_defs_tpl,
        {
            "%{cuda_is_configured}": "False",
            "%{cuda_extra_copts}": "[]",
            "%{cuda_gpu_architectures}": "[]",
            "%{cuda_version}": "0.0",
        },
    )

    repository_ctx.template(
        "cuda/BUILD",
        repository_ctx.attr.dummy_cuda_build_tpl,
        {
            "%{cuda_driver_lib}": lib_name("cuda", cpu_value),
            "%{cudart_static_lib}": lib_name(
                "cudart_static",
                cpu_value,
                static = True,
            ),
            "%{cudart_static_linkopt}": cudart_static_linkopt(cpu_value),
            "%{cudart_lib}": lib_name("cudart", cpu_value),
            "%{cublas_lib}": lib_name("cublas", cpu_value),
            "%{cublasLt_lib}": lib_name("cublasLt", cpu_value),
            "%{cusolver_lib}": lib_name("cusolver", cpu_value),
            "%{cudnn_lib}": lib_name("cudnn", cpu_value),
            "%{cufft_lib}": lib_name("cufft", cpu_value),
            "%{curand_lib}": lib_name("curand", cpu_value),
            "%{cupti_lib}": lib_name("cupti", cpu_value),
            "%{cusparse_lib}": lib_name("cusparse", cpu_value),
            "%{cub_actual}": ":cuda_headers",
            "%{copy_rules}": """
filegroup(name="cuda-include")
filegroup(name="cublas-include")
filegroup(name="cusolver-include")
filegroup(name="cufft-include")
filegroup(name="cusparse-include")
filegroup(name="curand-include")
filegroup(name="cudnn-include")
""",
        },
    )

    # Create dummy files for the CUDA toolkit since they are still required by
    # tensorflow/tsl/platform/default/build_config:cuda.
    repository_ctx.file("cuda/cuda/include/cuda.h")
    repository_ctx.file("cuda/cuda/include/cublas.h")
    repository_ctx.file("cuda/cuda/include/cudnn.h")
    repository_ctx.file("cuda/cuda/extras/CUPTI/include/cupti.h")
    repository_ctx.file("cuda/cuda/nvml/include/nvml.h")
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cuda", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cudart", cpu_value))
    repository_ctx.file(
        "cuda/cuda/lib/%s" % lib_name("cudart_static", cpu_value),
    )
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cublas", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cublasLt", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cusolver", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cudnn", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("curand", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cufft", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cupti", cpu_value))
    repository_ctx.file("cuda/cuda/lib/%s" % lib_name("cusparse", cpu_value))

    # Set up cuda_config.h, which is used by
    # tensorflow/compiler/xla/stream_executor/dso_loader.cc.
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
        py_tmpl_dict({}),
    )

    # If cuda_configure is not configured to build with GPU support, and the user
    # attempts to build with --config=cuda, add a dummy build rule to intercept
    # this and fail with an actionable error message.
    repository_ctx.file(
        "crosstool/error_gpu_disabled.bzl",
        _DUMMY_CROSSTOOL_BZL_FILE,
    )
    repository_ctx.file("crosstool/BUILD", _DUMMY_CROSSTOOL_BUILD_FILE)

def _create_local_cuda_repository(repository_ctx):
    """Creates the repository containing files set up to build with CUDA."""
    cuda_config = _get_cuda_config(repository_ctx)

    # Set up BUILD file for cuda/
    repository_ctx.template(
        "cuda/build_defs.bzl",
        repository_ctx.attr.build_defs_tpl,
        {
            "%{cuda_is_configured}": "True",
            "%{cuda_extra_copts}": compute_cuda_extra_copts(
                repository_ctx,
                cuda_config.compute_capabilities,
            ),
            "%{cuda_gpu_architectures}": str(cuda_config.compute_capabilities),
            "%{cuda_version}": cuda_config.cuda_version,
        },
    )

    cub_actual = "@cub_archive//:cub"
    if int(cuda_config.cuda_version_major) >= 11:
        cub_actual = ":cuda_headers"

    repository_ctx.template(
        "cuda/BUILD",
        repository_ctx.attr.cuda_build_tpl,
        {
            "%{cudart_static_linkopt}": cudart_static_linkopt(cuda_config.cpu_value),
            "%{cub_actual}": cub_actual,
            "%{cccl_repo_name}": repository_ctx.attr.cccl_archive_workspace.repo_name,
            "%{cublas_repo_name}": repository_ctx.attr.cublas_archive_workspace.repo_name,
            "%{cudart_repo_name}": repository_ctx.attr.cudart_archive_workspace.repo_name,
            "%{cudnn_repo_name}": repository_ctx.attr.cudnn_archive_workspace.repo_name,
            "%{cufft_repo_name}": repository_ctx.attr.cufft_archive_workspace.repo_name,
            "%{cupti_repo_name}": repository_ctx.attr.cupti_archive_workspace.repo_name,
            "%{curand_repo_name}": repository_ctx.attr.curand_archive_workspace.repo_name,
            "%{cusolver_repo_name}": repository_ctx.attr.cusolver_archive_workspace.repo_name,
            "%{cusparse_repo_name}": repository_ctx.attr.cusparse_archive_workspace.repo_name,
            "%{nvcc_repo_name}": repository_ctx.attr.nvcc_archive_workspace.repo_name,
            "%{nvjitlink_repo_name}": repository_ctx.attr.nvjitlink_archive_workspace.repo_name,
            "%{nvml_repo_name}": repository_ctx.attr.nvml_archive_workspace.repo_name,
            "%{nvtx_repo_name}": repository_ctx.attr.nvtx_archive_workspace.repo_name,
            "%{nvprune_repo_name}": repository_ctx.attr.nvprune_archive_workspace.repo_name,
        },
    )

    is_nvcc_and_clang = use_nvcc_and_clang(repository_ctx)
    tf_sys_root = tf_sysroot(repository_ctx)

    should_download_clang = flag_enabled(
        repository_ctx,
        TF_DOWNLOAD_CLANG,
    )
    if should_download_clang:
        download_clang(repository_ctx, "crosstool/extra_tools")

    # Set up crosstool/
    cc = find_cc(repository_ctx, use_cuda_clang = True)
    cc_fullpath = cc if not should_download_clang else "crosstool/" + cc

    host_compiler_includes = get_cxx_inc_directories(
        repository_ctx,
        cc_fullpath,
        tf_sys_root,
    )
    cuda_defines = {}
    cuda_defines["%{builtin_sysroot}"] = tf_sys_root
    cuda_defines["%{cuda_toolkit_path}"] = repository_ctx.attr.nvcc_archive_workspace.workspace_root
    cuda_defines["%{compiler}"] = "clang"
    host_compiler_prefix = "/usr/bin"
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
    cuda_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(host_compiler_includes)
    cuda_defines["%{additional_files}"] = "if_cuda([\"@{nvcc_archive}//:bin\", \"@{nvcc_archive}//:nvvm\"])".format(nvcc_archive = repository_ctx.attr.nvcc_archive_workspace.repo_name)
    if not is_nvcc_and_clang:
        cuda_defines["%{host_compiler_path}"] = str(cc)
        cuda_defines["%{host_compiler_warnings}"] = """
        # Some parts of the codebase set -Werror and hit this warning, so
        # switch it off for now.
        "-Wno-invalid-partial-specialization"
    """
        cuda_defines["%{compiler_deps}"] = ":cuda_nvcc_files"
        cuda_defines["%{win_compiler_deps}"] = ":cuda_nvcc_files"
        repository_ctx.file(
            "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
            "",
        )
        repository_ctx.file("crosstool/windows/msvc_wrapper_for_nvcc.py", "")
    else:
        cuda_defines["%{host_compiler_path}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"
        cuda_defines["%{host_compiler_warnings}"] = ""

        file_ext = ".exe" if is_windows(repository_ctx) else ""
        nvcc_path = "%s/nvcc%s" % (cuda_config.config["nvcc_binary_dir"], file_ext)
        nvcc_root = repository_ctx.attr.nvcc_archive_workspace.workspace_root
        nvcc_path_relative = nvcc_path[nvcc_path.find(nvcc_root) + len(nvcc_root):]
        cuda_defines["%{compiler_deps}"] = ":crosstool_wrapper_driver_is_not_gcc"
        cuda_defines["%{win_compiler_deps}"] = ":windows_msvc_wrapper_files"

        wrapper_defines = {
            "%{cpu_compiler}": str(cc),
            "%{cuda_version}": cuda_config.cuda_version,
            "%{nvcc_path}": nvcc_root + nvcc_path_relative,
            "%{host_compiler_path}": str(cc),
            "%{use_clang_compiler}": "True",
            "%{nvcc_tmp_dir}": get_nvcc_tmp_dir_for_windows(repository_ctx),
        }
        repository_ctx.template(
            "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
            repository_ctx.attr.crosstool_wrapper_driver_is_not_gcc_tpl,
            wrapper_defines,
        )

        repository_ctx.file(
            "crosstool/windows/msvc_wrapper_for_nvcc.bat",
            content = "@echo OFF\n{} -B external/local_config_cuda/crosstool/windows/msvc_wrapper_for_nvcc.py %*".format(
                get_python_bin(repository_ctx),
            ),
        )
        repository_ctx.template(
            "crosstool/windows/msvc_wrapper_for_nvcc.py",
            repository_ctx.attr.crosstool_msvc_wrapper_for_nvcc_tpl,
            wrapper_defines,
        )

    cuda_defines.update(get_win_cuda_defines(repository_ctx))

    verify_build_defines(cuda_defines)

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
        py_tmpl_dict({
            "cuda_version": cuda_config.cuda_version,
            "cudnn_version": cuda_config.cudnn_version,
            "cuda_compute_capabilities": cuda_config.compute_capabilities,
            "cpu_compiler": str(cc),
        }),
    )

def _cuda_autoconf_impl(repository_ctx):
    """Implementation of the cuda_autoconf repository rule."""
    build_file = repository_ctx.attr.local_config_cuda_build_file

    if not enable_cuda(repository_ctx):
        _create_dummy_repository(repository_ctx)
    else:
        _create_local_cuda_repository(repository_ctx)

    repository_ctx.symlink(build_file, "BUILD")

_ENVIRONS = [
    CLANG_CUDA_COMPILER_PATH,
    TF_NEED_CUDA,
    TF_NVCC_CLANG,
    TF_DOWNLOAD_CLANG,
    TF_CUDA_VERSION,
    TF_CUDA_COMPUTE_CAPABILITIES,
    PYTHON_BIN_PATH,
    "TMP",
    "TMPDIR",
] + MSVC_ENVVARS

hermetic_cuda_configure = repository_rule(
    implementation = _cuda_autoconf_impl,
    environ = _ENVIRONS,
    attrs = {
        "environ": attr.string_dict(),
        "cccl_archive_workspace": attr.label(default = Label("@cuda_cccl//:WORKSPACE")),
        "cublas_archive_workspace": attr.label(default = Label("@cuda_cublas//:WORKSPACE")),
        "cudart_archive_workspace": attr.label(default = Label("@cuda_cudart//:WORKSPACE")),
        "cudnn_archive_workspace": attr.label(default = Label("@cuda_cudnn//:WORKSPACE")),
        "cufft_archive_workspace": attr.label(default = Label("@cuda_cufft//:WORKSPACE")),
        "cupti_archive_workspace": attr.label(default = Label("@cuda_cupti//:WORKSPACE")),
        "curand_archive_workspace": attr.label(default = Label("@cuda_curand//:WORKSPACE")),
        "cusolver_archive_workspace": attr.label(default = Label("@cuda_cusolver//:WORKSPACE")),
        "cusparse_archive_workspace": attr.label(default = Label("@cuda_cusparse//:WORKSPACE")),
        "nvcc_archive_workspace": attr.label(default = Label("@cuda_nvcc//:WORKSPACE")),
        "nvjitlink_archive_workspace": attr.label(default = Label("@cuda_nvjitlink//:WORKSPACE")),
        "nvml_archive_workspace": attr.label(default = Label("@cuda_nvml//:WORKSPACE")),
        "nvprune_archive_workspace": attr.label(default = Label("@cuda_nvprune//:WORKSPACE")),
        "nvtx_archive_workspace": attr.label(default = Label("@cuda_nvtx//:WORKSPACE")),
        "local_config_cuda_build_file": attr.label(default = Label("//third_party/gpus:local_config_cuda.BUILD")),
        "build_defs_tpl": attr.label(default = Label("//third_party/gpus/cuda:build_defs.bzl.tpl")),
        "cuda_build_tpl": attr.label(default = Label("//third_party/gpus/cuda:BUILD.hermetic.tpl")),
        "dummy_cuda_build_tpl": attr.label(default = Label("//third_party/gpus/cuda:BUILD.tpl")),
        "cuda_config_tpl": attr.label(default = Label("//third_party/gpus/cuda:cuda_config.h.tpl")),
        "cuda_config_py_tpl": attr.label(default = Label("//third_party/gpus/cuda:cuda_config.py.tpl")),
        "crosstool_wrapper_driver_is_not_gcc_tpl": attr.label(default = Label("//third_party/gpus/crosstool:clang/bin/crosstool_wrapper_driver_is_not_gcc.tpl")),
        "crosstool_build_tpl": attr.label(default = Label("//third_party/gpus/crosstool:BUILD.tpl")),
        "cc_toolchain_config_tpl": attr.label(default = Label("//third_party/gpus/crosstool:cc_toolchain_config.bzl.tpl")),
        "crosstool_msvc_wrapper_for_nvcc_tpl": attr.label(default = Label("//third_party/gpus/crosstool:windows/msvc_wrapper_for_nvcc.py.tpl")),
    },
)
"""Detects and configures the hermetic CUDA toolchain.

Add the following to your WORKSPACE FILE:

```python
hermetic cuda_configure(name = "local_config_cuda")
```

Args:
  name: A unique name for this workspace rule.
"""
