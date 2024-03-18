"""Repository rule for CUDA autoconfiguration.

`cuda_configure` depends on the following environment variables:

  * `TF_NEED_CUDA`: Whether to enable building with CUDA.
  * `GCC_HOST_COMPILER_PATH`: The GCC host compiler path
  * `TF_CUDA_CLANG`: Whether to use clang for C++ and Cuda compilation.
  * `TF_NVCC_CLANG`: Whether to use clang for C++ and NVCC for Cuda compilation.
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
    "//third_party/remote_config:common.bzl",
    "config_repo_label",
    "err_out",
    "execute",
    "get_bash_bin",
    "get_cpu_value",
    "get_host_environ",
    "get_python_bin",
    "is_windows",
    "raw_exec",
    "read_dir",
    "realpath",
)
load(
    ":cuda_common_tools.bzl",
    "CLANG_CUDA_COMPILER_PATH",
    "CUDA_TOOLKIT_PATH",
    "CUDNN_INSTALL_PATH",
    "GCC_HOST_COMPILER_PATH",
    "GCC_HOST_COMPILER_PREFIX",
    "MSVC_ENVVARS",
    "PYTHON_BIN_PATH",
    "TF_CUDA_CLANG",
    "TF_CUDA_COMPUTE_CAPABILITIES",
    "TF_CUDA_CONFIG_REPO",
    "TF_CUDA_VERSION",
    "TF_CUDNN_VERSION",
    "TF_DOWNLOAD_CLANG",
    "TF_NEED_CUDA",
    "TF_NVCC_CLANG",
    "auto_configure_fail",
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
    "use_cuda_clang",
    "use_nvcc_and_clang",
    "verify_build_defines",
)

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

    # The expected exit code of this command is non-zero. Bazel remote execution
    # only caches commands with zero exit code. So force a zero exit code.
    cmd = "%s -v /dev/null -o /dev/null ; [ $? -eq 1 ]" % str(nvcc_path)
    result = raw_exec(repository_ctx, [get_bash_bin(repository_ctx), "-c", cmd])
    target_dir = ""
    for one_line in err_out(result).splitlines():
        if one_line.startswith("#$ _TARGET_DIR_="):
            target_dir = (
                cuda_config.cuda_toolkit_path + "/" + one_line.replace(
                    "#$ _TARGET_DIR_=",
                    "",
                ) + "/include"
            )
    inc_entries = []
    if target_dir != "":
        inc_entries.append(realpath(repository_ctx, target_dir))
    inc_entries.append(realpath(repository_ctx, cuda_config.cuda_toolkit_path + "/include"))
    return inc_entries

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

def _lib_path(lib, cpu_value, basedir, version, static):
    file_name = lib_name(lib, cpu_value, version, static)
    return "%s/%s" % (basedir, file_name)

def _should_check_soname(version, static):
    return version and not static

def _check_cuda_lib_params(lib, cpu_value, basedir, version, static = False):
    return (
        _lib_path(lib, cpu_value, basedir, version, static),
        _should_check_soname(version, static),
    )

def _check_cuda_libs(repository_ctx, script_path, libs):
    python_bin = get_python_bin(repository_ctx)
    contents = repository_ctx.read(script_path).splitlines()

    cmd = "from os import linesep;"
    cmd += "f = open('script.py', 'w');"
    for line in contents:
        cmd += "f.write('%s' + linesep);" % line
    cmd += "f.close();"
    cmd += "from os import system;"
    args = " ".join(["\"" + path + "\" " + str(check) for path, check in libs])
    cmd += "system('%s script.py %s');" % (python_bin, args)

    all_paths = [path for path, _ in libs]
    checked_paths = execute(repository_ctx, [python_bin, "-c", cmd]).stdout.splitlines()

    # Filter out empty lines from splitting on '\r\n' on Windows
    checked_paths = [path for path in checked_paths if len(path) > 0]
    if all_paths != checked_paths:
        auto_configure_fail("Error with installed CUDA libs. Expected '%s'. Actual '%s'." % (all_paths, checked_paths))

def _find_libs(repository_ctx, check_cuda_libs_script, cuda_config):
    """Returns the CUDA and cuDNN libraries on the system.

      Also, verifies that the script actually exist.

      Args:
        repository_ctx: The repository context.
        check_cuda_libs_script: The path to a script verifying that the cuda
          libraries exist on the system.
        cuda_config: The CUDA config as returned by _get_cuda_config

      Returns:
        Map of library names to structs of filename and path.
      """
    cpu_value = cuda_config.cpu_value
    stub_dir = "" if is_windows(repository_ctx) else "/stubs"

    check_cuda_libs_params = {
        "cuda": _check_cuda_lib_params(
            "cuda",
            cpu_value,
            cuda_config.config["cuda_library_dir"] + stub_dir,
            version = None,
            static = False,
        ),
        "cudart": _check_cuda_lib_params(
            "cudart",
            cpu_value,
            cuda_config.config["cuda_library_dir"],
            cuda_config.cudart_version,
            static = False,
        ),
        "cudart_static": _check_cuda_lib_params(
            "cudart_static",
            cpu_value,
            cuda_config.config["cuda_library_dir"],
            cuda_config.cudart_version,
            static = True,
        ),
        "cublas": _check_cuda_lib_params(
            "cublas",
            cpu_value,
            cuda_config.config["cublas_library_dir"],
            cuda_config.cublas_version,
            static = False,
        ),
        "cublasLt": _check_cuda_lib_params(
            "cublasLt",
            cpu_value,
            cuda_config.config["cublas_library_dir"],
            cuda_config.cublas_version,
            static = False,
        ),
        "cusolver": _check_cuda_lib_params(
            "cusolver",
            cpu_value,
            cuda_config.config["cusolver_library_dir"],
            cuda_config.cusolver_version,
            static = False,
        ),
        "curand": _check_cuda_lib_params(
            "curand",
            cpu_value,
            cuda_config.config["curand_library_dir"],
            cuda_config.curand_version,
            static = False,
        ),
        "cufft": _check_cuda_lib_params(
            "cufft",
            cpu_value,
            cuda_config.config["cufft_library_dir"],
            cuda_config.cufft_version,
            static = False,
        ),
        "cudnn": _check_cuda_lib_params(
            "cudnn",
            cpu_value,
            cuda_config.config["cudnn_library_dir"],
            cuda_config.cudnn_version,
            static = False,
        ),
        "cupti": _check_cuda_lib_params(
            "cupti",
            cpu_value,
            cuda_config.config["cupti_library_dir"],
            cuda_config.cupti_version,
            static = False,
        ),
        "cusparse": _check_cuda_lib_params(
            "cusparse",
            cpu_value,
            cuda_config.config["cusparse_library_dir"],
            cuda_config.cusparse_version,
            static = False,
        ),
    }

    # Verify that the libs actually exist at their locations.
    _check_cuda_libs(repository_ctx, check_cuda_libs_script, check_cuda_libs_params.values())

    paths = {filename: v[0] for (filename, v) in check_cuda_libs_params.items()}
    return paths

# TODO(csigg): Only call once instead of from here, tensorrt_configure.bzl,
# and nccl_configure.bzl.
def find_cuda_config(repository_ctx, cuda_libraries):
    """Returns CUDA config dictionary from running find_cuda_config.py"""
    python_bin = get_python_bin(repository_ctx)
    exec_result = execute(repository_ctx, [python_bin, repository_ctx.attr._find_cuda_config] + cuda_libraries)
    if exec_result.return_code:
        auto_configure_fail("Failed to run find_cuda_config.py: %s" % err_out(exec_result))

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
          cudart_version: The CUDA runtime version on the system.
          cudnn_version: The version of cuDNN on the system.
          compute_capabilities: A list of the system's CUDA compute capabilities.
          cpu_value: The name of the host operating system.
      """
    config = find_cuda_config(repository_ctx, ["cuda", "cudnn"])
    cpu_value = get_cpu_value(repository_ctx)
    toolkit_path = config["cuda_toolkit_path"]

    is_windows = cpu_value == "Windows"
    cuda_version = config["cuda_version"].split(".")
    cuda_major = cuda_version[0]
    cuda_minor = cuda_version[1]

    cuda_version = ("64_%s%s" if is_windows else "%s.%s") % (cuda_major, cuda_minor)
    cudnn_version = ("64_%s" if is_windows else "%s") % config["cudnn_version"]

    if int(cuda_major) >= 11:
        # The libcudart soname in CUDA 11.x is versioned as 11.0 for backward compatability.
        if int(cuda_major) == 11:
            cudart_version = "64_110" if is_windows else "11.0"
            cupti_version = cuda_version
        else:
            cudart_version = ("64_%s" if is_windows else "%s") % cuda_major
            cupti_version = cudart_version
        cublas_version = ("64_%s" if is_windows else "%s") % config["cublas_version"].split(".")[0]
        cusolver_version = ("64_%s" if is_windows else "%s") % config["cusolver_version"].split(".")[0]
        curand_version = ("64_%s" if is_windows else "%s") % config["curand_version"].split(".")[0]
        cufft_version = ("64_%s" if is_windows else "%s") % config["cufft_version"].split(".")[0]
        cusparse_version = ("64_%s" if is_windows else "%s") % config["cusparse_version"].split(".")[0]
    elif (int(cuda_major), int(cuda_minor)) >= (10, 1):
        # cuda_lib_version is for libraries like cuBLAS, cuFFT, cuSOLVER, etc.
        # It changed from 'x.y' to just 'x' in CUDA 10.1.
        cuda_lib_version = ("64_%s" if is_windows else "%s") % cuda_major
        cudart_version = cuda_version
        cupti_version = cuda_version
        cublas_version = cuda_lib_version
        cusolver_version = cuda_lib_version
        curand_version = cuda_lib_version
        cufft_version = cuda_lib_version
        cusparse_version = cuda_lib_version
    else:
        cudart_version = cuda_version
        cupti_version = cuda_version
        cublas_version = cuda_version
        cusolver_version = cuda_version
        curand_version = cuda_version
        cufft_version = cuda_version
        cusparse_version = cuda_version

    return struct(
        cuda_toolkit_path = toolkit_path,
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
            "%{cuda_gpu_architectures}": "[]",
            "%{cuda_version}": "0.0",
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
    _tpl(
        repository_ctx,
        "cuda:cuda_config.h",
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
        "cuda/cuda/cuda_config.h",
    )

    # Set up cuda_config.py, which is used by gen_build_info to provide
    # static build environment info to the API
    _tpl(
        repository_ctx,
        "cuda:cuda_config.py",
        py_tmpl_dict({}),
        "cuda/cuda/cuda_config.py",
    )

    # If cuda_configure is not configured to build with GPU support, and the user
    # attempts to build with --config=cuda, add a dummy build rule to intercept
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

def make_copy_dir_rule(repository_ctx, name, src_dir, out_dir, exceptions = None):
    """Returns a rule to recursively copy a directory.
    If exceptions is not None, it must be a list of files or directories in
    'src_dir'; these will be excluded from copying.
    """
    src_dir = _norm_path(src_dir)
    out_dir = _norm_path(out_dir)
    outs = read_dir(repository_ctx, src_dir)
    post_cmd = ""
    if exceptions != None:
        outs = [x for x in outs if not any([
            x.startswith(src_dir + "/" + y)
            for y in exceptions
        ])]
    outs = [('        "%s",' % out.replace(src_dir, out_dir)) for out in outs]

    # '@D' already contains the relative path for a single file, see
    # http://docs.bazel.build/versions/master/be/make-variables.html#predefined_genrule_variables
    out_dir = "$(@D)/%s" % out_dir if len(outs) > 1 else "$(@D)"
    if exceptions != None:
        for x in exceptions:
            post_cmd += " ; rm -fR " + out_dir + "/" + x
    return """genrule(
    name = "%s",
    outs = [
%s
    ],
    cmd = \"""cp -rLf "%s/." "%s/" %s\""",
)""" % (name, "\n".join(outs), src_dir, out_dir, post_cmd)

def _tpl_path(repository_ctx, filename):
    return repository_ctx.path(Label("//third_party/gpus/%s.tpl" % filename))

def _basename(repository_ctx, path_str):
    """Returns the basename of a path of type string.

    This method is different from path.basename in that it also works if
    the host platform is different from the execution platform
    i.e. linux -> windows.
    """

    num_chars = len(path_str)
    is_win = is_windows(repository_ctx)
    for i in range(num_chars):
        r_i = num_chars - 1 - i
        if (is_win and path_str[r_i] == "\\") or path_str[r_i] == "/":
            return path_str[r_i + 1:]
    return path_str

def _create_local_cuda_repository(repository_ctx):
    """Creates the repository containing files set up to build with CUDA."""

    # Resolve all labels before doing any real work. Resolving causes the
    # function to be restarted with all previous state being lost. This
    # can easily lead to a O(n^2) runtime in the number of labels.
    # See https://github.com/tensorflow/tensorflow/commit/62bd3534525a036f07d9851b3199d68212904778
    tpl_paths = {filename: _tpl_path(repository_ctx, filename) for filename in [
        "cuda:build_defs.bzl",
        "crosstool:clang/bin/crosstool_wrapper_driver_is_not_gcc",
        "crosstool:windows/msvc_wrapper_for_nvcc.py",
        "crosstool:BUILD",
        "crosstool:cc_toolchain_config.bzl",
        "cuda:cuda_config.h",
        "cuda:cuda_config.py",
    ]}
    tpl_paths["cuda:BUILD"] = _tpl_path(repository_ctx, "cuda:BUILD.windows" if is_windows(repository_ctx) else "cuda:BUILD")

    cuda_config = _get_cuda_config(repository_ctx)

    cuda_include_path = cuda_config.config["cuda_include_dir"]
    cublas_include_path = cuda_config.config["cublas_include_dir"]
    cudnn_header_dir = cuda_config.config["cudnn_include_dir"]
    cupti_header_dir = cuda_config.config["cupti_include_dir"]
    nvvm_libdevice_dir = cuda_config.config["nvvm_library_dir"]
    nvml_header_dir = cuda_config.config["nvml_header_dir"]

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
        make_copy_dir_rule(
            repository_ctx,
            name = "nvml",
            src_dir = nvml_header_dir,
            out_dir = "cuda/nvml/include",
        ),
    ]

    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "cublas-include",
        srcs = [
            cublas_include_path + "/cublas.h",
            cublas_include_path + "/cublas_v2.h",
            cublas_include_path + "/cublas_api.h",
            cublas_include_path + "/cublasLt.h",
        ],
        outs = [
            "cublas/include/cublas.h",
            "cublas/include/cublas_v2.h",
            "cublas/include/cublas_api.h",
            "cublas/include/cublasLt.h",
        ],
    ))

    cusolver_include_path = cuda_config.config["cusolver_include_dir"]
    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "cusolver-include",
        srcs = [
            cusolver_include_path + "/cusolver_common.h",
            cusolver_include_path + "/cusolverDn.h",
        ],
        outs = [
            "cusolver/include/cusolver_common.h",
            "cusolver/include/cusolverDn.h",
        ],
    ))

    cufft_include_path = cuda_config.config["cufft_include_dir"]
    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "cufft-include",
        srcs = [
            cufft_include_path + "/cufft.h",
        ],
        outs = [
            "cufft/include/cufft.h",
        ],
    ))

    cusparse_include_path = cuda_config.config["cusparse_include_dir"]
    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "cusparse-include",
        srcs = [
            cusparse_include_path + "/cusparse.h",
        ],
        outs = [
            "cusparse/include/cusparse.h",
        ],
    ))

    curand_include_path = cuda_config.config["curand_include_dir"]
    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "curand-include",
        srcs = [
            curand_include_path + "/curand.h",
        ],
        outs = [
            "curand/include/curand.h",
        ],
    ))

    check_cuda_libs_script = repository_ctx.path(Label("@local_tsl//third_party/gpus:check_cuda_libs.py"))
    cuda_libs = _find_libs(repository_ctx, check_cuda_libs_script, cuda_config)
    cuda_lib_srcs = []
    cuda_lib_outs = []
    for path in cuda_libs.values():
        cuda_lib_srcs.append(path)
        cuda_lib_outs.append("cuda/lib/" + _basename(repository_ctx, path))
    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "cuda-lib",
        srcs = cuda_lib_srcs,
        outs = cuda_lib_outs,
    ))

    # copy files mentioned in third_party/nccl/build_defs.bzl.tpl
    file_ext = ".exe" if is_windows(repository_ctx) else ""
    bin_files = (
        ["crt/link.stub"] +
        [f + file_ext for f in ["bin2c", "fatbinary", "nvlink", "nvprune"]]
    )
    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "cuda-bin",
        srcs = [cuda_config.cuda_toolkit_path + "/bin/" + f for f in bin_files],
        outs = ["cuda/bin/" + f for f in bin_files],
    ))

    # Select the headers based on the cuDNN version (strip '64_' for Windows).
    cudnn_headers = ["cudnn.h"]
    if cuda_config.cudnn_version.rsplit("_", 1)[-1] >= "9":
        cudnn_headers += [
            "cudnn_adv.h",
            "cudnn_backend.h",
            "cudnn_cnn.h",
            "cudnn_graph.h",
            "cudnn_ops.h",
            "cudnn_version.h",
        ]
    elif cuda_config.cudnn_version.rsplit("_", 1)[-1] >= "8":
        cudnn_headers += [
            "cudnn_backend.h",
            "cudnn_adv_infer.h",
            "cudnn_adv_train.h",
            "cudnn_cnn_infer.h",
            "cudnn_cnn_train.h",
            "cudnn_ops_infer.h",
            "cudnn_ops_train.h",
            "cudnn_version.h",
        ]

    cudnn_srcs = []
    cudnn_outs = []
    for header in cudnn_headers:
        cudnn_srcs.append(cudnn_header_dir + "/" + header)
        cudnn_outs.append("cudnn/include/" + header)

    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "cudnn-include",
        srcs = cudnn_srcs,
        outs = cudnn_outs,
    ))

    # Set up BUILD file for cuda/
    repository_ctx.template(
        "cuda/build_defs.bzl",
        tpl_paths["cuda:build_defs.bzl"],
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
        tpl_paths["cuda:BUILD"],
        {
            "%{cuda_driver_lib}": _basename(repository_ctx, cuda_libs["cuda"]),
            "%{cudart_static_lib}": _basename(repository_ctx, cuda_libs["cudart_static"]),
            "%{cudart_static_linkopt}": cudart_static_linkopt(cuda_config.cpu_value),
            "%{cudart_lib}": _basename(repository_ctx, cuda_libs["cudart"]),
            "%{cublas_lib}": _basename(repository_ctx, cuda_libs["cublas"]),
            "%{cublasLt_lib}": _basename(repository_ctx, cuda_libs["cublasLt"]),
            "%{cusolver_lib}": _basename(repository_ctx, cuda_libs["cusolver"]),
            "%{cudnn_lib}": _basename(repository_ctx, cuda_libs["cudnn"]),
            "%{cufft_lib}": _basename(repository_ctx, cuda_libs["cufft"]),
            "%{curand_lib}": _basename(repository_ctx, cuda_libs["curand"]),
            "%{cupti_lib}": _basename(repository_ctx, cuda_libs["cupti"]),
            "%{cusparse_lib}": _basename(repository_ctx, cuda_libs["cusparse"]),
            "%{cub_actual}": cub_actual,
            "%{copy_rules}": "\n".join(copy_rules),
        },
    )

    is_cuda_clang = use_cuda_clang(repository_ctx)
    is_nvcc_and_clang = use_nvcc_and_clang(repository_ctx)
    tf_sys_root = tf_sysroot(repository_ctx)

    should_download_clang = is_cuda_clang and flag_enabled(
        repository_ctx,
        TF_DOWNLOAD_CLANG,
    )
    if should_download_clang:
        download_clang(repository_ctx, "crosstool/extra_tools")

    # Set up crosstool/
    cc = find_cc(repository_ctx, is_cuda_clang)
    cc_fullpath = cc if not should_download_clang else "crosstool/" + cc

    host_compiler_includes = get_cxx_inc_directories(
        repository_ctx,
        cc_fullpath,
        tf_sys_root,
    )
    cuda_defines = {}
    cuda_defines["%{builtin_sysroot}"] = tf_sys_root
    cuda_defines["%{cuda_toolkit_path}"] = ""
    cuda_defines["%{compiler}"] = "unknown"
    if is_cuda_clang:
        cuda_defines["%{cuda_toolkit_path}"] = cuda_config.config["cuda_toolkit_path"]
        cuda_defines["%{compiler}"] = "clang"

    host_compiler_prefix = get_host_environ(repository_ctx, GCC_HOST_COMPILER_PREFIX)
    if not host_compiler_prefix:
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
    cuda_defines["%{additional_files}"] = "[]"
    if is_cuda_clang and not is_nvcc_and_clang:
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
            ) + [cupti_header_dir, cudnn_header_dir, nvml_header_dir],
        )

        # For gcc, do not canonicalize system header paths; some versions of gcc
        # pick the shortest possible path for system includes when creating the
        # .d file - given that includes that are prefixed with "../" multiple
        # time quickly grow longer than the root of the tree, this can lead to
        # bazel's header check failing.
        if not is_cuda_clang:
            cuda_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-fno-canonical-system-headers\""

        file_ext = ".exe" if is_windows(repository_ctx) else ""
        nvcc_path = "%s/nvcc%s" % (cuda_config.config["cuda_binary_dir"], file_ext)
        cuda_defines["%{compiler_deps}"] = ":crosstool_wrapper_driver_is_not_gcc"
        cuda_defines["%{win_compiler_deps}"] = ":windows_msvc_wrapper_files"

        wrapper_defines = {
            "%{cpu_compiler}": str(cc),
            "%{cuda_version}": cuda_config.cuda_version,
            "%{nvcc_path}": nvcc_path,
            "%{host_compiler_path}": str(cc),
            "%{use_clang_compiler}": str(is_nvcc_and_clang),
            "%{nvcc_tmp_dir}": get_nvcc_tmp_dir_for_windows(repository_ctx),
        }
        repository_ctx.template(
            "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
            tpl_paths["crosstool:clang/bin/crosstool_wrapper_driver_is_not_gcc"],
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
            tpl_paths["crosstool:windows/msvc_wrapper_for_nvcc.py"],
            wrapper_defines,
        )

    cuda_defines.update(get_win_cuda_defines(repository_ctx))

    verify_build_defines(cuda_defines)

    # Only expand template variables in the BUILD file
    repository_ctx.template(
        "crosstool/BUILD",
        tpl_paths["crosstool:BUILD"],
        cuda_defines,
    )

    # No templating of cc_toolchain_config - use attributes and templatize the
    # BUILD file.
    repository_ctx.template(
        "crosstool/cc_toolchain_config.bzl",
        tpl_paths["crosstool:cc_toolchain_config.bzl"],
        {},
    )

    # Set up cuda_config.h, which is used by
    # tensorflow/compiler/xla/stream_executor/dso_loader.cc.
    repository_ctx.template(
        "cuda/cuda/cuda_config.h",
        tpl_paths["cuda:cuda_config.h"],
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
            "%{cuda_toolkit_path}": cuda_config.cuda_toolkit_path,
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
        tpl_paths["cuda:cuda_config.py"],
        py_tmpl_dict({
            "cuda_version": cuda_config.cuda_version,
            "cudnn_version": cuda_config.cudnn_version,
            "cuda_compute_capabilities": cuda_config.compute_capabilities,
            "cpu_compiler": str(cc),
        }),
    )

def _create_remote_cuda_repository(repository_ctx, remote_config_repo):
    """Creates pointers to a remotely configured repo set up to build with CUDA."""
    _tpl(
        repository_ctx,
        "cuda:build_defs.bzl",
        {
            "%{cuda_is_configured}": "True",
            "%{cuda_extra_copts}": compute_cuda_extra_copts(
                repository_ctx,
                compute_capabilities(repository_ctx),
            ),
            "%{cuda_version}": get_host_environ(repository_ctx, TF_CUDA_VERSION),
        },
    )
    repository_ctx.template(
        "cuda/BUILD",
        config_repo_label(remote_config_repo, "cuda:BUILD"),
        {},
    )
    repository_ctx.template(
        "cuda/build_defs.bzl",
        config_repo_label(remote_config_repo, "cuda:build_defs.bzl"),
        {},
    )
    repository_ctx.template(
        "cuda/cuda/cuda_config.h",
        config_repo_label(remote_config_repo, "cuda:cuda/cuda_config.h"),
        {},
    )
    repository_ctx.template(
        "cuda/cuda/cuda_config.py",
        config_repo_label(remote_config_repo, "cuda:cuda/cuda_config.py"),
        py_tmpl_dict({}),
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

def _cuda_autoconf_impl(repository_ctx):
    """Implementation of the cuda_autoconf repository rule."""
    build_file = Label("//third_party/gpus:local_config_cuda.BUILD")

    if not enable_cuda(repository_ctx):
        _create_dummy_repository(repository_ctx)
    elif get_host_environ(repository_ctx, TF_CUDA_CONFIG_REPO) != None:
        has_cuda_version = get_host_environ(repository_ctx, TF_CUDA_VERSION) != None
        has_cudnn_version = get_host_environ(repository_ctx, TF_CUDNN_VERSION) != None
        if not has_cuda_version or not has_cudnn_version:
            auto_configure_fail("%s and %s must also be set if %s is specified" %
                                (TF_CUDA_VERSION, TF_CUDNN_VERSION, TF_CUDA_CONFIG_REPO))
        _create_remote_cuda_repository(
            repository_ctx,
            get_host_environ(repository_ctx, TF_CUDA_CONFIG_REPO),
        )
    else:
        _create_local_cuda_repository(repository_ctx)

    repository_ctx.symlink(build_file, "BUILD")

_ENVIRONS = [
    GCC_HOST_COMPILER_PATH,
    GCC_HOST_COMPILER_PREFIX,
    CLANG_CUDA_COMPILER_PATH,
    TF_NEED_CUDA,
    TF_CUDA_CLANG,
    TF_NVCC_CLANG,
    TF_DOWNLOAD_CLANG,
    CUDA_TOOLKIT_PATH,
    CUDNN_INSTALL_PATH,
    TF_CUDA_VERSION,
    TF_CUDNN_VERSION,
    TF_CUDA_COMPUTE_CAPABILITIES,
    "NVVMIR_LIBRARY_DIR",
    PYTHON_BIN_PATH,
    "TMP",
    "TMPDIR",
    "TF_CUDA_PATHS",
] + MSVC_ENVVARS

remote_cuda_configure = repository_rule(
    implementation = _create_local_cuda_repository,
    environ = _ENVIRONS,
    remotable = True,
    attrs = {
        "environ": attr.string_dict(),
        "_find_cuda_config": attr.label(
            default = Label("@local_tsl//third_party/gpus:find_cuda_config.py"),
        ),
    },
)

cuda_configure = repository_rule(
    implementation = _cuda_autoconf_impl,
    environ = _ENVIRONS + [TF_CUDA_CONFIG_REPO],
    attrs = {
        "_find_cuda_config": attr.label(
            default = Label("@local_tsl//third_party/gpus:find_cuda_config.py"),
        ),
    },
)
"""Detects and configures the local CUDA toolchain.

Add the following to your WORKSPACE FILE:

```python
cuda_configure(name = "local_config_cuda")
```

Args:
  name: A unique name for this workspace rule.
"""
