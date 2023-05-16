"""Repository rule for ROCm autoconfiguration.

`rocm_configure` depends on the following environment variables:

  * `TF_NEED_ROCM`: Whether to enable building with ROCm.
  * `GCC_HOST_COMPILER_PATH`: The GCC host compiler path
  * `ROCM_PATH`: The path to the ROCm toolkit. Default is `/opt/rocm`.
  * `TF_ROCM_AMDGPU_TARGETS`: The AMDGPU targets.
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
    "get_python_bin",
    "raw_exec",
    "realpath",
    "which",
)


_GCC_HOST_COMPILER_PATH = "GCC_HOST_COMPILER_PATH"
_GCC_HOST_COMPILER_PREFIX = "GCC_HOST_COMPILER_PREFIX"
_ROCM_TOOLKIT_PATH = "ROCM_PATH"
_TF_ROCM_AMDGPU_TARGETS = "TF_ROCM_AMDGPU_TARGETS"
_TF_ROCM_CONFIG_REPO = "TF_ROCM_CONFIG_REPO"

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

def _rocm_include_path(repository_ctx, rocm_config, bash_bin):
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

    # Add HSA headers (needs to match $HSA_PATH)
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hsa/include")

    # Add HIP headers (needs to match $HIP_PATH)
    inc_dirs.append(rocm_config.rocm_toolkit_path + "/hip/include")
    if int(rocm_config.rocm_version_number) >= 50200:
        inc_dirs.append(rocm_config.rocm_toolkit_path + "/include")
        inc_dirs.append(rocm_config.rocm_toolkit_path + "/include/hip")
        inc_dirs.append(rocm_config.rocm_toolkit_path + "/include/rocprim")
        inc_dirs.append(rocm_config.rocm_toolkit_path + "/include/rocsolver")
        inc_dirs.append(rocm_config.rocm_toolkit_path + "/include/rocblas")

    # Add HIP-Clang headers (realpath relative to compiler binary)
    rocm_toolkit_path = realpath(repository_ctx, rocm_config.rocm_toolkit_path, bash_bin)
    inc_dirs.append(rocm_toolkit_path + "/llvm/lib/clang/8.0/include")
    inc_dirs.append(rocm_toolkit_path + "/llvm/lib/clang/9.0.0/include")
    inc_dirs.append(rocm_toolkit_path + "/llvm/lib/clang/10.0.0/include")
    inc_dirs.append(rocm_toolkit_path + "/llvm/lib/clang/11.0.0/include")
    inc_dirs.append(rocm_toolkit_path + "/llvm/lib/clang/12.0.0/include")
    inc_dirs.append(rocm_toolkit_path + "/llvm/lib/clang/13.0.0/include")
    inc_dirs.append(rocm_toolkit_path + "/llvm/lib/clang/14.0.0/include")
    inc_dirs.append(rocm_toolkit_path + "/llvm/lib/clang/15.0.0/include")
    inc_dirs.append(rocm_toolkit_path + "/llvm/lib/clang/16.0.0/include")
    inc_dirs.append(rocm_toolkit_path + "/llvm/lib/clang/17.0.0/include")

    # Support hcc based off clang 10.0.0 (for ROCm 3.3)
    inc_dirs.append(rocm_toolkit_path + "/hcc/compiler/lib/clang/10.0.0/include/")
    inc_dirs.append(rocm_toolkit_path + "/hcc/lib/clang/10.0.0/include")

    # Add hcc headers
    inc_dirs.append(rocm_toolkit_path + "/hcc/include")

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

def _find_libs(repository_ctx, rocm_config, hipfft_or_rocfft, miopen_path, rccl_path, bash_bin):
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
            ("amdhip64", rocm_config.rocm_toolkit_path + "/hip"),
            ("rocblas", rocm_config.rocm_toolkit_path),
            (hipfft_or_rocfft, rocm_config.rocm_toolkit_path),
            ("hiprand", rocm_config.rocm_toolkit_path),
            ("MIOpen", miopen_path),
            ("rccl", rccl_path),
            ("hipsparse", rocm_config.rocm_toolkit_path),
            ("roctracer64", rocm_config.rocm_toolkit_path + "/roctracer"),
            ("rocsolver", rocm_config.rocm_toolkit_path),
        ]
    ]
    if int(rocm_config.rocm_version_number) >= 40500:
        libs_paths.append(("hipsolver", _rocm_lib_paths(repository_ctx, "hipsolver", rocm_config.rocm_toolkit_path)))
        libs_paths.append(("hipblas", _rocm_lib_paths(repository_ctx, "hipblas", rocm_config.rocm_toolkit_path)))
    return _select_rocm_lib_paths(repository_ctx, libs_paths, bash_bin)

def _exec_find_rocm_config(repository_ctx, script_path):
    python_bin = get_python_bin(repository_ctx)

    # If used with remote execution then repository_ctx.execute() can't
    # access files from the source tree. A trick is to read the contents
    # of the file in Starlark and embed them as part of the command. In
    # this case the trick is not sufficient as the find_cuda_config.py
    # script has more than 8192 characters. 8192 is the command length
    # limit of cmd.exe on Windows. Thus we additionally need to compress
    # the contents locally and decompress them as part of the execute().
    compressed_contents = repository_ctx.read(script_path)
    decompress_and_execute_cmd = (
        "from zlib import decompress;" +
        "from base64 import b64decode;" +
        "from os import system;" +
        "script = decompress(b64decode('%s'));" % compressed_contents +
        "f = open('script.py', 'wb');" +
        "f.write(script);" +
        "f.close();" +
        "system('\"%s\" script.py');" % (python_bin)
    )

    return execute(repository_ctx, [python_bin, "-c", decompress_and_execute_cmd])

def find_rocm_config(repository_ctx, script_path):
    """Returns ROCm config dictionary from running find_rocm_config.py"""
    exec_result = _exec_find_rocm_config(repository_ctx, script_path)
    if exec_result.return_code:
        auto_configure_fail("Failed to run find_rocm_config.py: %s" % err_out(exec_result))

    # Parse the dict from stdout.
    return dict([tuple(x.split(": ")) for x in exec_result.stdout.splitlines()])

def _get_rocm_config(repository_ctx, bash_bin, find_rocm_config_script):
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
    config = find_rocm_config(repository_ctx, find_rocm_config_script)
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
            "%{rocm_gpu_architectures}": "[]",
            "%{rocm_version_number}": "0",
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
            "%{hipfft_or_rocfft}": _lib_name("hipfft"),
            "%{hipfft_or_rocfft_lib}": _lib_name("hipfft"),
            "%{hiprand_lib}": _lib_name("hiprand"),
            "%{hipsparse_lib}": _lib_name("hipsparse"),
            "%{roctracer_lib}": _lib_name("roctracer64"),
            "%{rocsolver_lib}": _lib_name("rocsolver"),
            "%{hipsolver_lib}": _lib_name("hipsolver"),
            "%{copy_rules}": "",
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
    amdgpu_target_flags = ["--amdgpu-target=" +
                           amdgpu_target for amdgpu_target in amdgpu_targets]
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

    find_rocm_config_script = repository_ctx.path(Label("@org_tensorflow//third_party/gpus:find_rocm_config.py.gz.base64"))

    bash_bin = get_bash_bin(repository_ctx)
    rocm_config = _get_rocm_config(repository_ctx, bash_bin, find_rocm_config_script)

    # For ROCm 4.1 and above use hipfft, older ROCm versions use rocfft
    rocm_version_number = int(rocm_config.rocm_version_number)
    hipfft_or_rocfft = "rocfft" if rocm_version_number < 40100 else "hipfft"

    # For ROCm 5.2 and above, find MIOpen and RCCL in the main rocm lib path
    miopen_path = rocm_config.rocm_toolkit_path + "/miopen" if rocm_version_number < 50200 else rocm_config.rocm_toolkit_path
    rccl_path = rocm_config.rocm_toolkit_path + "/rccl" if rocm_version_number < 50200 else rocm_config.rocm_toolkit_path

    # Copy header and library files to execroot.
    # rocm_toolkit_path
    rocm_toolkit_path = rocm_config.rocm_toolkit_path
    copy_rules = [
        make_copy_dir_rule(
            repository_ctx,
            name = "rocm-include",
            src_dir = rocm_toolkit_path + "/include",
            out_dir = "rocm/include",
            exceptions = ["gtest", "gmock"],
        ),
    ]

    # explicitly copy (into the local_config_rocm repo) the $ROCM_PATH/hiprand/include and
    # $ROCM_PATH/rocrand/include dirs, only once the softlink to them in $ROCM_PATH/include
    # dir has been removed. This removal will happen in a near-future ROCm release.
    hiprand_include = ""
    hiprand_include_softlink = rocm_config.rocm_toolkit_path + "/include/hiprand"
    softlink_exists = files_exist(repository_ctx, [hiprand_include_softlink], bash_bin)
    if not softlink_exists[0]:
        hiprand_include = '":hiprand-include",\n'
        copy_rules.append(
            make_copy_dir_rule(
                repository_ctx,
                name = "hiprand-include",
                src_dir = rocm_toolkit_path + "/hiprand/include",
                out_dir = "rocm/include/hiprand",
            ),
        )

    rocrand_include = ""
    rocrand_include_softlink = rocm_config.rocm_toolkit_path + "/include/rocrand"
    softlink_exists = files_exist(repository_ctx, [rocrand_include_softlink], bash_bin)
    if not softlink_exists[0]:
        rocrand_include = '":rocrand-include",\n'
        copy_rules.append(
            make_copy_dir_rule(
                repository_ctx,
                name = "rocrand-include",
                src_dir = rocm_toolkit_path + "/rocrand/include",
                out_dir = "rocm/include/rocrand",
            ),
        )

    rocm_libs = _find_libs(repository_ctx, rocm_config, hipfft_or_rocfft, miopen_path, rccl_path, bash_bin)
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

    clang_offload_bundler_path = rocm_toolkit_path + "/llvm/bin/clang-offload-bundler"

    # copy files mentioned in third_party/gpus/rocm/BUILD
    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "rocm-bin",
        srcs = [
            clang_offload_bundler_path,
        ],
        outs = [
            "rocm/bin/" + "clang-offload-bundler",
        ],
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
            "%{rocm_gpu_architectures}": str(rocm_config.amdgpu_targets),
            "%{rocm_version_number}": str(rocm_version_number),
        },
    )

    repository_dict = {
        "%{hip_lib}": rocm_libs["amdhip64"].file_name,
        "%{rocblas_lib}": rocm_libs["rocblas"].file_name,
        "%{hipfft_or_rocfft}": hipfft_or_rocfft,
        "%{hipfft_or_rocfft_lib}": rocm_libs[hipfft_or_rocfft].file_name,
        "%{hiprand_lib}": rocm_libs["hiprand"].file_name,
        "%{miopen_lib}": rocm_libs["MIOpen"].file_name,
        "%{rccl_lib}": rocm_libs["rccl"].file_name,
        "%{hipsparse_lib}": rocm_libs["hipsparse"].file_name,
        "%{roctracer_lib}": rocm_libs["roctracer64"].file_name,
        "%{rocsolver_lib}": rocm_libs["rocsolver"].file_name,
        "%{copy_rules}": "\n".join(copy_rules),
        "%{rocm_headers}": ('":rocm-include",\n' +
                            hiprand_include +
                            rocrand_include),
    }
    if rocm_version_number >= 40500:
        repository_dict["%{hipsolver_lib}"] = rocm_libs["hipsolver"].file_name
        repository_dict["%{hipblas_lib}"] = rocm_libs["hipblas"].file_name

    repository_ctx.template(
        "rocm/BUILD",
        tpl_paths["rocm:BUILD"],
        repository_dict,
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
    )

    repository_ctx.template(
        "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
        tpl_paths["crosstool:clang/bin/crosstool_wrapper_driver_rocm"],
        {
            "%{cpu_compiler}": str(cc),
            "%{hipcc_path}": rocm_config.rocm_toolkit_path + "/hip/bin/hipcc",
            "%{hipcc_env}": _hipcc_env(repository_ctx),
            "%{rocr_runtime_path}": rocm_config.rocm_toolkit_path + "/lib",
            "%{rocr_runtime_library}": "hsa-runtime64",
            "%{hip_runtime_path}": rocm_config.rocm_toolkit_path + "/hip/lib",
            "%{hip_runtime_library}": "amdhip64",
            "%{crosstool_verbose}": _crosstool_verbose(repository_ctx),
            "%{gcc_host_compiler_path}": str(cc),
        },
    )

    # Set up rocm_config.h, which is used by
    # tensorflow/compiler/xla/stream_executor/dso_loader.cc.
    repository_ctx.template(
        "rocm/rocm/rocm_config.h",
        tpl_paths["rocm:rocm_config.h"],
        {
            "%{rocm_amdgpu_targets}": ",".join(
                ["\"%s\"" % c for c in rocm_config.amdgpu_targets],
            ),
            "%{rocm_toolkit_path}": rocm_config.rocm_toolkit_path,
            "%{rocm_version_number}": rocm_config.rocm_version_number,
            "%{miopen_version_number}": rocm_config.miopen_version_number,
            "%{hipruntime_version_number}": rocm_config.hipruntime_version_number,
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
