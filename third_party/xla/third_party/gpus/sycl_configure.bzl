"""Repository rule for SYCL autoconfiguration.
`sycl_configure` depends on the following environment variables:
  * `TF_NEED_SYCL`: Whether to enable building with SYCL.
  * `GCC_HOST_COMPILER_PATH`: The GCC host compiler path
"""

load("//third_party/gpus/sycl:sycl_dl_essential.bzl", "sycl_redist")
load(
    "//third_party/remote_config:common.bzl",
    "err_out",
    "execute",
    "files_exist",
    "get_bash_bin",
    "get_host_environ",
    "get_python_bin",
    "raw_exec",
    "realpath",
    "which",
)
load(
    ":compiler_common_tools.bzl",
    "to_list_of_strings",
)
load(
    ":cuda_configure.bzl",
    "make_copy_dir_rule",
    "make_copy_files_rule",
)

_GCC_HOST_COMPILER_PATH = "GCC_HOST_COMPILER_PATH"
_GCC_HOST_COMPILER_PREFIX = "GCC_HOST_COMPILER_PREFIX"

def _mkl_path(sycl_config):
    return sycl_config.sycl_basekit_path + "/mkl/" + sycl_config.sycl_basekit_version_number

def _sycl_header_path(repository_ctx, sycl_config, bash_bin):
    sycl_header_path = sycl_config.sycl_basekit_path + "/compiler/" + sycl_config.sycl_basekit_version_number
    include_dir = sycl_header_path + "/include"
    if not files_exist(repository_ctx, [include_dir], bash_bin)[0]:
        sycl_header_path = sycl_header_path + "/linux"
        include_dir = sycl_header_path + "/include"
        if not files_exist(repository_ctx, [include_dir], bash_bin)[0]:
            auto_configure_fail("Cannot find sycl headers in {}".format(include_dir))
    return sycl_header_path

def _sycl_include_path(repository_ctx, sycl_config, bash_bin):
    """Generates the cxx_builtin_include_directory entries for sycl inc dirs.
    Args:
      repository_ctx: The repository context.
      sycl_config: The path to the gcc host compiler.
    Returns:
      A string containing the Starlark string for each of the gcc
      host compiler include directories, which can be added to the CROSSTOOL
      file.
    """
    inc_dirs = []

    inc_dirs.append(_mkl_path(sycl_config) + "/include")
    inc_dirs.append(_sycl_header_path(repository_ctx, sycl_config, bash_bin) + "/include")
    inc_dirs.append(_sycl_header_path(repository_ctx, sycl_config, bash_bin) + "/include/sycl")

    return inc_dirs

def enable_sycl(repository_ctx):
    """Returns whether to build with SYCL support."""
    return int(get_host_environ(repository_ctx, "TF_NEED_SYCL", False))

def auto_configure_fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sAuto-Configuration Error:%s %s\n" % (red, no_color, msg))

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

def find_sycl_root(repository_ctx, sycl_config):
    sycl_name = str(repository_ctx.path(sycl_config.sycl_toolkit_path.strip()).realpath)
    if sycl_name.startswith("/"):
        return sycl_name
    fail("Cannot find SYCL compiler, please correct your path")

def find_sycl_include_path(repository_ctx, sycl_config):
    base_path = find_sycl_root(repository_ctx, sycl_config)
    bin_path = repository_ctx.path(base_path + "/" + "bin" + "/" + "icpx")
    icpx_extra = ""
    if not bin_path.exists:
        bin_path = repository_ctx.path(base_path + "/" + "bin" + "/" + "clang")
        if not bin_path.exists:
            fail("Cannot find SYCL compiler, please correct your path")
    else:
        icpx_extra = "-fsycl"
    gcc_path = repository_ctx.which("gcc")
    gcc_install_dir = repository_ctx.execute([gcc_path, "-print-libgcc-file-name"])
    gcc_install_dir_opt = "--gcc-install-dir=" + str(repository_ctx.path(gcc_install_dir.stdout.strip()).dirname)
    cmd_out = repository_ctx.execute([bin_path, icpx_extra, gcc_install_dir_opt, "-xc++", "-E", "-v", "/dev/null", "-o", "/dev/null"])
    outlist = cmd_out.stderr.split("\n")
    real_base_path = str(repository_ctx.path(base_path).realpath).strip()
    include_dirs = []
    for l in outlist:
        if l.startswith(" ") and l.strip().startswith("/") and str(repository_ctx.path(l.strip()).realpath) not in include_dirs:
            include_dirs.append(str(repository_ctx.path(l.strip()).realpath))
    return include_dirs

def _lib_name(lib, version = "", static = False):
    """Constructs the name of a library on Linux.
    Args:
      lib: The name of the library, such as "mkl"
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

def _sycl_lib_paths(repository_ctx, lib, basedir):
    file_name = _lib_name(lib, version = "", static = False)
    return [
        repository_ctx.path("%s/lib/%s" % (basedir, file_name)),
        repository_ctx.path("%s/lib/intel64/%s" % (basedir, file_name)),
    ]

def _batch_files_exist(repository_ctx, libs_paths, bash_bin):
    all_paths = []
    for _, lib_paths in libs_paths:
        for lib_path in lib_paths:
            all_paths.append(lib_path)
    return files_exist(repository_ctx, all_paths, bash_bin)

def _select_sycl_lib_paths(repository_ctx, libs_paths, bash_bin):
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
            auto_configure_fail("Cannot find sycl library %s in %s" % (name, path))

        libs[name] = struct(file_name = selected_path.basename, path = realpath(repository_ctx, selected_path, bash_bin))

    return libs

def _find_libs(repository_ctx, sycl_config, bash_bin):
    """Returns the SYCL libraries on the system.
    Args:
      repository_ctx: The repository context.
      sycl_config: The SYCL config as returned by _get_sycl_config
      bash_bin: the path to the bash interpreter
    Returns:
      Map of library names to structs of filename and path
    """
    mkl_path = _mkl_path(sycl_config)
    sycl_path = _sycl_header_path(repository_ctx, sycl_config, bash_bin)
    libs_paths = [
        (name, _sycl_lib_paths(repository_ctx, name, path))
        for name, path in [
            ("sycl", sycl_path),
            ("OpenCL", sycl_path),
            ("mkl_intel_ilp64", mkl_path),
            ("mkl_sequential", mkl_path),
            ("mkl_core", mkl_path),
        ]
    ]
    if sycl_config.sycl_basekit_version_number < "2024":
        libs_paths.append(("mkl_sycl", _sycl_lib_paths(repository_ctx, "mkl_sycl", mkl_path)))
    else:
        libs_paths.append(("mkl_sycl_blas", _sycl_lib_paths(repository_ctx, "mkl_sycl_blas", mkl_path)))
        libs_paths.append(("mkl_sycl_lapack", _sycl_lib_paths(repository_ctx, "mkl_sycl_lapack", mkl_path)))
        libs_paths.append(("mkl_sycl_sparse", _sycl_lib_paths(repository_ctx, "mkl_sycl_sparse", mkl_path)))
        libs_paths.append(("mkl_sycl_dft", _sycl_lib_paths(repository_ctx, "mkl_sycl_dft", mkl_path)))
        libs_paths.append(("mkl_sycl_vm", _sycl_lib_paths(repository_ctx, "mkl_sycl_vm", mkl_path)))
        libs_paths.append(("mkl_sycl_rng", _sycl_lib_paths(repository_ctx, "mkl_sycl_rng", mkl_path)))
        libs_paths.append(("mkl_sycl_stats", _sycl_lib_paths(repository_ctx, "mkl_sycl_stats", mkl_path)))
        libs_paths.append(("mkl_sycl_data_fitting", _sycl_lib_paths(repository_ctx, "mkl_sycl_data_fitting", mkl_path)))
    return _select_sycl_lib_paths(repository_ctx, libs_paths, bash_bin)

def find_sycl_config(repository_ctx):
    """Returns SYCL config dictionary from running find_sycl_config.py"""
    python_bin = get_python_bin(repository_ctx)
    exec_result = execute(repository_ctx, [python_bin, repository_ctx.attr._find_sycl_config])
    if exec_result.return_code:
        auto_configure_fail("Failed to run find_sycl_config.py: %s" % err_out(exec_result))

    # Parse the dict from stdout.
    return dict([tuple(x.split(": ")) for x in exec_result.stdout.splitlines()])

def _get_sycl_config(repository_ctx, bash_bin):
    """Detects and returns information about the SYCL installation on the system.
    Args:
      repository_ctx: The repository context.
      bash_bin: the path to the path interpreter
    """
    config = find_sycl_config(repository_ctx)
    sycl_basekit_path = config["sycl_basekit_path"]
    sycl_toolkit_path = config["sycl_toolkit_path"]
    sycl_version_number = config["sycl_version_number"]
    sycl_basekit_version_number = config["sycl_basekit_version_number"]
    return struct(
        sycl_basekit_path = sycl_basekit_path,
        sycl_toolkit_path = sycl_toolkit_path,
        sycl_version_number = sycl_version_number,
        sycl_basekit_version_number = sycl_basekit_version_number,
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

_INC_DIR_MARKER_BEGIN = "#include <...>"

def _cxx_inc_convert(path):
    """Convert path returned by cc -E xc++ in a complete path."""
    path = path.strip()
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
        return "\"" + path[len(crosstool_folder) + 1:] + "\""
    return "\"" + path + "\""

def _get_cxx_inc_directories_impl(repository_ctx, cc, lang_is_cpp):
    """Compute the list of default C or C++ include directories."""
    if lang_is_cpp:
        lang = "c++"
    else:
        lang = "c"

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

_DUMMY_CROSSTOOL_BZL_FILE = """
def error_gpu_disabled():
  fail("ERROR: Building with --config=sycl but TensorFlow is not configured " +
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
    # Set up BUILD file for sycl/.
    _tpl(repository_ctx, "sycl:build_defs.bzl")
    _tpl(
        repository_ctx,
        "sycl:BUILD",
        {
            "%{mkl_intel_ilp64_lib}": _lib_name("mkl_intel_ilp64"),
            "%{mkl_sequential_lib}": _lib_name("mkl_sequential"),
            "%{mkl_core_lib}": _lib_name("mkl_core"),
            "%{mkl_sycl_libs}": "",
            "%{core_sycl_libs}": "",
            "%{copy_rules}": "",
            "%{sycl_headers}": "",
        },
    )

    # If sycl_configure is not configured to build with SYCL support, and the user
    # attempts to build with --config=sycl, add a dummy build rule to intercept
    # this and fail with an actionable error message.
    repository_ctx.file(
        "crosstool/error_gpu_disabled.bzl",
        _DUMMY_CROSSTOOL_BZL_FILE,
    )
    repository_ctx.file("crosstool/BUILD", _DUMMY_CROSSTOOL_BUILD_FILE)

    _tpl(
        repository_ctx,
        "sycl:build_defs.bzl",
        {
            "%{sycl_is_configured}": "False",
            "%{sycl_build_is_configured}": "False",
        },
    )

def _extract_file_name_from_url(url):
    """Extract the file name from a URL by finding the last slash '/'."""
    return url[url.rfind("/") + 1:]

def _download_and_extract_archive(repository_ctx, archive_info, distribution_path):
    """Downloads and extracts a tar.gz ONEAPI redistributable (cached by Bazel)."""

    archive_url = archive_info.get("url")
    archive_sha256 = archive_info.get("sha256")
    if not archive_url or not archive_sha256:
        fail("Missing required archive metadata: 'url' or 'sha256'")

    repository_ctx.report_progress("Installing oneAPI Basekit to: %s" % distribution_path)

    archive_override = repository_ctx.getenv("oneAPI_ARCHIVE_OVERRIDE")
    if archive_override:
        repository_ctx.report_progress("Using overridden archive: %s" % archive_override)
        repository_ctx.extract(archive_override, distribution_path)
    else:
        repository_ctx.download_and_extract(
            url = archive_url,
            sha256 = archive_sha256,
            output = distribution_path,
        )

    repository_ctx.report_progress("oneAPI Basekit archive extracted and installed.")

def _create_local_sycl_repository(repository_ctx):
    tpl_paths = {labelname: _tpl_path(repository_ctx, labelname) for labelname in [
        "sycl:build_defs.bzl",
        "sycl:BUILD",
        "crosstool:BUILD.sycl",
        "crosstool:sycl_cc_toolchain_config.bzl",
        "crosstool:clang/bin/crosstool_wrapper_driver_sycl",
    ]}

    bash_bin = get_bash_bin(repository_ctx)

    hermetic = repository_ctx.getenv("SYCL_BUILD_HERMETIC") == "1"
    if hermetic:
        oneapi_version = repository_ctx.getenv("ONEAPI_VERSION", "")
        os_id = repository_ctx.getenv("OS", "")
        if not oneapi_version or not os_id:
            fail("ONEAPI_VERSION and OS must be set via --repo_env for hermetic build.")
        redist_info = sycl_redist.get(os_id, {}).get(oneapi_version, {}).get("sycl_dl_essential")
        if not redist_info:
            fail("Missing redistributable definition for %s on %s" % (oneapi_version, os_id))
        archive_info = redist_info["archives"][0]
        install_path = str(repository_ctx.path(redist_info["root"]))
        _download_and_extract_archive(repository_ctx, archive_info, install_path)

        sycl_config = struct(
            sycl_basekit_path = install_path + "/oneapi/",
            sycl_toolkit_path = install_path + "/oneapi/compiler/" + oneapi_version,
            # TODO(Intel)
            # hard coded for this oneAPI version 2025.1,
            # change to auto-detect the sycl_version_number
            sycl_version_number = "80000",
            sycl_basekit_version_number = oneapi_version,
        )
    else:
        install_path = repository_ctx.getenv("SYCL_TOOLKIT_PATH") or "/opt/intel/oneapi/compiler/2025.1"
        repository_ctx.report_progress("Falling back to default SYCL path: %s" % install_path)
        sycl_config = _get_sycl_config(repository_ctx, bash_bin)

    # Copy header and library files to execroot.
    copy_rules = [
        make_copy_dir_rule(
            repository_ctx,
            name = "sycl-include",
            src_dir = _sycl_header_path(repository_ctx, sycl_config, bash_bin) + "/include",
            out_dir = "sycl/include",
        ),
    ]
    copy_rules.append(make_copy_dir_rule(
        repository_ctx,
        name = "mkl-include",
        src_dir = _mkl_path(sycl_config) + "/include",
        out_dir = "sycl/include",
    ))

    sycl_libs = _find_libs(repository_ctx, sycl_config, bash_bin)
    sycl_lib_srcs = []
    sycl_lib_outs = []
    for lib in sycl_libs.values():
        sycl_lib_srcs.append(lib.path)
        sycl_lib_outs.append("sycl/lib/" + lib.file_name)
    copy_rules.append(make_copy_files_rule(
        repository_ctx,
        name = "sycl-lib",
        srcs = sycl_lib_srcs,
        outs = sycl_lib_outs,
    ))

    # Set up BUILD file for sycl/
    repository_ctx.template(
        "sycl/build_defs.bzl",
        tpl_paths["sycl:build_defs.bzl"],
        {
            "%{sycl_is_configured}": "True",
            "%{sycl_build_is_configured}": "True",
        },
    )

    if sycl_config.sycl_basekit_version_number < "2024":
        mkl_sycl_libs = '"{}"'.format(
            "sycl/lib/" + sycl_libs["mkl_sycl"].file_name,
        )
    else:
        mkl_sycl_libs = '"{}",\n"{}",\n"{}",\n"{}",\n"{}",\n"{}",\n"{}",\n"{}"'.format(
            "sycl/lib/" + sycl_libs["mkl_sycl_blas"].file_name,
            "sycl/lib/" + sycl_libs["mkl_sycl_lapack"].file_name,
            "sycl/lib/" + sycl_libs["mkl_sycl_sparse"].file_name,
            "sycl/lib/" + sycl_libs["mkl_sycl_dft"].file_name,
            "sycl/lib/" + sycl_libs["mkl_sycl_vm"].file_name,
            "sycl/lib/" + sycl_libs["mkl_sycl_rng"].file_name,
            "sycl/lib/" + sycl_libs["mkl_sycl_stats"].file_name,
            "sycl/lib/" + sycl_libs["mkl_sycl_data_fitting"].file_name,
        )
    core_sycl_libs = '"{}",\n"{}"'.format(
        "sycl/lib/" + sycl_libs["sycl"].file_name,
        "sycl/lib/" + sycl_libs["OpenCL"].file_name,
    )
    repository_dict = {
        "%{mkl_intel_ilp64_lib}": sycl_libs["mkl_intel_ilp64"].file_name,
        "%{mkl_sequential_lib}": sycl_libs["mkl_sequential"].file_name,
        "%{mkl_core_lib}": sycl_libs["mkl_core"].file_name,
        "%{mkl_sycl_libs}": mkl_sycl_libs,
        "%{core_sycl_libs}": core_sycl_libs,
        "%{copy_rules}": "\n".join(copy_rules),
        "%{sycl_headers}": ('":mkl-include",\n":sycl-include",\n'),
    }
    repository_ctx.template(
        "sycl/BUILD",
        tpl_paths["sycl:BUILD"],
        repository_dict,
    )

    # Set up crosstool/

    cc = find_cc(repository_ctx)

    host_compiler_includes = get_cxx_inc_directories(repository_ctx, cc)

    host_compiler_prefix = get_host_environ(repository_ctx, _GCC_HOST_COMPILER_PREFIX, "/usr/bin")

    sycl_defines = {}

    sycl_defines["%{host_compiler_prefix}"] = host_compiler_prefix
    sycl_defines["%{host_compiler_path}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"

    sycl_defines["%{cpu_compiler}"] = str(cc)
    sycl_defines["%{linker_bin_path}"] = "/usr/bin"

    sycl_internal_inc_dirs = find_sycl_include_path(repository_ctx, sycl_config)
    cxx_builtin_includes_list = sycl_internal_inc_dirs + _sycl_include_path(repository_ctx, sycl_config, bash_bin) + host_compiler_includes

    sycl_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(cxx_builtin_includes_list)
    sycl_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-fno-canonical-system-headers\""
    sycl_defines["%{unfiltered_compile_flags}"] = to_list_of_strings([
        "-DTENSORFLOW_USE_SYCL=1",
        "-DMKL_ILP64",
        "-fPIC",
    ])
    sycl_defines["%{sycl_compiler_root}"] = str(sycl_config.sycl_toolkit_path)
    sycl_defines["%{SYCL_ROOT_DIR}"] = str(sycl_config.sycl_toolkit_path)
    sycl_defines["%{basekit_path}"] = str(sycl_config.sycl_basekit_path)
    sycl_defines["%{basekit_version}"] = str(sycl_config.sycl_basekit_version_number)
    sycl_defines["%{MKL_PATH}"] = _mkl_path(sycl_config)

    # Only expand template variables in the BUILD file
    repository_ctx.template(
        "crosstool/BUILD",
        tpl_paths["crosstool:BUILD.sycl"],
        sycl_defines,
    )

    # No templating of cc_toolchain_config - use attributes and templatize the
    # BUILD file.
    repository_ctx.template(
        "crosstool/cc_toolchain_config.bzl",
        tpl_paths["crosstool:sycl_cc_toolchain_config.bzl"],
        sycl_defines,
    )

    repository_ctx.template(
        "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
        tpl_paths["crosstool:clang/bin/crosstool_wrapper_driver_sycl"],
        sycl_defines,
    )

def _sycl_autoconf_imp(repository_ctx):
    """Implementation of the sycl_autoconf rule."""
    if not enable_sycl(repository_ctx):
        _create_dummy_repository(repository_ctx)
    else:
        _create_local_sycl_repository(repository_ctx)

sycl_configure = repository_rule(
    # Detects and configures the local SYCL toolchain.
    # Add the following to your WORKSPACE FILE:
    # ```python
    # sycl_configure(name = "local_config_sycl")
    # ```
    # Args:
    #   name: A unique name for this workspace rule.
    implementation = _sycl_autoconf_imp,
    local = True,
    attrs = {
        "_find_sycl_config": attr.label(
            default = Label("//third_party/gpus:find_sycl_config.py"),
        ),
    },
)
