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

_GCC_HOST_COMPILER_PATH = "GCC_HOST_COMPILER_PATH"
_ROCM_TOOLKIT_PATH = "ROCM_TOOLKIT_PATH"
_TF_ROCM_VERSION = "TF_ROCM_VERSION"
_TF_MIOPEN_VERSION = "TF_MIOPEN_VERSION"
_TF_ROCM_AMDGPU_TARGETS = "TF_ROCM_AMDGPU_TARGETS"
_TF_ROCM_CONFIG_REPO = "TF_ROCM_CONFIG_REPO"

_DEFAULT_ROCM_VERSION = ""
_DEFAULT_MIOPEN_VERSION = ""
_DEFAULT_ROCM_TOOLKIT_PATH = "/opt/rocm"
_DEFAULT_ROCM_AMDGPU_TARGETS = ["gfx803", "gfx900"]

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
  result = repository_ctx.execute([cc, "-no-canonical-prefixes",
                                   "-E", "-x" + lang, "-", "-v"])
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

  return [str(repository_ctx.path(_cxx_inc_convert(p)))
          for p in inc_dirs.split("\n")]

def get_cxx_inc_directories(repository_ctx, cc):
  """Compute the list of default C and C++ include directories."""
  # For some reason `clang -xc` sometimes returns include paths that are
  # different from the ones from `clang -xc++`. (Symlink and a dir)
  # So we run the compiler with both `-xc` and `-xc++` and merge resulting lists
  includes_cpp = _get_cxx_inc_directories_impl(repository_ctx, cc, True)
  includes_c = _get_cxx_inc_directories_impl(repository_ctx, cc, False)

  includes_cpp_set = depset(includes_cpp)
  return includes_cpp + [inc for inc in includes_c
                         if inc not in includes_cpp_set]

def auto_configure_fail(msg):
  """Output failure message when rocm configuration fails."""
  red = "\033[0;31m"
  no_color = "\033[0m"
  fail("\n%sCuda Configuration Error:%s %s\n" % (red, no_color, msg))
# END cc_configure common functions (see TODO above).

def _host_compiler_includes(repository_ctx, cc):
  """Generates the cxx_builtin_include_directory entries for gcc inc dirs.

  Args:
    repository_ctx: The repository context.
    cc: The path to the gcc host compiler.

  Returns:
    A string containing the cxx_builtin_include_directory for each of the gcc
    host compiler include directories, which can be added to the CROSSTOOL
    file.
  """
  inc_dirs = get_cxx_inc_directories(repository_ctx, cc)

  # Add numpy headers
  inc_dirs.append("/usr/lib/python2.7/dist-packages/numpy/core/include")

  entries = []
  for inc_dir in inc_dirs:
    entries.append("  cxx_builtin_include_directory: \"%s\"" % inc_dir)

  # define TENSORFLOW_USE_ROCM
  entries.append("  unfiltered_cxx_flag: \"-DTENSORFLOW_USE_ROCM\"")

  return "\n".join(entries)

def _rocm_include_path(repository_ctx, rocm_config):
  """Generates the cxx_builtin_include_directory entries for rocm inc dirs.

  Args:
    repository_ctx: The repository context.
    cc: The path to the gcc host compiler.

  Returns:
    A string containing the cxx_builtin_include_directory for each of the gcc
    host compiler include directories, which can be added to the CROSSTOOL
    file.
  """
  inc_dirs = []

  # general ROCm include path
  inc_dirs.append(rocm_config.rocm_toolkit_path + '/include')

  # Add HSA headers
  inc_dirs.append("/opt/rocm/hsa/include")

  # Add HIP headers
  inc_dirs.append("/opt/rocm/include/hip")
  inc_dirs.append("/opt/rocm/include/hip/hcc_detail")

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

  inc_entries = []
  for inc_dir in inc_dirs:
    inc_entries.append("  cxx_builtin_include_directory: \"%s\"" % inc_dir)
  return "\n".join(inc_entries)

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

def _cpu_value(repository_ctx):
  """Returns the name of the host operating system.

  Args:
    repository_ctx: The repository context.

  Returns:
    A string containing the name of the host operating system.
  """
  os_name = repository_ctx.os.name.lower()
  result = repository_ctx.execute(["uname", "-s"])
  return result.stdout.strip()

def _lib_name(lib, cpu_value, version="", static=False):
  """Constructs the platform-specific name of a library.

  Args:
    lib: The name of the library, such as "hip"
    cpu_value: The name of the host operating system.
    version: The version of the library.
    static: True the library is static or False if it is a shared object.

  Returns:
    The platform-specific name of the library.
  """
  if cpu_value in ("Linux"):
    if static:
      return "lib%s.a" % lib
    else:
      if version:
        version = ".%s" % version
      return "lib%s.so%s" % (lib, version)
  else:
    auto_configure_fail("Invalid cpu_value: %s" % cpu_value)

def _find_rocm_lib(lib, repository_ctx, cpu_value, basedir, version="",
                   static=False):
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
      return struct(file_name=file_name, path=str(path.realpath))
    path = repository_ctx.path("%s/lib64/stubs/%s" % (basedir, file_name))
    if path.exists:
      return struct(file_name=file_name, path=str(path.realpath))
    path = repository_ctx.path(
        "%s/lib/x86_64-linux-gnu/%s" % (basedir, file_name))
    if path.exists:
      return struct(file_name=file_name, path=str(path.realpath))

  path = repository_ctx.path("%s/lib/%s" % (basedir, file_name))
  if path.exists:
    return struct(file_name=file_name, path=str(path.realpath))
  path = repository_ctx.path("%s/%s" % (basedir, file_name))
  if path.exists:
    return struct(file_name=file_name, path=str(path.realpath))

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
          "hip_hcc", repository_ctx, cpu_value, rocm_config.rocm_toolkit_path),
      "rocblas": _find_rocm_lib(
          "rocblas", repository_ctx, cpu_value, rocm_config.rocm_toolkit_path),
      "rocfft": _find_rocm_lib(
          "rocfft", repository_ctx, cpu_value, rocm_config.rocm_toolkit_path + "/rocfft"),
      "hiprand": _find_rocm_lib(
          "hiprand", repository_ctx, cpu_value, rocm_config.rocm_toolkit_path + "/hiprand"),
      "miopen": _find_rocm_lib(
          "MIOpen", repository_ctx, cpu_value, rocm_config.rocm_toolkit_path + "/miopen"),
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
      cpu_value = cpu_value)

def _tpl(repository_ctx, tpl, substitutions={}, out=None):
  if not out:
    out = tpl.replace(":", "/")
  repository_ctx.template(
      out,
      Label("//third_party/gpus/%s.tpl" % tpl),
      substitutions)


def _file(repository_ctx, label):
  repository_ctx.template(
      label.replace(":", "/"),
      Label("//third_party/gpus/%s.tpl" % label),
      {})


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
  _tpl(repository_ctx, "rocm:build_defs.bzl",
       {
           "%{rocm_is_configured}": "False",
           "%{rocm_extra_copts}": "[]"
       })
  _tpl(repository_ctx, "rocm:BUILD",
       {
           "%{hip_lib}": _lib_name("hip", cpu_value),
           "%{rocblas_lib}": _lib_name("rocblas", cpu_value),
           "%{miopen_lib}": _lib_name("miopen", cpu_value),
           "%{rocfft_lib}": _lib_name("rocfft", cpu_value),
           "%{hiprand_lib}": _lib_name("hiprand", cpu_value),
           "%{rocm_include_genrules}": '',
           "%{rocm_headers}": '',
       })

  # Create dummy files for the ROCm toolkit since they are still required by
  # tensorflow/core/platform/default/build_config:rocm.
  repository_ctx.file("rocm/hip/include/hip/hip_runtime.h", "")

  # Set up rocm_config.h, which is used by
  # tensorflow/stream_executor/dso_loader.cc.
  _tpl(repository_ctx, "rocm:rocm_config.h",
       {
           "%{rocm_toolkit_path}": _DEFAULT_ROCM_TOOLKIT_PATH,
       }, "rocm/rocm/rocm_config.h")

  # If rocm_configure is not configured to build with GPU support, and the user
  # attempts to build with --config=rocm, add a dummy build rule to intercept
  # this and fail with an actionable error message.
  repository_ctx.file("crosstool/error_gpu_disabled.bzl",
                      _DUMMY_CROSSTOOL_BZL_FILE)
  repository_ctx.file("crosstool/BUILD", _DUMMY_CROSSTOOL_BUILD_FILE)

def _execute(repository_ctx, cmdline, error_msg=None, error_details=None,
             empty_stdout_fine=False):
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
            error_details if error_details else ""]))
  return result

def _norm_path(path):
  """Returns a path with '/' and remove the trailing slash."""
  path = path.replace("\\", "/")
  if path[-1] == "/":
    path = path[:-1]
  return path

def _symlink_genrule_for_dir(repository_ctx, src_dir, dest_dir, genrule_name,
    src_files = [], dest_files = []):
  """Returns a genrule to symlink(or copy if on Windows) a set of files.

  If src_dir is passed, files will be read from the given directory; otherwise
  we assume files are in src_files and dest_files
  """
  if src_dir != None:
    src_dir = _norm_path(src_dir)
    dest_dir = _norm_path(dest_dir)
    files = _read_dir(repository_ctx, src_dir)
    # Create a list with the src_dir stripped to use for outputs.
    dest_files = files.replace(src_dir, '').splitlines()
    src_files = files.splitlines()
  command = []
  # We clear folders that might have been generated previously to avoid
  # undesired inclusions
  command.append('if [ -d "$(@D)/include" ]; then rm $(@D)/include -drf; fi')
  command.append('if [ -d "$(@D)/lib" ]; then rm $(@D)/lib -drf; fi')
  outs = []
  for i in range(len(dest_files)):
    if dest_files[i] != "":
      # If we have only one file to link we do not want to use the dest_dir, as
      # $(@D) will include the full path to the file.
      dest = '$(@D)/' + dest_dir + dest_files[i] if len(dest_files) != 1 else '$(@D)/' + dest_files[i]
      # On Windows, symlink is not supported, so we just copy all the files.
      cmd = 'ln -s'
      command.append(cmd + ' "%s" "%s"' % (src_files[i] , dest))
      outs.append('        "' + dest_dir + dest_files[i] + '",')
  genrule = _genrule(src_dir, genrule_name, " && ".join(command),
                     "\n".join(outs))
  return genrule

def _genrule(src_dir, genrule_name, command, outs):
  """Returns a string with a genrule.

  Genrule executes the given command and produces the given outputs.
  """
  return (
      'genrule(\n' +
      '    name = "' +
      genrule_name + '",\n' +
      '    outs = [\n' +
      outs +
      '\n    ],\n' +
      '    cmd = """\n' +
      command +
      '\n   """,\n' +
      ')\n'
  )

def _read_dir(repository_ctx, src_dir):
  """Returns a string with all files in a directory.

  Finds all files inside a directory, traversing subfolders and following
  symlinks. The returned string contains the full path of all files
  separated by line breaks.
  """
  find_result = _execute(
      repository_ctx, ["find", src_dir, "-follow", "-type", "f"],
      empty_stdout_fine=True)
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

  # Set up symbolic links for the rocm toolkit by creating genrules to do
  # symlinking. We create one genrule for each directory we want to track under
  # rocm_toolkit_path
  rocm_toolkit_path = rocm_config.rocm_toolkit_path
  rocm_include_path = rocm_toolkit_path + "/include"
  genrules = [_symlink_genrule_for_dir(repository_ctx,
      rocm_include_path, "rocm/include", "rocm-include")]
  genrules.append(_symlink_genrule_for_dir(repository_ctx,
      rocm_toolkit_path + "/rocfft/include", "rocm/include/rocfft", "rocfft-include"))
  genrules.append(_symlink_genrule_for_dir(repository_ctx,
      rocm_toolkit_path + "/rocblas/include", "rocm/include/rocblas", "rocblas-include"))
  genrules.append(_symlink_genrule_for_dir(repository_ctx,
      rocm_toolkit_path + "/miopen/include", "rocm/include/miopen", "miopen-include"))

  rocm_libs = _find_libs(repository_ctx, rocm_config)
  rocm_lib_src = []
  rocm_lib_dest = []
  for lib in rocm_libs.values():
    rocm_lib_src.append(lib.path)
    rocm_lib_dest.append("rocm/lib/" + lib.file_name)
  genrules.append(_symlink_genrule_for_dir(repository_ctx, None, "", "rocm-lib",
                                       rocm_lib_src, rocm_lib_dest))

  included_files = _read_dir(repository_ctx, rocm_include_path).replace(
      rocm_include_path, '').splitlines()

  # Set up BUILD file for rocm/
  _tpl(repository_ctx, "rocm:build_defs.bzl",
       {
           "%{rocm_is_configured}": "True",
           "%{rocm_extra_copts}": _compute_rocm_extra_copts(
               repository_ctx, rocm_config.amdgpu_targets),

       })
  _tpl(repository_ctx, "rocm:BUILD",
       {
           "%{hip_lib}": rocm_libs["hip"].file_name,
           "%{rocblas_lib}": rocm_libs["rocblas"].file_name,
           "%{rocfft_lib}": rocm_libs["rocfft"].file_name,
           "%{hiprand_lib}": rocm_libs["hiprand"].file_name,
           "%{miopen_lib}": rocm_libs["miopen"].file_name,
           "%{rocm_include_genrules}": "\n".join(genrules),
           "%{rocm_headers}": ('":rocm-include",\n' +
                               '":rocfft-include",\n' +
                               '":rocblas-include",\n' +
                               '":miopen-include",'),
       })
  # Set up crosstool/
  _tpl(repository_ctx, "crosstool:BUILD", {"%{linker_files}": ":empty", "%{win_linker_files}": ":empty"})
  cc = find_cc(repository_ctx)
  host_compiler_includes = _host_compiler_includes(repository_ctx, cc)
  rocm_defines = {
           "%{rocm_include_path}": _rocm_include_path(repository_ctx,
                                                      rocm_config),
           "%{host_compiler_includes}": host_compiler_includes,
           "%{clang_path}": str(cc),
       }

  _tpl(repository_ctx, "crosstool:CROSSTOOL_hipcc", rocm_defines, out="crosstool/CROSSTOOL")

  _tpl(repository_ctx,
       "crosstool:clang/bin/crosstool_wrapper_driver_rocm",
       {
           "%{cpu_compiler}": str(cc),
           "%{hipcc_path}": "/opt/rocm/bin/hipcc",
           "%{gcc_host_compiler_path}": str(cc),
           "%{rocm_amdgpu_targets}": ",".join(
               ["\"%s\"" % c for c in rocm_config.amdgpu_targets]),
       })

  # Set up rocm_config.h, which is used by
  # tensorflow/stream_executor/dso_loader.cc.
  _tpl(repository_ctx, "rocm:rocm_config.h",
       {
           "%{rocm_amdgpu_targets}": ",".join(
               ["\"%s\"" % c for c in rocm_config.amdgpu_targets]),
           "%{rocm_toolkit_path}": rocm_config.rocm_toolkit_path,
       }, "rocm/rocm/rocm_config.h")


def _create_remote_rocm_repository(repository_ctx, remote_config_repo):
  """Creates pointers to a remotely configured repo set up to build with ROCm."""
  _tpl(repository_ctx, "rocm:build_defs.bzl",
       {
           "%{rocm_is_configured}": "True",
           "%{rocm_extra_copts}": _compute_rocm_extra_copts(
               repository_ctx, #_compute_capabilities(repository_ctx)
            ),

       })
  _tpl(repository_ctx, "rocm:remote.BUILD",
       {
           "%{remote_rocm_repo}": remote_config_repo,
       }, "rocm/BUILD")
  _tpl(repository_ctx, "crosstool:remote.BUILD", {
           "%{remote_rocm_repo}": remote_config_repo,
       }, "crosstool/BUILD")

def _rocm_autoconf_impl(repository_ctx):
  """Implementation of the rocm_autoconf repository rule."""
  if not _enable_rocm(repository_ctx):
    _create_dummy_repository(repository_ctx)
  else:
    if _TF_ROCM_CONFIG_REPO in repository_ctx.os.environ:
      _create_remote_rocm_repository(repository_ctx,
          repository_ctx.os.environ[_TF_ROCM_CONFIG_REPO])
    else:
      _create_local_rocm_repository(repository_ctx)


rocm_configure = repository_rule(
    implementation = _rocm_autoconf_impl,
    environ = [
        _GCC_HOST_COMPILER_PATH,
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
