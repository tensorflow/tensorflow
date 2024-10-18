if 'LD_LIBRARY_PATH' in environ_cp and environ_cp.get(
    'LD_LIBRARY_PATH') != '1':
  write_action_env_to_bazelrc('LD_LIBRARY_PATH',
                              environ_cp.get('LD_LIBRARY_PATH'))

set_tf_cuda_clang(environ_cp)
if environ_cp.get('TF_CUDA_CLANG') == '1':
  # Set up which clang we should use as the cuda / host compiler.
  clang_cuda_compiler_path = set_clang_cuda_compiler_path(environ_cp)
  clang_version = retrieve_clang_version(clang_cuda_compiler_path)
  disable_clang_offsetof_extension(clang_version)
else:
  # Set up which gcc nvcc should use as the host compiler
  # No need to set this on Windows
  if not is_windows():
    set_gcc_host_compiler_path(environ_cp)
set_other_cuda_vars(environ_cp)
