#!/usr/bin/env python3
"""Crosstool wrapper for compiling ROCm programs.

SYNOPSIS:
  crosstool_wrapper_driver_rocm [options passed in by cc_library()
                                or cc_binary() rule]

DESCRIPTION:
  This script is expected to be called by the cc_library() or cc_binary() bazel
  rules. When the option "-x rocm" is present in the list of arguments passed
  to this script, it invokes the hipcc compiler. Most arguments are passed
  as is as a string to --compiler-options of hipcc. When "-x rocm" is not
  present, this wrapper invokes gcc with the input arguments as is.
"""

__author__ = 'whchung@gmail.com (Wen-Heng (Jack) Chung)'

from argparse import ArgumentParser
import os
import subprocess
import re
import sys

# Template values set by rocm_configure.bzl or environment
AMDGPU_TARGETS = ('%{rocm_amdgpu_targets}')
CPU_COMPILER = os.environ.get('HOST_COMPILER', '/usr/bin/clang')

HIPCC_PATH = '%{rocm_root}/bin/hipcc'
HIPCC_ENV = '%{hipcc_env}'
HIP_RUNTIME_PATH = '%{rocm_root}/lib'
HIP_RUNTIME_LIBRARY = '%{rocm_root}/lib'
ROCR_RUNTIME_PATH = '%{rocm_root}/lib'
ROCR_RUNTIME_LIBRARY = '%{rocr_runtime_library}'
TMPDIR= '%{tmpdir}'
VERBOSE = '%{crosstool_verbose}'=='1'


def _parse_hipcc_env(hipcc_env_str):
  """Parse HIPCC_ENV string into a list of 'KEY=VALUE' strings for env command.

  Args:
    hipcc_env_str: Semicolon-separated env assignments like 'KEY="val"; KEY2="val2"'

  Returns:
    A list of strings like ['KEY=val', 'KEY2=val2'], suitable for ['env', ...] prefix.
  """
  env_list = []
  if not hipcc_env_str or not hipcc_env_str.strip():
    return env_list
  for assignment in hipcc_env_str.split(';'):
    assignment = assignment.strip()
    if assignment and '=' in assignment:
      key, _, value = assignment.partition('=')
      value = value.strip().strip('"')
      env_list.append('{0}={1}'.format(key.strip(), value))
  return env_list


def Log(s):
  print('gpus/crosstool: {0}'.format(s))


def GetOptionValue(argv, option):
  """Extract the list of values for option from the argv list.

  Args:
    argv: A list of strings, possibly the argv passed to main().
    option: The option whose value to extract, without the leading '-'.

  Returns:
    A list of values, either directly following the option,
    (eg., -opt val1 val2) or values collected from multiple occurrences of
    the option (eg., -opt val1 -opt val2).
  """

  parser = ArgumentParser()
  parser.add_argument('-' + option, nargs='*', action='append')
  args, _ = parser.parse_known_args(argv)
  if not args or not vars(args)[option]:
    return []
  else:
    return sum(vars(args)[option], [])


def GetHostCompilerOptions(argv):
  """Collect the -isystem, -iquote, and --sysroot option values from argv.

  Args:
    argv: A list of strings, possibly the argv passed to main().

  Returns:
    A list of compiler option strings.
  """

  parser = ArgumentParser()
  parser.add_argument('-isystem', nargs='*', action='append')
  parser.add_argument('-iquote', nargs='*', action='append')
  parser.add_argument('--sysroot', nargs=1)
  parser.add_argument('-g', nargs='*', action='append')
  parser.add_argument('-no-canonical-prefixes', action='store_true')
  parser.add_argument('--genco', action='store_true')

  args, _ = parser.parse_known_args(argv)

  opts = []

  if args.isystem:
    for p in sum(args.isystem, []):
      opts.extend(['-isystem', p])
  if args.iquote:
    for p in sum(args.iquote, []):
      opts.extend(['-iquote', p])
  if args.g:
    for g in sum(args.g, []):
      opts.append('-g' + g)
  if args.no_canonical_prefixes:
    opts.append('-no-canonical-prefixes')
  if args.sysroot:
    opts.extend(['--sysroot', args.sysroot[0]])
  if args.genco:
    opts.append('--genco')

  return opts

def GetHipccOptions(argv):
  """Collect the -hipcc_options values from argv.
  Args:
    argv: A list of strings, possibly the argv passed to main().
  Returns:
    A list of option strings that can be passed directly to hipcc.
  """

  parser = ArgumentParser()
  parser.add_argument('-gline-tables-only', action='store_true')

  args, _ = parser.parse_known_args(argv)

  hipcc_opts = ['-gline-tables-only'] if args.gline_tables_only else []
  for target in AMDGPU_TARGETS.split(','):
    hipcc_opts.append('--offload-arch=' + target.strip())

  return hipcc_opts

def system(cmd):
  """Invokes command safely using subprocess with list-based arguments.

  Args:
    cmd: A list of strings (command and arguments).

  Returns:
    The exit code of the process.
  """
  try:
    return subprocess.call(cmd)
  except OSError as e:
    Log('Error executing command: {0}'.format(e))
    return 1


def InvokeHipcc(argv, log=False):
  """Call hipcc with arguments assembled from argv.

  Args:
    argv: A list of strings, possibly the argv passed to main().
    log: True if logging is requested.

  Returns:
    The return value of the subprocess call.
  """

  host_compiler_options = GetHostCompilerOptions(argv)
  hipcc_compiler_options = GetHipccOptions(argv)
  opt_option = GetOptionValue(argv, 'O')
  m_options = GetOptionValue(argv, 'm')
  m_options = ['-m' + m for m in m_options if m in ['32', '64']]
  include_options = GetOptionValue(argv, 'I')
  out_file = GetOptionValue(argv, 'o')
  depfiles = GetOptionValue(argv, 'MF')
  defines = GetOptionValue(argv, 'D')
  defines = ['-D' + define for define in defines]
  undefines = GetOptionValue(argv, 'U')
  undefines = ['-U' + define for define in undefines]
  std_options = GetOptionValue(argv, 'std')
  hipcc_allowed_std_options = ["c++11", "c++14", "c++17"]
  std_options = ['-std=' + define
      for define in std_options if define in hipcc_allowed_std_options]

  # The list of source files get passed after the -c option. I don't know of
  # any other reliable way to just get the list of source files to be compiled.
  src_files = GetOptionValue(argv, 'c')

  if len(src_files) == 0:
    return 1
  if len(out_file) != 1:
    return 1

  opt = ['-O2'] if (len(opt_option) > 0 and int(opt_option[0]) > 0) else ['-g']

  includes = []
  for inc in include_options:
    includes.extend(['-I', inc])

  # Unfortunately, there are other options that have -c prefix too.
  # So allowing only those look like C/C++ files.
  src_files = [f for f in src_files if
               re.search(r'\.cpp$|\.cc$|\.c$|\.cxx$|\.C$', f)]

  hipccopts = list(hipcc_compiler_options)
  # In hip-clang environment, we need to make sure that hip header is included
  # before some standard math header like <complex> is included in any source.
  # Otherwise, we get build error.
  # Also we need to retain warning about uninitialised shared variable as
  # warning only, even when -Werror option is specified.
  hipccopts.append('--include=hip/hip_runtime.h')
  # Force C++17 dialect
  hipccopts.append('--std=c++17')
  # Use -fno-gpu-rdc by default for early GPU kernel finalization
  # This flag would trigger GPU kernels be generated at compile time, instead
  # of link time. This allows the default host compiler (gcc) be used as the
  # linker for TensorFlow on ROCm platform.
  hipccopts.append('-fno-gpu-rdc')
  hipccopts.append('-fcuda-flush-denormals-to-zero')
  hipccopts.extend(undefines)
  hipccopts.extend(defines)
  hipccopts.extend(std_options)
  hipccopts.extend(m_options)
  hipccopts.append('--rocm-path=%{rocm_root}')

  env_prefix = _parse_hipcc_env(HIPCC_ENV)

  if depfiles:
    # Generate the dependency file
    depfile = depfiles[0]
    cmd = [HIPCC_PATH] + hipccopts + host_compiler_options
    cmd.extend(['-I', '.'])
    cmd.extend(includes)
    cmd.extend(src_files)
    cmd.extend(['-M', '-o', depfile])
    if env_prefix:
      cmd = ['env'] + env_prefix + cmd
    if log: Log(' '.join(cmd))
    if VERBOSE: print(' '.join(cmd))
    exit_status = system(cmd)
    if exit_status != 0:
      return exit_status

  cmd = [HIPCC_PATH] + hipccopts + host_compiler_options
  cmd.extend(['-fPIC', '-I', '.'])
  cmd.extend(opt)
  cmd.extend(includes)
  cmd.append('-c')
  cmd.extend(src_files)
  cmd.extend(['-o', out_file[0]])
  if env_prefix:
    cmd = ['env'] + env_prefix + cmd
  if log: Log(' '.join(cmd))
  if VERBOSE: print(' '.join(cmd))
  return system(cmd)


def main():
  if TMPDIR:
    os.environ['TMPDIR'] = TMPDIR
  # ignore PWD env var
  os.environ['PWD']=''

  parser = ArgumentParser(fromfile_prefix_chars='@')
  parser.add_argument('-x', nargs=1)
  parser.add_argument('--rocm_log', action='store_true')
  parser.add_argument('-pass-exit-codes', action='store_true')
  args, leftover = parser.parse_known_args(sys.argv[1:])

  if VERBOSE: print('PWD=' + os.getcwd())
  if VERBOSE: print('HIPCC_ENV=' + HIPCC_ENV)

  if args.x and args.x[0] == 'rocm':
    # compilation for GPU objects
    if args.rocm_log: Log('-x rocm')
    if args.rocm_log: Log('using hipcc')
    return InvokeHipcc(leftover, log=args.rocm_log)

  elif args.pass_exit_codes:
    # link
    # with hipcc compiler invoked with -fno-gpu-rdc by default now, it's ok to
    # use host compiler as linker, but we have to link with HCC/HIP runtime.
    # Such restriction would be revised further as the bazel script get
    # improved to fine tune dependencies to ROCm libraries.
    gpu_linker_flags = [flag for flag in sys.argv[1:]
                               if not flag.startswith(('--rocm_log'))]

    gpu_linker_flags.append('-L' + ROCR_RUNTIME_PATH)
    gpu_linker_flags.append('-Wl,-rpath=' + ROCR_RUNTIME_PATH)
    gpu_linker_flags.append('-l' + ROCR_RUNTIME_LIBRARY)
    gpu_linker_flags.append('-L' + HIP_RUNTIME_PATH)
    gpu_linker_flags.append('-Wl,-rpath=' + HIP_RUNTIME_PATH)
    gpu_linker_flags.append('-l' + HIP_RUNTIME_LIBRARY)
    gpu_linker_flags.append("-lrt")
    gpu_linker_flags.append("-lstdc++")

    if VERBOSE: print(' '.join([CPU_COMPILER] + gpu_linker_flags))
    return subprocess.call([CPU_COMPILER] + gpu_linker_flags)

  else:
    # compilation for host objects

    # Strip our flags before passing through to the CPU compiler for files which
    # are not -x rocm. We can't just pass 'leftover' because it also strips -x.
    # We not only want to pass -x to the CPU compiler, but also keep it in its
    # relative location in the argv list (the compiler is actually sensitive to
    # this).
    cpu_compiler_flags = [flag for flag in sys.argv[1:]
                               if not flag.startswith(('--rocm_log'))]

    # XXX: SE codes need to be built with gcc, but need this macro defined
    cpu_compiler_flags.append("-D__HIP_PLATFORM_HCC__")
    if VERBOSE: print(' '.join([CPU_COMPILER] + cpu_compiler_flags))
    return subprocess.call([CPU_COMPILER] + cpu_compiler_flags)

if __name__ == '__main__':
  sys.exit(main())
