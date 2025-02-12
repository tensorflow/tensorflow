#!/usr/bin/env python
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Crosstool wrapper for compiling CUDA programs with nvcc on Windows.

DESCRIPTION:
  This script is the Windows version of //third_party/gpus/crosstool/crosstool_wrapper_is_not_gcc
"""

from argparse import ArgumentParser
import os
import subprocess
import re
import sys
import tempfile

# Template values set by cuda_autoconf.
CPU_COMPILER = ('%{cpu_compiler}')
GCC_HOST_COMPILER_PATH = ('%{gcc_host_compiler_path}')

NVCC_PATH = '%{nvcc_path}'
NVCC_VERSION = '%{cuda_version}'
NVCC_TEMP_DIR = "%{nvcc_tmp_dir}"

def Log(s):
  print('gpus/crosstool: {0}'.format(s))


def GetOptionValue(argv, option):
  """Extract the list of values for option from options.

  Args:
    option: The option whose value to extract.

  Returns:
    1. A list of values, either directly following the option,
    (eg., /opt val1 val2) or values collected from multiple occurrences of
    the option (eg., /opt val1 /opt val2).
    2. The leftover options.
  """

  parser = ArgumentParser(prefix_chars='-/')
  parser.add_argument(option, nargs='*', action='append')
  option = option.lstrip('-/').replace('-', '_')
  args, leftover = parser.parse_known_args(argv)
  if args and vars(args)[option]:
    return (sum(vars(args)[option], []), leftover)
  return ([], leftover)

def _update_options(nvcc_options):
  if NVCC_VERSION in ("7.0",):
    return nvcc_options

  update_options = { "relaxed-constexpr" : "expt-relaxed-constexpr" }
  return [ update_options[opt] if opt in update_options else opt
                    for opt in nvcc_options ]

def GetNvccOptions(argv):
  """Collect the -nvcc_options values from argv.

  Args:
    argv: A list of strings, possibly the argv passed to main().

  Returns:
    1. The string that can be passed directly to nvcc.
    2. The leftover options.
  """

  parser = ArgumentParser()
  parser.add_argument('-nvcc_options', nargs='*', action='append')

  args, leftover = parser.parse_known_args(argv)

  if args.nvcc_options:
    options = _update_options(sum(args.nvcc_options, []))
    return (['--' + a for a in options], leftover)
  return ([], leftover)


def InvokeNvcc(argv, log=False):
  """Call nvcc with arguments assembled from argv.

  Args:
    argv: A list of strings, possibly the argv passed to main().
    log: True if logging is requested.

  Returns:
    The return value of calling os.system('nvcc ' + args)
  """

  src_files = [f for f in argv if
               re.search('\.cpp$|\.cc$|\.c$|\.cxx$|\.C$', f)]
  if len(src_files) == 0:
    raise Error('No source files found for cuda compilation.')

  out_file = [ f for f in argv if f.startswith('/Fo') ]
  if len(out_file) != 1:
    raise Error('Please specify exactly one output file for cuda compilation.')
  out = ['-o', out_file[0][len('/Fo'):]]

  nvcc_compiler_options, argv = GetNvccOptions(argv)

  opt_option, argv = GetOptionValue(argv, '/O')
  opt = ['-g']
  if (len(opt_option) > 0 and opt_option[0] != 'd'):
    opt = ['-O2']

  include_options, argv = GetOptionValue(argv, '/I')
  includes = ["-I " + include for include in include_options]

  defines, argv = GetOptionValue(argv, '/D')
  defines = [
      '-D' + define
      for define in defines
      if 'BAZEL_CURRENT_REPOSITORY' not in define
  ]

  undefines, argv = GetOptionValue(argv, '/U')
  undefines = ['-U' + define for define in undefines]

  fatbin_options, argv = GetOptionValue(argv, '-Xcuda-fatbinary')
  fatbin_options = ['--fatbin-options=' + option for option in fatbin_options]

  # The rest of the unrecognized options should be passed to host compiler
  host_compiler_options = [option for option in argv if option not in (src_files + out_file)]

  m_options = ["-m64"]

  nvccopts = ['-D_FORCE_INLINES']
  compute_capabilities, argv = GetOptionValue(argv, "--cuda-gpu-arch")
  for capability in compute_capabilities:
    capability = capability[len('sm_'):]
    nvccopts += [
        r'-gencode=arch=compute_%s,"code=sm_%s"' % (capability, capability)
    ]
  compute_capabilities, argv = GetOptionValue(argv, '--cuda-include-ptx')
  for capability in compute_capabilities:
    capability = capability[len('sm_'):]
    nvccopts += [
        r'-gencode=arch=compute_%s,"code=compute_%s"' % (capability, capability)
    ]
  _, argv = GetOptionValue(argv, '--no-cuda-include-ptx')

  # nvcc doesn't respect the INCLUDE and LIB env vars from MSVC,
  # so we explicity specify the system include paths and library search paths.
  if 'INCLUDE' in os.environ:
    nvccopts += [('--system-include="%s"' % p) for p in os.environ['INCLUDE'].split(";")]
  if 'LIB' in os.environ:
    nvccopts += [('--library-path="%s"' % p) for p in os.environ['LIB'].split(";")]

  nvccopts += nvcc_compiler_options
  nvccopts += undefines
  nvccopts += defines
  nvccopts += m_options
  nvccopts += fatbin_options
  nvccopts += ['--compiler-options=' + ",".join(host_compiler_options)]
  nvccopts += ['-x', 'cu'] + opt + includes + out + ['-c'] + src_files
  # Specify a unique temp directory for nvcc to generate intermediate files,
  # then Bazel can ignore files under NVCC_TEMP_DIR during dependency check
  # http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-guiding-compiler-driver
  # Different actions are sharing NVCC_TEMP_DIR, so we cannot remove it if the directory already exists.
  if os.path.isfile(NVCC_TEMP_DIR):
    os.remove(NVCC_TEMP_DIR)
  if not os.path.exists(NVCC_TEMP_DIR):
    os.makedirs(NVCC_TEMP_DIR)
  # Provide a unique dir for each compiling action to avoid conflicts.
  tempdir = tempfile.mkdtemp(dir = NVCC_TEMP_DIR)
  nvccopts += ['--keep', '--keep-dir', tempdir]
  # Force C++17 dialect (note, everything in just one string!)
  nvccopts += ['--std c++17']
  # This is so that nvcc does not complain about MSVC or CLANG.
  nvccopts += ['-allow-unsupported-compiler']
  nvccopts += ['--expt-extended-lambda', '--expt-relaxed-constexpr']
  if log:
    Log([NVCC_PATH] + nvccopts)

  # Store command line options in a file to avoid hitting the character limit.
  optsfile = tempfile.NamedTemporaryFile(mode='w', dir=tempdir, delete=False)
  optsfile.write("\n".join(nvccopts))
  optsfile.close()

  proc = subprocess.Popen([NVCC_PATH, "--options-file", optsfile.name],
                          stdout=sys.stdout,
                          stderr=sys.stderr,
                          env=os.environ.copy(),
                          shell=True)
  proc.wait()
  return proc.returncode

def ExpandParamsFileForArgv():
  new_argv = []
  for arg in sys.argv:
    if arg.startswith("@"):
      with open(arg.strip("@")) as f:
        new_argv.extend([l.strip() for l in f.readlines()])
    else:
      new_argv.append(arg)

  sys.argv = new_argv

def ProcessFlagForCommandFile(flag):
  if flag.startswith("/D") or flag.startswith("-D"):
    # We need to re-escape /DFOO="BAR" as /DFOO=\"BAR\", so that we get
    # `#define FOO "BAR"` after expansion as a string literal define
    if flag.endswith('"') and not flag.endswith('\\"'):
      flag = '\\"'.join(flag.split('"', 1))
      flag = '\\"'.join(flag.rsplit('"', 1))
      return flag
  return flag

def main():
  ExpandParamsFileForArgv()
  parser = ArgumentParser()
  parser.add_argument('-x', nargs=1)
  parser.add_argument('--cuda_log', action='store_true')
  args, leftover = parser.parse_known_args(sys.argv[1:])

  if args.x and args.x[0] == 'cuda':
    if args.cuda_log: Log('-x cuda')
    if args.cuda_log: Log('using nvcc')
    return InvokeNvcc(leftover, log=args.cuda_log)

  # Strip our flags before passing through to the CPU compiler for files which
  # are not -x cuda. We can't just pass 'leftover' because it also strips -x.
  # We not only want to pass -x to the CPU compiler, but also keep it in its
  # relative location in the argv list (the compiler is actually sensitive to
  # this).
  cpu_compiler_flags = [flag for flag in sys.argv[1:]
                             if not flag.startswith(('--cuda_log'))
                             and not flag.startswith(('-nvcc_options'))]
  output = [flag for flag in cpu_compiler_flags if flag.startswith("/Fo")]

  # Store command line options in a file to avoid hitting the character limit.
  if len(output) == 1:
    commandfile_path = output[0][3:] + ".msvc_params"
    commandfile = open(commandfile_path, "w")
    cpu_compiler_flags = [ProcessFlagForCommandFile(flag) for flag in cpu_compiler_flags]
    commandfile.write("\n".join(cpu_compiler_flags))
    commandfile.close()
    return subprocess.call([CPU_COMPILER, "@" + commandfile_path])
  else:
    return subprocess.call([CPU_COMPILER] + cpu_compiler_flags)
  return subprocess.call([CPU_COMPILER] + cpu_compiler_flags)

if __name__ == '__main__':
  sys.exit(main())
