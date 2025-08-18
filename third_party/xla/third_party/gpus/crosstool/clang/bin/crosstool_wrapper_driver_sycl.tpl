#!/usr/bin/env python

"""Crosstool wrapper for compiling DPC++ program
SYNOPSIS:
  crosstool_wrapper_driver_sycl [options passed in by cc_library()
                            or cc_binary() rule]

DESCRIPTION:
  call DPC++ compiler for device-side code, and host
  compiler for other code.
"""

from __future__ import print_function
from argparse import ArgumentParser
import os
import subprocess
import re
import sys
import shlex
import tempfile

def check_is_intel_llvm(path):
  cmd = path + " -dM -E -x c /dev/null | grep '__INTEL_LLVM_COMPILER'"
  check_result = subprocess.getoutput(cmd)
  if len(check_result) > 0 and check_result.find('__INTEL_LLVM_COMPILER') > -1:
    return True
  return False

SYCL_PATH = os.path.join("%{sycl_compiler_root}", "bin/icpx")

if not os.path.exists(SYCL_PATH):
  SYCL_PATH = os.path.join('%{sycl_compiler_root}', 'bin/compiler/clang')
  if not os.path.exists(SYCL_PATH) or check_is_intel_llvm(SYCL_PATH):
    raise RuntimeError("compiler not found or invalid")

CPU_COMPILER = ('%{cpu_compiler}')
if ('%{tf_icpx_clang}')=="False":
 USE_ICPX_CLANG = False
else:
 USE_ICPX_CLANG = True
basekit_path = "%{basekit_path}"
basekit_version = "%{basekit_version}"

result = subprocess.run(["which", "ar"], capture_output=True, text=True)
if result.returncode == 0:
    AR_PATH = result.stdout.strip()  # Remove any trailing newline
else:
    raise RuntimeError("ar not found or invalid")

def system(cmd):
  """Invokes cmd with os.system()"""
  
  ret = os.system(cmd)
  if os.WIFEXITED(ret):
    return os.WEXITSTATUS(ret)
  else:
    return -os.WTERMSIG(ret)

def GetHostCompilerOptions(argv):
  parser = ArgumentParser()
  args, leftover = parser.parse_known_args(argv)
  sycl_host_compile_flags = leftover
  sycl_host_compile_flags.append('-std=c++17')
  host_flags = ['-fsycl-host-compiler-options=\'%s\'' % (' '.join(sycl_host_compile_flags))]
  return host_flags

def call_compiler(argv, link = False, sycl_compile = True):
  parser = ArgumentParser()
  parser.add_argument('-c', nargs=1, action='append')
  parser.add_argument('-o', nargs=1, action='append')
  args, leftover = parser.parse_known_args(argv)

  flags = leftover

  sycl_device_only_flags = ['-fsycl']
  sycl_device_only_flags.append('-fno-sycl-unnamed-lambda')
  sycl_device_only_flags.append('-fsycl-targets=spir64_gen,spir64')
  sycl_device_only_flags.append('-sycl-std=2020')
  sycl_device_only_flags.append('-fhonor-infinities')
  sycl_device_only_flags.append('-fhonor-nans')
  sycl_device_only_flags.append('-fno-sycl-use-footer')
  sycl_device_only_flags.append('-Xclang -fdenormal-fp-math=preserve-sign')
  sycl_device_only_flags.append('-Xclang -cl-mad-enable')
  sycl_device_only_flags.append('-cl-fp32-correctly-rounded-divide-sqrt')
  sycl_device_only_flags.append('-fsycl-device-code-split=per_source')

  common_flags = []
  # ref: https://github.com/intel/llvm/blob/sycl/clang/docs/UsersManual.rst#controlling-floating-point-behavior
  common_flags.append("-fno-finite-math-only")
  common_flags.append("-fno-fast-math")
  common_flags.append("-fexceptions")

  compile_flags = []
  compile_flags.append('-DDNNL_GRAPH_WITH_SYCL=1')
  compile_flags.append("-std=c++17")

  link_flags = ['-fPIC']
  link_flags.append('-lsycl')
  link_flags.append("-Wl,-no-as-needed")
  link_flags.append("-Wl,--enable-new-dtags")
  link_flags.append("-L%{SYCL_ROOT_DIR}/lib/")
  link_flags.append("-L%{SYCL_ROOT_DIR}/compiler/lib/intel64_lin/")
  link_flags.append("-lOpenCL")

  sycl_link_flags = ['-fPIC']
  sycl_link_flags.append("-fsycl")
  sycl_link_flags.append('-fsycl-max-parallel-link-jobs=8')
  sycl_link_flags.append('-fsycl-link')

  # host_flags = GetHostCompilerOptions(flags+common_flags)

  in_files, out_files = [], []
  if sycl_compile:
    flags = [shlex.quote(s) for s in flags]
    # device compilation
    if args.c:
      in_files.append('-c')
      in_files.extend(args.c[0])
      assert len(args.c[0]) == 1
    if args.o:
      out_files.append('-o')
      out_files.extend(args.o[0])
      assert len(args.o[0]) == 1

    in_file = args.c[0][0]
    out_file = args.o[0][0]
    out_file_name = out_file.split('/')[-1].split('.')[0]

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
      object_file = os.path.join(temp_dir, out_file_name + '.compile.o')
      dev_file = os.path.join(temp_dir, out_file_name + '.dev.o')

      # compile object file
      # icx -fsycl -c kernel.cpp -o kernel.compile.o
      sycl_compile_flags = [" -c {} -o {} ".format(in_file, object_file)]
      sycl_compile_flags += (flags + common_flags + compile_flags + sycl_device_only_flags)
      compile_cmd = ('env ' + SYCL_PATH + ' ' + ' '.join(sycl_compile_flags))
      exit_status = system(compile_cmd)
      if exit_status != 0:
        return exit_status

      # generate device object file that can be used by host compiler
      # icx -fsycl -fPIC -fsycl-link kernel.compile.o -o kernel.dev.o
      sycl_link_flags_dev = [" {} -o {} ".format(object_file, dev_file)]
      sycl_link_flags_dev += (common_flags + sycl_link_flags)
      link_cmd = ('env ' + SYCL_PATH + ' ' + ' '.join(sycl_link_flags_dev))
      exit_status = system(link_cmd)
      if exit_status != 0:
        return exit_status

      # archive object files
      # ar rcsD output.o kernel.compile.o kernel.dev.o
      ar_flags = " rcsD {} {} {}".format(out_file, object_file, dev_file)
      ar_cmd = ('env ' + AR_PATH + ar_flags)
      return system(ar_cmd)
  elif link:
    # Expand @params file if present
    expanded_flags = []
    for s in flags:
        if s.startswith("@") and os.path.isfile(s[1:]):
            with open(s[1:], "r") as f:
                expanded_flags += shlex.split(f.read())
        else:
            expanded_flags.append(s)

    seen = set()
    ordered_flags = []
    
    # Track the type of each unique flag and preserve order
    for s in expanded_flags:
        if s not in seen:
            seen.add(s)
            flag_type = "whole" if s.endswith((".o", ".lo")) else "regular"
            ordered_flags.append((flag_type, shlex.quote(s)))
    
    deduped_flags = []
    in_whole_archive = False
    
    for flag_type, flag in ordered_flags:
        if flag_type == "whole":
            if not in_whole_archive:
                deduped_flags.append("-Wl,--whole-archive")
                in_whole_archive = True
            deduped_flags.append(flag)
        else:
            if in_whole_archive:
                deduped_flags.append("-Wl,--no-whole-archive")
                in_whole_archive = False
            deduped_flags.append(flag)
    
    # Close any open --whole-archive section
    if in_whole_archive:
        deduped_flags.append("-Wl,--no-whole-archive")
    
    # Output file
    if args.o:
        out_files.append('-o')
        out_files.extend(args.o[0])
    deduped_flags += common_flags + in_files + out_files + link_flags

    # Write to response file to avoid command line length issues
    # This saves all flags into a temporary .params file like /tmp/tmpabc123.params to avoid blowing up command line limits.
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".params") as f:
        f.write('\n'.join(deduped_flags))
        response_file = f.name
    cmd = f'env {CPU_COMPILER} @{response_file}' # env icpx @/tmp/tmpabc123.params
    return system(cmd)
  else:
    # host compilation
    if args.c:
      in_files.append('-c')
      in_files.extend(args.c[0])
      input_file = os.path.basename(args.c[0][0])
    if args.o:
      out_files.append('-o')
      out_files.extend(args.o[0])
    flags += (common_flags + in_files + out_files)
    VERBOSE = os.environ.get("VERBOSE", "0") == "1"
    if USE_ICPX_CLANG:
      if VERBOSE:
        print(' '.join([CPU_COMPILER] + flags))
      return subprocess.call([CPU_COMPILER] + flags)
    else:
      if VERBOSE:
        print(' '.join([SYCL_PATH] + flags))
      return subprocess.call([SYCL_PATH] + flags)

def main():
  parser = ArgumentParser()
  parser = ArgumentParser(fromfile_prefix_chars='@')
  parser.add_argument('-sycl_compile', action='store_true')
  parser.add_argument('-link_stage', action='store_true')
  args, leftover = parser.parse_known_args(sys.argv[1:])

  return call_compiler(leftover, link=args.link_stage, sycl_compile=args.sycl_compile)

if __name__ == '__main__':
  sys.exit(main())
