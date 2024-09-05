#!/usr/bin/env python
"""Crosstool wrapper for compiling SYCL program
SYNOPSIS:
  crosstool_wrapper_driver_sycl [options passed in by cc_library()
                                or cc_binary() rule]
DESCRIPTION:
  This script is expected to be called by the cc_library() or cc_binary() bazel
  rules. When the option "sycl_compile" is present in the list of arguments passed
  to this script, it invokes the sycl compiler. When "sycl_compile" is not
  present, this wrapper invokes gcc with the input arguments as is.
"""

from __future__ import print_function
from argparse import ArgumentParser
import os
import subprocess
import sys
import shlex

CPU_COMPILER = ('%{cpu_compiler}')

def system(cmd):
  """Invokes cmd with os.system()"""

  ret = os.system(cmd)
  if os.WIFEXITED(ret):
    return os.WEXITSTATUS(ret)
  else:
    return -os.WTERMSIG(ret)

def call_compiler(argv):
  parser = ArgumentParser()
  parser.add_argument('-c', nargs=1, action='append')
  parser.add_argument('-o', nargs=1, action='append')
  args, leftover = parser.parse_known_args(argv)

  flags = leftover

  common_flags = []
  common_flags.append("-fno-finite-math-only")
  common_flags.append("-fno-fast-math")
  common_flags.append("-fexceptions")

  in_files, out_files = [], []
  if args.c:
    in_files.append('-c')
    in_files.extend(args.c[0])
  if args.o:
    out_files.append('-o')
    out_files.extend(args.o[0])
  flags += (common_flags + in_files + out_files)
  print("cmd: ", " ".join([CPU_COMPILER] + flags))
  return subprocess.call([CPU_COMPILER] + flags)

def main():
  parser = ArgumentParser()
  parser = ArgumentParser(fromfile_prefix_chars='@')
  parser.add_argument('-sycl_compile', action='store_true')
  args, leftover = parser.parse_known_args(sys.argv[1:])

  return call_compiler(leftover)

if __name__ == '__main__':
  sys.exit(main())
