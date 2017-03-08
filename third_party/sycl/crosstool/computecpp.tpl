#!/usr/bin/env python

from argparse import ArgumentParser
import os
import subprocess
import re
import sys
import pipes

CPU_CXX_COMPILER = ('%{host_cxx_compiler}')
CPU_C_COMPILER = ('%{host_c_compiler}')

CURRENT_DIR = os.path.dirname(sys.argv[0])
COMPUTECPP_ROOT = CURRENT_DIR +"/../sycl/"
COMPUTECPP_DRIVER= COMPUTECPP_ROOT+"bin/compute++"
COMPUTECPP_INCLUDE = COMPUTECPP_ROOT+"include"

def main():
  computecpp_compiler_flags = [""]
  computecpp_compiler_flags = [flag for flag in sys.argv[1:]]
  computecpp_compiler_flags = computecpp_compiler_flags + ["-D_GLIBCXX_USE_CXX11_ABI=0"]

  output_file_index = computecpp_compiler_flags.index("-o") +1
  output_file_name = computecpp_compiler_flags[output_file_index]

  if(output_file_index == 1):
    # we are linking
    return subprocess.call([CPU_CXX_COMPILER] +computecpp_compiler_flags )

  # find what we compile
  compiling_cpp = 0
  if("-c" in computecpp_compiler_flags):
      compiled_file_index = computecpp_compiler_flags.index("-c") +1
      compited_file_name = computecpp_compiler_flags[compiled_file_index]
      if(compited_file_name.endswith(('.cc', '.c++', '.cpp', '.CPP', '.C', '.cxx'))):
          compiling_cpp = 1;

  if(compiling_cpp == 1):
      filename, file_extension = os.path.splitext(output_file_name)
      bc_out = filename + ".sycl"

      computecpp_compiler_flags = ['-sycl-compress-name', '-DTENSORFLOW_USE_SYCL', '-Wno-unused-variable','-I', COMPUTECPP_INCLUDE,'-isystem',
      COMPUTECPP_INCLUDE, "-std=c++11", "-sycl", "-emit-llvm", "-no-serial-memop"] + computecpp_compiler_flags

      # dont want that in case of compiling with computecpp first
      host_compiler_flags = [""]
      host_compiler_flags = [flag for flag in sys.argv[1:]
                                if not flag.startswith(('-MF','-MD',))
                                if not ".d" in flag]

      x = subprocess.call([COMPUTECPP_DRIVER] +computecpp_compiler_flags )
      if(x == 0):
          host_compiler_flags = ['-D_GLIBCXX_USE_CXX11_ABI=0', '-DTENSORFLOW_USE_SYCL', '-Wno-unused-variable', '-I', COMPUTECPP_INCLUDE, "--include",bc_out] + host_compiler_flags
          return subprocess.call([CPU_CXX_COMPILER] +host_compiler_flags )
      return x
  else:
    # compile for C
    return subprocess.call([CPU_C_COMPILER] +computecpp_compiler_flags)

if __name__ == '__main__':
  sys.exit(main())
