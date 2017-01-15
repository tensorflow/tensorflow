#!/usr/bin/env python

import os
import subprocess
import sys

CPU_CXX_COMPILER = ('%{host_cxx_compiler}')
CPU_C_COMPILER = ('%{host_c_compiler}')

CURRENT_DIR = os.path.dirname(sys.argv[0])
COMPUTECPP_ROOT = CURRENT_DIR + '/../sycl/'
COMPUTECPP_DRIVER= COMPUTECPP_ROOT + 'bin/compute++'
COMPUTECPP_INCLUDE = COMPUTECPP_ROOT + 'include'

def main():
  compiler_flags = []

  # remove -fsamotoze-coverage from string
  if CPU_CXX_COMPILER.find("g++") != -1:
    compiler_flags = [flag for flag in sys.argv[1:] if not flag.startswith(('-Wl,--no-undefined', '-fsanitize-coverage', '-Wno-unused-but-set-variable', '-Wignored-attributes'))]
  else:
    compiler_flags = [flag for flag in sys.argv[1:] if not flag.startswith(('-Wl,--no-undefined', '-Wno-unused-but-set-variable', '-Wignored-attributes'))]

  output_file_index = compiler_flags.index('-o') + 1
  output_file_name = compiler_flags[output_file_index]

  if(output_file_index == 1):
    # we are linking
    return subprocess.call([CPU_CXX_COMPILER] + compiler_flags)

  compiler_flags = compiler_flags + ['-D_GLIBCXX_USE_CXX11_ABI=0', '-DEIGEN_USE_SYCL=1']

  # find what we compile
  compiling_cpp = 0
  if('-c' in compiler_flags):
      compiled_file_index = compiler_flags.index('-c') + 1
      compited_file_name = compiler_flags[compiled_file_index]
      if(compited_file_name.endswith(('.cc', '.c++', '.cpp', '.CPP', '.C', '.cxx'))):
          compiling_cpp = 1;

  if(compiling_cpp == 1):
      filename, file_extension = os.path.splitext(output_file_name)
      bc_out = filename + '.sycl'

      # strip asan for the device
      computecpp_device_compiler_flags = [flag for flag in compiler_flags if not flag.startswith(('-fsanitize'))]
      computecpp_device_compiler_flags = ['-sycl-compress-name', '-DTENSORFLOW_USE_SYCL', '-Wno-unused-variable', '-I', COMPUTECPP_INCLUDE, '-isystem',
      COMPUTECPP_INCLUDE, '-std=c++11', '-sycl', '-emit-llvm', '-no-serial-memop', '-Xclang', '-cl-denorms-are-zero', '-Xclang', '-cl-fp32-correctly-rounded-divide-sqrt'] + computecpp_device_compiler_flags

      x = subprocess.call([COMPUTECPP_DRIVER] + computecpp_device_compiler_flags )
      if(x == 0):
          # dont want that in case of compiling with computecpp first
          host_compiler_flags = [flag for flag in compiler_flags
                                    if not flag.startswith(('-MF', '-MD',))
                                    if not '.d' in flag]

          host_compiler_flags = ['-D_GLIBCXX_USE_CXX11_ABI=0', '-DTENSORFLOW_USE_SYCL', '-Wno-unused-variable', '-I', COMPUTECPP_INCLUDE, '--include', bc_out] + host_compiler_flags
          x = subprocess.call([CPU_CXX_COMPILER] + host_compiler_flags)
      return x
  else:
    # compile for C
    return subprocess.call([CPU_C_COMPILER] + compiler_flags)

if __name__ == '__main__':
  sys.exit(main())
