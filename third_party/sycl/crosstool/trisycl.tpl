#!/usr/bin/env python

import os
import sys
import tempfile
from subprocess import call

CPU_CXX_COMPILER = ('%{host_cxx_compiler}')
CPU_C_COMPILER = ('%{host_c_compiler}')

CURRENT_DIR = os.path.dirname(sys.argv[0])
TRISYCL_INCLUDE_DIR = CURRENT_DIR + '/../sycl/include'

def main():
  compiler_flags = []

  remove_flags = ('-Wl,--no-undefined', '-Wno-unused-but-set-variable', '-Wignored-attributes', '-fno-exceptions')
  # remove -fsamotoze-coverage from string with g++
  if 'g++' in CPU_CXX_COMPILER:
    remove_flags += ('-fsanitize-coverage',)
    compiler_flags += ['-fopenmp']
  else:
    compiler_flags += ['-fopenmp=libomp']

  compiler_flags += [flag for flag in sys.argv[1:] if not flag.startswith(remove_flags)]


  output_file_index = compiler_flags.index('-o') + 1
  output_file_name = compiler_flags[output_file_index]

  if(output_file_index == 1):
    # we are linking
    return call([CPU_CXX_COMPILER] + compiler_flags +
                ['-Wl,--no-undefined'])

  # find what we compile
  compiling_cpp = 0
  if('-c' in compiler_flags):
      compiled_file_index = compiler_flags.index('-c') + 1
      compiled_file_name = compiler_flags[compiled_file_index]
      if(compiled_file_name.endswith(('.cc', '.c++', '.cpp', '.CPP',
                                      '.C', '.cxx'))):
        compiling_cpp = 1;

  debug_flags = ['-DTRISYCL_DEBUG', '-DBOOST_LOG_DYN_LINK', '-DTRISYCL_TRACE_KERNEL', '-lpthread', '-lboost_log', '-g', '-rdynamic']

  opt_flags = ['-DNDEBUG', '-DBOOST_DISABLE_ASSERTS', '-O3']

  compiler_flags = compiler_flags + ['-DEIGEN_USE_SYCL=1',
                                     '-DEIGEN_HAS_C99_MATH',
                                     '-DEIGEN_MAX_ALIGN_BYTES=16',
                                     '-DTENSORFLOW_USE_SYCL'] + opt_flags

  if(compiling_cpp == 1):
    # create a blacklist of folders that will be skipped when compiling
    # with triSYCL
    skip_extensions = [".cu.cc"]
    skip_folders = ["tensorflow/compiler", "tensorflow/docs_src", "tensorflow/tensorboard", "third_party", "external", "hexagon"]
    skip_folders = [(folder + '/') for folder in skip_folders]
    # if compiling external project skip triSYCL
    if any(compiled_file_name.endswith(_ext) for _ext in skip_extensions) or any(_folder in output_file_name for _folder in skip_folders):
      return call([CPU_CXX_COMPILER] + compiler_flags)

    host_compiler_flags = ['-xc++', '-Wno-unused-variable',
                           '-I', TRISYCL_INCLUDE_DIR] + compiler_flags
    x = call([CPU_CXX_COMPILER] + host_compiler_flags)
    return x
  else:
    # compile for C
    return call([CPU_C_COMPILER] + compiler_flags)

if __name__ == '__main__':
  sys.exit(main())
