#!/usr/bin/env python

import os
import sys
import tempfile
from subprocess import call, Popen, PIPE

CPU_CXX_COMPILER = ('%{host_cxx_compiler}')
CPU_C_COMPILER = ('%{host_c_compiler}')

CURRENT_DIR = os.path.dirname(sys.argv[0])
COMPUTECPP_ROOT = CURRENT_DIR + '/../sycl/'
COMPUTECPP_DRIVER= COMPUTECPP_ROOT + 'bin/compute++'
COMPUTECPP_INCLUDE = COMPUTECPP_ROOT + 'include'

def main():
  remove_flags = ('-Wl,--no-undefined', '-Wno-unused-but-set-variable', '-Wignored-attributes')
    # remove -fsamotoze-coverage from string with g++
  if 'g++' in CPU_CXX_COMPILER:
    remove_flags += ('-fsanitize-coverage',)
  compiler_flags = [flag for flag in sys.argv[1:] if not flag.startswith(remove_flags)]

  output_file_index = compiler_flags.index('-o') + 1
  output_file_name = compiler_flags[output_file_index]

  if output_file_index == 1:
    # we are linking
    return call([CPU_CXX_COMPILER] + compiler_flags + ['-Wl,--no-undefined'])

  # find what we compile
  compiling_cpp = False
  if '-c' in compiler_flags:
    compiled_file_index = compiler_flags.index('-c') + 1
    compiled_file_name = compiler_flags[compiled_file_index]
    compiling_cpp = compiled_file_name.endswith(('.cc', '.c++', '.cpp', '.CPP', '.C', '.cxx'))

  # add -D_GLIBCXX_USE_CXX11_ABI=0 to the command line if you have custom installation of GCC/Clang
  compiler_flags = compiler_flags + ['-DEIGEN_USE_SYCL=1', '-DTENSORFLOW_USE_SYCL', '-DEIGEN_HAS_C99_MATH']

  if not compiling_cpp:
    # compile for C
    return call([CPU_C_COMPILER] + compiler_flags)

  # create a blacklist of folders that will be skipped when compiling with ComputeCpp
  skip_extensions = [".cu.cc"]
  skip_folders = ["tensorflow/compiler", "tensorflow/docs_src", "tensorflow/tensorboard", "third_party", "external", "hexagon"]
  skip_folders = [(folder + '/') for folder in skip_folders]
  # if compiling external project skip computecpp
  if any(compiled_file_name.endswith(_ext) for _ext in skip_extensions) or any(_folder in output_file_name for _folder in skip_folders):
    return call([CPU_CXX_COMPILER] + compiler_flags)

  # this is an optimisation that will check if compiled file has to be compiled with ComputeCpp
  flags_without_output = list(compiler_flags)
  del flags_without_output[output_file_index]   # remove output_file_name
  del flags_without_output[output_file_index - 1] # remove '-o'
  # create preprocessed of the file and store it for later use
  pipe = Popen([CPU_CXX_COMPILER] + flags_without_output + ["-E"], stdout=PIPE)
  preprocessed_file_str = pipe.communicate()[0]
  if pipe.returncode != 0:
    return pipe.returncode

  # check if it has parallel_for in it
  if not '.parallel_for' in preprocessed_file_str:
    # call CXX compiler like usual
    with tempfile.NamedTemporaryFile(suffix=".ii") as preprocessed_file: # Force '.ii' extension so that g++ does not preprocess the file again
      preprocessed_file.write(preprocessed_file_str)
      preprocessed_file.flush()
      compiler_flags[compiled_file_index] = preprocessed_file.name
      return call([CPU_CXX_COMPILER] + compiler_flags)
  del preprocessed_file_str   # save some memory as this string can be quite big

  filename, file_extension = os.path.splitext(output_file_name)
  bc_out = filename + '.sycl'

  # strip asan for the device
  computecpp_device_compiler_flags = ['-sycl-compress-name', '-Wno-unused-variable',
                                      '-I', COMPUTECPP_INCLUDE, '-isystem', COMPUTECPP_INCLUDE,
                                      '-std=c++11', '-sycl', '-emit-llvm', '-no-serial-memop',
                                      '-Xclang', '-cl-denorms-are-zero', '-Xclang', '-cl-fp32-correctly-rounded-divide-sqrt']
  # disable flags enabling SIMD instructions
  computecpp_device_compiler_flags += [flag for flag in compiler_flags if \
    not any(x in flag.lower() for x in ('-fsanitize', '=native', '=core2', 'msse', 'vectorize', 'mavx', 'mmmx', 'm3dnow', 'fma'))]

  x = call([COMPUTECPP_DRIVER] + computecpp_device_compiler_flags)
  if x == 0:
    # dont want that in case of compiling with computecpp first
    host_compiler_flags = [flag for flag in compiler_flags if (not flag.startswith(('-MF', '-MD',)) and not '.d' in flag)]
    host_compiler_flags[host_compiler_flags.index('-c')] = "--include"
    host_compiler_flags = ['-xc++', '-Wno-unused-variable', '-I', COMPUTECPP_INCLUDE, '-c', bc_out] + host_compiler_flags
    x = call([CPU_CXX_COMPILER] + host_compiler_flags)
  return x

if __name__ == '__main__':
  sys.exit(main())
