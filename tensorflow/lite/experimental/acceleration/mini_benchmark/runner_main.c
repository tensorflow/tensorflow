/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// LINT.IfChange
#define DLOPEN_FAILED 11
#define SYMBOL_LOOKUP_FAILED 12
#define TOO_FEW_ARGUMENTS 13
#define UNSUPPORTED_PLATFORM 14
// LINT.ThenChange(//tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h)

#ifndef _WIN32
#include <dlfcn.h>
#include <stdio.h>

/**
 * Mini-benchmark helper executable entry point.
 *
 * This is a minimal executable that loads the shared library that contains the
 * actual logic as not to duplicate code between the mini-benchmark and
 * applicaiton code that uses TFLite.
 *
 * The executable takes 2 or more arguments:
 *   executable path_to_shared_library symbol_to_call ...
 * It loads the path_to_shared_library and calls symbol_to_call with all the
 * arguments (using original argc and argv).
 *
 * Exit codes:
 * - Happy path exits with the return value from symbol_to_call().
 * - If library loading fails, exits with 1.
 * - If symbol lookup fails, exits with 2.
 * - If less than 2 arguments are passed, exits with 3.
 */
int main(int argc, char** argv) {
  if (argc < 3) {
    return TOO_FEW_ARGUMENTS;
  }
  void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    printf("dlopen failed for %s: %s\n", argv[1], dlerror());
    return DLOPEN_FAILED;
  }
  void* sym = dlsym(lib, argv[2]);
  if (!sym) {
    printf("dlsym failed for %s on library %s with error: %s\n", argv[2],
           argv[1], dlerror());
    return SYMBOL_LOOKUP_FAILED;
  }
  int (*f)(int argc, char** argv) = (int (*)(int, char**))sym;
  int exitcode = (*f)(argc, argv);
  return exitcode;
}

#else   // _WIN32
int main(int argc, char** argv) { return UNSUPPORTED_PLATFORM; }
#endif  // !_WIN32
