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

// This version of the allocation utilities uses a fixed pool of memory so that
// a standard library heap isn't required. Include this in the list of files to
// compile instead of the memory_util_stdlib.c version if your platform doesn't
// have a heap available.

#include "tensorflow/lite/experimental/microfrontend/lib/memory_util.h"

// This size has been determined by experimentation, based on the largest
// allocations used by the micro speech example and tests.
#define FIXED_POOL_SIZE (30 * 1024)

void* microfrontend_alloc(size_t size) {
  static unsigned char fixed_pool[FIXED_POOL_SIZE];
  static int fixed_pool_used = 0;

  int next_used = fixed_pool_used + size;
  if (next_used > FIXED_POOL_SIZE) {
    return 0;
  }

  void* result = &fixed_pool[fixed_pool_used];
  fixed_pool_used += size;

  return result;
}

void microfrontend_free(void* ptr) {
  // Do nothing.
}
