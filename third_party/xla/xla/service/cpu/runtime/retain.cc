/* Copyright 2023 The OpenXLA Authors.

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

#include <cstdint>
#include <cstdlib>
#include <utility>

extern "C" void retainBuffers(int64_t numAllocs, void** allocBuffers,
                              int64_t numRetained, void** retainedBuffers) {
  for (int64_t i = 0; i < numRetained; ++i) {
    void* retained = retainedBuffers[i];
    retainedBuffers[i] = nullptr;
    for (int64_t j = 0; j < numAllocs; ++j) {
      if (allocBuffers[j] == retained) {
        std::swap(allocBuffers[j], retainedBuffers[i]);
        break;
      }
    }
  }

  for (int64_t i = 0; i < numAllocs; ++i) {
    if (allocBuffers[i]) {
      free(allocBuffers[i]);
    }
  }
}
