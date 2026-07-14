/* Copyright 2025 The OpenXLA Authors.

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

#include <cstddef>

#include "tsl/platform/mem.h"
#include "tsl/platform/numa.h"

namespace tsl {
namespace port {

bool NUMAEnabled() { return false; }

int NUMANumNodes() { return 1; }

void NUMASetThreadNodeAffinity(int node) {}

int NUMAGetThreadNodeAffinity() { return kNUMANoAffinity; }

void* NUMAMalloc(int node, size_t size, int minimum_alignment) {
  return AlignedMalloc(size, static_cast<std::align_val_t>(minimum_alignment));
}

void NUMAFree(void* ptr, size_t size) { ::tsl::port::Free(ptr); }

int NUMAGetMemAffinity(const void* ptr) { return kNUMANoAffinity; }

}  // namespace port
}  // namespace tsl
