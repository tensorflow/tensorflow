/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/cpu_function_runtime.h"

#include <cstddef>
#include <cstdint>

#include "absl/base/dynamic_annotations.h"
#include "xla/util.h"

namespace xla {
namespace {

xla::FreeDeleter freeDeleter;

constexpr size_t align_to(size_t n, size_t align) {
  return (((n - 1) / align) + 1) * align;
}
}  // namespace

namespace cpu_function_runtime {
size_t AlignedBufferBytes(const BufferInfo* buffer_infos, size_t n,
                          bool allocate_entry_params) {
  size_t total = 0;
  for (size_t i = 0; i < n; ++i) {
    bool should_allocate =
        buffer_infos[i].is_temp_buffer() ||
        (buffer_infos[i].is_entry_parameter() && allocate_entry_params);

    if (should_allocate) {
      total += align_to(buffer_infos[i].size(), Align());
    }
  }
  return total;
}

void* MallocContiguousBuffers(const BufferInfo* buffer_infos, size_t n,
                              bool allocate_entry_params, void** bufs,
                              bool annotate_initialized) {
  const size_t total =
      AlignedBufferBytes(buffer_infos, n, allocate_entry_params);
  void* contiguous = nullptr;
  if (total > 0) {
    contiguous = xla::AlignedAlloc(Align(), total).release();
    if (annotate_initialized) {
      // Since the memory for temp buffers is written to by JITed code, msan has
      // no way of knowing the memory was initialized, so explicitly mark it.
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(contiguous, total);
    }
  }
  uintptr_t pos = reinterpret_cast<uintptr_t>(contiguous);
  for (size_t i = 0; i < n; ++i) {
    bool should_allocate =
        buffer_infos[i].is_temp_buffer() ||
        (buffer_infos[i].is_entry_parameter() && allocate_entry_params);
    if (should_allocate) {
      bufs[i] = reinterpret_cast<void*>(pos);
      pos += align_to(buffer_infos[i].size(), Align());
    } else {
      bufs[i] = nullptr;
    }
  }
  return contiguous;
}

void FreeContiguous(void* contiguous) {
  if (contiguous != nullptr) {
    freeDeleter(contiguous);
  }
}
}  // namespace cpu_function_runtime
}  // namespace xla
