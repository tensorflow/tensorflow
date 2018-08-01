/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/aot/runtime.h"

#include <stdlib.h>

#include "tensorflow/core/platform/dynamic_annotations.h"

namespace tensorflow {
namespace tfcompile {
namespace runtime {

namespace {

// Inline memory allocation routines here, because depending on '//base' brings
// in libraries which use c++ streams, which adds considerable code size on
// android.
inline void* aligned_malloc(size_t size, int minimum_alignment) {
#if defined(__ANDROID__) || defined(OS_ANDROID) || defined(OS_CYGWIN)
  return memalign(minimum_alignment, size);
#elif defined(_WIN32)
  return _aligned_malloc(size, minimum_alignment);
#else  // !__ANDROID__ && !OS_ANDROID && !OS_CYGWIN
  void* ptr = nullptr;
  // posix_memalign requires that the requested alignment be at least
  // sizeof(void*). In this case, fall back on malloc which should return memory
  // aligned to at least the size of a pointer.
  const int required_alignment = sizeof(void*);
  if (minimum_alignment < required_alignment) return malloc(size);
  if (posix_memalign(&ptr, minimum_alignment, size) != 0)
    return nullptr;
  else
    return ptr;
#endif
}

inline void aligned_free(void* aligned_memory) {
#if defined(_WIN32)
  _aligned_free(aligned_memory);
#else
  free(aligned_memory);
#endif
}

size_t align_to(size_t n, size_t align) {
  return (((n - 1) / align) + 1) * align;
}

}  // namespace

size_t aligned_buffer_bytes(const intptr_t* sizes, size_t n) {
  size_t total = 0;
  for (size_t i = 0; i < n; ++i) {
    if (sizes[i] != -1) {
      total += align_to(sizes[i], kAlign);
    }
  }
  return total;
}

void* MallocContiguousBuffers(const intptr_t* sizes, size_t n, void** bufs,
                              bool annotate_initialized) {
  const size_t total = aligned_buffer_bytes(sizes, n);
  void* contiguous = nullptr;
  if (total > 0) {
    contiguous = aligned_malloc(total, kAlign);
    if (annotate_initialized) {
      // Since the memory for temp buffers is written to by JITed code, msan has
      // no way of knowing the memory was initialized, so explicitly mark it.
      TF_ANNOTATE_MEMORY_IS_INITIALIZED(contiguous, total);
    }
  }
  uintptr_t pos = reinterpret_cast<uintptr_t>(contiguous);
  for (size_t i = 0; i < n; ++i) {
    if (sizes[i] == -1) {
      bufs[i] = nullptr;
    } else {
      bufs[i] = reinterpret_cast<void*>(pos);
      pos += align_to(sizes[i], kAlign);
    }
  }
  return contiguous;
}

void FreeContiguous(void* contiguous) {
  if (contiguous != nullptr) {
    aligned_free(contiguous);
  }
}

}  // namespace runtime
}  // namespace tfcompile
}  // namespace tensorflow
