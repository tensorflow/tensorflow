/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PLATFORM_MEM_H_
#define TENSORFLOW_PLATFORM_MEM_H_

#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"

// Include appropriate platform-dependent implementation of
// TF_ANNOTATE_MEMORY_IS_INITIALIZED.
#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/build_config/dynamic_annotations.h"
#elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) || \
    defined(PLATFORM_GOOGLE_ANDROID)
#include "tensorflow/core/platform/default/dynamic_annotations.h"
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

namespace tensorflow {
namespace port {

// Aligned allocation/deallocation
void* aligned_malloc(size_t size, int minimum_alignment);
void aligned_free(void* aligned_memory);

// Tries to release num_bytes of free memory back to the operating
// system for reuse.  Use this routine with caution -- to get this
// memory back may require faulting pages back in by the OS, and
// that may be slow.
//
// Currently, if a malloc implementation does not support this
// routine, this routine is a no-op.
void MallocExtension_ReleaseToSystem(std::size_t num_bytes);

// Returns the actual number N of bytes reserved by the malloc for the
// pointer p.  This number may be equal to or greater than the number
// of bytes requested when p was allocated.
//
// This routine is just useful for statistics collection.  The
// client must *not* read or write from the extra bytes that are
// indicated by this call.
//
// Example, suppose the client gets memory by calling
//    p = malloc(10)
// and GetAllocatedSize(p) may return 16.  The client must only use the
// first 10 bytes p[0..9], and not attempt to read or write p[10..15].
//
// Currently, if a malloc implementation does not support this
// routine, this routine returns 0.
std::size_t MallocExtension_GetAllocatedSize(const void* p);

// Prefetching support
//
// Defined behavior on some of the uarchs:
// PREFETCH_HINT_T0:
//   prefetch to all levels of the hierarchy (except on p4: prefetch to L2)
// PREFETCH_HINT_NTA:
//   p4: fetch to L2, but limit to 1 way (out of the 8 ways)
//   core: skip L2, go directly to L1
//   k8 rev E and later: skip L2, can go to either of the 2-ways in L1
enum PrefetchHint {
  PREFETCH_HINT_T0 = 3,  // More temporal locality
  PREFETCH_HINT_T1 = 2,
  PREFETCH_HINT_T2 = 1,  // Less temporal locality
  PREFETCH_HINT_NTA = 0  // No temporal locality
};
template <PrefetchHint hint>
void prefetch(const void* x);

// ---------------------------------------------------------------------------
// Inline implementation
// ---------------------------------------------------------------------------
template <PrefetchHint hint>
inline void prefetch(const void* x) {
#if defined(__llvm__) || defined(COMPILER_GCC)
  __builtin_prefetch(x, 0, hint);
#else
// You get no effect.  Feel free to add more sections above.
#endif
}

}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_MEM_H_
