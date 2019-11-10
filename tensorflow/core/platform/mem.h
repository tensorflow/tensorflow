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

#ifndef TENSORFLOW_CORE_PLATFORM_MEM_H_
#define TENSORFLOW_CORE_PLATFORM_MEM_H_

// TODO(cwhipkey): remove this when callers use annotations directly.
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace port {

// Aligned allocation/deallocation. `minimum_alignment` must be a power of 2
// and a multiple of sizeof(void*).
void* AlignedMalloc(size_t size, int minimum_alignment);
void AlignedFree(void* aligned_memory);

void* Malloc(size_t size);
void* Realloc(void* ptr, size_t size);
void Free(void* ptr);

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

// Returns the amount of RAM available in bytes, or INT64_MAX if unknown.
int64 AvailableRam();

}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_MEM_H_
