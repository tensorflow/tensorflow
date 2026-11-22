/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_GPU_VMM_ALLOCATOR_GATE_H_
#define XLA_PJRT_GPU_VMM_ALLOCATOR_GATE_H_

namespace xla {

// Returns whether the GPU client may create the VMM allocator.
//
// The VMM allocator is experimental outside of open source builds, so it is
// only allowed there, or while a ScopedAllowVmmAllocatorForTesting is alive.
// This library is restricted to its package, so only the GPU client and its
// tests can depend on it.
bool IsVmmAllocatorAllowed();

// Allows the VMM allocator for as long as this object is alive, so tests can
// cover it in builds where it is otherwise disabled. Instances may nest.
class ScopedAllowVmmAllocatorForTesting {
 public:
  ScopedAllowVmmAllocatorForTesting();
  ~ScopedAllowVmmAllocatorForTesting();

  ScopedAllowVmmAllocatorForTesting(const ScopedAllowVmmAllocatorForTesting&) =
      delete;
  ScopedAllowVmmAllocatorForTesting& operator=(
      const ScopedAllowVmmAllocatorForTesting&) = delete;
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_VMM_ALLOCATOR_GATE_H_
