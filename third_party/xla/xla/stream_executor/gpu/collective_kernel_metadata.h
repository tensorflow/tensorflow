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

#ifndef XLA_STREAM_EXECUTOR_GPU_COLLECTIVE_KERNEL_METADATA_H_
#define XLA_STREAM_EXECUTOR_GPU_COLLECTIVE_KERNEL_METADATA_H_

#include <stdint.h>

// Metadata parameter which is passed to the collective kernel.
// The metadata allows to compute the address of a peer's buffer in the
// collective kernel and get the current rank of a peer device.
// Right now two root pointers are getting passed. One is used for buffers
// allocated by the buffer assignment and allows kernel to address input and
// output buffers. The second one is used for buffers allocated within the
// collective kernel thunk.
struct CollectiveKernelMetadata {
  constexpr static int kMaxNumDevices = 8;
  int64_t rank;
  // Root pointer for buffers.
  int64_t buffer_root_ptrs[kMaxNumDevices];

  // Root pointer for multicast buffer for current device.
  int64_t multicast_buffer_ptr;
};

#endif  // XLA_STREAM_EXECUTOR_GPU_COLLECTIVE_KERNEL_METADATA_H_
