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
// For each kernel parameter `param_to_peers` contains the N peer pointers to
// the same parameter at the peer device, where N is the number of devices
// participating in the collective kernel.
// This information is structured as the
// single dimensional array with the following layout:
// [
//   param0_peer0, param0_peer1, ..., param0_peerN,
//   param1_peer0, param1_peer1, ..., param1_peerN,
//   ...
// ]
//
// Each parameter might also have an assosiated pointer to the root of the
// multicast address space for the current device. These parameters are stored
// in the same order as the parameters themselves in the array:
// [param0_multicast_root, param1_multicast_root, ...]
//
// `param_to_multimem_addresses` is a pointer to the array of the multicast root
// pointers for each parameter.
struct CollectiveKernelMetadata {
  uint64_t rank;
  void** param_to_peers;
  void** param_to_multimem_addresses;
};

#endif  // XLA_STREAM_EXECUTOR_GPU_COLLECTIVE_KERNEL_METADATA_H_
