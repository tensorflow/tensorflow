/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_CORE_COLLECTIVES_COLLECTIVES_H_
#define XLA_CORE_COLLECTIVES_COLLECTIVES_H_

namespace xla {

// Collectives is a base class for host-initiated collective operations in XLA.
//
// Host-initiated collective operations are collective operations that are
// initiated by the host runtime, i.e. in XLA:GPU the default collectives
// implementation uses NCCL and Thunks initiate collective operations of the
// runtime-managed streams.
//
// IMPORTANT: XLA also supports device-initiated collective operations, which
// are collective operations for communication between device kernels. In
// XLA:GPU device-initiated collective operations are implemented using NVSHMEM.
class Collectives {
 public:
  virtual ~Collectives() = default;
};

}  // namespace xla
#endif  // XLA_CORE_COLLECTIVES_COLLECTIVES_H_
