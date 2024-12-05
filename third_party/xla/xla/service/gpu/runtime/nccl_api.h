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

#ifndef XLA_SERVICE_GPU_RUNTIME_NCCL_API_H_
#define XLA_SERVICE_GPU_RUNTIME_NCCL_API_H_

#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// NcclApi
//===----------------------------------------------------------------------===//

// NcclApi hides implementation detail of collective operations built on top of
// NCCL library so that no other parts of XLA should include nccl.h header
// directly (or indirectly).

class NcclApi : public GpuCollectives {
 public:
  virtual ~NcclApi() = default;

  // Returns a default NcclApi for a current process. Can be a real one based on
  // NCCL or a stub if XLA compiled without NCCL or CUDA support.
  static NcclApi* Default();

  // Returns true if XLA is compiled with NCCL support, otherwise returns false.
  // If false, Default() will return a stub implementation.
  static bool HasNcclSupport();
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_NCCL_API_H_
