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

#include "xla/backends/gpu/collectives/gpu_collectives_stub.h"
#include "xla/service/gpu/runtime/nccl_api.h"

namespace xla::gpu {

// This is a NCCL API stub that is linked into the process when XLA compiled
// without NCCL or CUDA support. It returns errors from all API calls. This stub
// makes it always safe to include NCCL API headers everywhere in XLA without
// #ifdefs or complex build rules magic. All magic handled by `:nccl_api`.

//===----------------------------------------------------------------------===//
// NcclApiStub
//===----------------------------------------------------------------------===//

class NcclApiStub final : public GpuCollectivesStub {
 public:
};

NcclApi* NcclApi::Default() {
  static auto* nccl_api = new NcclApiStub();
  return nccl_api;
}

bool NcclApi::HasNcclSupport() { return false; }

}  // namespace xla::gpu
