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

#include "xla/service/gpu/runtime/nccl_api.h"

#include "xla/backends/gpu/collectives/nccl_collectives.h"
#include "xla/xla_data.pb.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200
#else
#include "third_party/nccl/nccl.h"
#endif  // TENSORFLOW_USE_ROCM

#if (defined(PLATFORM_GOOGLE) && !defined(TENSORFLOW_USE_ROCM))
#define WITH_PERSISTENT_PLAN_ALLOCATOR_SUPPORT true
#else
#define WITH_PERSISTENT_PLAN_ALLOCATOR_SUPPORT false
#endif

namespace xla::gpu {

//==-----------------------------------------------------------------------===//
// NcclApi
//==-----------------------------------------------------------------------===//

// This a default NCCL API implementation that forwards all API calls to NCCL
// itself. It is available only if NCCL + CUDA are configured at compile time.
class DefaultNcclApi final : public NcclCollectives {
 public:
};

NcclApi* NcclApi::Default() {
  static auto* nccl_api = new DefaultNcclApi();
  return nccl_api;
}

bool NcclApi::HasNcclSupport() { return true; }

}  // namespace xla::gpu
