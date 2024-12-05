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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_STUB_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_STUB_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/util.h"

namespace xla::gpu {

// A stub for GPU collectives when XLA:GPU compiled without collectives support.
class GpuCollectivesStub : public NcclApi {
 public:
  bool IsGlobalConfig() const final { return false; }

  absl::StatusOr<CliqueId> CreateUniqueCliqueId() const final {
    return UnimplementedError();
  }

  absl::StatusOr<const CliqueIdCallback*> GetCliqueIdCallback(
      const CliqueIdCallback*, bool) final {
    return UnimplementedError();
  }

  absl::Status GroupStart() final { return UnimplementedError(); }
  absl::Status GroupEnd() final { return UnimplementedError(); }

 protected:
  static absl::Status UnimplementedError() {
    return Unimplemented("XLA compiled without GPU collectives support");
  }
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_STUB_H_
