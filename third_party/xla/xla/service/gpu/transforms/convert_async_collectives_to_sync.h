/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_CONVERT_ASYNC_COLLECTIVES_TO_SYNC_H_
#define XLA_SERVICE_GPU_TRANSFORMS_CONVERT_ASYNC_COLLECTIVES_TO_SYNC_H_

#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/collectives/convert_async_collectives_to_sync.h"

namespace xla {
namespace gpu {

class GpuConvertAsyncCollectivesToSync : public ConvertAsyncCollectivesToSync {
 public:
  using ConvertAsyncCollectivesToSync::ConvertAsyncCollectivesToSync;
  absl::string_view name() const override {
    return "gpu-convert-async-collectives-to-sync";
  }

  absl::Status ConvertAsyncInstructionsToSync(
      HloComputation* computation,
      absl::Span<const std::pair<HloInstruction*, HloInstruction*>> async_pairs)
      const override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_CONVERT_ASYNC_COLLECTIVES_TO_SYNC_H_
