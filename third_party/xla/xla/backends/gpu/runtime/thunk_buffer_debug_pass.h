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

#ifndef XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_DEBUG_PASS_H_
#define XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_DEBUG_PASS_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// Adds buffer debug tracing to thunks.
class ThunkBufferDebugPass : public ThunkPassInterface {
 public:
  enum class Mode {
    kChecksum,
    kFloatChecker,
    kBufferSaver,
  };

  // Returns an error if any of the provided module_output_slices is a tuple.
  static absl::StatusOr<std::unique_ptr<ThunkBufferDebugPass>> Create(
      Mode mode, std::vector<ShapedSlice> module_output_slices);

  absl::string_view name() const override { return "thunk-buffer-debug"; }

  absl::StatusOr<bool> Run(ThunkSequence* thunk_sequence,
                           const DebugOptions& debug_options,
                           const HloModule* absl_nullable hlo_module,
                           const se::DeviceDescription& device_info,
                           ThunkPassBufferAllocator& allocator) override;

 private:
  explicit ThunkBufferDebugPass(Mode mode,
                                std::vector<ShapedSlice> module_output_slices)
      : mode_(mode), module_output_slices_(std::move(module_output_slices)) {}

  Mode mode_;
  // Outputs of the entire HLO module graph.
  std::vector<ShapedSlice> module_output_slices_;
};

absl::StatusOr<std::vector<ShapedSlice>> GetOutputShapedBuffers(
    const HloModule* hlo_module, const BufferAssignment* buffer_assignment);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_DEBUG_PASS_H_
