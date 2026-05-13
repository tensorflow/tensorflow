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

#ifndef XLA_BACKENDS_GPU_RUNTIME_DYNAMIC_SLICE_FUSION_V2_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_DYNAMIC_SLICE_FUSION_V2_THUNK_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/device_address.h"

namespace xla::gpu {

// A dynamic slice fusion thunk that wraps an embedded thunk sequence and at
// run time adjusts buffer slices for arguments and results based on the current
// while loop iteration. The byte offset for each sliced buffer is computed as:
//
//   offset + loop_iteration[loop_index] * stride
//
// This allows the embedded thunks to operate on a different slice of the buffer
// at each loop iteration without any device-to-host synchronization.
//
// Optionally, the thunk can verify at runtime that the statically-annotated
// offsets (from DynamicSliceConfig) match the actual offsets computed on
// device. This is a debugging aid and must not be enabled by default, as it
// requires a synchronous device-to-host copy of the offset scalars.
class DynamicSliceFusionV2Thunk : public Thunk {
 public:
  DynamicSliceFusionV2Thunk(
      ThunkInfo thunk_info,
      std::vector<DynamicSliceFusion::Parameter> parameters,
      std::vector<DynamicSliceFusion::Result> results,
      std::vector<BufferAllocation::Slice> parameter_buffers,
      std::vector<BufferAllocation::Slice> result_buffers,
      std::vector<BufferAllocation> slice_allocations,
      ThunkSequence embedded_thunks, bool verify_offsets = false);

  absl::Status Prepare(const PrepareParams& params) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  BufferUses buffer_uses() const override;

  const ThunkSequence& thunks() const { return executor_.thunks(); }

  absl::Span<const DynamicSliceFusion::Parameter> parameters() const {
    return parameters_;
  }
  absl::Span<const DynamicSliceFusion::Result> results() const {
    return results_;
  }

  std::string ToString(int indent) const override;

  absl::StatusOr<ThunkProto> ToProto() const override;
  static absl::StatusOr<std::unique_ptr<DynamicSliceFusionV2Thunk>> FromProto(
      ThunkInfo thunk_info, const DynamicSliceFusionThunkProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      const DeserializerWithCustomAllocations& deserializer);

 protected:
  absl::Status WalkNested(Walker callback) override;
  absl::Status TransformNested(Transformer callback) override;

 private:
  std::vector<se::DeviceAddressBase> BuildDynamicSliceBuffers(
      const BufferAllocations& orig_allocs,
      absl::Span<const WhileLoopState> loop_nest) const;

  std::vector<DynamicSliceFusion::Parameter> parameters_;
  std::vector<DynamicSliceFusion::Result> results_;

  std::vector<BufferAllocation::Slice> parameter_buffers_;
  std::vector<BufferAllocation::Slice> result_buffers_;

  // Buffer allocations for the embedded thunks. These buffer allocations match
  // the parameter and result slicing configs defined by the fusion.
  std::vector<BufferAllocation> slice_allocations_;

  ThunkExecutor executor_;

  bool verify_offsets_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_DYNAMIC_SLICE_FUSION_V2_THUNK_H_
