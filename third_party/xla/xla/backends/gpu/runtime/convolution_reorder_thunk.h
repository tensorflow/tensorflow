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

#ifndef XLA_BACKENDS_GPU_RUNTIME_CONVOLUTION_REORDER_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_CONVOLUTION_REORDER_THUNK_H_

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/convolution_filter_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/dnn.h"

namespace xla {
namespace gpu {

// Launches the kernel that reorders input data for int8x32 convolutions.
class ConvolutionReorderThunk : public Thunk {
 public:
  struct BiasBuffers {
    BufferAllocation::Slice bias_input;
    BufferAllocation::Slice bias_output;
  };

  ConvolutionReorderThunk(ThunkInfo thunk_info,
                          ConvolutionFilterDimensions filter_dimensions,
                          BufferAllocation::Slice filter_input,
                          BufferAllocation::Slice filter_output,
                          std::optional<BiasBuffers> biases);

  ConvolutionReorderThunk(const ConvolutionReorderThunk&) = delete;
  ConvolutionReorderThunk& operator=(const ConvolutionReorderThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  BufferUses buffer_uses() const override {
    BufferUses res{
        BufferUse::Read(filter_input_),
        BufferUse::Write(filter_output_),
    };
    if (biases_.has_value()) {
      res.push_back(BufferUse::Read(biases_->bias_input));
      res.push_back(BufferUse::Write(biases_->bias_output));
    }
    return res;
  }

  static absl::StatusOr<std::unique_ptr<ConvolutionReorderThunk>> FromProto(
      ThunkInfo thunk_info, const ConvolutionReorderThunkProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  const ConvolutionFilterDimensions filter_dimensions_;
  const se::dnn::FilterDescriptor filter_descriptor_;
  BufferAllocation::Slice filter_input_;
  BufferAllocation::Slice filter_output_;
  std::optional<BiasBuffers> biases_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_CONVOLUTION_REORDER_THUNK_H_
