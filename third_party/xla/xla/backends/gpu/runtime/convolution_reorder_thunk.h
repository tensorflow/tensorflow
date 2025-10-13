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


#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/convolution_filter_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/dnn.h"

namespace xla {
namespace gpu {

// Launches the kernel that reorders input data for int8x32 convolutions.
class ConvolutionReorderThunk : public Thunk {
 public:
  ConvolutionReorderThunk(
      ThunkInfo thunk_info, ConvolutionFilterDimensions filter_dimensions,
      absl::InlinedVector<BufferAllocation::Slice, 2> operand_slices,
      absl::InlinedVector<BufferAllocation::Slice, 2> result_slices);

  ConvolutionReorderThunk(const ConvolutionReorderThunk&) = delete;
  ConvolutionReorderThunk& operator=(const ConvolutionReorderThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  // TODO: b/431980836 - Store the filter dimensions to use for serialization.
  const se::dnn::FilterDescriptor filter_descriptor_;
  absl::InlinedVector<BufferAllocation::Slice, 2> operand_buffers_;
  absl::InlinedVector<BufferAllocation::Slice, 2> result_buffers_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_CONVOLUTION_REORDER_THUNK_H_
