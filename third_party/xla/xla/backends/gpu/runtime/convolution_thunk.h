/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_CONVOLUTION_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_CONVOLUTION_THUNK_H_

#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/traced_command.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

// This class stores everything that StreamExecutor needs to launch a DNN
// convolution. It implements both Thunk (via ExecuteOnStream) and Command
// (via TracedCommand) so it can be used directly in command buffers without
// a separate ConvolutionCmd wrapper. The default Record() inherited from
// TracedCommand traces ExecuteOnStream on the trace stream.
class ConvolutionThunk : public TracedCommand {
 public:
  // Constructs a thunk for launching a DNN convolution.
  //
  // operand_slices should be in the same order as cudnn_call->operands().
  static absl::StatusOr<std::unique_ptr<ConvolutionThunk>> Create(
      ThunkInfo thunk_info, GpuConvDescriptor descriptor,
      std::vector<ShapedSlice> operand_slices,
      std::vector<ShapedSlice> result_slices,
      BufferAllocation::Slice scratch_slice);

  ConvolutionThunk(const ConvolutionThunk&) = delete;
  ConvolutionThunk& operator=(const ConvolutionThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  BufferUses buffer_uses() const override;

  static absl::StatusOr<std::unique_ptr<ConvolutionThunk>> FromProto(
      ThunkInfo thunk_info, const ConvolutionThunkProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  ConvolutionThunk(ThunkInfo thunk_info, GpuConvDescriptor descriptor,
                   GpuConvConfig config,
                   std::vector<ShapedSlice> operand_slices,
                   std::vector<ShapedSlice> result_slices,
                   BufferAllocation::Slice scratch_slice);

  RunConvOptions GetOrCreate(const GpuConvConfig& config,
                             const se::Stream* stream);

  absl::Mutex mu_;
  absl::flat_hash_map<const se::StreamExecutor*,
                      std::unique_ptr<GenericConvRunner>>
      cache_ ABSL_GUARDED_BY(mu_);

  std::vector<ShapedSlice> operand_buffers_;
  std::vector<ShapedSlice> result_buffers_;
  BufferAllocation::Slice scratch_buffer_;
  // Technically this is only needed during initialization to create the
  // GpuConvConfig, but the actual GpuConvConfig is hard to serialize. So we
  // keep the descriptor around for serialization purposes.
  const GpuConvDescriptor descriptor_;
  // Convolution config
  const GpuConvConfig config_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_CONVOLUTION_THUNK_H_
