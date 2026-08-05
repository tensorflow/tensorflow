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

// oneDNN library support, implementing the general DnnSupport interface.

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_DNN_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_DNN_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace sycl {

// onednn-library based DNN support. For details on overridden interface
// functions, see dnn.h.
class OnednnSupport : public dnn::DnnSupport {
 public:
  explicit OnednnSupport(StreamExecutor* parent);

  absl::Status Init() override;
  absl::StatusOr<stream_executor::dnn::VersionInfo> GetVersion() override;

  // Static helper to get oneDNN version without requiring an instance.
  static absl::StatusOr<stream_executor::dnn::VersionInfo> GetOnednnVersion();

  absl::Status DoConvolveWithGpuConfig(
      Stream* stream, const xla::gpu::GpuConvConfig& config,
      absl::Span<const DeviceMemoryBase> operand_se_buffers,
      DeviceMemoryBase result_se_buffer, ScratchAllocator* scratch_allocator);

  absl::Status DoPoolForward(dnn::DataType element_type, Stream* stream,
                             const dnn::PoolingDescriptor& pooling_dimensions,
                             const dnn::BatchDescriptor& input_dimensions,
                             DeviceMemoryBase input_data,
                             const dnn::BatchDescriptor& output_dimensions,
                             DeviceMemoryBase output_data,
                             ScratchAllocator* workspace_allocator) override {
    return absl::UnimplementedError(
        "DoPoolForward is not implemented for SYCL");
  }

  absl::Status DoPoolBackward(dnn::DataType element_type, Stream* stream,
                              const dnn::PoolingDescriptor& pooling_dimensions,
                              const dnn::BatchDescriptor& input_dimensions,
                              DeviceMemoryBase input_data,
                              const dnn::BatchDescriptor& output_dimensions,
                              DeviceMemoryBase output_data,
                              DeviceMemoryBase input_diff_data,
                              DeviceMemoryBase output_diff_data,
                              ScratchAllocator* workspace_allocator) override {
    return absl::UnimplementedError(
        "DoPoolBackward is not implemented for SYCL");
  }

  absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>> ConvolveRunnerFromDesc(
      Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
      dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor) override {
    return absl::UnimplementedError(
        "ConvolveRunnerFromDesc is not implemented for SYCL");
  }

  absl::Status DoCtcLoss(Stream* stream, dnn::DataType element_type,
                         const dnn::RnnStateTensorDescriptor& probs_desc,
                         DeviceMemoryBase probs_data,
                         absl::Span<const int> labels_data,
                         absl::Span<const int> labels_lengths_data,
                         absl::Span<const int> input_lengths_data,
                         DeviceMemoryBase costs_data,
                         const dnn::RnnStateTensorDescriptor& grads_desc,
                         DeviceMemoryBase grads_data,
                         DeviceMemory<uint8_t> scratch_memory,
                         int ctc_loss_algo_id) override {
    return absl::UnimplementedError("DoCtcLoss is not implemented for SYCL");
  }

 private:
  StreamExecutor* parent_;  // Parent executor object. Not owned.

  OnednnSupport(const OnednnSupport&) = delete;
  void operator=(const OnednnSupport&) = delete;
};

}  // namespace sycl
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_DNN_H_
