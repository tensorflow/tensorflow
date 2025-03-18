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

#ifndef XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_CONVOLUTION_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_CONVOLUTION_THUNK_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/convolution_lib.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_fusion_thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// Convolution operation implemented on top of XNNPACK.
class XnnConvolutionThunk final : public XnnFusionThunk {
 public:
  static absl::StatusOr<std::unique_ptr<XnnConvolutionThunk>> Create(
      Options options, Info info, BufferAllocation::Slice input_buffer,
      const Shape& input_shape, BufferAllocation::Slice kernel_buffer,
      const Shape& kernel_shape, BufferAllocation::Slice output_buffer,
      const Shape& output_shape, const ConvolutionDimensionNumbers& dnums,
      const Window& window, int64_t feature_group_count);

  ConvolutionDimensionNumbers dnums() const { return dnums_; }
  Window window() const { return window_; }

  int64_t feature_group_count() const {
    return convolution_canonical_dims_.feature_group_count;
  }

  const ConvolutionSlices& convolution_slices() const {
    return convolution_slices_;
  }

 protected:
  std::string fusion_kind() const final;
  std::string fusion_description() const final;

  bool has_fusion_details() const final { return true; }
  std::vector<std::string> fusion_details() const final;

  std::string argument_name(size_t index) const final;
  std::string result_name(size_t index) const final;

 private:
  XnnConvolutionThunk(Options options, Info info,
                      ConvolutionSlices convolution_slices,
                      ConvolutionCanonicalDims convolution_canonical_dims,
                      ConvolutionDimensionNumbers dnums, Window window);

  absl::StatusOr<xnn_subgraph_t> BuildConvolutionSubgraph(
      absl::Span<const Argument> arguments, absl::Span<const Result> results,
      absl::Span<const se::DeviceMemoryBase> arguments_buffers,
      absl::Span<const se::DeviceMemoryBase> results_buffers);

  ConvolutionSlices convolution_slices_;
  ConvolutionCanonicalDims convolution_canonical_dims_;

  // Convolution operation parameters that were used to construct this thunk. We
  // only keep them around to be able to serialize/deserialize thunk.
  ConvolutionDimensionNumbers dnums_;
  Window window_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_CONVOLUTION_THUNK_H_
