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

#ifndef XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_THUNK_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/convolution_lib.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// Performs 1D, 2D or 3D convolution.
class ConvolutionThunk final : public Thunk {
 public:
  struct Options {
    bool multi_threaded = false;
  };

  static absl::StatusOr<std::unique_ptr<ConvolutionThunk>> Create(
      Info info, Options options, BufferAllocation::Slice input_buffer,
      const Shape& input_shape, BufferAllocation::Slice kernel_buffer,
      const Shape& kernel_shape, BufferAllocation::Slice output_buffer,
      const Shape& output_shape, const ConvolutionDimensionNumbers& dnums,
      const Window& window, int64_t feature_group_count);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  Thunk::BufferUses buffer_uses() const final {
    return ConvolutionBufferUses(convolution_slices_);
  }

  ConvolutionDimensionNumbers dnums() const { return dnums_; }
  Window window() const { return window_; }
  int64_t feature_group_count() const {
    return convolution_canonical_dims_.feature_group_count;
  }
  const Options& options() const { return options_; }

  const ConvolutionSlices& convolution_slices() const {
    return convolution_slices_;
  }

 private:
  ConvolutionThunk(Info info, Options options,
                   ConvolutionSlices convolution_slices,
                   ConvolutionCanonicalDims convolution_canonical_dims,
                   ConvolutionDimensionNumbers dnums, Window window);

  tsl::AsyncValueRef<Thunk::ExecuteEvent> HandleEigen2DConvolution(
      const ExecuteParams& params, se::DeviceMemoryBase input,
      se::DeviceMemoryBase kernel, se::DeviceMemoryBase output);

  tsl::AsyncValueRef<Thunk::ExecuteEvent> HandleEigen3DConvolution(
      const ExecuteParams& params, se::DeviceMemoryBase input,
      se::DeviceMemoryBase kernel, se::DeviceMemoryBase output);

  Options options_;
  ConvolutionSlices convolution_slices_;
  ConvolutionCanonicalDims convolution_canonical_dims_;

  // Convolution operation parameters that were used to construct this thunk. We
  // only keep them around to be able to serialize/deserialize thunk.
  ConvolutionDimensionNumbers dnums_;
  Window window_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_THUNK_H_
