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

#ifndef XLA_SERVICE_CPU_RUNTIME_CONVOLUTION_THUNK_H_
#define XLA_SERVICE_CPU_RUNTIME_CONVOLUTION_THUNK_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// Performs 1D, 2D or 3D convolution.
class ConvolutionThunk final : public Thunk {
 public:
  struct Options {
    bool multi_threaded = false;
    bool use_acl = false;
  };
  static absl::StatusOr<std::unique_ptr<ConvolutionThunk>> Create(
      Info info, Options options, BufferAllocation::Slice input_buffer,
      const Shape& input_shape, BufferAllocation::Slice kernel_buffer,
      const Shape& kernel_shape, BufferAllocation::Slice output_buffer,
      const Shape& output_shape, const ConvolutionDimensionNumbers& dnums,
      const Window& window, int64_t feature_group_count);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  Thunk::BufferUses buffer_uses() const final {
    return {{input_buffer_, BufferUse::kRead},
            {kernel_buffer_, BufferUse::kRead},
            {output_buffer_, BufferUse::kWrite}};
  }

 private:
  ConvolutionThunk(Info info, BufferAllocation::Slice input_buffer,
                   const Shape& input_shape,
                   BufferAllocation::Slice kernel_buffer,
                   const Shape& kernel_shape,
                   BufferAllocation::Slice output_buffer,
                   const Shape& output_shape, int64_t input_batch,
                   const absl::InlinedVector<int64_t, 2>& input_dims,
                   int64_t input_channels,
                   const absl::InlinedVector<int64_t, 2>& kernel_dims,
                   int64_t kernel_channels, int64_t kernel_filters,
                   const absl::InlinedVector<int64_t, 2>& output_dims,
                   const absl::InlinedVector<int64_t, 2>& strides,
                   const absl::InlinedVector<int64_t, 2>& padding_before,
                   const absl::InlinedVector<int64_t, 2>& padding_after,
                   const absl::InlinedVector<int64_t, 2>& base_dilation,
                   const absl::InlinedVector<int64_t, 2>& window_dilation,
                   int64_t feature_group_count, Options options);

  void HandleACLConvolution(const ExecuteParams& params,
                            se::DeviceMemoryBase input,
                            se::DeviceMemoryBase kernel,
                            se::DeviceMemoryBase output);
  void HandleEigen2DConvolution(const ExecuteParams& params,
                                se::DeviceMemoryBase input,
                                se::DeviceMemoryBase kernel,
                                se::DeviceMemoryBase output);
  void HandleEigen3DConvolution(const ExecuteParams& params,
                                se::DeviceMemoryBase input,
                                se::DeviceMemoryBase kernel,
                                se::DeviceMemoryBase output);

  // A helper struct to store the x, y and z dimensions of a tensor, introduced
  // for readability.
  // In case of 2D convolution, only the x and y dimensions are used and z is
  // set to 0.
  struct Dims {
    explicit Dims(const absl::InlinedVector<int64_t, 2>& dims);

    int64_t x;
    int64_t y;
    int64_t z;
  };

  BufferAllocation::Slice input_buffer_;
  Shape input_shape_;

  BufferAllocation::Slice kernel_buffer_;
  Shape kernel_shape_;

  BufferAllocation::Slice output_buffer_;
  Shape output_shape_;

  int64_t input_batch_;
  Dims input_dims_;
  int64_t input_channels_;
  Dims kernel_dims_;
  int64_t kernel_channels_;
  int64_t kernel_filters_;
  Dims output_dims_;
  Dims strides_;
  Dims padding_before_;
  Dims padding_after_;
  Dims base_dilation_;
  Dims window_dilation_;
  int64_t feature_group_count_;
  int convolution_rank_;
  Options options_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_CONVOLUTION_THUNK_H_
