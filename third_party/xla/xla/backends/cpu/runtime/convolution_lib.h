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

#ifndef XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_LIB_H_
#define XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_LIB_H_

#include <cstddef>
#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// Allocation slices of the convolution operation.
struct ConvolutionSlices {
  BufferAllocation::Slice input_buffer;
  Shape input_shape;

  BufferAllocation::Slice kernel_buffer;
  Shape kernel_shape;

  BufferAllocation::Slice output_buffer;
  Shape output_shape;
};

// Returns buffer uses of the dot operation.
absl::InlinedVector<BufferUse, 4> ConvolutionBufferUses(
    const ConvolutionSlices& slices);

// Convolution dimensions in canonical form inferred from the operands shapes
// and convolution parameters.
struct ConvolutionCanonicalDims {
  // A helper struct to store the x, y and z dimensions of a tensor, introduced
  // for readability. In case of 2D convolution, only the x and y dimensions are
  // used and z is set to 0.
  struct Dims {
    explicit Dims(absl::Span<const int64_t> dims);

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Dims& d);

    int64_t rank;
    int64_t x;
    int64_t y;
    int64_t z;
  };

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ConvolutionCanonicalDims& d);

  size_t convolution_rank() const { return input_dims.rank; }

  int64_t input_batch;
  Dims input_dims;
  int64_t input_channels;

  Dims kernel_dims;
  int64_t kernel_channels;
  int64_t kernel_filters;

  Dims output_dims;

  Dims strides;
  Dims padding_before;
  Dims padding_after;
  Dims base_dilation;
  Dims window_dilation;

  int64_t feature_group_count;
};

// Get convolution dimensions in canonical form inferred from the operands
// shapes and convolution parameters.
absl::StatusOr<ConvolutionCanonicalDims> GetConvolutionCanonicalDims(
    const ConvolutionSlices& slices, const ConvolutionDimensionNumbers& dnums,
    const Window& window, int64_t feature_group_count);

template <typename Sink>
void AbslStringify(Sink& sink, const ConvolutionCanonicalDims::Dims& d) {
  switch (d.rank) {
    case 2:
      absl::Format(&sink, "[%d,%d]", d.x, d.y);
      break;
    case 3:
      absl::Format(&sink, "[%d,%d,%d]", d.x, d.y, d.z);
      break;
    default:
      absl::Format(&sink, "[invalid rank %d]", d.rank);
  }
}

template <typename Sink>
void AbslStringify(Sink& sink, const ConvolutionCanonicalDims& d) {
  absl::Format(&sink,
               "convolution_rank=%d input_batch=%d input_dims=%v "
               "input_channels=%d kernel_dims=%v kernel_channels=%d "
               "kernel_filters=%d output_dims=%v strides=%v padding_before=%v "
               "padding_after=%v base_dilation=%v window_dilation=%v "
               "feature_group_count=%d",
               d.convolution_rank(), d.input_batch, d.input_dims,
               d.input_channels, d.kernel_dims, d.kernel_channels,
               d.kernel_filters, d.output_dims, d.strides, d.padding_before,
               d.padding_after, d.base_dilation, d.window_dilation,
               d.feature_group_count);
}

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_LIB_H_
