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

#ifndef XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_THUNK_TEST_UTIL_H_
#define XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_THUNK_TEST_UTIL_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/convolution_thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

// Convolution dimensions to be used in the tests.
struct ConvolutionDimensions {
  explicit ConvolutionDimensions(int convolution_rank = 2)
      : convolution_rank(convolution_rank) {}

  int convolution_rank = 2;
  int batch_size = 1;
  int input_size = 3;
  int input_channels = 5;
  int kernel_size = 3;
  int output_channels = 3;
  // Correct for 0 padding, default stride, default dilation.
  int output_size = input_size - kernel_size + 1;
};

static std::vector<int64_t> MakeInputDims(
    ConvolutionDimensions dims = ConvolutionDimensions()) {
  std::vector<int64_t> input_dims = {dims.batch_size};
  for (int i = 0; i < dims.convolution_rank; ++i) {
    input_dims.push_back(dims.input_size);
  }
  input_dims.push_back(dims.input_channels);
  return input_dims;
}

static std::vector<int64_t> MakeKernelDims(
    ConvolutionDimensions dims = ConvolutionDimensions()) {
  std::vector<int64_t> kernel_dims = {};
  for (int i = 0; i < dims.convolution_rank; ++i) {
    kernel_dims.push_back(dims.kernel_size);
  }
  kernel_dims.push_back(dims.input_channels);
  kernel_dims.push_back(dims.output_channels);
  return kernel_dims;
}

static std::vector<int64_t> MakeOutputDims(
    ConvolutionDimensions dims = ConvolutionDimensions()) {
  std::vector<int64_t> output_dims = {dims.batch_size};
  for (int i = 0; i < dims.convolution_rank; ++i) {
    output_dims.push_back(dims.output_size);
  }
  output_dims.push_back(dims.output_channels);
  return output_dims;
}

template <typename ElementType>
static std::vector<ElementType> MakeDataVector(
    const std::vector<int64_t>& dims) {
  auto size = absl::c_accumulate(dims, 1, std::multiplies<int>());
  return std::vector<ElementType>(size, ElementType(0.0));
}

static ConvolutionThunk::Options MakeConvolutionOptions() {
  ConvolutionThunk::Options options;
  options.multi_threaded = false;
  return options;
}

static ConvolutionDimensionNumbers MakeConvolutionDimensionNumbers(
    int convolution_rank) {
  ConvolutionDimensionNumbers dnums;
  // Input dimensions.
  int dim = 0;
  dnums.set_input_batch_dimension(dim++);
  for (int i = 0; i < convolution_rank; ++i) {
    dnums.add_input_spatial_dimensions(dim++);
  }
  dnums.set_input_feature_dimension(dim++);

  // Kernel dimensions.
  dim = 0;
  for (int i = 0; i < convolution_rank; ++i) {
    dnums.add_kernel_spatial_dimensions(dim++);
  }
  dnums.set_kernel_input_feature_dimension(dim++);
  dnums.set_kernel_output_feature_dimension(dim++);

  // Output dimensions.
  dim = 0;
  dnums.set_output_batch_dimension(dim++);
  for (int i = 0; i < convolution_rank; ++i) {
    dnums.add_output_spatial_dimensions(dim++);
  }
  dnums.set_output_feature_dimension(dim++);

  return dnums;
}

static Window MakeWindow(int convolution_rank) {
  Window window;
  for (int i = 0; i < convolution_rank; ++i) {
    WindowDimension* window_dim = window.add_dimensions();
    window_dim->set_stride(1);
    window_dim->set_padding_low(0);
    window_dim->set_padding_high(0);
    window_dim->set_window_dilation(1);
    window_dim->set_base_dilation(1);
  }
  return window;
}

// This class is used to build ConvolutionThunk and execute it. It stores all
// the variables that are needed to create and execute the thunk, so it must be
// kept alive until the end of the execution.
template <typename ElementType>
class ConvolutionThunkBuilder {
 public:
  ConvolutionThunkBuilder(ConvolutionThunkBuilder&&) = delete;
  ConvolutionThunkBuilder& operator=(ConvolutionThunkBuilder&&) = delete;

  explicit ConvolutionThunkBuilder(
      ConvolutionDimensions dims = ConvolutionDimensions())
      : ConvolutionThunkBuilder(MakeInputDims(dims), MakeKernelDims(dims),
                                MakeOutputDims(dims)) {}

  ConvolutionThunkBuilder(absl::Span<const int64_t> input_dims,
                          absl::Span<const int64_t> kernel_dims,
                          absl::Span<const int64_t> output_dims) {
    // Convolution rank inferred from the input dimensions.
    int convolution_rank = input_dims.size() - 2;

    // Convolution parameters.
    dnums_ = MakeConvolutionDimensionNumbers(convolution_rank);
    window_ = MakeWindow(convolution_rank);

    // Actual data.
    input_ = LiteralUtil::CreateFull(input_dims, ElementType(0.0));
    kernel_ = LiteralUtil::CreateFull(kernel_dims, ElementType(0.0));
    output_ = LiteralUtil::CreateFull(output_dims, ElementType(0.0));

    input_alloc_ = CreateBufferAllocation(0, input_);
    kernel_alloc_ = CreateBufferAllocation(1, kernel_);
    output_alloc_ = CreateBufferAllocation(2, output_);
  }

  // Set convolution options. If not called before Build(), default options are
  // used.
  void SetOptions(ConvolutionThunk::Options options) {
    options_ = std::move(options);
  }

  BufferAllocations GetAllocations() {
    return CreateBufferAllocations(input_, kernel_, output_);
  }

  auto Build() {
    auto [input_slice, kernel_slice, output_slice] =
        CreateBufferAllocationSlice(*input_alloc_, *kernel_alloc_,
                                    *output_alloc_);
    return ConvolutionThunk::Create(
        {"convolution"}, options_, input_slice, input_.shape(), kernel_slice,
        kernel_.shape(), output_slice, output_.shape(), dnums_, window_,
        /*feature_group_count=*/1);
  }

 private:
  ConvolutionDimensionNumbers dnums_;
  Window window_;

  Literal input_;
  Literal kernel_;
  Literal output_;

  std::optional<BufferAllocation> input_alloc_;
  std::optional<BufferAllocation> kernel_alloc_;
  std::optional<BufferAllocation> output_alloc_;

  ConvolutionThunk::Options options_ = MakeConvolutionOptions();
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_CONVOLUTION_THUNK_TEST_UTIL_H_
