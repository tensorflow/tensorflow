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

#include "xla/service/cpu/runtime/convolution_thunk.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "Eigen/Core"  // from @eigen_archive
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/buffer_allocations.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

// NOTE: This file serves to verify the basic functionality of the convolution
// thunk. Comprehensive tests cases are common for all backends and are covered
// in xla/tests/convolution_test.cc file.

// Convolution dimensions to be used in the tests.
struct ConvolutionDimensions {
  int batch_size = 1;
  int input_size = 3;
  int input_channels = 5;
  int kernel_size = 3;
  int output_channels = 3;
  // Correct for 0 padding, default stride, default dilation.
  int output_size = input_size - kernel_size + 1;
};

template <typename T>
class ConvolutionThunkTypedTest : public ::testing::Test {};

using CorrectTypes = ::testing::Types<float, Eigen::half>;
TYPED_TEST_SUITE(ConvolutionThunkTypedTest, CorrectTypes);

std::vector<int64_t> MakeInputDims(
    int convolution_rank,
    ConvolutionDimensions dims = ConvolutionDimensions()) {
  std::vector<int64_t> input_dims = {dims.batch_size};
  for (int i = 0; i < convolution_rank; ++i) {
    input_dims.push_back(dims.input_size);
  }
  input_dims.push_back(dims.input_channels);
  return input_dims;
}

std::vector<int64_t> MakeKernelDims(
    int convolution_rank,
    ConvolutionDimensions dims = ConvolutionDimensions()) {
  std::vector<int64_t> kernel_dims = {};
  for (int i = 0; i < convolution_rank; ++i) {
    kernel_dims.push_back(dims.kernel_size);
  }
  kernel_dims.push_back(dims.input_channels);
  kernel_dims.push_back(dims.output_channels);
  return kernel_dims;
}

std::vector<int64_t> MakeOutputDims(
    int convolution_rank,
    ConvolutionDimensions dims = ConvolutionDimensions()) {
  std::vector<int64_t> output_dims = {dims.batch_size};
  for (int i = 0; i < convolution_rank; ++i) {
    output_dims.push_back(dims.output_size);
  }
  output_dims.push_back(dims.output_channels);
  return output_dims;
}

template <typename ElementType>
std::vector<ElementType> MakeDataVector(const std::vector<int64_t>& dims) {
  auto size = absl::c_accumulate(dims, 1, std::multiplies<int>());
  return std::vector<ElementType>(size, ElementType(0.0));
}

template <typename ElementType>
std::vector<MaybeOwningDeviceMemory> MakeBuffers(
    const std::vector<ElementType>& input,
    const std::vector<ElementType>& kernel,
    const std::vector<ElementType>& output) {
  std::vector<MaybeOwningDeviceMemory> buffers;
  size_t input_size_in_bytes = input.size() * sizeof(ElementType);
  buffers.emplace_back(se::DeviceMemoryBase(input.data(), input_size_in_bytes));
  size_t kernel_size_in_bytes = kernel.size() * sizeof(ElementType);
  buffers.emplace_back(
      se::DeviceMemoryBase(kernel.data(), kernel_size_in_bytes));
  size_t output_size_in_bytes = output.size() * sizeof(ElementType);
  buffers.emplace_back(
      se::DeviceMemoryBase(output.data(), output_size_in_bytes));
  return buffers;
}

ConvolutionThunk::Options MakeConvolutionOptions() {
  ConvolutionThunk::Options options;
  options.multi_threaded = false;
  options.use_acl = false;
  return options;
}

ConvolutionDimensionNumbers MakeConvolutionDimensionNumbers(
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

Window MakeWindow(int convolution_rank) {
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
  auto Build(int convolution_rank,
             ConvolutionDimensions dims = ConvolutionDimensions()) {
    // Data dimensions.
    auto input_dims = MakeInputDims(convolution_rank, dims);
    auto kernel_dims = MakeKernelDims(convolution_rank, dims);
    auto output_dims = MakeOutputDims(convolution_rank, dims);

    // Actual data.
    input_ = MakeDataVector<ElementType>(input_dims);
    kernel_ = MakeDataVector<ElementType>(kernel_dims);
    output_ = MakeDataVector<ElementType>(output_dims);

    // Buffers.
    size_t input_size_in_bytes = input_.size() * sizeof(ElementType);
    buffers_.emplace_back(
        se::DeviceMemoryBase(input_.data(), input_size_in_bytes));
    size_t kernel_size_in_bytes = kernel_.size() * sizeof(ElementType);
    buffers_.emplace_back(
        se::DeviceMemoryBase(kernel_.data(), kernel_size_in_bytes));
    size_t output_size_in_bytes = output_.size() * sizeof(ElementType);
    buffers_.emplace_back(
        se::DeviceMemoryBase(output_.data(), output_size_in_bytes));

    // Buffer allocations.
    allocations_ = std::make_unique<BufferAllocations>(buffers_);

    input_alloc_ =
        std::make_unique<BufferAllocation>(0, input_size_in_bytes, 0);
    kernel_alloc_ =
        std::make_unique<BufferAllocation>(1, kernel_size_in_bytes, 0);
    output_alloc_ =
        std::make_unique<BufferAllocation>(2, output_size_in_bytes, 0);

    BufferAllocation::Slice input_slice(input_alloc_.get(), 0,
                                        input_size_in_bytes);
    BufferAllocation::Slice kernel_slice(kernel_alloc_.get(), 0,
                                         kernel_size_in_bytes);
    BufferAllocation::Slice output_slice(output_alloc_.get(), 0,
                                         output_size_in_bytes);

    // Shapes.
    auto primitive_type = primitive_util::NativeToPrimitiveType<ElementType>();
    Shape input_shape = ShapeUtil::MakeShape(primitive_type, input_dims);
    Shape kernel_shape = ShapeUtil::MakeShape(primitive_type, kernel_dims);
    Shape output_shape = ShapeUtil::MakeShape(primitive_type, output_dims);

    // Convolution parameters.
    auto options = MakeConvolutionOptions();
    auto dnums = MakeConvolutionDimensionNumbers(convolution_rank);
    auto window = MakeWindow(convolution_rank);

    // Create thunk.
    return ConvolutionThunk::Create(
        {"convolution"}, options, std::move(input_slice), input_shape,
        std::move(kernel_slice), kernel_shape, std::move(output_slice),
        output_shape, dnums, window,
        /*feature_group_count=*/1);
  }

  // Get execution parameters for the last created thunk.
  auto GetExecutionParams() {
    return Thunk::ExecuteParams{nullptr, allocations_.get()};
  }

 private:
  std::vector<ElementType> input_;
  std::vector<ElementType> kernel_;
  std::vector<ElementType> output_;
  std::vector<MaybeOwningDeviceMemory> buffers_;

  // Unique pointers, because they are created only when needed.
  std::unique_ptr<BufferAllocations> allocations_;
  std::unique_ptr<BufferAllocation> input_alloc_;
  std::unique_ptr<BufferAllocation> kernel_alloc_;
  std::unique_ptr<BufferAllocation> output_alloc_;
};

template <typename ElementType>
void SuccessfulConvolution(int convolution_rank) {
  ConvolutionThunkBuilder<ElementType> builder;
  TF_ASSERT_OK_AND_ASSIGN(auto thunk, builder.Build(convolution_rank))

  // Execute thunk and wait for completion.
  Thunk::ExecuteParams params = builder.GetExecutionParams();
  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);

  // Verify that the execution was successful.
  // NOTE: We don't check the correctness of the output here, just that it
  // executes without errors. Numerics is verified in generic convolution tests.
  ASSERT_FALSE(execute_event.IsError()) << execute_event.GetError();
}

TYPED_TEST(ConvolutionThunkTypedTest, SuccessfulConvolution1D) {
  SuccessfulConvolution<TypeParam>(/*convolution_rank=*/1);
}

TYPED_TEST(ConvolutionThunkTypedTest, SuccessfulConvolution2D) {
  SuccessfulConvolution<TypeParam>(/*convolution_rank=*/2);
}

TYPED_TEST(ConvolutionThunkTypedTest, SuccessfulConvolution3D) {
  SuccessfulConvolution<TypeParam>(/*convolution_rank=*/3);
}

TEST(ConvolutionThunkTest, CreationErrorOnUnsupportedType) {
  ConvolutionThunkBuilder<int> builder;

  auto status_or_thunk = builder.Build(/*convolution_rank=*/2);
  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_thunk.status().message(),
              ::testing::HasSubstr("Unsupported element type (S32)"));
}

TEST(ConvolutionThunkTest, CreationErrorOnIncorrectConvolutionRank) {
  ConvolutionThunkBuilder<float> builder;

  auto status_or_thunk = builder.Build(/*convolution_rank=*/4);
  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_thunk.status().message(),
              ::testing::HasSubstr("Incorrect convolution rank (4)"));
}

}  // namespace
}  // namespace xla::cpu
