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

#include "xla/backends/cpu/runtime/convolution_thunk.h"

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/convolution_thunk_test_util.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::cpu {
namespace {

// NOTE: This file serves to verify the basic functionality of the convolution
// thunk. Comprehensive tests cases are common for all backends and are covered
// in xla/tests/convolution_test.cc file.

template <typename T>
class ConvolutionThunkTypedTest : public ::testing::Test {};

using CorrectTypes = ::testing::Types<float, Eigen::half>;
TYPED_TEST_SUITE(ConvolutionThunkTypedTest, CorrectTypes);

template <typename ElementType>
void SuccessfulConvolution(int convolution_rank) {
  ConvolutionThunkBuilder<ElementType> builder(
      ConvolutionDimensions{convolution_rank});
  TF_ASSERT_OK_AND_ASSIGN(auto thunk, builder.Build());
  BufferAllocations allocations = builder.GetAllocations();

  // Execute thunk and wait for completion.
  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

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

  auto status_or_thunk = builder.Build();
  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_thunk.status().message(),
              ::testing::HasSubstr("Unsupported element type (S32)"));
}

TEST(ConvolutionThunkTest, CreationErrorOnTooHighConvolutionRank) {
  ConvolutionThunkBuilder<float> builder(
      ConvolutionDimensions(/*convolution_rank=*/4));

  auto status_or_thunk = builder.Build();
  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_thunk.status().message(),
              ::testing::HasSubstr("Incorrect convolution rank (4)"));
}

TEST(ConvolutionThunkTest, CreationErrorOnTooLowConvolutionRank) {
  ConvolutionThunkBuilder<float> builder(
      ConvolutionDimensions(/*convolution_rank=*/0));

  auto status_or_thunk = builder.Build();
  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_thunk.status().message(),
              ::testing::HasSubstr("Incorrect convolution rank (0)"));
}

TEST(ConvolutionThunkTest, CreationErrorOnMismatchedKernelBufferRank) {
  ConvolutionDimensions dims_2d(/*convolution_rank=*/2);
  auto input_dims = MakeInputDims(dims_2d);
  auto output_dims = MakeOutputDims(dims_2d);

  // Create kernel buffer with mismatched rank.
  ConvolutionDimensions dims_3d(/*convolution_rank=*/3);
  auto kernel_dims = MakeKernelDims(dims_3d);

  ConvolutionThunkBuilder<float> builder(input_dims, kernel_dims, output_dims);

  auto status_or_thunk = builder.Build();
  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_thunk.status().message(),
              ::testing::HasSubstr("Buffer ranks mismatch. Input rank (4) vs "
                                   "kernel rank (5) vs output rank (4)"));
}

TEST(ConvolutionThunkTest, CreationErrorOnMismatchedOutputBufferRank) {
  ConvolutionDimensions dims_2d(/*convolution_rank=*/2);
  auto input_dims = MakeInputDims(dims_2d);
  auto kernel_dims = MakeKernelDims(dims_2d);

  // Create output buffer with mismatched rank.
  ConvolutionDimensions dims_3d(/*convolution_rank=*/3);
  auto output_dims = MakeOutputDims(dims_3d);

  ConvolutionThunkBuilder<float> builder(input_dims, kernel_dims, output_dims);
  auto status_or_thunk = builder.Build();

  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_thunk.status().message(),
              ::testing::HasSubstr("Buffer ranks mismatch. Input rank (4) vs "
                                   "kernel rank (4) vs output rank (5)"));
}

TEST(ConvolutionThunkTest, CreationErrorOnBatchSizeMismatch) {
  ConvolutionDimensions dims;
  dims.batch_size = 1;
  auto input_dims = MakeInputDims(dims);
  auto kernel_dims = MakeKernelDims(dims);

  // Create output with mismatched batch size.
  dims.batch_size = 2;
  auto output_dims = MakeOutputDims(dims);

  ConvolutionThunkBuilder<float> builder(input_dims, kernel_dims, output_dims);
  auto status_or_thunk = builder.Build();

  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status_or_thunk.status().message(),
              ::testing::HasSubstr(
                  "Batch sizes mismatch. Input batch (1) vs output batch (2)"));
}

TEST(ConvolutionThunkTest, CreationErrorOnOutputChannelsMismatch) {
  ConvolutionDimensions dims;
  dims.output_channels = 3;
  auto input_dims = MakeInputDims(dims);
  auto kernel_dims = MakeKernelDims(dims);

  // Create output with output channels different than the kernel filters count.
  dims.output_channels = 4;
  auto output_dims = MakeOutputDims(dims);

  ConvolutionThunkBuilder<float> builder(input_dims, kernel_dims, output_dims);
  auto status_or_thunk = builder.Build();

  EXPECT_EQ(status_or_thunk.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status_or_thunk.status().message(),
      ::testing::HasSubstr("Output channels mismatch. Kernel filters count (3) "
                           "should be the same as output channels count (4)"));
}

TEST(ConvolutionThunkTest,
     ExecutionErrorOnMissingThreadPoolInMultiThreadedMode) {
  ConvolutionThunkBuilder<float> builder;

  auto options = MakeConvolutionOptions();
  options.multi_threaded = true;
  builder.SetOptions(options);

  TF_ASSERT_OK_AND_ASSIGN(auto thunk, builder.Build());
  BufferAllocations allocations = builder.GetAllocations();

  // Execute thunk and wait for completion.
  Thunk::ExecuteParams params;
  params.intra_op_threadpool = nullptr;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);

  // Verify that the execution was not successful.
  ASSERT_TRUE(execute_event.IsError());
  auto status = execute_event.GetError();
  EXPECT_EQ(absl::StatusCode::kInternal, status.code());
  EXPECT_EQ(
      "Intra-op threadpool must be provided for ConvolutionThunk in "
      "multi-threaded mode.",
      status.message());
}

}  // namespace
}  // namespace xla::cpu
