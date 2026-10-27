/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/maxpooling_op_gpu.h"

#include <cstdint>

#include "absl/status/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_format.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

namespace tensorflow {
namespace functor {
namespace {

using ::testing::HasSubstr;

constexpr int kBatch = 220;
constexpr int kHeight = 487;
constexpr int kWidth = 487;
constexpr int kChannels = 42;
constexpr int kPooledSize = 31;
constexpr int kWindow = 3;
constexpr int kStride = 16;

class MaxPoolingOpGpuTest : public ::testing::Test {
 protected:
  Eigen::GpuStreamDevice stream_;
  Eigen::GpuDevice device_{&stream_};
  float dummy_ = 0.0f;
  int64_t dummy_mask_ = 0;
};

TEST_F(MaxPoolingOpGpuTest, ForwardRejectsInputOverInt32Max) {
  EXPECT_THAT(
      MaxPoolForwardWithOptionalArgmax<float>()(
          &dummy_, kBatch, kHeight, kWidth, kChannels, kPooledSize, kPooledSize,
          kWindow, kWindow, kStride, kStride, /*pad_t=*/0,
          /*pad_l=*/0, &dummy_, &dummy_mask_, device_,
          /*propagate_nans=*/false, /*include_batch_in_index=*/false),
      absl_testing::StatusIs(absl::StatusCode::kInternal,
                             HasSubstr("input size exceeds int32 bounds")));
}

TEST_F(MaxPoolingOpGpuTest, GradBackwardNoMaskRejectsInputOverInt32Max) {
  for (TensorFormat data_format : {FORMAT_NHWC, FORMAT_NCHW}) {
    EXPECT_THAT(
        MaxPoolGradBackwardNoMask<float>()(
            data_format, &dummy_, &dummy_, kBatch, kPooledSize, kPooledSize,
            kChannels, kHeight, kWidth, kWindow, kWindow, kStride, kStride,
            /*pad_t=*/0, /*pad_l=*/0, &dummy_, &dummy_, device_),
        absl_testing::StatusIs(absl::StatusCode::kInternal,
                               HasSubstr("input size exceeds int32 bounds")));
  }
}

#if GOOGLE_CUDA
TEST_F(MaxPoolingOpGpuTest, ForwardNchwVectCRejectsInputOverInt32Max) {
  int32_t dummy_int = 0;
  EXPECT_THAT(
      MaxPoolForwardNoMask_NCHW_VECT_C()(
          &dummy_int, kBatch, kHeight, kWidth, kChannels, kPooledSize,
          kPooledSize, kWindow, kWindow, kStride, kStride, /*pad_t=*/0,
          /*pad_l=*/0, &dummy_int, device_),
      absl_testing::StatusIs(absl::StatusCode::kInternal,
                             HasSubstr("input size exceeds int32 bounds")));
}
#endif  // GOOGLE_CUDA

TEST_F(MaxPoolingOpGpuTest, EmptyOutputWithLargeInputSucceeds) {
  TF_EXPECT_OK(MaxPoolForwardWithOptionalArgmax<float>()(
      &dummy_, kBatch, kHeight, kWidth, kChannels, /*pooled_height=*/0,
      kPooledSize, kWindow, kWindow, kStride, kStride, /*pad_t=*/0,
      /*pad_l=*/0, &dummy_, &dummy_mask_, device_, /*propagate_nans=*/false,
      /*include_batch_in_index=*/false));
  TF_EXPECT_OK(MaxPoolGradBackwardNoMask<float>()(
      FORMAT_NHWC, &dummy_, &dummy_, kBatch, /*pooled_height=*/0, kPooledSize,
      kChannels, kHeight, kWidth, kWindow, kWindow, kStride, kStride,
      /*pad_t=*/0, /*pad_l=*/0, &dummy_, &dummy_, device_));
}

}  // namespace
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
