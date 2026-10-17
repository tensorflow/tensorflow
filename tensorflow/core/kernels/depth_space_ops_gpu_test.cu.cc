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

#include "absl/status/status.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/depthtospace_op.h"
#include "tensorflow/core/kernels/spacetodepth_op.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_format.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

namespace tensorflow {
namespace functor {
namespace {

using GPUDevice = Eigen::GpuDevice;
using Dims = Eigen::DSizes<Eigen::DenseIndex, 4>;
using ::testing::HasSubstr;

class DepthSpaceOpGpuTest : public ::testing::Test {
 protected:
  template <typename Functor>
  absl::Status RunFunctor(const Dims& input_dims, const Dims& output_dims) {
    TTypes<float, 4>::ConstTensor input(&dummy_, input_dims);
    TTypes<float, 4>::Tensor output(&dummy_, output_dims);
    return Functor()(device_, input, /*block_size=*/2, output);
  }

  Eigen::GpuStreamDevice stream_;
  GPUDevice device_{&stream_};
  float dummy_ = 0.0f;
};

TEST_F(DepthSpaceOpGpuTest, SpaceToDepthRejectsInputOverInt32Max) {
  EXPECT_THAT(
      (RunFunctor<SpaceToDepthOpFunctor<GPUDevice, float, FORMAT_NHWC>>(
          Dims(1, 2, 2, 600000000), Dims(1, 1, 1, 2400000000))),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("input size exceeds int32 bounds")));
  EXPECT_THAT(
      (RunFunctor<SpaceToDepthOpFunctor<GPUDevice, float, FORMAT_NCHW>>(
          Dims(1, 600000000, 2, 2), Dims(1, 2400000000, 1, 1))),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("input size exceeds int32 bounds")));
}

TEST_F(DepthSpaceOpGpuTest, DepthToSpaceRejectsInputOverInt32Max) {
  EXPECT_THAT(
      (RunFunctor<DepthToSpaceOpFunctor<GPUDevice, float, FORMAT_NHWC>>(
          Dims(1, 1, 1, 2400000000), Dims(1, 2, 2, 600000000))),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("input size exceeds int32 bounds")));
  EXPECT_THAT(
      (RunFunctor<DepthToSpaceOpFunctor<GPUDevice, float, FORMAT_NCHW>>(
          Dims(1, 2400000000, 1, 1), Dims(1, 600000000, 2, 2))),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("input size exceeds int32 bounds")));
}

TEST_F(DepthSpaceOpGpuTest, EmptyInputWithDimensionOverInt32MaxSucceeds) {
  TF_EXPECT_OK(
      (RunFunctor<SpaceToDepthOpFunctor<GPUDevice, float, FORMAT_NHWC>>(
          Dims(3000000000, 0, 2, 2), Dims(3000000000, 0, 1, 4))));
  TF_EXPECT_OK(
      (RunFunctor<SpaceToDepthOpFunctor<GPUDevice, float, FORMAT_NCHW>>(
          Dims(3000000000, 0, 2, 2), Dims(3000000000, 0, 1, 1))));
  TF_EXPECT_OK(
      (RunFunctor<DepthToSpaceOpFunctor<GPUDevice, float, FORMAT_NHWC>>(
          Dims(3000000000, 0, 1, 4), Dims(3000000000, 0, 2, 1))));
  TF_EXPECT_OK(
      (RunFunctor<DepthToSpaceOpFunctor<GPUDevice, float, FORMAT_NCHW>>(
          Dims(3000000000, 0, 1, 1), Dims(3000000000, 0, 2, 2))));
}

}  // namespace
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
