/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/image/sampling_kernels.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace functor {
namespace {

class KernelsTest : public ::testing::Test {
 protected:
  template <typename KernelType>
  void TestKernelValues(const KernelType& kernel, const std::vector<float>& x,
                        const std::vector<float>& expected) const {
    ASSERT_EQ(x.size(), expected.size());
    for (int i = 0; i < x.size(); ++i) {
      constexpr float kTolerance = 1e-3;
      EXPECT_NEAR(kernel(x[i]), expected[i], kTolerance);
      EXPECT_NEAR(kernel(-x[i]), expected[i], kTolerance);
    }
  }
};

TEST_F(KernelsTest, TestKernelValues) {
  // Tests kernel values against a set of known golden values
  TestKernelValues(CreateLanczos1Kernel(), {0.0f, 0.5f, 1.0f, 1.5},
                   {1.0f, 0.4052f, 0.0f, 0.0f});
  TestKernelValues(CreateLanczos3Kernel(), {0.0f, 0.5f, 1.0f, 1.5f, 2.5f, 3.5},
                   {1.0f, 0.6079f, 0.0f, -0.1351f, 0.0243f, 0.0f});
  TestKernelValues(
      CreateLanczos5Kernel(), {0.0f, 0.5f, 1.0f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5},
      {1.0f, 0.6262f, 0.0f, -0.1822f, 0.0810569f, -0.0334f, 0.0077f, 0.0f});
  TestKernelValues(CreateGaussianKernel(), {0.0f, 0.5f, 1.0f, 1.5},
                   {1.0f, 0.6065f, 0.1353f, 0.0f});

  TestKernelValues(CreateBoxKernel(), {0.0f, 0.25f, 0.5f, 1.0f},
                   {1.0f, 1.0f, 0.5f, 0.0f});
  TestKernelValues(CreateTriangleKernel(), {0.0f, 0.5f, 1.0f},
                   {1.0f, 0.5f, 0.0f});

  TestKernelValues(CreateKeysCubicKernel(), {0.0f, 0.5f, 1.0f, 1.5f, 2.5},
                   {1.0f, 0.5625f, 0.0f, -0.0625f, 0.0f});
  TestKernelValues(CreateMitchellCubicKernel(), {0.0f, 0.5f, 1.0f, 1.5f, 2.5},
                   {0.8889f, 0.5347f, 0.0556f, -0.0347f, 0.0f});
}

TEST(SamplingKernelTypeFromStringTest, Works) {
  EXPECT_EQ(SamplingKernelTypeFromString("lanczos1"), Lanczos1Kernel);
  EXPECT_EQ(SamplingKernelTypeFromString("lanczos3"), Lanczos3Kernel);
  EXPECT_EQ(SamplingKernelTypeFromString("lanczos5"), Lanczos5Kernel);
  EXPECT_EQ(SamplingKernelTypeFromString("gaussian"), GaussianKernel);
  EXPECT_EQ(SamplingKernelTypeFromString("box"), BoxKernel);
  EXPECT_EQ(SamplingKernelTypeFromString("triangle"), TriangleKernel);
  EXPECT_EQ(SamplingKernelTypeFromString("mitchellcubic"), MitchellCubicKernel);
  EXPECT_EQ(SamplingKernelTypeFromString("keyscubic"), KeysCubicKernel);
  EXPECT_EQ(SamplingKernelTypeFromString("not a kernel"),
            SamplingKernelTypeEnd);
}

}  // namespace
}  // namespace functor
}  // namespace tensorflow
