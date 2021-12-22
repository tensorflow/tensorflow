/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"

#include <complex>
#include <limits>
#include <string>

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/device_memory.h"

namespace xla {
namespace gpu {
namespace {

class BufferComparatorTest : public testing::Test {
 protected:
  BufferComparatorTest()
      : platform_(
            se::MultiPlatformManager::PlatformWithName("cuda").ValueOrDie()),
        stream_exec_(platform_->ExecutorForDevice(0).ValueOrDie()) {}

  // Take floats only for convenience. Still uses ElementType internally.
  template <typename ElementType>
  bool CompareEqualBuffers(const std::vector<ElementType>& lhs,
                           const std::vector<ElementType>& rhs) {
    se::Stream stream(stream_exec_);
    stream.Init();

    se::ScopedDeviceMemory<ElementType> lhs_buffer =
        stream_exec_->AllocateOwnedArray<ElementType>(lhs.size());
    se::ScopedDeviceMemory<ElementType> rhs_buffer =
        stream_exec_->AllocateOwnedArray<ElementType>(rhs.size());

    stream.ThenMemcpy(lhs_buffer.ptr(), lhs.data(), lhs_buffer->size());
    stream.ThenMemcpy(rhs_buffer.ptr(), rhs.data(), rhs_buffer->size());
    TF_CHECK_OK(stream.BlockHostUntilDone());

    BufferComparator comparator(
        ShapeUtil::MakeShape(
            primitive_util::NativeToPrimitiveType<ElementType>(),
            {static_cast<int64_t>(lhs_buffer->ElementCount())}),
        HloModuleConfig());
    return comparator.CompareEqual(&stream, *lhs_buffer, *rhs_buffer)
        .ConsumeValueOrDie();
  }

  // Take floats only for convenience. Still uses ElementType internally.
  template <typename ElementType>
  bool CompareEqualFloatBuffers(const std::vector<float>& lhs_float,
                                const std::vector<float>& rhs_float) {
    std::vector<ElementType> lhs(lhs_float.begin(), lhs_float.end());
    std::vector<ElementType> rhs(rhs_float.begin(), rhs_float.end());
    return CompareEqualBuffers(lhs, rhs);
  }

  template <typename ElementType>
  bool CompareEqualComplex(const std::vector<std::complex<ElementType>>& lhs,
                           const std::vector<std::complex<ElementType>>& rhs) {
    return CompareEqualBuffers<std::complex<ElementType>>(lhs, rhs);
  }

  se::Platform* platform_;
  se::StreamExecutor* stream_exec_;
};

TEST_F(BufferComparatorTest, TestComplex) {
  EXPECT_FALSE(
      CompareEqualComplex<float>({{0.1, 0.2}, {2, 3}}, {{0.1, 0.2}, {6, 7}}));
  EXPECT_TRUE(CompareEqualComplex<float>({{0.1, 0.2}, {2, 3}},
                                         {{0.1, 0.2}, {2.2, 3.3}}));
  EXPECT_TRUE(
      CompareEqualComplex<float>({{0.1, 0.2}, {2, 3}}, {{0.1, 0.2}, {2, 3}}));

  EXPECT_FALSE(
      CompareEqualComplex<float>({{0.1, 0.2}, {2, 3}}, {{0.1, 0.2}, {6, 3}}));

  EXPECT_FALSE(
      CompareEqualComplex<float>({{0.1, 0.2}, {2, 3}}, {{0.1, 0.2}, {6, 7}}));

  EXPECT_FALSE(
      CompareEqualComplex<float>({{0.1, 0.2}, {2, 3}}, {{0.1, 6}, {2, 3}}));
  EXPECT_TRUE(CompareEqualComplex<double>({{0.1, 0.2}, {2, 3}},
                                          {{0.1, 0.2}, {2.2, 3.3}}));
  EXPECT_FALSE(
      CompareEqualComplex<double>({{0.1, 0.2}, {2, 3}}, {{0.1, 0.2}, {2, 7}}));
}

TEST_F(BufferComparatorTest, TestNaNs) {
  EXPECT_TRUE(
      CompareEqualFloatBuffers<Eigen::half>({std::nanf("")}, {std::nanf("")}));
  // NaN values with different bit patterns should compare equal.
  EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>({std::nanf("")},
                                                    {std::nanf("1234")}));
  EXPECT_FALSE(CompareEqualFloatBuffers<Eigen::half>({std::nanf("")}, {1.}));

  EXPECT_TRUE(
      CompareEqualFloatBuffers<float>({std::nanf("")}, {std::nanf("")}));
  // NaN values with different bit patterns should compare equal.
  EXPECT_TRUE(
      CompareEqualFloatBuffers<float>({std::nanf("")}, {std::nanf("1234")}));
  EXPECT_FALSE(CompareEqualFloatBuffers<float>({std::nanf("")}, {1.}));

  EXPECT_TRUE(
      CompareEqualFloatBuffers<double>({std::nanf("")}, {std::nanf("")}));
  // NaN values with different bit patterns should compare equal.
  EXPECT_TRUE(
      CompareEqualFloatBuffers<double>({std::nanf("")}, {std::nanf("1234")}));
  EXPECT_FALSE(CompareEqualFloatBuffers<double>({std::nanf("")}, {1.}));
}

TEST_F(BufferComparatorTest, TestInfs) {
  const auto inf = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(CompareEqualFloatBuffers<Eigen::half>({inf}, {std::nanf("")}));
  EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>({inf}, {inf}));
  EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>({inf}, {65504}));
  EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>({-inf}, {-65504}));
  EXPECT_FALSE(CompareEqualFloatBuffers<Eigen::half>({inf}, {-65504}));
  EXPECT_FALSE(CompareEqualFloatBuffers<Eigen::half>({-inf}, {65504}));
  EXPECT_FALSE(CompareEqualFloatBuffers<Eigen::half>({inf}, {20}));
  EXPECT_FALSE(CompareEqualFloatBuffers<Eigen::half>({inf}, {-20}));
  EXPECT_FALSE(CompareEqualFloatBuffers<Eigen::half>({-inf}, {20}));
  EXPECT_FALSE(CompareEqualFloatBuffers<Eigen::half>({-inf}, {-20}));

  EXPECT_FALSE(CompareEqualFloatBuffers<float>({inf}, {std::nanf("")}));
  EXPECT_TRUE(CompareEqualFloatBuffers<float>({inf}, {inf}));
  EXPECT_FALSE(CompareEqualFloatBuffers<float>({inf}, {65504}));
  EXPECT_FALSE(CompareEqualFloatBuffers<float>({-inf}, {-65504}));
  EXPECT_FALSE(CompareEqualFloatBuffers<float>({inf}, {-65504}));
  EXPECT_FALSE(CompareEqualFloatBuffers<float>({-inf}, {65504}));
  EXPECT_FALSE(CompareEqualFloatBuffers<float>({inf}, {20}));
  EXPECT_FALSE(CompareEqualFloatBuffers<float>({inf}, {-20}));
  EXPECT_FALSE(CompareEqualFloatBuffers<float>({-inf}, {20}));
  EXPECT_FALSE(CompareEqualFloatBuffers<float>({-inf}, {-20}));

  EXPECT_FALSE(CompareEqualFloatBuffers<double>({inf}, {std::nanf("")}));
  EXPECT_TRUE(CompareEqualFloatBuffers<double>({inf}, {inf}));
  EXPECT_FALSE(CompareEqualFloatBuffers<double>({inf}, {65504}));
  EXPECT_FALSE(CompareEqualFloatBuffers<double>({-inf}, {-65504}));
  EXPECT_FALSE(CompareEqualFloatBuffers<double>({inf}, {-65504}));
  EXPECT_FALSE(CompareEqualFloatBuffers<double>({-inf}, {65504}));
  EXPECT_FALSE(CompareEqualFloatBuffers<double>({inf}, {20}));
  EXPECT_FALSE(CompareEqualFloatBuffers<double>({inf}, {-20}));
  EXPECT_FALSE(CompareEqualFloatBuffers<double>({-inf}, {20}));
  EXPECT_FALSE(CompareEqualFloatBuffers<double>({-inf}, {-20}));
}

TEST_F(BufferComparatorTest, TestNumbers) {
  EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>({20}, {20.1}));
  EXPECT_FALSE(CompareEqualFloatBuffers<Eigen::half>({0}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>({0.9}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>({9}, {10}));
  EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>({10}, {9}));

  EXPECT_TRUE(CompareEqualFloatBuffers<float>({20}, {20.1}));
  EXPECT_FALSE(CompareEqualFloatBuffers<float>({0}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<float>({0.9}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<float>({9}, {10}));
  EXPECT_TRUE(CompareEqualFloatBuffers<float>({10}, {9}));

  EXPECT_TRUE(CompareEqualFloatBuffers<double>({20}, {20.1}));
  EXPECT_FALSE(CompareEqualFloatBuffers<double>({0}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<double>({0.9}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<double>({9}, {10}));
  EXPECT_TRUE(CompareEqualFloatBuffers<double>({10}, {9}));

  EXPECT_TRUE(CompareEqualFloatBuffers<int8_t>({100}, {101}));
  EXPECT_FALSE(CompareEqualFloatBuffers<int8_t>({0}, {10}));
  EXPECT_TRUE(CompareEqualFloatBuffers<int8_t>({9}, {10}));
  EXPECT_TRUE(CompareEqualFloatBuffers<int8_t>({90}, {100}));
  EXPECT_TRUE(CompareEqualFloatBuffers<int8_t>({100}, {90}));
  EXPECT_FALSE(CompareEqualFloatBuffers<int8_t>({-128}, {127}));
}

TEST_F(BufferComparatorTest, TestMultiple) {
  {
    EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>(
        {20, 30, 40, 50, 60}, {20.1, 30.1, 40.1, 50.1, 60.1}));
    std::vector<float> lhs(200);
    std::vector<float> rhs(200);
    for (int i = 0; i < 200; i++) {
      EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>(lhs, rhs))
          << "should be the same at index " << i;
      lhs[i] = 3;
      rhs[i] = 5;
      EXPECT_FALSE(CompareEqualFloatBuffers<Eigen::half>(lhs, rhs))
          << "should be the different at index " << i;
      lhs[i] = 0;
      rhs[i] = 0;
    }
  }

  {
    EXPECT_TRUE(CompareEqualFloatBuffers<float>(
        {20, 30, 40, 50, 60}, {20.1, 30.1, 40.1, 50.1, 60.1}));
    std::vector<float> lhs(200);
    std::vector<float> rhs(200);
    for (int i = 0; i < 200; i++) {
      EXPECT_TRUE(CompareEqualFloatBuffers<float>(lhs, rhs))
          << "should be the same at index " << i;
      lhs[i] = 3;
      rhs[i] = 5;
      EXPECT_FALSE(CompareEqualFloatBuffers<float>(lhs, rhs))
          << "should be the different at index " << i;
      lhs[i] = 0;
      rhs[i] = 0;
    }
  }

  {
    EXPECT_TRUE(CompareEqualFloatBuffers<double>(
        {20, 30, 40, 50, 60}, {20.1, 30.1, 40.1, 50.1, 60.1}));
    std::vector<float> lhs(200);
    std::vector<float> rhs(200);
    for (int i = 0; i < 200; i++) {
      EXPECT_TRUE(CompareEqualFloatBuffers<double>(lhs, rhs))
          << "should be the same at index " << i;
      lhs[i] = 3;
      rhs[i] = 5;
      EXPECT_FALSE(CompareEqualFloatBuffers<double>(lhs, rhs))
          << "should be the different at index " << i;
      lhs[i] = 0;
      rhs[i] = 0;
    }
  }

  {
    EXPECT_TRUE(CompareEqualFloatBuffers<int8_t>({20, 30, 40, 50, 60},
                                                 {21, 31, 41, 51, 61}));
    std::vector<float> lhs(200);
    std::vector<float> rhs(200);
    for (int i = 0; i < 200; i++) {
      EXPECT_TRUE(CompareEqualFloatBuffers<int8_t>(lhs, rhs))
          << "should be the same at index " << i;
      lhs[i] = 3;
      rhs[i] = 5;
      EXPECT_FALSE(CompareEqualFloatBuffers<int8_t>(lhs, rhs))
          << "should be the different at index " << i;
      lhs[i] = 0;
      rhs[i] = 0;
    }
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
