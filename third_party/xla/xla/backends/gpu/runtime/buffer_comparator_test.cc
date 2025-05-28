/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/buffer_comparator.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

constexpr double kDefaultTolerance = 0.1;

class BufferComparatorTest : public testing::Test {
 protected:
  BufferComparatorTest() {
    auto name = absl::AsciiStrToUpper(
        xla::PlatformUtil::CanonicalPlatformName("gpu").value());
    platform_ = se::PlatformManager::PlatformWithName(name).value();
    stream_exec_ = platform_->ExecutorForDevice(0).value();
  }

  // Take floats only for convenience. Still uses ElementType internally.
  template <typename ElementType>
  bool CompareEqualBuffers(absl::Span<const ElementType> current,
                           absl::Span<const ElementType> expected,
                           double tolerance) {
    auto stream = stream_exec_->CreateStream().value();

    se::DeviceMemoryHandle current_buffer(
        stream_exec_, stream_exec_->AllocateArray<ElementType>(current.size()));
    se::DeviceMemoryHandle expected_buffer(
        stream_exec_,
        stream_exec_->AllocateArray<ElementType>(expected.size()));

    TF_CHECK_OK(stream->Memcpy(current_buffer.memory_ptr(), current.data(),
                               current_buffer.memory().size()));
    TF_CHECK_OK(stream->Memcpy(expected_buffer.memory_ptr(), expected.data(),
                               expected_buffer.memory().size()));
    TF_CHECK_OK(stream->BlockHostUntilDone());

    BufferComparator comparator(
        ShapeUtil::MakeShape(
            primitive_util::NativeToPrimitiveType<ElementType>(),
            {static_cast<int64_t>(current.size())}),
        tolerance);
    return comparator
        .CompareEqual(stream.get(), current_buffer.memory(),
                      expected_buffer.memory())
        .value();
  }

  // Take floats only for convenience. Still uses ElementType internally.
  template <typename ElementType>
  bool CompareEqualFloatBuffers(const std::vector<float>& lhs_float,
                                const std::vector<float>& rhs_float,
                                double tolerance = kDefaultTolerance) {
    std::vector<ElementType> lhs(lhs_float.begin(), lhs_float.end());
    std::vector<ElementType> rhs(rhs_float.begin(), rhs_float.end());
    return CompareEqualBuffers<ElementType>(lhs, rhs, tolerance);
  }

  template <typename ElementType>
  bool CompareEqualComplex(const std::vector<std::complex<ElementType>>& lhs,
                           const std::vector<std::complex<ElementType>>& rhs) {
    return CompareEqualBuffers<std::complex<ElementType>>(lhs, rhs,
                                                          kDefaultTolerance);
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

TEST_F(BufferComparatorTest, TestPred) {
  EXPECT_TRUE(CompareEqualBuffers<bool>({false, true}, {false, true}, 0.0));
  EXPECT_FALSE(CompareEqualBuffers<bool>({false, true}, {false, false}, 0.0));
  EXPECT_FALSE(CompareEqualBuffers<bool>({false, true}, {true, true}, 0.0));
  EXPECT_FALSE(CompareEqualBuffers<bool>({false, true}, {true, false}, 0.0));
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

  EXPECT_TRUE(
      CompareEqualFloatBuffers<tsl::float8_e4m3fn>({inf}, {std::nanf("")}));
  EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>({inf}, {inf}));
  EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>({inf}, {-inf}));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>({inf}, {448}));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>({inf}, {-448}));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>({inf}, {20}));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>({inf}, {-20}));

  EXPECT_FALSE(
      CompareEqualFloatBuffers<tsl::float8_e5m2>({inf}, {std::nanf("")}));
  EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e5m2>({inf}, {inf}));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e5m2>({inf}, {-inf}));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e5m2>({inf}, {57344}));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e5m2>({-inf}, {-57344}));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e5m2>({inf}, {20}));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e5m2>({inf}, {-20}));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e5m2>({-inf}, {20}));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e5m2>({-inf}, {-20}));
}

TEST_F(BufferComparatorTest, TestNumbers) {
  EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>({20}, {20.1}));
  EXPECT_FALSE(CompareEqualFloatBuffers<Eigen::half>({20}, {23.0}));
  EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>({20}, {23.0}, 0.2));
  EXPECT_FALSE(CompareEqualFloatBuffers<Eigen::half>({20}, {26.0}, 0.2));
  EXPECT_FALSE(CompareEqualFloatBuffers<Eigen::half>({0}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>({0.9}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>({9}, {10}));
  EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>({10}, {9}));

  EXPECT_TRUE(CompareEqualFloatBuffers<float>({20}, {20.1}));
  EXPECT_FALSE(CompareEqualFloatBuffers<float>({20}, {23.0}));
  EXPECT_TRUE(CompareEqualFloatBuffers<float>({20}, {23.0}, 0.2));
  EXPECT_FALSE(CompareEqualFloatBuffers<float>({20}, {26.0}, 0.2));
  EXPECT_FALSE(CompareEqualFloatBuffers<float>({0}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<float>({0.9}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<float>({9}, {10}));
  EXPECT_TRUE(CompareEqualFloatBuffers<float>({10}, {9}));

  EXPECT_TRUE(CompareEqualFloatBuffers<double>({20}, {20.1}));
  EXPECT_FALSE(CompareEqualFloatBuffers<double>({20}, {23.0}));
  EXPECT_TRUE(CompareEqualFloatBuffers<double>({20}, {23.0}, 0.2));
  EXPECT_FALSE(CompareEqualFloatBuffers<double>({20}, {26.0}, 0.2));
  EXPECT_FALSE(CompareEqualFloatBuffers<double>({0}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<double>({0.9}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<double>({9}, {10}));
  EXPECT_TRUE(CompareEqualFloatBuffers<double>({10}, {9}));

  EXPECT_TRUE(CompareEqualFloatBuffers<int8_t>({100}, {101}));
  EXPECT_FALSE(CompareEqualFloatBuffers<int8_t>({100}, {120}));
  EXPECT_TRUE(CompareEqualFloatBuffers<int8_t>({100}, {120}, 0.2));
  EXPECT_FALSE(CompareEqualFloatBuffers<int8_t>({90}, {120}, 0.2));
  EXPECT_FALSE(CompareEqualFloatBuffers<int8_t>({0}, {10}));
  EXPECT_TRUE(CompareEqualFloatBuffers<int8_t>({9}, {10}));
  EXPECT_TRUE(CompareEqualFloatBuffers<int8_t>({90}, {100}));
  EXPECT_TRUE(CompareEqualFloatBuffers<int8_t>({100}, {90}));
  EXPECT_FALSE(CompareEqualFloatBuffers<int8_t>({-128}, {127}));

  EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>({20}, {20.1}));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>({20}, {23.0}));
  EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>({20}, {23.0}, 0.2));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>({20}, {26.0}, 0.2));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>({0}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>({0.9}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>({9}, {10}));
  EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>({9}, {10}));

  EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e5m2>({20}, {20.1}));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e5m2>({20}, {23.0}));
  EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e5m2>({20}, {23.0}, 0.2));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e5m2>({20}, {30.0}, 0.2));
  EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e5m2>({0}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e5m2>({0.9}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e5m2>({11}, {12}));
  EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e5m2>({12}, {11}));

  // Rerunning tests with increased relative tolerance
  const double tol = 0.001;
  EXPECT_FALSE(CompareEqualFloatBuffers<Eigen::half>({0.9}, {1}, tol));
  EXPECT_TRUE(CompareEqualFloatBuffers<Eigen::half>({0.9}, {0.901}, tol));
  EXPECT_FALSE(CompareEqualFloatBuffers<float>({10}, {10.1}, tol));
  EXPECT_TRUE(CompareEqualFloatBuffers<float>({10}, {10.01}, tol));
  EXPECT_FALSE(CompareEqualFloatBuffers<int8_t>({100}, {101}, tol));
  EXPECT_FALSE(CompareEqualFloatBuffers<double>({20}, {20.1}, tol));
  EXPECT_TRUE(CompareEqualFloatBuffers<double>({20}, {20.01}, tol));
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
  {
    EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>(
        {20, 30, 40, 50, 60}, {20.1, 30.1, 40.1, 50.1, 60.1}));
    std::vector<float> lhs(200);
    std::vector<float> rhs(200);
    for (int i = 0; i < 200; i++) {
      EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>(lhs, rhs))
          << "should be the same at index " << i;
      lhs[i] = 3;
      rhs[i] = 5;
      EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e4m3fn>(lhs, rhs))
          << "should be the different at index " << i;
      lhs[i] = 0;
      rhs[i] = 0;
    }
  }

  {
    EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e5m2>(
        {20, 30, 40, 50, 60}, {20.1, 30.1, 40.1, 50.1, 60.1}));
    std::vector<float> lhs(200);
    std::vector<float> rhs(200);
    for (int i = 0; i < 200; i++) {
      EXPECT_TRUE(CompareEqualFloatBuffers<tsl::float8_e5m2>(lhs, rhs))
          << "should be the same at index " << i;
      lhs[i] = 3;
      rhs[i] = 5;
      EXPECT_FALSE(CompareEqualFloatBuffers<tsl::float8_e5m2>(lhs, rhs))
          << "should be the different at index " << i;
      lhs[i] = 0;
      rhs[i] = 0;
    }
  }
}

TEST_F(BufferComparatorTest, BF16) {
  const int element_count = 3123;
  int64_t rng_state = 0;

  auto stream = stream_exec_->CreateStream().value();

  se::DeviceMemoryHandle lhs(
      stream_exec_,
      stream_exec_->AllocateArray<Eigen::bfloat16>(element_count));
  InitializeBuffer(stream.get(), BF16, &rng_state, lhs.memory());

  se::DeviceMemoryHandle rhs(
      stream_exec_,
      stream_exec_->AllocateArray<Eigen::bfloat16>(element_count));
  InitializeBuffer(stream.get(), BF16, &rng_state, rhs.memory());

  BufferComparator comparator(ShapeUtil::MakeShape(BF16, {element_count}));
  EXPECT_FALSE(comparator.CompareEqual(stream.get(), lhs.memory(), rhs.memory())
                   .value());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
