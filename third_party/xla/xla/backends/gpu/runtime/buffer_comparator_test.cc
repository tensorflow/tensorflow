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

#include "xla/backends/gpu/runtime/buffer_comparator_test.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "Eigen/Core"
#include "xla/backends/gpu/runtime/buffer_comparator.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"

namespace xla {
namespace gpu {
namespace {

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

TEST_F(BufferComparatorTest, TestScalar) {
  EXPECT_TRUE(CompareEqualScalar<std::complex<double>>({1, 1}, {1, 1}));
  EXPECT_FALSE(CompareEqualScalar<std::complex<double>>({1, 1}, {1, 2}));
  EXPECT_FALSE(CompareEqualScalar<std::complex<double>>({1, 1}, {2, 1}));
  EXPECT_TRUE(CompareEqualScalar<std::complex<float>>({1, 1}, {1, 1}));
  EXPECT_FALSE(CompareEqualScalar<std::complex<float>>({1, 1}, {1, 2}));
  EXPECT_FALSE(CompareEqualScalar<std::complex<float>>({1, 1}, {2, 1}));
  EXPECT_TRUE(CompareEqualScalar<float>(1, 1));
  EXPECT_FALSE(CompareEqualScalar<float>(1, 2));
  EXPECT_TRUE(CompareEqualScalar<double>(1, 1));
  EXPECT_FALSE(CompareEqualScalar<double>(1, 2));
  EXPECT_TRUE(CompareEqualScalar<bool>(true, true));
  EXPECT_TRUE(CompareEqualScalar<bool>(false, false));
  EXPECT_FALSE(CompareEqualScalar<bool>(true, false));
  EXPECT_TRUE(CompareEqualScalar<int8_t>(1, 1));
  EXPECT_FALSE(CompareEqualScalar<int8_t>(1, 2));
  EXPECT_TRUE(CompareEqualScalar<int32_t>(1, 1));
  EXPECT_FALSE(CompareEqualScalar<int32_t>(1, 2));
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       stream_exec_->CreateStream());

  se::DeviceAddressHandle lhs(
      stream_exec_,
      stream_exec_->AllocateArray<Eigen::bfloat16>(element_count));
  InitializeBuffer(stream.get(), BF16, &rng_state, lhs.address());

  se::DeviceAddressHandle rhs(
      stream_exec_,
      stream_exec_->AllocateArray<Eigen::bfloat16>(element_count));
  InitializeBuffer(stream.get(), BF16, &rng_state, rhs.address());

  BufferComparator comparator(ShapeUtil::MakeShape(BF16, {element_count}));
  ASSERT_OK_AND_ASSIGN(
      bool equal,
      comparator.CompareEqual(stream.get(), lhs.address(), rhs.address()));
  EXPECT_FALSE(equal);
}

TEST_F(BufferComparatorTest, VeryLargeArray) {
  constexpr PrimitiveType number_type = U8;
  using NT = primitive_util::PrimitiveTypeToNative<number_type>::type;

  // Set non-power-of-two element count on purpose, use aligned buffer size.
  int64_t n_elems = (1LL << 32) - 11,
          // Buffer size must be 4-bytes aligned for Memset32.
      buf_size = (((n_elems + 1) * sizeof(NT)) + 3) & ~3;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       stream_exec_->CreateStream());

  auto base = stream_exec_->Allocate(buf_size);
  EXPECT_TRUE(!base.is_null());
  auto cleanup =
      absl::MakeCleanup([this, &base] { stream_exec_->Deallocate(&base); });

  // We use overlapping lhs and rhs arrays to reduce memory usage, also this
  // serves as an extra test for possible pointer aliasing problems.
  se::DeviceAddressBase lhs(base.opaque(), n_elems * sizeof(NT)),
      rhs(static_cast<NT*>(base.opaque()) + 1, lhs.size());

  constexpr uint32_t pattern = 0xABABABAB;
  CHECK_OK(stream->Memset32(&lhs, pattern, buf_size));

  // First we do "positive" test to make sure lhs and rhs are indeed equal:
  // disable host comparison here since it could take a while for ~4GB array
  BufferComparator comparator(ShapeUtil::MakeShape(number_type, {n_elems}),
                              /*tolerance*/ 0.1, /* verbose */ false,
                              /*run_host_compare*/ false);
  ASSERT_OK_AND_ASSIGN(bool equal1,
                       comparator.CompareEqual(stream.get(), lhs, rhs));
  EXPECT_TRUE(equal1);

  se::DeviceAddressBase last_word(
      static_cast<uint8_t*>(base.opaque()) + (n_elems & ~3), sizeof(uint32_t));
  // Change only the very last entry of rhs to verify that the whole arrays are
  // compared (if the grid dimensions are not computed correctly, this might
  // not be the case).
  CHECK_OK(stream->Memset32(&last_word, 0x11223344, last_word.size()));
  ASSERT_OK_AND_ASSIGN(bool equal2,
                       comparator.CompareEqual(stream.get(), lhs, rhs));
  EXPECT_FALSE(equal2);

  // Make *every* element mismatch and verify the comparator still reports
  // inequality. This is a regression test for a 32-bit overflow: the device
  // kernel accumulates into an int32 counter, so if it summed the per-element
  // mismatch count (~2^32 here) it would wrap modulo 2^32 and could land back
  // on 0, making DeviceCompare spuriously report equality. The kernel instead
  // counts mismatching *blocks* (bounded by the grid size), which cannot
  // overflow. The byte pattern 0x00FF00FF alternates 0xFF/0x00 bytes, so
  // buffer[i] != buffer[i + 1] for every i, i.e. lhs[i] != rhs[i] everywhere
  // (rhs is lhs shifted by one element).
  constexpr uint32_t alternating_pattern = 0x00FF00FF;
  CHECK_OK(stream->Memset32(&lhs, alternating_pattern, buf_size));
  ASSERT_OK_AND_ASSIGN(bool equal3,
                       comparator.CompareEqual(stream.get(), lhs, rhs));
  EXPECT_FALSE(equal3);
}

TEST_F(BufferComparatorTest, ErrorReportGeneratedOnMismatch) {
  std::vector<float> current = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected = {1.0f, 2.5f, 3.0f, 4.0f};
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       stream_exec_->CreateStream());

  se::DeviceAddressHandle current_buffer(
      stream_exec_, stream_exec_->AllocateArray<float>(current.size()));
  se::DeviceAddressHandle expected_buffer(
      stream_exec_, stream_exec_->AllocateArray<float>(expected.size()));

  ASSERT_OK(stream->Memcpy(current_buffer.address_ptr(), current.data(),
                           current_buffer.address().size()));
  ASSERT_OK(stream->Memcpy(expected_buffer.address_ptr(), expected.data(),
                           expected_buffer.address().size()));
  ASSERT_OK(stream->BlockHostUntilDone());

  BufferComparator comparator(
      ShapeUtil::MakeShape(F32, {static_cast<int64_t>(current.size())}),
      /*tolerance=*/0.01);
  std::string error_report;
  ASSERT_OK_AND_ASSIGN(
      bool equal,
      comparator.CompareEqual(stream.get(), current_buffer.address(),
                              expected_buffer.address(), &error_report));
  EXPECT_FALSE(equal);
  EXPECT_THAT(error_report, ::testing::HasSubstr("Mismatch count: 1 / 4"));
  EXPECT_THAT(error_report, ::testing::HasSubstr("Max relative difference:"));
  EXPECT_THAT(error_report, ::testing::HasSubstr("Max absolute difference:"));
  EXPECT_THAT(error_report, ::testing::HasSubstr("at [1] (linear index 1)"));
  EXPECT_THAT(error_report,
              ::testing::HasSubstr("First 1 mismatch sample(s):"));
}

TEST_F(BufferComparatorTest, ErrorReportEmptyOnMatch) {
  std::vector<float> current = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       stream_exec_->CreateStream());

  se::DeviceAddressHandle current_buffer(
      stream_exec_, stream_exec_->AllocateArray<float>(current.size()));
  se::DeviceAddressHandle expected_buffer(
      stream_exec_, stream_exec_->AllocateArray<float>(expected.size()));

  ASSERT_OK(stream->Memcpy(current_buffer.address_ptr(), current.data(),
                           current_buffer.address().size()));
  ASSERT_OK(stream->Memcpy(expected_buffer.address_ptr(), expected.data(),
                           expected_buffer.address().size()));
  ASSERT_OK(stream->BlockHostUntilDone());

  BufferComparator comparator(
      ShapeUtil::MakeShape(F32, {static_cast<int64_t>(current.size())}),
      /*tolerance=*/0.01);
  std::string error_report = "prior_stale_content";
  ASSERT_OK_AND_ASSIGN(
      bool equal,
      comparator.CompareEqual(stream.get(), current_buffer.address(),
                              expected_buffer.address(), &error_report));
  EXPECT_TRUE(equal);
  EXPECT_TRUE(error_report.empty());
}

TEST_F(BufferComparatorTest, ErrorReportOnMultidimensionalMismatch) {
  // 2x2 matrix with mismatch at (1, 0) -> linear index 2
  std::vector<float> current = {1.0f, 2.0f, 10.0f, 4.0f};
  std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       stream_exec_->CreateStream());

  se::DeviceAddressHandle current_buffer(
      stream_exec_, stream_exec_->AllocateArray<float>(current.size()));
  se::DeviceAddressHandle expected_buffer(
      stream_exec_, stream_exec_->AllocateArray<float>(expected.size()));

  ASSERT_OK(stream->Memcpy(current_buffer.address_ptr(), current.data(),
                           current_buffer.address().size()));
  ASSERT_OK(stream->Memcpy(expected_buffer.address_ptr(), expected.data(),
                           expected_buffer.address().size()));
  ASSERT_OK(stream->BlockHostUntilDone());

  BufferComparator comparator(ShapeUtil::MakeShape(F32, {2, 2}),
                              /*tolerance=*/0.01);
  std::string error_report;
  ASSERT_OK_AND_ASSIGN(
      bool equal,
      comparator.CompareEqual(stream.get(), current_buffer.address(),
                              expected_buffer.address(), &error_report));
  EXPECT_FALSE(equal);
  EXPECT_THAT(error_report, ::testing::HasSubstr("[1, 0] (linear index 2)"));
  EXPECT_THAT(error_report, ::testing::HasSubstr("Max relative difference:"));
  EXPECT_THAT(error_report, ::testing::HasSubstr("Max absolute difference:"));
}

// The mismatch counter used by DeviceCompare is allocated in host memory
// (via StreamExecutor::HostMemoryAllocate) and MemZero'd before each kernel
// launch. This test runs many comparisons back-to-back on the same executor,
// alternating between equal and mismatching buffers, to pin that the counter
// is correctly re-zeroed between calls and that a prior mismatch never leaks
// into a subsequent equal comparison (or vice versa).
TEST_F(BufferComparatorTest, CounterIsRezeroedAcrossComparisons) {
  const std::vector<float> equal_a = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> equal_b = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> mismatch_b = {1.0f, 99.0f, 3.0f, 4.0f};

  for (int i = 0; i < 50; ++i) {
    // Equal comparison must return true even right after a mismatch.
    EXPECT_TRUE(CompareEqualFloatBuffers<float>(equal_a, equal_b, 0.01))
        << "equal comparison failed on iteration " << i;
    // Mismatch comparison must return false even right after an equal compare.
    EXPECT_FALSE(CompareEqualFloatBuffers<float>(equal_a, mismatch_b, 0.01))
        << "mismatch comparison failed on iteration " << i;
  }
}

// The following tests exercise the *parallel* host-compare path, which is only
// taken for buffers with at least `kParallelThreshold` (1 << 20) elements AND a
// non-null error_report. Because the parallel path is a second, independent
// implementation of the comparison logic, these tests pin its results against
// the same expectations the serial path satisfies, and pin down the
// deterministic (lowest-index) tie-breaking in the shard reduction.

// Must stay in sync with kParallelThreshold in buffer_comparator.cc.
constexpr int64_t kParallelThreshold = 1 << 20;

// A single mismatch in an otherwise-equal large buffer should be found and
// reported by the parallel path, at the correct linear index.
TEST_F(BufferComparatorTest, ParallelPathSingleMismatch) {
  const int64_t n = kParallelThreshold;
  std::vector<float> current(n, 1.0f);
  std::vector<float> expected(n, 1.0f);
  const int64_t mismatch_index = n - 7;  // Not in the first shard.
  current[mismatch_index] = 100.0f;

  HostCompareResult r =
      CompareEqualF32WithReport(current, expected, /*tolerance=*/0.01);
  EXPECT_FALSE(r.equal);
  EXPECT_THAT(r.error_report,
              ::testing::HasSubstr("Mismatch count: 1 / " + std::to_string(n)));
  EXPECT_THAT(
      r.error_report,
      ::testing::HasSubstr("linear index " + std::to_string(mismatch_index)));
  EXPECT_THAT(r.error_report,
              ::testing::HasSubstr("First 1 mismatch sample(s):"));
}

// Just below the threshold the serial path is taken; just at/above it the
// parallel path is taken. Both must find the single mismatch and agree on the
// reported linear index (pins the branch boundary).
TEST_F(BufferComparatorTest, ParallelPathBranchBoundary) {
  for (int64_t n : {kParallelThreshold - 1, kParallelThreshold}) {
    std::vector<float> current(n, 2.0f);
    std::vector<float> expected(n, 2.0f);
    const int64_t mismatch_index = n - 1;
    current[mismatch_index] = 5.0f;

    HostCompareResult r =
        CompareEqualF32WithReport(current, expected, /*tolerance=*/0.01);
    EXPECT_FALSE(r.equal) << "n=" << n;
    EXPECT_THAT(r.error_report, ::testing::HasSubstr("Mismatch count: 1 / " +
                                                     std::to_string(n)))
        << "n=" << n;
    EXPECT_THAT(
        r.error_report,
        ::testing::HasSubstr("linear index " + std::to_string(mismatch_index)))
        << "n=" << n;
  }
}

// Every element mismatches: the parallel reduction must count all of them and
// report exactly the first kMaxSamples (=5) lowest-index samples.
TEST_F(BufferComparatorTest, ParallelPathAllMismatch) {
  const int64_t n = kParallelThreshold;
  std::vector<float> current(n, 1.0f);
  std::vector<float> expected(n, 100.0f);

  HostCompareResult r =
      CompareEqualF32WithReport(current, expected, /*tolerance=*/0.01);
  EXPECT_FALSE(r.equal);
  EXPECT_THAT(r.error_report,
              ::testing::HasSubstr("Mismatch count: " + std::to_string(n) +
                                   " / " + std::to_string(n)));
  // The first (lowest-index) samples must be 0..4 regardless of shard order.
  EXPECT_THAT(r.error_report,
              ::testing::HasSubstr("First 5 mismatch sample(s):"));
  EXPECT_THAT(r.error_report, ::testing::HasSubstr("linear index 0"));
  EXPECT_THAT(r.error_report, ::testing::HasSubstr("linear index 1"));
  EXPECT_THAT(r.error_report, ::testing::HasSubstr("linear index 4"));
}

// NaN/Inf-only mismatches exercise the linear_index != -1 guards in
// merge_stats: the max relative/absolute difference samples are non-finite, so
// they must be reported as N/A while the NaN/Inf counts are still tallied.
TEST_F(BufferComparatorTest, ParallelPathNanInfOnly) {
  const int64_t n = kParallelThreshold;
  const float nan = std::nanf("");
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> current(n, 1.0f);
  std::vector<float> expected(n, 1.0f);
  // A NaN mismatch and an Inf mismatch, both past the first shard.
  const int64_t nan_index = n - 3;
  const int64_t inf_index = n - 5;
  current[nan_index] = nan;  // expected is finite -> NaN mismatch.
  current[inf_index] = inf;  // expected is finite -> Inf mismatch.

  HostCompareResult r =
      CompareEqualF32WithReport(current, expected, /*tolerance=*/0.01);
  EXPECT_FALSE(r.equal);
  EXPECT_THAT(r.error_report,
              ::testing::HasSubstr("Mismatch count: 2 / " + std::to_string(n)));
  EXPECT_THAT(r.error_report, ::testing::HasSubstr("NaN count: 1"));
  EXPECT_THAT(r.error_report, ::testing::HasSubstr("Inf count: 1"));
  // All mismatches are NaN/Inf, so both max-diff samples are N/A.
  EXPECT_THAT(r.error_report,
              ::testing::HasSubstr(
                  "Max relative difference: N/A (all mismatches are NaN/Inf)"));
  EXPECT_THAT(r.error_report,
              ::testing::HasSubstr(
                  "Max absolute difference: N/A (all mismatches are NaN/Inf)"));
}

// Many equal maxima: several elements share the exact same (maximal) diff. The
// reported "Max ... difference at ..." sample must be the lowest linear index
// among them, deterministically, no matter how shards are scheduled.
TEST_F(BufferComparatorTest, ParallelPathEqualMaximaTieBreak) {
  const int64_t n = kParallelThreshold;
  std::vector<float> current(n, 1.0f);
  std::vector<float> expected(n, 1.0f);
  // Give a set of elements, spread across shards, the identical large delta.
  const int64_t first_max_index = 3;
  for (int64_t idx : {first_max_index, n / 4, n / 2, 3 * n / 4, n - 2}) {
    current[idx] = 1000.0f;  // Identical value -> identical abs/rel diff.
  }

  HostCompareResult r =
      CompareEqualF32WithReport(current, expected, /*tolerance=*/0.01);
  EXPECT_FALSE(r.equal);
  EXPECT_THAT(r.error_report,
              ::testing::HasSubstr("Mismatch count: 5 / " + std::to_string(n)));
  // The max-diff samples must tie-break to the lowest linear index (3).
  EXPECT_THAT(r.error_report,
              ::testing::HasSubstr("Max relative difference: "));
  EXPECT_THAT(
      r.error_report,
      ::testing::HasSubstr("linear index " + std::to_string(first_max_index)));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
