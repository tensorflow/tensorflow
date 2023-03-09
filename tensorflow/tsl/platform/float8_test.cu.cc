/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// Enable CPU or GPU device, depending on build configuration.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#else
#define EIGEN_USE_THREADS
#endif

#include "tensorflow/tsl/platform/float8.h"

#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/tsl/platform/test.h"

namespace tsl {
namespace {

template <typename Float8_>
class Float8Test : public ::testing::Test {};

// Helper utility for prettier test names.
struct Float8TestParamNames {
  template <typename TypeParam>
  static std::string GetName(int idx) {
    if constexpr (std::is_same_v<TypeParam, float8_e4m3fn>) {
      return "float8_e4m3fn";
    } else if constexpr (std::is_same_v<TypeParam, float8_e4m3b11>) {
      return "float8_e4m3b11";
    } else if constexpr (std::is_same_v<TypeParam, float8_e5m2>) {
      return "float8_e5m2";
    }
    return absl::StrCat(idx);
  }
};

using Float8Types =
    ::testing::Types<float8_e4m3fn, float8_e5m2, float8_e4m3b11>;
TYPED_TEST_SUITE(Float8Test, Float8Types, Float8TestParamNames);

TEST(Float8E4m3Test, NumericLimits) {
  EXPECT_TRUE(
      Eigen::numext::isnan(std::numeric_limits<float8_e4m3fn>::quiet_NaN()));
  EXPECT_TRUE(Eigen::numext::isnan(
      std::numeric_limits<float8_e4m3fn>::signaling_NaN()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e4m3fn>::min()),
            std::exp2(-6));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e4m3fn>::max()), 448);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e4m3fn>::lowest()),
            -448);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e4m3fn>::epsilon()),
            0.125);
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<float8_e4m3fn>::round_error()),
      0.5);
  // No infinity, represent as NaN.
  EXPECT_TRUE(
      Eigen::numext::isnan(std::numeric_limits<float8_e4m3fn>::infinity()));
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<float8_e4m3fn>::denorm_min()),
      std::exp2(-9));
}

TEST(Float8E5m2Test, NumericLimits) {
  EXPECT_TRUE(
      Eigen::numext::isnan(std::numeric_limits<float8_e5m2>::quiet_NaN()));
  EXPECT_TRUE(
      Eigen::numext::isnan(std::numeric_limits<float8_e5m2>::signaling_NaN()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e5m2>::min()),
            std::exp2(-14));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e5m2>::max()), 57344);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e5m2>::lowest()),
            -57344);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e5m2>::epsilon()),
            0.25);
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e5m2>::round_error()),
            0.5);
  EXPECT_TRUE(
      Eigen::numext::isinf(std::numeric_limits<float8_e5m2>::infinity()));
  EXPECT_EQ(static_cast<float>(std::numeric_limits<float8_e5m2>::denorm_min()),
            std::exp2(-16));
}

TYPED_TEST(Float8Test, FromRep) {
  using Float8 = TypeParam;
  Float8 x = Float8::FromRep(0x4F);
  EXPECT_EQ(x.rep(), 0x4F);
}

TYPED_TEST(Float8Test, Negate) {
  using Float8 = TypeParam;
  Float8 x = -Float8::FromRep(0x4F);
  EXPECT_EQ(x.rep(), 0x80 | 0x4F);

  Float8 nan = -std::numeric_limits<Float8>::quiet_NaN();
  EXPECT_TRUE(Eigen::numext::isnan(nan));
}

TYPED_TEST(Float8Test, BitCasts) {
  using Float8 = TypeParam;
  Float8 x = Float8::FromRep(0x47);
  EXPECT_EQ(Eigen::numext::bit_cast<uint8_t>(x), 0x47);
  EXPECT_EQ(Eigen::numext::bit_cast<Float8>(x.rep()).rep(), 0x47);
}

TYPED_TEST(Float8Test, UpCasts) {
  using Float8 = TypeParam;

  // Loop through each float8 value.
  for (int i = 0x00; i <= 0xFF; ++i) {
    // Cast up to each other floating-point type, and verify they are the same.
    Float8 f8 = Float8::FromRep(i);
    double f64 = static_cast<double>(f8);
    float f32 = static_cast<float>(f8);
    Eigen::bfloat16 bf16 = static_cast<Eigen::bfloat16>(f8);
    Eigen::half f16 = static_cast<Eigen::half>(f8);

    if (Eigen::numext::isnan(f8)) {
      EXPECT_TRUE(Eigen::numext::isnan(f64));
      EXPECT_TRUE(Eigen::numext::isnan(f32));
      EXPECT_TRUE(Eigen::numext::isnan(bf16));
      EXPECT_TRUE(Eigen::numext::isnan(f16));
    } else {
      EXPECT_EQ(f64, f32);
      EXPECT_EQ(f32, bf16);
      EXPECT_EQ(bf16, f16);
    }
  }
}

TYPED_TEST(Float8Test, DownCasts) {
  using Float8 = TypeParam;
  for (int i = 0x00; i <= 0xFF; ++i) {
    float x = static_cast<float>(Float8::FromRep(i));

    Float8 f64 = static_cast<Float8>(static_cast<double>(x));
    Float8 f32 = static_cast<Float8>(static_cast<float>(x));
    Float8 bf16 = static_cast<Float8>(static_cast<Eigen::bfloat16>(x));
    Float8 f16 = static_cast<Float8>(static_cast<Eigen::half>(x));

    if (Eigen::numext::isnan(x)) {
      EXPECT_TRUE(Eigen::numext::isnan(f64));
      EXPECT_TRUE(Eigen::numext::isnan(f32));
      EXPECT_TRUE(Eigen::numext::isnan(bf16));
      EXPECT_TRUE(Eigen::numext::isnan(f16));
    } else {
      EXPECT_EQ(f64.rep(), i) << i;
      EXPECT_EQ(f32.rep(), i) << i;
      EXPECT_EQ(bf16.rep(), i) << i;
      EXPECT_EQ(f16.rep(), i) << i;
    }
  }
}

TYPED_TEST(Float8Test, ConvertFromWithSaturation) {
  using Float8 = TypeParam;

  // Saturation above max value.
  Float8 upper =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
          static_cast<float>(std::numeric_limits<Float8>::max()) * 2);
  EXPECT_EQ(upper, std::numeric_limits<Float8>::max());

  Float8 lower =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/false>(
          static_cast<float>(std::numeric_limits<Float8>::lowest()) * 2);
  EXPECT_EQ(lower, std::numeric_limits<Float8>::lowest());

  // Special values remain with saturation.
  Float8 nan =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/true>(
          std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(Eigen::numext::isnan(nan));
  Float8 inf =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/true>(
          std::numeric_limits<float>::infinity());
  // E4M3 doesn't have inf, so check inf -> NaN conversion.
  EXPECT_TRUE(std::numeric_limits<Float8>::has_infinity
                  ? Eigen::numext::isinf(inf)
                  : Eigen::numext::isnan(inf));
  Float8 ninf =
      Float8::template ConvertFrom</*kSaturate=*/true, /*kTruncate=*/true>(
          -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::numeric_limits<Float8>::has_infinity
                  ? Eigen::numext::isinf(ninf)
                  : Eigen::numext::isnan(ninf));
}

TYPED_TEST(Float8Test, ConvertFromWithTruncation) {
  using Float8 = TypeParam;

  // Truncation and rounding of a number ever-so-slightly less than 2.
  float less_than_two = Eigen::numext::bit_cast<float>(0x3FFFFFFF);
  Float8 truncated =
      Float8::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
          less_than_two);
  EXPECT_LT(static_cast<float>(truncated), 2);

  Float8 rounded =
      Float8::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
          less_than_two);
  EXPECT_EQ(static_cast<float>(rounded), 2);

  // Truncation and rounding of a subnormal.
  for (int i = 0x01; i < 0x04; ++i) {
    float less_than_subnorm =
        std::nexttoward(static_cast<float>(Float8::FromRep(i)), 0);

    Float8 truncated_subnorm =
        Float8::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
            less_than_subnorm);
    EXPECT_EQ(truncated_subnorm.rep(), i - 1);

    Float8 rounded_subnorm =
        Float8::template ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
            less_than_subnorm);
    EXPECT_EQ(rounded_subnorm.rep(), i);
  }
}

TYPED_TEST(Float8Test, ConvertTo) {
  using Float8 = TypeParam;

  // Converting to higher precision types doesn't result in either
  // truncation or saturation, so let's just ensure they all provide the
  // same results.
  for (int i = 0x00; i <= 0xFF; ++i) {
    // Cast up to each other floating-point type, and verify they are the same.
    Float8 f8 = Float8::FromRep(i);
    float f32 = static_cast<float>(f8);
    if (Eigen::numext::isnan(f8)) {
      EXPECT_TRUE(
          std::isnan(Float8::template ConvertTo<float, /*kSaturate=*/false,
                                                /*kTruncate=*/false>(f8)));
      EXPECT_TRUE(
          std::isnan(Float8::template ConvertTo<float, /*kSaturate=*/false,
                                                /*kTruncate=*/true>(f8)));
      EXPECT_TRUE(
          std::isnan(Float8::template ConvertTo<float, /*kSaturate=*/true,
                                                /*kTruncate=*/false>(f8)));
      EXPECT_TRUE(
          std::isnan(Float8::template ConvertTo<float, /*kSaturate=*/true,
                                                /*kTruncate=*/true>(f8)));
    } else {
      EXPECT_EQ(f32, (Float8::template ConvertTo<float, /*kSaturate=*/false,
                                                 /*kTruncate=*/false>(f8)));
      EXPECT_EQ(f32, (Float8::template ConvertTo<float, /*kSaturate=*/false,
                                                 /*kTruncate=*/true>(f8)));
      EXPECT_EQ(f32, (Float8::template ConvertTo<float, /*kSaturate=*/true,
                                                 /*kTruncate=*/false>(f8)));
      EXPECT_EQ(f32, (Float8::template ConvertTo<float, /*kSaturate=*/true,
                                                 /*kTruncate=*/true>(f8)));
    }
  }
}

TEST(Float8Test, Float8E5m2_To_Float8E4m3) {
  for (int i = 0x00; i <= 0xFF; ++i) {
    float8_e5m2 e5m2 = float8_e5m2::FromRep(i);
    float8_e4m3fn e4m3 = static_cast<float8_e4m3fn>(e5m2);
    float8_e4m3fn expected =
        static_cast<float8_e4m3fn>(static_cast<float>(e5m2));
    EXPECT_EQ(e4m3.rep(), expected.rep()) << i;
  }

  // Saturation.
  float8_e5m2 max = std::numeric_limits<float8_e5m2>::max();
  float8_e4m3fn saturated = float8_e4m3fn::ConvertFrom</*kSaturate=*/true>(max);
  EXPECT_EQ(saturated, std::numeric_limits<float8_e4m3fn>::max());
  saturated = float8_e5m2::ConvertTo<float8_e4m3fn, /*kSaturate=*/true>(max);
  EXPECT_EQ(saturated, std::numeric_limits<float8_e4m3fn>::max());

  // Truncation - only occurs for e4m3 subnormals.
  float8_e5m2 less_than_subnorm = float8_e5m2::FromRep(0x1F);  // 2^-7 - 2^-10.
  float8_e4m3fn rounded_subnorm =
      float8_e4m3fn::ConvertFrom</*kSaturate=*/false, /*kTruncate=*/false>(
          less_than_subnorm);
  EXPECT_EQ(rounded_subnorm.rep(), 0x04);
  float8_e4m3fn truncated_subnorm =
      float8_e4m3fn::ConvertFrom</*kSaturate=*/false, /*kTruncate=*/true>(
          less_than_subnorm);
  EXPECT_EQ(truncated_subnorm.rep(), 0x03);
}

TEST(Float8Test, Float8E4m3_To_Float8E5m2) {
  for (int i = 0x00; i <= 0xFF; ++i) {
    float8_e4m3fn e4m3 = float8_e4m3fn::FromRep(i);
    float8_e5m2 e5m2 = static_cast<float8_e5m2>(e4m3);
    float8_e5m2 expected = static_cast<float8_e5m2>(static_cast<float>(e4m3));
    EXPECT_EQ(e5m2.rep(), expected.rep()) << i;
  }

  // Truncation and rounding of a number ever-so-slightly less than 2.
  float8_e4m3fn less_than_two = float8_e4m3fn::FromRep(0x3F);
  float8_e5m2 truncated =
      float8_e5m2::template ConvertFrom</*kSaturate=*/false,
                                        /*kTruncate=*/true>(less_than_two);
  EXPECT_LT(static_cast<float>(truncated), 2);

  float8_e5m2 rounded =
      float8_e5m2::template ConvertFrom</*kSaturate=*/false,
                                        /*kTruncate=*/false>(less_than_two);
  EXPECT_EQ(static_cast<float>(rounded), 2);
}

TEST(Float8Test, Half_To_Float8E5m2) {
  // Special values, NaN.
  Eigen::half inf =
      Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(0x7C00));
  EXPECT_EQ(static_cast<float8_e5m2>(inf).rep(), 0x7C);
  Eigen::half ninf =
      Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(0xFC00));
  EXPECT_EQ(static_cast<float8_e5m2>(ninf).rep(), 0xFC);

  Eigen::half nan =
      Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(0x7C01));
  EXPECT_EQ(static_cast<float8_e5m2>(nan).rep(), 0x7D);
  Eigen::half nnan =
      Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(0xFC01));
  EXPECT_EQ(static_cast<float8_e5m2>(nnan).rep(), 0xFD);

  // Rounding vs truncation.
  Eigen::half less_than_two =
      Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ((float8_e5m2::ConvertFrom</*kSaturate=*/false,
                                      /*kTruncate=*/false>(less_than_two)
                 .rep()),
            0x40);
  EXPECT_EQ((float8_e5m2::ConvertFrom</*kSaturate=*/false,
                                      /*kTruncate=*/true>(less_than_two)
                 .rep()),
            0x3F);
  EXPECT_EQ((float8_e5m2::ConvertFrom</*kSaturate=*/false,
                                      /*kTruncate=*/false>(-less_than_two)
                 .rep()),
            0xC0);
  EXPECT_EQ((float8_e5m2::ConvertFrom</*kSaturate=*/false,
                                      /*kTruncate=*/true>(-less_than_two)
                 .rep()),
            0xBF);
}

using ::testing::Eq;
using ::testing::IsTrue;
MATCHER_P(EqOrIsNan, other, "") {
  if (Eigen::numext::isnan(other)) {
    return ExplainMatchResult(IsTrue(), Eigen::numext::isnan(arg),
                              result_listener);
  }
  return ExplainMatchResult(Eq(other), arg, result_listener);
}

TYPED_TEST(Float8Test, CallTheOperator) {
  using Float8 = TypeParam;

  for (int i = 0x00; i <= 0xFF; ++i) {
    Float8 a = Float8::FromRep(i);
    for (int j = 0x00; j <= 0xFF; ++j) {
      Float8 b = Float8::FromRep(j);

      EXPECT_THAT(a + b, EqOrIsNan(Float8{float{a} + float{b}}));
      EXPECT_THAT(a - b, EqOrIsNan(Float8{float{a} - float{b}}));
      EXPECT_THAT(a * b, EqOrIsNan(Float8{float{a} * float{b}}));
      EXPECT_THAT(a / b, EqOrIsNan(Float8{float{a} / float{b}}));

      Float8 c;
      EXPECT_THAT((c = a, c += b), EqOrIsNan(Float8{float{a} + float{b}}));
      EXPECT_THAT((c = a, c -= b), EqOrIsNan(Float8{float{a} - float{b}}));
      EXPECT_THAT((c = a, c *= b), EqOrIsNan(Float8{float{a} * float{b}}));
      EXPECT_THAT((c = a, c /= b), EqOrIsNan(Float8{float{a} / float{b}}));

      EXPECT_EQ(a == b, float{a} == float{b}) << float{a} << " vs " << float{b};
      EXPECT_EQ(a != b, float{a} != float{b});
      EXPECT_EQ(a < b, float{a} < float{b});
      EXPECT_EQ(a <= b, float{a} <= float{b});
      EXPECT_EQ(a > b, float{a} > float{b});
      EXPECT_EQ(a >= b, float{a} >= float{b});
    }
  }
}

TYPED_TEST(Float8Test, CallTheConstOperator) {
  using Float8 = TypeParam;

  for (int i = 0x00; i <= 0xFF; ++i) {
    const Float8 a = Float8::FromRep(i);
    for (int j = 0x00; j <= 0xFF; ++j) {
      const Float8 b = Float8::FromRep(j);

      EXPECT_THAT(a + b, EqOrIsNan(Float8{float{a} + float{b}}));
      EXPECT_THAT(a - b, EqOrIsNan(Float8{float{a} - float{b}}));
      EXPECT_THAT(a * b, EqOrIsNan(Float8{float{a} * float{b}}));
      EXPECT_THAT(a / b, EqOrIsNan(Float8{float{a} / float{b}}));

      Float8 c;
      EXPECT_THAT((c = a, c += b), EqOrIsNan(Float8{float{a} + float{b}}));
      EXPECT_THAT((c = a, c -= b), EqOrIsNan(Float8{float{a} - float{b}}));
      EXPECT_THAT((c = a, c *= b), EqOrIsNan(Float8{float{a} * float{b}}));
      EXPECT_THAT((c = a, c /= b), EqOrIsNan(Float8{float{a} / float{b}}));

      EXPECT_EQ(a == b, float{a} == float{b}) << float{a} << " vs " << float{b};
      EXPECT_EQ(a != b, float{a} != float{b});
      EXPECT_EQ(a < b, float{a} < float{b}) << float{a} << " vs " << float{b};
      EXPECT_EQ(a <= b, float{a} <= float{b});
      EXPECT_EQ(a > b, float{a} > float{b}) << float{a} << " vs " << float{b};
      EXPECT_EQ(a >= b, float{a} >= float{b});
    }
  }
}

// Helper utility for prettier test names.
struct Float8CastTestParamNames {
  template <typename TypeParam>
  static std::string GetName(int idx) {
    using first_type = typename TypeParam::first_type;
    using second_type = typename TypeParam::second_type;
    return absl::StrCat(::testing::internal::GetTypeName<first_type>(), "_",
                        ::testing::internal::GetTypeName<second_type>());
  }
};

using Float8CastTypePairs = ::testing::Types<
#if !defined(EIGEN_USE_GPU) && !defined(EIGEN_GPU_COMPILE_PHASE)
    // long double doesn't work on GPU - it is treated as a regular 8-byte
    // double, which differs in size from the 16-byte long double on intel CPU.
    std::pair<float8_e5m2, long double>, std::pair<float8_e4m3fn, long double>,
#endif
    std::pair<float8_e4m3fn, double>, std::pair<float8_e4m3fn, float>,
    std::pair<float8_e4m3fn, Eigen::bfloat16>,
    std::pair<float8_e4m3fn, Eigen::half>, std::pair<float8_e4m3fn, bool>,
    std::pair<float8_e4m3fn, int32_t>, std::pair<float8_e4m3fn, int64_t>,
    std::pair<float8_e5m2, double>, std::pair<float8_e5m2, float>,
    std::pair<float8_e5m2, Eigen::bfloat16>,
    std::pair<float8_e5m2, Eigen::half>, std::pair<float8_e5m2, bool>,
    std::pair<float8_e5m2, int32_t>, std::pair<float8_e5m2, int64_t> >;

template <typename CastPair>
class Float8CastTest : public ::testing::Test {};
TYPED_TEST_SUITE(Float8CastTest, Float8CastTypePairs, Float8CastTestParamNames);

TYPED_TEST(Float8CastTest, CastThroughFloat) {
  using Float8 = typename TypeParam::first_type;
  using DestType = typename TypeParam::second_type;

  for (int i = 0x00; i <= 0xFF; ++i) {
    Float8 f8 = Float8::FromRep(i);

    if ((!Eigen::numext::isnan(f8) ||
         std::numeric_limits<DestType>::has_quiet_NaN) &&
        (!Eigen::numext::isinf(f8) ||
         std::numeric_limits<DestType>::has_infinity)) {
      DestType dest = static_cast<DestType>(f8);
      DestType expected = static_cast<DestType>(static_cast<float>(f8));
      EXPECT_THAT(dest, EqOrIsNan(expected));
    }
  }
}

// Work-around for lack of consistent .synchronize() method in Eigen.
template <typename Device>
void synchronize(Device& device) {
  // Nothing.
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <>
void synchronize<Eigen::GpuDevice>(Eigen::GpuDevice& device) {
  device.synchronize();
}
#endif

TYPED_TEST(Float8CastTest, DeviceCast) {
  using Float8 = typename TypeParam::first_type;
  using DestType = typename TypeParam::second_type;

#if defined(EIGEN_USE_GPU)
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice device(&stream);
#elif defined(EIGEN_USE_THREADS)
  constexpr int kThreads = 4;
  Eigen::ThreadPool tp(kThreads);
  Eigen::ThreadPoolDevice device(&tp, kThreads);
#else
  Eigen::DefaultDevice device;
#endif

  const int kNumElems = 256;
  // Allocate device buffers and create device tensors.
  Float8* src_device_buffer =
      (Float8*)device.allocate(kNumElems * sizeof(Float8));
  DestType* dst_device_buffer =
      (DestType*)device.allocate(kNumElems * sizeof(DestType));

  Eigen::TensorMap<Eigen::Tensor<Float8, 1>, Eigen::Aligned> src_device(
      src_device_buffer, kNumElems);
  Eigen::TensorMap<Eigen::Tensor<DestType, 1>, Eigen::Aligned> dst_device(
      dst_device_buffer, kNumElems);

  // Allocate host buffers and initially src memory.
  Eigen::Tensor<Float8, 1> src_cpu(kNumElems);
  Eigen::Tensor<DestType, 1> dst_cpu(kNumElems);
  for (int i = 0; i < kNumElems; ++i) {
    src_cpu(i) = Eigen::numext::bit_cast<Float8>(static_cast<uint8_t>(i));
    // If src is inf or nan but DestType doesn't support these values
    // (e.g. integer types), replace the input with a zero.
    if ((!std::numeric_limits<DestType>::has_quiet_NaN &&
         Eigen::numext::isnan(src_cpu(i))) ||
        (!std::numeric_limits<DestType>::has_infinity &&
         Eigen::numext::isinf(src_cpu(i)))) {
      src_cpu(i) = Float8(0.0);
    }
  }

  // Transfer data to device, perform a cast to DestType, then transfer result
  // back to host.
  device.memcpyHostToDevice(src_device_buffer, src_cpu.data(),
                            kNumElems * sizeof(Float8));
  dst_device.device(device) = src_device.template cast<DestType>();
  device.memcpyDeviceToHost(dst_cpu.data(), dst_device_buffer,
                            kNumElems * sizeof(DestType));
  synchronize(device);

  for (int i = 0; i < kNumElems; ++i) {
    DestType expected = static_cast<DestType>(src_cpu(i));
    EXPECT_THAT(dst_cpu(i), EqOrIsNan(expected));
  }

  // Cast back from DestType to Float8.
  // First clear out the device src buffer, since that will be the destination.
  src_cpu.setZero();
  device.memcpyHostToDevice(src_device_buffer, src_cpu.data(),
                            kNumElems * sizeof(Float8));
  src_device.device(device) = dst_device.template cast<Float8>();
  device.memcpyDeviceToHost(src_cpu.data(), src_device_buffer,
                            kNumElems * sizeof(Float8));
  synchronize(device);

  for (int i = 0; i < kNumElems; ++i) {
    Float8 expected = static_cast<Float8>(dst_cpu(i));
    EXPECT_THAT(src_cpu(i), EqOrIsNan(expected));
  }

  // Clean up.
  device.deallocate(src_device_buffer);
  device.deallocate(dst_device_buffer);
  synchronize(device);
}

TEST(Float8Test, SmallCastToDenormal) {
  // Special edge-case where rounding to a normalized value would
  // normally round down, but rounding to a subnormal rounds up.
  float x = std::ldexp(1.3125, -15);
  float8_e5m2 y = static_cast<float8_e5m2>(x);
  float z = static_cast<float>(y);
  EXPECT_EQ(z, std::ldexp(1.5, -15));
}

}  // namespace
}  // namespace tsl
