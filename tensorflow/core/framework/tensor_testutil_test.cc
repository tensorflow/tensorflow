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

#include "tensorflow/core/framework/tensor_testutil.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace test {
namespace {

using internal::Expector;
using internal::Helper;

template <typename T>
static void TestEdgeCasesNear() {
  EXPECT_TRUE(Expector<T>::Near(Eigen::NumTraits<T>::infinity(),
                                Eigen::NumTraits<T>::infinity(), 0.0));
  EXPECT_TRUE(Expector<T>::Near(Eigen::NumTraits<T>::lowest(),
                                Eigen::NumTraits<T>::highest(),
                                Eigen::NumTraits<double>::infinity()));
  EXPECT_FALSE(Expector<T>::Near(Eigen::NumTraits<T>::lowest(),
                                 Eigen::NumTraits<T>::highest(),
                                 Eigen::NumTraits<double>::highest()));
  EXPECT_FALSE(Expector<T>::Near(Eigen::NumTraits<T>::quiet_NaN(),
                                 Eigen::NumTraits<T>::quiet_NaN(), 0.0));
  EXPECT_FALSE(Expector<T>::Near(Eigen::NumTraits<T>::quiet_NaN(),
                                 Eigen::NumTraits<T>::quiet_NaN(),
                                 Eigen::NumTraits<double>::infinity()));
}

// For debug printing. Example usage:
// dumpFloatingPointStorage<Eigen::half, uint16>(
//     static_cast<Eigen::half>(-2.71f));
// dumpFloatingPointStorage<float, uint32>(-2.718281f);
// dumpFloatingPointStorage <double, uint64>(-2.71828182846);
template <typename T, typename U>
static void dumpFloatingPointStorage(T value) {
  U* integral = reinterpret_cast<U*>(&value);
  int shift_amount = (sizeof(U) << 3) - 1;
  int exponent_bits = 2 + (log2(sizeof(U)) * 3);
  U mask = static_cast<U>(1) << shift_amount;
  for (int bits = 0; bits <= shift_amount; ++bits) {
    std::cout << ((*integral & mask) > 0);
    if (bits == 0 || bits == exponent_bits) std::cout << " ";
    mask >>= 1;
  }
  std::cout << std::endl;
  printf("%.20lf\n", static_cast<double>(value));
}

TEST(TensorTestUtilTest, ExpectTensorNearHalf) {
  // Eigen::half has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
  // The exponent is offset at 15.
  // https://en.wikipedia.org/wiki/Half-precision_floating-point_format
  typedef Eigen::half T;
#define HALF(x) static_cast<T>(x)

  // Trivial cases: equalities.
  EXPECT_TRUE(Expector<T>::Near(HALF(1.0f), HALF(1.0f), 0.0));
  EXPECT_TRUE(Expector<T>::Near(HALF(0.0f), HALF(-0.0f), 0.0));
  EXPECT_TRUE(Expector<T>::Near(HALF(3.141592f), HALF(3.141592f), 0.0));

  // 0 10010 0001111110 -> 1150/128 = 8.984375 vs
  // 0 10010 0001111111 -> 1151/128 = 8.9921875 (diff = 0.0078125)
  EXPECT_TRUE(Expector<T>::Near(HALF(8.9875f), HALF(8.99f), 0.0078125));
  EXPECT_FALSE(Expector<T>::Near(HALF(8.9875f), HALF(8.99f), 0.007));

  // 0 11000 0110100000 -> 1440/2 = 720 vs
  // 0 11000 0110100001 -> 1441/2 = 720.5 (diff = 0.5)
  EXPECT_TRUE(Expector<T>::Near(HALF(720.2f), HALF(720.3f), 0.5));
  EXPECT_FALSE(Expector<T>::Near(HALF(720.2f), HALF(720.3f), 0.4));

  // 0 11001 0011010010 -> 1234 vs
  // 0 11001 0011010011 -> 1235 (diff = 1)
  // Rounds to even (1234.5 -> 1234).
  EXPECT_TRUE(Expector<T>::Near(HALF(1234.f), HALF(1235.f), 1.0));
  EXPECT_FALSE(Expector<T>::Near(HALF(1234.5f), HALF(1235.f), 0.5));
  EXPECT_TRUE(Expector<T>::Near(HALF(1234.5f), HALF(1235.f), 1.0));

  // 1 10000 0101101100 -> -1388/512 = -2.7109375 vs
  // 1 10000 0101110001 -> -1393/512 = -2.720703125 (diff = 0.009765625)
  EXPECT_TRUE(Expector<T>::Near(HALF(-2.71f), HALF(-2.72f), 0.01));

#undef HALF

  // Some of the cases failed because Eigen::half doesn't behave as expected.
  // For example, (inf == inf) should have been true, but it returns false.
  // TODO(penporn): uncomment this test once we fix Eigen::half
  // TestEdgeCasesNear<T>();
}

TEST(TensorTestUtilTest, ExpectTensorNearFloat) {
  // float has 1 sign bit, 8 exponent bits, and 23 mantissa bits.
  // The exponent offset is 127.
  // https://en.wikipedia.org/wiki/Single-precision_floating-point_format
  typedef float T;
  // Trivial cases: equalities.
  EXPECT_TRUE(Expector<T>::Near(1.0f, 1.0f, 0.0));
  EXPECT_TRUE(Expector<T>::Near(0.0f, -0.0f, 0.0));
  EXPECT_TRUE(Expector<T>::Near(3.14159265359f, 3.14159265359f, 0.0));

  // 0 10000010 00011111100110011001101 -> 9,424,077/2^20 vs
  // 0 10000010 00011111100110100110110 -> 9,424,182/2^20
  // diff = 105/2^20 = 0.000100135803223
  EXPECT_TRUE(Expector<T>::Near(8.9875f, 8.9876f, 0.0001002));
  EXPECT_FALSE(Expector<T>::Near(8.9875f, 8.9876f, 0.0001));

  // 0 10001000 01101000000110011101001 -> 11,799,785/2^14 vs
  // 0 10001000 01101000000110011101010 -> 11,799,786/2^14
  // diff = 1/2^14 = 0.00006103515625
  EXPECT_TRUE(Expector<T>::Near(720.2017f, 720.2018f, 0.0001));
  EXPECT_FALSE(Expector<T>::Near(720.20175f, 720.20185f, 0.0001));
  EXPECT_TRUE(Expector<T>::Near(720.20175f, 720.20185f, 0.00013));

  // 0 10011001 11010110111100110100010 -> 15,432,098*2^3 vs
  // 0 10011001 11010110111100110100011 -> 15,432,099*2^3 (diff = 2^3 = 8)
  EXPECT_FALSE(Expector<T>::Near(123456788.f, 123456789.f, 4.0));
  EXPECT_TRUE(Expector<T>::Near(123456788.f, 123456789.f, 8.0));

  // 1 10000000 01011011111100001010001 -> 11,401,297/2^22 vs
  // 1 10000000 01011011111100001010101 -> 11,401,301/2^22
  // diff = 4/2^22 = 0.000000953674316
  EXPECT_TRUE(Expector<T>::Near(-2.718281f, -2.718282f, 0.1));

  TestEdgeCasesNear<T>();
}

TEST(TensorTestUtilTest, ExpectTensorNearDouble) {
  // double has 1 sign bit, 11 exponent bits, and 52 mantissa bits.
  // The exponent offset is 1,023.
  // https://en.wikipedia.org/wiki/Double-precision_floating-point_format
  typedef double T;
  // Trivial cases: equalities.
  EXPECT_TRUE(Expector<T>::Near(1.0, 1.0, 0.0));
  EXPECT_TRUE(Expector<T>::Near(0.0, -0.0, 0.0));
  EXPECT_TRUE(Expector<T>::Near(3.14159265359, 3.14159265359, 0.0));

  // 0 10000000010 0001111110011001100110011001100110011001100110011010
  //   -> 5,059,512,706,374,042/2^49 vs
  // 0 10000000010 0001111110011010011010110101000010110000111100101000
  //   -> 5,059,569,001,369,384/2^49
  // diff = 56,294,995,342/2^49 = 9.999999999976694198267E-5
  EXPECT_TRUE(Expector<T>::Near(8.9875, 8.9876, 0.0001));

  // 0 10000001111 1000100101110000001100111010100100101010001100000101
  //   -> 6,921,439,564,440,325/2^36
  // 0 10000001111 1000100101110000001100111010111110110111111010010001
  //   -> 6,921,439,571,312,273/2^36
  // diff = 6,871,948/2^36 = 1.000000047497451305389E-4
  EXPECT_FALSE(Expector<T>::Near(100720.2018, 100720.2019, 0.0001));
  EXPECT_TRUE(Expector<T>::Near(100720.2018, 100720.2019, 1.00000005e-4));

  // 0 10000110100 0101111011100010101000101110101101011010010111000100
  //   -> 6,172,839,450,617,284 * 2
  // 0 10000110100 0101111011100010101000101110101101011010010111000011
  //   -> 6,172,839,450,617,283 * 2
  // diff = 1 * 2 = 2
  EXPECT_FALSE(Expector<T>::Near(12345678901234567., 12345678901234566., 1.0));
  EXPECT_TRUE(Expector<T>::Near(12345678901234567., 12345678901234566., 2.0));

  // 1 10000000000 0101101111110000101010001011000101000101111111001111
  //   -> -6,121,026,514,870,223/2^51
  // 1 10000000000 0101101111110000101010001011000101001011011111000101
  //   -> -6,121,026,514,892,741/2^51
  // diff = 22,518/2^51 = 1.00000008274037099909E-11
  EXPECT_FALSE(Expector<T>::Near(-2.71828182846, -2.71828182847, 1.0e-11));
  EXPECT_TRUE(
      Expector<T>::Near(-2.71828182846, -2.71828182847, 1.00000009e-11));

  TestEdgeCasesNear<T>();
}

static const double kSlackFactor = 5.0;

template <typename T>
static void TestEdgeCasesClose() {
  T kZero = static_cast<T>(0.0);
  EXPECT_TRUE(Helper<T>::IsClose(Eigen::NumTraits<T>::infinity(),
                                 Eigen::NumTraits<T>::infinity(), kZero,
                                 kZero));
  EXPECT_TRUE(Helper<T>::IsClose(
      Eigen::NumTraits<T>::lowest(), Eigen::NumTraits<T>::highest(),
      Eigen::NumTraits<T>::infinity(), Eigen::NumTraits<T>::infinity()));
  EXPECT_TRUE(Helper<T>::IsClose(
      Eigen::NumTraits<T>::lowest(), Eigen::NumTraits<T>::highest(),
      Eigen::NumTraits<T>::highest(), Eigen::NumTraits<T>::highest()));
  EXPECT_FALSE(Helper<T>::IsClose(Eigen::NumTraits<T>::quiet_NaN(),
                                  Eigen::NumTraits<T>::quiet_NaN(), kZero,
                                  kZero));
  EXPECT_FALSE(Helper<T>::IsClose(
      Eigen::NumTraits<T>::quiet_NaN(), Eigen::NumTraits<T>::quiet_NaN(),
      Eigen::NumTraits<T>::infinity(), Eigen::NumTraits<T>::infinity()));
}

TEST(TensorTestUtilTest, ExpectTensorCloseHalf) {
  typedef Eigen::half T;
#define HALF(x) static_cast<T>(x)
  EXPECT_TRUE(
      Helper<T>::IsClose(HALF(1.0f), HALF(1.1f), HALF(0.1f), HALF(0.1f)));
  EXPECT_TRUE(
      Helper<T>::IsClose(HALF(1.0f), HALF(1.0f), HALF(0.0f), HALF(0.0f)));
  EXPECT_FALSE(
      Helper<T>::IsClose(HALF(1.0f), HALF(1.1f), HALF(0.0f), HALF(0.0f)));

  // Epsilon:      0 00010 0000000000 -> 2^-13  = 0.0001220703125
  // kDefaultTol:  0 00100 0100000000 -> 5/2^13 = 0.0006103515625
  const T kDefaultTol =
      static_cast<T>(kSlackFactor) * Eigen::NumTraits<T>::epsilon();

  // 1.234 -> 0 01111 0011110000 -> 1264/2^10 = 1.234375
  // 1.233 -> 0 01111 0011101111 -> 1263/2^10 = 1.2333984375
  // 1.235 -> 0 01111 0011110001 -> 1265/2^10 = 1.2353515625
  // 1.232 -> 0 01111 0011101110 -> 1262/2^10 = 1.232421875
  // 1.236 -> 0 01111 0011110010 -> 1266/2^10 = 1.236328125
  // 1/2^10 = 0.0009765625E
  // Threshold = 0.0013637542724609375
  EXPECT_TRUE(
      Helper<T>::IsClose(HALF(1.234f), HALF(1.234f), kDefaultTol, kDefaultTol));
  EXPECT_TRUE(
      Helper<T>::IsClose(HALF(1.234f), HALF(1.233f), kDefaultTol, kDefaultTol));
  EXPECT_TRUE(
      Helper<T>::IsClose(HALF(1.234f), HALF(1.235f), kDefaultTol, kDefaultTol));

  // Diff = 0.001953125
  EXPECT_FALSE(
      Helper<T>::IsClose(HALF(1.234f), HALF(1.232f), kDefaultTol, kDefaultTol));
  EXPECT_FALSE(
      Helper<T>::IsClose(HALF(1.234f), HALF(1.236f), kDefaultTol, kDefaultTol));
  EXPECT_TRUE(
      Helper<T>::IsClose(HALF(1.234f), HALF(1.232f), HALF(8e-4f), HALF(1e-3f)));
  EXPECT_TRUE(Helper<T>::IsClose(HALF(1.234f), HALF(1.236f), HALF(1.4e-3f),
                                 HALF(5e-4f)));

  // Too fine-grained: won't detect the difference
  EXPECT_TRUE(Helper<T>::IsClose(HALF(3.141592f), HALF(3.141593f), HALF(0.0),
                                 HALF(0.0)));

  // Trivial case.
  EXPECT_FALSE(
      Helper<T>::IsClose(HALF(1e4f), HALF(1e-4f), kDefaultTol, kDefaultTol));
#undef HALF

  // Some of the cases failed because Eigen::half doesn't behave as expected.
  // For example, (inf == inf) should have been true, but it returns false.
  // TODO(penporn): uncomment this test once we fix Eigen::half
  // TestEdgeCasesClose<T>();
}

TEST(TensorTestUtilTest, ExpectTensorCloseFloat) {
  typedef float T;

  EXPECT_TRUE(Helper<T>::IsClose(1.0f, 1.1f, 0.1f, 0.1f));
  EXPECT_TRUE(Helper<T>::IsClose(1.0f, 1.0f, 0.0f, 0.0f));
  EXPECT_FALSE(Helper<T>::IsClose(1.0f, 1.1f, 0.0f, 0.0f));

  // Epsilon:      2^-23  ~ 0.00000011920928955078
  // kDefaultTol:  5/2^23 ~ 0.00000059604644775391
  const T kDefaultTol =
      static_cast<T>(kSlackFactor) * Eigen::NumTraits<T>::epsilon();

  // 1.234567f -> 10,356,299/2^23 ~ 1.234567046165466308594
  // 1.234568f -> 10,356,307/2^23 ~ 1.234567999839782714844
  // 1.234566f -> 10,356,290/2^23 ~ 1.234565973281860351563
  // 1.234569f -> 10,356,315/2^23 ~ 1.234568953514099121094
  // 1.234565f -> 10,356,282/2^23 ~ 1.234565019607543945313
  // Threshold ~ 0.00000133190576434572
  EXPECT_TRUE(
      Helper<T>::IsClose(1.234567f, 1.234567f, kDefaultTol, kDefaultTol));
  EXPECT_TRUE(
      Helper<T>::IsClose(1.234567f, 1.234568f, kDefaultTol, kDefaultTol));
  EXPECT_TRUE(
      Helper<T>::IsClose(1.234567f, 1.234566f, kDefaultTol, kDefaultTol));
  EXPECT_FALSE(
      Helper<T>::IsClose(1.234567f, 1.234569f, kDefaultTol, kDefaultTol));
  EXPECT_FALSE(
      Helper<T>::IsClose(1.234567f, 1.234565f, kDefaultTol, kDefaultTol));
  EXPECT_TRUE(Helper<T>::IsClose(1.234567f, 1.234569f, 8e-7f, 1e-6f));
  EXPECT_TRUE(Helper<T>::IsClose(1.234567f, 1.234565f, 3e-7f, 1.5e-6f));

  // Too fine-grained: won't detect the difference
  EXPECT_TRUE(Helper<T>::IsClose(3.14159265f, 3.14159266f, 0.0f, 0.0f));

  // Trivial cases
  EXPECT_FALSE(Helper<T>::IsClose(1e8f, 1e-8f, kDefaultTol, kDefaultTol));
  EXPECT_FALSE(Helper<T>::IsClose(1e15f, 1e-15f, kDefaultTol, kDefaultTol));

  TestEdgeCasesClose<T>();
}

TEST(TensorTestUtilTest, ExpectTensorCloseDouble) {
  typedef double T;

  EXPECT_TRUE(Helper<T>::IsClose(1.0, 1.1, 0.1, 0.1));
  EXPECT_TRUE(Helper<T>::IsClose(1.0, 1.0, 0.0, 0.0));
  EXPECT_FALSE(Helper<T>::IsClose(1.0, 1.1, 0.0, 0.0));

  // Epsilon:      2^-52  ~ 2.220446049250313080847E-16
  // kDefaultTol:  5/2^52 ~ 1.110223024625156540424E-15
  const T kDefaultTol =
      static_cast<T>(kSlackFactor) * Eigen::NumTraits<T>::epsilon();

  // 1.234567890123456 -> 5,559,999,489,923,576/2^52 ~ 1.234567890123456024298
  // 1.234567890123457 -> 5,559,999,489,923,580/2^52 ~ 1.234567890123456912477
  // 1.234567890123455 -> 5,559,999,489,923,571/2^52 ~ 1.234567890123454914075
  // 1.234567890123458 -> 5,559,999,489,923,585/2^52 ~ 1.2345678901234580227
  // 1.234567890123454 -> 5,559,999,489,923,567/2^52 ~ 1.234567890123454025897
  // 1.234567890123459 -> 5,559,999,489,923,589/2^52 ~ 1.234567890123458910878
  // 1.234567890123453 -> 5,559,999,489,923,562/2^52 ~ 1.234567890123452915674
  // Threshold ~ 2.480868721703117812159E-15
  EXPECT_TRUE(Helper<T>::IsClose(1.234567890123456, 1.234567890123456,
                                 kDefaultTol, kDefaultTol));
  EXPECT_TRUE(Helper<T>::IsClose(1.234567890123456, 1.234567890123457,
                                 kDefaultTol, kDefaultTol));
  EXPECT_TRUE(Helper<T>::IsClose(1.234567890123456, 1.234567890123455,
                                 kDefaultTol, kDefaultTol));
  EXPECT_TRUE(Helper<T>::IsClose(1.234567890123456, 1.234567890123458,
                                 kDefaultTol, kDefaultTol));
  EXPECT_TRUE(Helper<T>::IsClose(1.234567890123456, 1.234567890123454,
                                 kDefaultTol, kDefaultTol));
  EXPECT_FALSE(Helper<T>::IsClose(1.234567890123456, 1.234567890123459,
                                  kDefaultTol, kDefaultTol));
  EXPECT_FALSE(Helper<T>::IsClose(1.234567890123456, 1.234567890123453,
                                  kDefaultTol, kDefaultTol));
  EXPECT_TRUE(Helper<T>::IsClose(1.234567890123456, 1.234567890123459, 9.5e-16,
                                 1.6e-15));
  EXPECT_TRUE(
      Helper<T>::IsClose(1.234567890123456, 1.234567890123453, 7e-16, 2e-15));

  // Too fine-grained: won't detect the difference
  EXPECT_TRUE(
      Helper<T>::IsClose(3.141592653589793238, 3.141592653589793239, 0.0, 0.0));

  // Trivial cases
  EXPECT_FALSE(Helper<T>::IsClose(1e15, 1e-15, kDefaultTol, kDefaultTol));
  EXPECT_FALSE(Helper<T>::IsClose(1e30, 1e-30, kDefaultTol, kDefaultTol));

  TestEdgeCasesClose<T>();
}

}  // namespace
}  // namespace test
}  // namespace tensorflow
