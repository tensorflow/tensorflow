/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "hwy/targets.h"
#include "ml_dtypes/include/float8.h"
#include "xla/hlo/testlib/test.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/denormal.h"
#include "tsl/platform/ml_dtypes.h"

namespace xla {
namespace {

using ::testing::Each;
using ::testing::ElementsAre;

TEST(UtilTest, Product) {
  EXPECT_EQ(Product({}), 1);
  EXPECT_EQ(Product({1}), 1);
  EXPECT_EQ(Product({2, 3}), 2 * 3);
  EXPECT_EQ(Product({2, 7, 9}), 2 * 7 * 9);
}

TEST(UtilTest, ToMixedRadix) {
  EXPECT_THAT(ToMixedRadix<std::vector<int64_t>>(0, {2, 3, 4}),
              ElementsAre(0, 0, 0));
  EXPECT_THAT(ToMixedRadix<std::vector<int64_t>>(1, {2, 3, 4}),
              ElementsAre(0, 0, 1));
  EXPECT_THAT(ToMixedRadix<std::vector<int64_t>>(19, {2, 3, 4}),
              ElementsAre(19 / (3 * 4), 7 / 4, 3));
  EXPECT_THAT(ToMixedRadix<std::vector<int64_t>>(23, {3, 2, 4}),
              ElementsAre(23 / (2 * 4), 7 / 4, 3));
}

// Verifies that, even with a different number of leading spaces, the
// Reindent routine turns them into a uniform number of leading spaces.
//
// Also throws in some trailing whitespace on the original to show it is
// removed.
TEST(UtilTest, ReindentsDifferentNumberOfLeadingSpacesUniformly) {
  std::string original = R"(   hello there
      world)";
  std::string got = Reindent(original, "  ");
  std::string want = R"(  hello there
  world)";
  EXPECT_EQ(want, got);
}

TEST(UtilTest, HumanReadableNumFlopsExample) {
  ASSERT_EQ("1.00GFLOP/s", HumanReadableNumFlops(1e9, 1e9));
}

TEST(UtilTest, CommaSeparatedString) {
  EXPECT_EQ(CommaSeparatedString({}), "");
  EXPECT_EQ(CommaSeparatedString({"hello world"}), "hello world");
  EXPECT_EQ(CommaSeparatedString({1, 57, 2}, "foo", "bar"), "foo1, 57, 2bar");
}

TEST(UtilTest, VectorString) {
  std::list<int64_t> empty_list;
  EXPECT_EQ(VectorString(empty_list), "()");

  std::vector<float> float_vector = {5.5};
  EXPECT_EQ(VectorString(float_vector), "(5.5)");

  std::set<absl::string_view> string_set = {absl::string_view("a"),
                                            absl::string_view("b")};
  EXPECT_EQ(VectorString(string_set), "(a, b)");

  EXPECT_EQ(VectorString({}), "()");
  EXPECT_EQ(VectorString({1, 57, 2}), "(1, 57, 2)");
}

TEST(UtilTest, CommonFactors) {
  struct {
    std::vector<int64_t> a, b;
    absl::InlinedVector<std::pair<int64_t, int64_t>, 8> expected;
  } test_cases[] = {
      {/*.a =*/{0}, /*.b =*/{0}, /*.expected =*/{{0, 0}, {1, 1}}},
      {/*.a =*/{1}, /*.b =*/{}, /*.expected =*/{{0, 0}, {1, 0}}},
      {/*.a =*/{}, /*.b =*/{1}, /*.expected =*/{{0, 0}, {0, 1}}},
      {/*.a =*/{0, 10}, /*.b =*/{0, 10, 3}, /*.expected =*/{{0, 0}, {2, 3}}},
      {/*.a =*/{1, 0}, /*.b =*/{1, 0, 1},
       /*.expected =*/{{0, 0}, {1, 1}, {2, 2}, {2, 3}}},
      {/*.a =*/{0, 1}, /*.b =*/{0, 1}, /*.expected =*/{{0, 0}, {1, 1}, {2, 2}}},
      {/*.a =*/{}, /*.b =*/{}, /*.expected =*/{{0, 0}}},
      {/*.a =*/{2, 5, 1, 3},
       /*.b =*/{1, 10, 3, 1},
       /*.expected =*/{{0, 0}, {0, 1}, {2, 2}, {3, 2}, {4, 3}, {4, 4}}},
      {/*.a =*/{1, 1, 3},
       /*.b =*/{1, 1, 3},
       /*.expected =*/{{0, 0}, {1, 1}, {2, 2}, {3, 3}}},
      // Splitting and combining dimensions.
      {/*.a =*/{2, 6},
       /*.b =*/{4, 3},
       /*.expected =*/{{0, 0}, {2, 2}}},
      {/*.a =*/{1, 2, 6},
       /*.b =*/{4, 1, 3, 1},
       /*.expected =*/{{0, 0}, {1, 0}, {3, 3}, {3, 4}}},
      // Extra degenerated dimension (second and third dims in the output) forms
      // single common factor group.
      {/*.a =*/{1, 2, 1},
       /*.b =*/{1, 1, 1, 2},
       /*.expected =*/{{0, 0}, {1, 1}, {1, 2}, {1, 3}, {2, 4}, {3, 4}}}};
  for (const auto& test_case : test_cases) {
    EXPECT_EQ(test_case.expected, CommonFactors(test_case.a, test_case.b));
  }
}

TEST(UtilTest, SanitizeFileName) {
  EXPECT_EQ(SanitizeFileName(""), "");
  EXPECT_EQ(SanitizeFileName("abc"), "abc");
  EXPECT_EQ(SanitizeFileName("/\\[]"), "____");
  EXPECT_EQ(SanitizeFileName("/A\\B[C]"), "_A_B_C_");
}

TEST(UtilTest, RoundTripFpToString) {
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<tsl::float8_e5m2>(
                false, QuietNanWithoutPayload<tsl::float8_e5m2>())),
            "nan");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<tsl::float8_e5m2>(
                true, QuietNanWithoutPayload<tsl::float8_e5m2>())),
            "-nan");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<tsl::float8_e4m3>(
                false, QuietNanWithoutPayload<tsl::float8_e4m3>())),
            "nan");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<tsl::float8_e4m3>(
                true, QuietNanWithoutPayload<tsl::float8_e4m3>())),
            "-nan");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<tsl::float8_e3m4>(
                false, QuietNanWithoutPayload<tsl::float8_e3m4>())),
            "nan");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<tsl::float8_e3m4>(
                true, QuietNanWithoutPayload<tsl::float8_e3m4>())),
            "-nan");
  EXPECT_EQ(
      RoundTripFpToString(std::numeric_limits<tsl::float8_e4m3fn>::quiet_NaN()),
      "nan");
  EXPECT_EQ(RoundTripFpToString(
                -std::numeric_limits<tsl::float8_e4m3fn>::quiet_NaN()),
            "-nan");
  EXPECT_EQ(RoundTripFpToString(
                std::numeric_limits<tsl::float8_e4m3b11fnuz>::quiet_NaN()),
            "-nan");
  EXPECT_EQ(RoundTripFpToString(
                std::numeric_limits<tsl::float8_e4m3fnuz>::quiet_NaN()),
            "-nan");
  EXPECT_EQ(RoundTripFpToString(
                std::numeric_limits<tsl::float8_e5m2fnuz>::quiet_NaN()),
            "-nan");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<half>(
                false, QuietNanWithoutPayload<half>())),
            "nan");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<half>(
                true, QuietNanWithoutPayload<half>())),
            "-nan");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<bfloat16>(
                false, QuietNanWithoutPayload<bfloat16>())),
            "nan");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<bfloat16>(
                true, QuietNanWithoutPayload<bfloat16>())),
            "-nan");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<float>(
                false, QuietNanWithoutPayload<float>())),
            "nan");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<float>(
                true, QuietNanWithoutPayload<float>())),
            "-nan");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<double>(
                false, QuietNanWithoutPayload<double>())),
            "nan");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<double>(
                true, QuietNanWithoutPayload<double>())),
            "-nan");

  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<half>(false, 0x1)),
            "nan(0x1)");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<half>(true, 0x1)),
            "-nan(0x1)");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<bfloat16>(false, 0x1)),
            "nan(0x1)");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<bfloat16>(true, 0x1)),
            "-nan(0x1)");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<float>(false, 0x1)),
            "nan(0x1)");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<float>(true, 0x1)),
            "-nan(0x1)");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<double>(false, 0x1)),
            "nan(0x1)");
  EXPECT_EQ(RoundTripFpToString(NanWithSignAndPayload<double>(true, 0x1)),
            "-nan(0x1)");
}

TEST(UtilTest, RoundTripFpToStringOnFlushDenormalThread) {
  // tsl::thread::ThreadPool workers run their tasks in this mode.
  tsl::port::ScopedFlushDenormal flush;
  EXPECT_EQ(RoundTripFpToString(std::numeric_limits<float>::denorm_min()),
            "1.4013e-45");
  EXPECT_EQ(RoundTripFpToString(std::numeric_limits<double>::denorm_min()),
            "4.94065645841247e-324");
  EXPECT_EQ(RoundTripFpToString(std::numeric_limits<bfloat16>::denorm_min()),
            "9.184e-41");
}

namespace {
template <typename T>
void TotalOrderHelper(T x, T y) {
  auto x_sm = ToSignMagnitude(x);
  auto y_sm = ToSignMagnitude(y);
  bool x_sign = static_cast<bool>(SignAndMagnitude(x).first);
  bool y_sign = static_cast<bool>(SignAndMagnitude(y).first);
  if (x_sign && !y_sign) {
    EXPECT_LT(x_sm, y_sm) << x << " " << y;
  }
  if (!x_sign && y_sign) {
    EXPECT_GT(x_sm, y_sm) << x << " " << y;
  }
  if (x == y && x_sign == y_sign) {
    EXPECT_EQ(x_sm, y_sm) << x << " " << y;
  }
  if (x < y) {
    EXPECT_LT(x_sm, y_sm) << x << " " << y;
  }
  if (x > y) {
    EXPECT_GT(x_sm, y_sm) << x << " " << y;
  }
  if (Eigen::numext::isnan(x) && x_sign && !Eigen::numext::isnan(y)) {
    EXPECT_LT(x_sm, y_sm) << x << " " << y;
  }
  if (Eigen::numext::isnan(x) && !x_sign && !Eigen::numext::isnan(y)) {
    EXPECT_GT(x_sm, y_sm) << x << " " << y;
  }
  if (Eigen::numext::isnan(y) && y_sign && !Eigen::numext::isnan(x)) {
    EXPECT_GT(x_sm, y_sm) << x << " " << y;
  }
  if (Eigen::numext::isnan(y) && !y_sign && !Eigen::numext::isnan(x)) {
    EXPECT_LT(x_sm, y_sm) << x << " " << y;
  }
}
}  // namespace

TEST(UtilTest, TotalOrder_F4E2M1FN) {
  for (int a = 0; a < 16; ++a) {
    tsl::float4_e2m1fn x =
        Eigen::numext::bit_cast<tsl::float4_e2m1fn>(static_cast<uint8_t>(a));
    for (int b = 0; b < 16; ++b) {
      tsl::float4_e2m1fn y =
          Eigen::numext::bit_cast<tsl::float4_e2m1fn>(static_cast<uint8_t>(b));
      TotalOrderHelper(x, y);
    }
  }
}

TEST(UtilTest, TotalOrder_F8E5M2) {
  for (int a = 0; a < 256; ++a) {
    tsl::float8_e5m2 x =
        Eigen::numext::bit_cast<tsl::float8_e5m2>(static_cast<uint8_t>(a));
    for (int b = 0; b < 256; ++b) {
      tsl::float8_e5m2 y =
          Eigen::numext::bit_cast<tsl::float8_e5m2>(static_cast<uint8_t>(b));
      TotalOrderHelper(x, y);
    }
  }
}

TEST(UtilTest, TotalOrder_F8E4M3) {
  for (int a = 0; a < 256; ++a) {
    tsl::float8_e4m3 x =
        Eigen::numext::bit_cast<tsl::float8_e4m3>(static_cast<uint8_t>(a));
    for (int b = 0; b < 256; ++b) {
      tsl::float8_e4m3 y =
          Eigen::numext::bit_cast<tsl::float8_e4m3>(static_cast<uint8_t>(b));
      TotalOrderHelper(x, y);
    }
  }
}

TEST(UtilTest, TotalOrder_F8E4M3FN) {
  for (int a = 0; a < 256; ++a) {
    tsl::float8_e4m3fn x =
        Eigen::numext::bit_cast<tsl::float8_e4m3fn>(static_cast<uint8_t>(a));
    for (int b = 0; b < 256; ++b) {
      tsl::float8_e4m3fn y =
          Eigen::numext::bit_cast<tsl::float8_e4m3fn>(static_cast<uint8_t>(b));
      TotalOrderHelper(x, y);
    }
  }
}

TEST(UtilTest, TotalOrder_F8E4M3B11) {
  for (int a = 0; a < 256; ++a) {
    tsl::float8_e4m3b11fnuz x =
        Eigen::numext::bit_cast<tsl::float8_e4m3b11fnuz>(
            static_cast<uint8_t>(a));
    for (int b = 0; b < 256; ++b) {
      tsl::float8_e4m3b11fnuz y =
          Eigen::numext::bit_cast<tsl::float8_e4m3b11fnuz>(
              static_cast<uint8_t>(b));
      TotalOrderHelper(x, y);
    }
  }
}

TEST(UtilTest, TotalOrder_F8E4M3FNUZ) {
  for (int a = 0; a < 256; ++a) {
    tsl::float8_e4m3fnuz x =
        Eigen::numext::bit_cast<tsl::float8_e4m3fnuz>(static_cast<uint8_t>(a));
    for (int b = 0; b < 256; ++b) {
      tsl::float8_e4m3fnuz y = Eigen::numext::bit_cast<tsl::float8_e4m3fnuz>(
          static_cast<uint8_t>(b));
      TotalOrderHelper(x, y);
    }
  }
}

TEST(UtilTest, TotalOrder_F8E5M2FNUZ) {
  for (int a = 0; a < 256; ++a) {
    tsl::float8_e5m2fnuz x =
        Eigen::numext::bit_cast<tsl::float8_e5m2fnuz>(static_cast<uint8_t>(a));
    for (int b = 0; b < 256; ++b) {
      tsl::float8_e5m2fnuz y = Eigen::numext::bit_cast<tsl::float8_e5m2fnuz>(
          static_cast<uint8_t>(b));
      TotalOrderHelper(x, y);
    }
  }
}

TEST(UtilTest, TotalOrder_F8E3M4) {
  for (int a = 0; a < 256; ++a) {
    tsl::float8_e3m4 x =
        Eigen::numext::bit_cast<tsl::float8_e3m4>(static_cast<uint8_t>(a));
    for (int b = 0; b < 256; ++b) {
      tsl::float8_e3m4 y =
          Eigen::numext::bit_cast<tsl::float8_e3m4>(static_cast<uint8_t>(b));
      TotalOrderHelper(x, y);
    }
  }
}

TEST(UtilTest, TotalOrder_F8E8M0FNU) {
  for (int a = 0; a < 256; ++a) {
    tsl::float8_e8m0fnu x =
        Eigen::numext::bit_cast<tsl::float8_e8m0fnu>(static_cast<uint8_t>(a));
    for (int b = 0; b < 256; ++b) {
      tsl::float8_e8m0fnu y =
          Eigen::numext::bit_cast<tsl::float8_e8m0fnu>(static_cast<uint8_t>(b));
      TotalOrderHelper(x, y);
    }
  }
}

TEST(UtilTest, TotalOrder_F8E5M3FNU) {
  for (int a = 0; a < 256; ++a) {
    tsl::float8_e5m3fnu x =
        Eigen::numext::bit_cast<tsl::float8_e5m3fnu>(static_cast<uint8_t>(a));
    for (int b = 0; b < 256; ++b) {
      tsl::float8_e5m3fnu y =
          Eigen::numext::bit_cast<tsl::float8_e5m3fnu>(static_cast<uint8_t>(b));
      TotalOrderHelper(x, y);
    }
  }
}

void PackInt4(absl::Span<const char> input, absl::Span<char> output) {
  CHECK_EQ(output.size(), CeilOfRatio(input.size(), size_t{2}));
  for (size_t i = 0; i < input.size(); ++i) {
    // Mask out the high-order 4 bits in case they have extraneous data.
    char val = input[i] & 0xf;
    if (i % 2 == 0) {
      output[i / 2] = val;
    } else {
      output[i / 2] |= val << 4;
    }
  }
}

TEST(UtilTest, PackInt4) {
  for (int size : {7, 15, 63, 64, 127, 128, 1024}) {
    std::vector<char> input(size);

    absl::c_iota(input, 0);

    std::vector<char> output_ref(CeilOfRatio<int64_t>(input.size(), 2));
    PackInt4(input, absl::MakeSpan(output_ref));

    std::vector<char> output_dut(CeilOfRatio<int64_t>(input.size(), 2));
    PackIntN(4, input, absl::MakeSpan(output_dut));
    for (size_t i = 0; i < output_dut.size(); ++i) {
      EXPECT_EQ(output_ref[i], output_dut[i])
          << "Size: " << size << " i: " << i;
    }

    std::vector<char> unpacked(input.size());
    UnpackIntN(4, output_ref, absl::MakeSpan(unpacked));
    for (size_t i = 0; i < input.size(); ++i) {
      EXPECT_EQ(unpacked[i], input[i] & 0xf) << "Size: " << size << " i: " << i;
    }
  }
}

TEST(UtilTest, PackInt4HexCheck) {
  std::vector<char> input = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7,
                             0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf};

  std::vector<char> expected_output = {
      static_cast<char>(0x10), static_cast<char>(0x32), static_cast<char>(0x54),
      static_cast<char>(0x76), static_cast<char>(0x98), static_cast<char>(0xba),
      static_cast<char>(0xdc), static_cast<char>(0xfe)};

  std::vector<char> output(expected_output.size(), 0);
  PackIntN(4, input, absl::MakeSpan(output));

  EXPECT_EQ(output, expected_output);
}

TEST(UtilTest, UnpackInt4HexCheck) {
  std::vector<char> input = {static_cast<char>(0x10), static_cast<char>(0x32),
                             static_cast<char>(0x54), static_cast<char>(0x76),
                             static_cast<char>(0x98), static_cast<char>(0xba),
                             static_cast<char>(0xdc), static_cast<char>(0xfe)};

  std::vector<char> expected_output = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7,
                                       0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf};

  std::vector<char> output(expected_output.size(), 0);
  UnpackIntN(4, input, absl::MakeSpan(output));

  EXPECT_EQ(output, expected_output);
}

TEST(UtilTest, PackInt4HexCheckOdd) {
  std::vector<char> input = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8,
                             0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x9};

  std::vector<char> expected_output = {
      static_cast<char>(0x10), static_cast<char>(0x32),
      static_cast<char>(0x54), static_cast<char>(0x76),
      static_cast<char>(0x98), static_cast<char>(0xba),
      static_cast<char>(0xdc), static_cast<char>(0xfe),
      static_cast<char>(0x09)};

  std::vector<char> output(expected_output.size(), 0);
  PackIntN(4, input, absl::MakeSpan(output));

  EXPECT_EQ(output, expected_output);
}

TEST(UtilTest, UnpackInt4HexCheckOdd) {
  std::vector<char> input = {static_cast<char>(0x10), static_cast<char>(0x32),
                             static_cast<char>(0x54), static_cast<char>(0x76),
                             static_cast<char>(0x98), static_cast<char>(0xba),
                             static_cast<char>(0xdc), static_cast<char>(0xfe),
                             static_cast<char>(0xf9)};

  std::vector<char> expected_output = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5,
                                       0x6, 0x7, 0x8, 0x9, 0xa, 0xb,
                                       0xc, 0xd, 0xe, 0xf, 0x9};

  std::vector<char> output(expected_output.size(), 0);
  UnpackIntN(4, input, absl::MakeSpan(output));

  EXPECT_EQ(output, expected_output);
}

TEST(UtilTest, PackInt2HexCheck) {
  std::vector<char> input = {0x0, 0x1, 0x2, 0x3, 0x3, 0x2, 0x1, 0x0,
                             0x1, 0x2, 0x3, 0x0, 0x2, 0x3, 0x0, 0x1};

  std::vector<char> expected_output = {
      static_cast<char>(0xe4), static_cast<char>(0x1b), static_cast<char>(0x39),
      static_cast<char>(0x4e)};

  std::vector<char> output(expected_output.size(), 0);
  PackIntN(2, input, absl::MakeSpan(output));

  EXPECT_EQ(output, expected_output);
}

TEST(UtilTest, UnpackInt2HexCheck) {
  std::vector<char> input = {static_cast<char>(0xe4), static_cast<char>(0x1b),
                             static_cast<char>(0x39), static_cast<char>(0x4e)};

  std::vector<char> expected_output = {0x0, 0x1, 0x2, 0x3, 0x3, 0x2, 0x1, 0x0,
                                       0x1, 0x2, 0x3, 0x0, 0x2, 0x3, 0x0, 0x1};

  std::vector<char> output(expected_output.size(), 0);
  UnpackIntN(2, input, absl::MakeSpan(output));

  EXPECT_EQ(output, expected_output);
}

TEST(UtilTest, PackInt2HexCheckOdd) {
  std::vector<char> input = {0x0, 0x1, 0x2, 0x3, 0x3, 0x2, 0x1, 0x0, 0x1,
                             0x2, 0x3, 0x0, 0x2, 0x3, 0x0, 0x1, 0x2};

  std::vector<char> expected_output = {
      static_cast<char>(0xe4), static_cast<char>(0x1b), static_cast<char>(0x39),
      static_cast<char>(0x4e), static_cast<char>(0x02)};

  std::vector<char> output(expected_output.size(), 0);
  PackIntN(2, input, absl::MakeSpan(output));

  EXPECT_EQ(output, expected_output);
}

TEST(UtilTest, UnpackInt2HexCheckOdd) {
  std::vector<char> input = {static_cast<char>(0xe4), static_cast<char>(0x1b),
                             static_cast<char>(0x39), static_cast<char>(0x4e),
                             static_cast<char>(0xfa)};

  std::vector<char> expected_output = {0x0, 0x1, 0x2, 0x3, 0x3, 0x2,
                                       0x1, 0x0, 0x1, 0x2, 0x3, 0x0,
                                       0x2, 0x3, 0x0, 0x1, 0x2};

  std::vector<char> output(expected_output.size(), 0);
  UnpackIntN(2, input, absl::MakeSpan(output));

  EXPECT_EQ(output, expected_output);
}

TEST(UtilTest, PackInt1HexCheck) {
  std::vector<char> input = {0x0, 0x1, 0x0, 0x1, 0x0, 0x1, 0x0, 0x1,
                             0x0, 0x1, 0x0, 0x1, 0x0, 0x1, 0x0, 0x1};

  std::vector<char> expected_output = {static_cast<char>(0xaa),
                                       static_cast<char>(0xaa)};

  std::vector<char> output(expected_output.size(), 0);
  PackIntN(1, input, absl::MakeSpan(output));

  EXPECT_EQ(output, expected_output);
}

TEST(UtilTest, UnpackInt1HexCheck) {
  std::vector<char> input = {static_cast<char>(0xaa), static_cast<char>(0xaa)};

  std::vector<char> expected_output = {0x0, 0x1, 0x0, 0x1, 0x0, 0x1, 0x0, 0x1,
                                       0x0, 0x1, 0x0, 0x1, 0x0, 0x1, 0x0, 0x1};

  std::vector<char> output(expected_output.size(), 0);
  UnpackIntN(1, input, absl::MakeSpan(output));

  EXPECT_EQ(output, expected_output);
}

TEST(UtilTest, PackInt1HexCheckOdd) {
  std::vector<char> input = {0x0, 0x1, 0x0, 0x1, 0x0, 0x1, 0x0, 0x1, 0x0,
                             0x1, 0x0, 0x1, 0x0, 0x1, 0x0, 0x1, 0x1};

  std::vector<char> expected_output = {static_cast<char>(0xaa),
                                       static_cast<char>(0xaa),
                                       static_cast<char>(0x01)};

  std::vector<char> output(expected_output.size(), 0);
  PackIntN(1, input, absl::MakeSpan(output));

  EXPECT_EQ(output, expected_output);
}

TEST(UtilTest, UnpackInt1HexCheckOdd) {
  std::vector<char> input = {static_cast<char>(0xaa), static_cast<char>(0xaa),
                             static_cast<char>(0xff)};

  std::vector<char> expected_output = {0x0, 0x1, 0x0, 0x1, 0x0, 0x1,
                                       0x0, 0x1, 0x0, 0x1, 0x0, 0x1,
                                       0x0, 0x1, 0x0, 0x1, 0x1};

  std::vector<char> output(expected_output.size(), 0);
  UnpackIntN(1, input, absl::MakeSpan(output));

  EXPECT_EQ(output, expected_output);
}

class PackUnpackIntNTest : public testing::TestWithParam<int> {};

TEST_P(PackUnpackIntNTest, RoundTrip) {
  const int bitwidth = GetParam();
  std::vector<int> sizes = {255,  256,  257,  511,  512,  513,  1023,
                            1024, 1025, 2048, 4096, 8192, 10000};
  for (int size = 0; size <= 145; ++size) {
    sizes.push_back(size);
  }
  for (int size : sizes) {
    std::vector<char> input(size);

    for (int i = 0; i < input.size(); ++i) {
      input[i] = i & LsbMask<uint8_t>(bitwidth);
    }

    std::vector<char> packed(CeilOfRatio<int64_t>(input.size(), 8 / bitwidth));
    PackIntN(bitwidth, input, absl::MakeSpan(packed));
    std::vector<char> unpacked(input.size());
    UnpackIntN(bitwidth, packed, absl::MakeSpan(unpacked));
    for (size_t i = 0; i < input.size(); ++i) {
      EXPECT_EQ(unpacked[i], input[i])
          << "Bitwidth: " << bitwidth << " Size: " << size << " i: " << i;
    }
  }
}

TEST_P(PackUnpackIntNTest, RoundTripWithDirtyHighBits) {
  const int bitwidth = GetParam();
  std::vector<int> sizes = {256, 1024, 4096};
  for (int size = 1; size <= 145; ++size) {
    sizes.push_back(size);
  }
  for (int size : sizes) {
    std::vector<char> input(size);
    std::vector<char> expected(size);

    for (int i = 0; i < input.size(); ++i) {
      expected[i] = i & LsbMask<uint8_t>(bitwidth);
      // Fill upper bits with garbage
      input[i] = static_cast<char>(expected[i] | (0xF0 ^ (i * 17)));
    }

    std::vector<char> packed(CeilOfRatio<int64_t>(input.size(), 8 / bitwidth));
    PackIntN(bitwidth, input, absl::MakeSpan(packed));
    std::vector<char> unpacked(input.size());
    UnpackIntN(bitwidth, packed, absl::MakeSpan(unpacked));
    for (size_t i = 0; i < input.size(); ++i) {
      EXPECT_EQ(unpacked[i], expected[i])
          << "Bitwidth: " << bitwidth << " Size: " << size << " i: " << i;
    }
  }
}

// Checks every compiled Highway target that the CPU supports against a scalar
// reference, not just the target that dynamic dispatch would pick.
TEST_P(PackUnpackIntNTest, MatchesReferenceOnAllTargets) {
  const int bitwidth = GetParam();
  const int elements_per_byte = 8 / bitwidth;
  constexpr char kGuard = 0x5A;
  for (int64_t target : hwy::SupportedAndGeneratedTargets()) {
    hwy::SetSupportedTargetsForTest(target);
    SCOPED_TRACE(hwy::TargetName(target));
    for (int size :
         {0,   1,   2,   3,   7,   15,  16,  17,  24,  31,   32,  33,
          47,  48,  49,  63,  64,  65,  79,  80,  95,  96,   111, 112,
          127, 128, 129, 143, 144, 145, 255, 257, 511, 1000, 4099}) {
      SCOPED_TRACE(size);
      std::vector<char> input(size);
      std::vector<char> expected_unpacked(size);
      std::vector<char> expected_packed(
          CeilOfRatio<int64_t>(size, elements_per_byte), 0);
      for (int i = 0; i < size; ++i) {
        const int value = (i * 7 + i / 16) & LsbMask<uint8_t>(bitwidth);
        expected_unpacked[i] = static_cast<char>(value);
        // Packing must ignore the bits above `bitwidth`.
        input[i] = static_cast<char>(value | ((i * 29) << bitwidth));
        expected_packed[i / elements_per_byte] |=
            static_cast<char>(value << (bitwidth * (i % elements_per_byte)));
      }

      // The trailing guard byte detects writes past the end of the output.
      std::vector<char> packed(expected_packed.size() + 1, kGuard);
      PackIntN(bitwidth, input,
               absl::MakeSpan(packed).first(expected_packed.size()));
      EXPECT_EQ(packed.back(), kGuard);
      packed.pop_back();
      EXPECT_EQ(packed, expected_packed);

      std::vector<char> unpacked(size + 1, kGuard);
      UnpackIntN(bitwidth, expected_packed,
                 absl::MakeSpan(unpacked).first(size));
      EXPECT_EQ(unpacked.back(), kGuard);
      unpacked.pop_back();
      EXPECT_EQ(unpacked, expected_unpacked);
    }
  }
  hwy::SetSupportedTargetsForTest(0);
}

TEST_P(PackUnpackIntNTest, RandomizedRoundTripOnAllTargets) {
  const int bitwidth = GetParam();
  const int elements_per_byte = 8 / bitwidth;
  const uint8_t value_mask = LsbMask<uint8_t>(bitwidth);
  constexpr char kGuard = 0x5A;
  constexpr int kNumTrials = 300;
  constexpr size_t kMaxOffset = 64;

  std::mt19937_64 rng(0x5EED1234ULL + static_cast<uint64_t>(bitwidth));
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_int_distribution<size_t> offset_dist(0, kMaxOffset - 1);
  // Short lengths mostly exercise the tails; longer ones run the main loops.
  std::uniform_int_distribution<int> tiny_len_dist(0, 150);
  std::uniform_int_distribution<int> medium_len_dist(151, 1200);
  std::uniform_int_distribution<int> large_len_dist(1201, 16384);

  for (int64_t target : hwy::SupportedAndGeneratedTargets()) {
    hwy::SetSupportedTargetsForTest(target);
    SCOPED_TRACE(hwy::TargetName(target));
    for (int trial = 0; trial < kNumTrials; ++trial) {
      int size = 0;
      if (trial % 3 == 0) {
        size = tiny_len_dist(rng);
      } else if (trial % 3 == 1) {
        size = medium_len_dist(rng);
      } else {
        size = large_len_dist(rng);
      }
      const size_t in_offset = offset_dist(rng);
      const size_t packed_offset = offset_dist(rng);
      const size_t unpacked_offset = offset_dist(rng);
      SCOPED_TRACE(testing::Message()
                   << "trial=" << trial << " size=" << size << " offsets=("
                   << in_offset << "," << packed_offset << ","
                   << unpacked_offset << ")");

      const size_t packed_size = CeilOfRatio<int64_t>(size, elements_per_byte);

      // Random raw bytes (including dirty high bits) at an unaligned offset.
      std::vector<char> input_buf(in_offset + size + kMaxOffset, kGuard);
      std::vector<char> expected_unpacked(size);
      std::vector<char> expected_packed(packed_size, 0);
      for (int i = 0; i < size; ++i) {
        const uint8_t raw = static_cast<uint8_t>(byte_dist(rng));
        const uint8_t value = raw & value_mask;
        input_buf[in_offset + i] = static_cast<char>(raw);
        expected_unpacked[i] = static_cast<char>(value);
        expected_packed[i / elements_per_byte] |=
            static_cast<char>(value << (bitwidth * (i % elements_per_byte)));
      }

      // Pack into an unaligned buffer surrounded by guard bytes.
      std::vector<char> packed_buf(packed_offset + packed_size + kMaxOffset,
                                   kGuard);
      absl::Span<char> packed_span =
          absl::MakeSpan(packed_buf).subspan(packed_offset, packed_size);
      PackIntN(bitwidth,
               absl::MakeConstSpan(input_buf).subspan(in_offset, size),
               packed_span);
      EXPECT_THAT(absl::MakeConstSpan(packed_buf).first(packed_offset),
                  Each(kGuard));
      EXPECT_THAT(
          absl::MakeConstSpan(packed_buf).subspan(packed_offset + packed_size),
          Each(kGuard));
      EXPECT_EQ(absl::MakeConstSpan(packed_span),
                absl::MakeConstSpan(expected_packed));

      // Unpack the packed bytes back into an unaligned guarded buffer.
      std::vector<char> unpacked_buf(unpacked_offset + size + kMaxOffset,
                                     kGuard);
      absl::Span<char> unpacked_span =
          absl::MakeSpan(unpacked_buf).subspan(unpacked_offset, size);
      UnpackIntN(bitwidth, packed_span, unpacked_span);
      EXPECT_THAT(absl::MakeConstSpan(unpacked_buf).first(unpacked_offset),
                  Each(kGuard));
      EXPECT_THAT(
          absl::MakeConstSpan(unpacked_buf).subspan(unpacked_offset + size),
          Each(kGuard));
      EXPECT_EQ(absl::MakeConstSpan(unpacked_span),
                absl::MakeConstSpan(expected_unpacked));

      // If the last packed byte has unused high bits, verify UnpackIntN
      // ignores garbage in those bits and does not write past `size`.
      const int rem = size % elements_per_byte;
      if (rem != 0) {
        const uint8_t dirty_tail = static_cast<uint8_t>(byte_dist(rng))
                                   << (bitwidth * rem);
        packed_span.back() |= static_cast<char>(dirty_tail);
        std::fill(unpacked_buf.begin(), unpacked_buf.end(), kGuard);
        UnpackIntN(bitwidth, packed_span, unpacked_span);
        EXPECT_THAT(absl::MakeConstSpan(unpacked_buf).first(unpacked_offset),
                    Each(kGuard));
        EXPECT_THAT(
            absl::MakeConstSpan(unpacked_buf).subspan(unpacked_offset + size),
            Each(kGuard));
        EXPECT_EQ(absl::MakeConstSpan(unpacked_span),
                  absl::MakeConstSpan(expected_unpacked));
      }
    }
  }
  hwy::SetSupportedTargetsForTest(0);
}

INSTANTIATE_TEST_SUITE_P(PackUnpackIntNTest, PackUnpackIntNTest,
                         testing::Values(1, 2, 4));

TEST(UtilTest, PrintAllFields) {
  // Here we are using one of the bool fields that has the default value to
  // false and ensuring that it is always printed.
  ExecutionProfile execution_profile;
  execution_profile.set_compilation_cache_hit(true);
  std::string result = PrintAllFields(execution_profile);
  EXPECT_TRUE(absl::StrContains(result, "compilation_cache_hit: true"));
  execution_profile.set_compilation_cache_hit(false);
  result = PrintAllFields(execution_profile);
  EXPECT_TRUE(absl::StrContains(result, "compilation_cache_hit: false"));
}

TEST(UtilTest, ScopedLoggingTimerLazyEvaluation) {
  int counter = 0;
  auto get_label = [&]() {
    counter++;
    return "lazy_label";
  };

  // Case 1: Condition is false, should not evaluate label.
  {
    XLA_SCOPED_LOGGING_TIMER_IF(get_label(), false);
  }
  EXPECT_EQ(counter, 0);

  // Case 2: Level is very high (disabled), should not evaluate label.
  {
    XLA_SCOPED_LOGGING_TIMER_LEVEL(get_label(), 100);
  }
  EXPECT_EQ(counter, 0);
}

TEST(UtilTest, ErrorWithStrCatAndBacktrace) {
  absl::Status invalid_arg = InvalidArgumentStrCat("bad arg: ", 42);
  EXPECT_EQ(invalid_arg.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(invalid_arg.message(), ::testing::HasSubstr("bad arg: 42"));

  absl::Status internal = InternalStrCat("internal failure: ", "oom");
  EXPECT_EQ(internal.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(internal.message(),
              ::testing::HasSubstr("internal failure: oom"));

  absl::Status precondition = FailedPreconditionStrCat("state=", 1);
  EXPECT_EQ(precondition.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(precondition.message(), ::testing::HasSubstr("state=1"));

  absl::Status not_found = NotFoundStrCat("missing key: ", "foo");
  EXPECT_EQ(not_found.code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(not_found.message(), ::testing::HasSubstr("missing key: foo"));
}

void BM_PackIntN(::testing::benchmark::State& state) {
  const int bitwidth = state.range(0);
  const size_t num_elements = state.range(1);

  std::vector<char> input(num_elements);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = i & LsbMask<uint8_t>(bitwidth);
  }

  std::vector<char> packed(CeilOfRatio<int64_t>(num_elements, 8 / bitwidth));

  for (auto s : state) {
    PackIntN(bitwidth, input, absl::MakeSpan(packed));
    ::benchmark::DoNotOptimize(packed);
  }

  state.SetItemsProcessed(state.iterations() * num_elements);
}

BENCHMARK(BM_PackIntN)
    ->ArgPair(1, 10000)
    ->ArgPair(1, 10000000)
    ->ArgPair(1, 100000000)
    ->ArgPair(1, 500000000)
    ->ArgPair(1, 1000000000)
    ->ArgPair(2, 10000)
    ->ArgPair(2, 10000000)
    ->ArgPair(2, 100000000)
    ->ArgPair(2, 500000000)
    ->ArgPair(2, 1000000000)
    ->ArgPair(4, 37)
    ->ArgPair(4, 64)
    ->ArgPair(4, 100)
    ->ArgPair(4, 128)
    ->ArgPair(4, 131)
    ->ArgPair(4, 256)
    ->ArgPair(4, 1024)
    ->ArgPair(4, 10000)
    ->ArgPair(4, 10000000)
    ->ArgPair(4, 100000000)
    ->ArgPair(4, 500000000)
    ->ArgPair(4, 1000000000);

void BM_UnpackIntN(::testing::benchmark::State& state) {
  const int bitwidth = state.range(0);
  const size_t num_elements = state.range(1);

  std::vector<char> packed(CeilOfRatio<int64_t>(num_elements, 8 / bitwidth));
  std::vector<char> unpacked(num_elements);

  for (auto s : state) {
    UnpackIntN(bitwidth, packed, absl::MakeSpan(unpacked));
    ::benchmark::DoNotOptimize(unpacked);
  }

  state.SetItemsProcessed(state.iterations() * num_elements);
}

BENCHMARK(BM_UnpackIntN)
    ->ArgPair(1, 10000)
    ->ArgPair(1, 10000000)
    ->ArgPair(1, 100000000)
    ->ArgPair(1, 500000000)
    ->ArgPair(1, 1000000000)
    ->ArgPair(2, 10000)
    ->ArgPair(2, 10000000)
    ->ArgPair(2, 100000000)
    ->ArgPair(2, 500000000)
    ->ArgPair(2, 1000000000)
    ->ArgPair(4, 37)
    ->ArgPair(4, 64)
    ->ArgPair(4, 100)
    ->ArgPair(4, 128)
    ->ArgPair(4, 131)
    ->ArgPair(4, 256)
    ->ArgPair(4, 1024)
    ->ArgPair(4, 10000)
    ->ArgPair(4, 10000000)
    ->ArgPair(4, 100000000)
    ->ArgPair(4, 500000000)
    ->ArgPair(4, 1000000000);

}  // namespace
}  // namespace xla
