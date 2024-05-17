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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xla/maybe_owning.h"
#include "xla/test.h"
#include "xla/types.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/ml_dtypes.h"

namespace xla {
namespace {

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

  std::set<std::string_view> string_set = {std::string_view("a"),
                                           std::string_view("b")};
  EXPECT_EQ(VectorString(string_set), "(a, b)");

  EXPECT_EQ(VectorString({}), "()");
  EXPECT_EQ(VectorString({1, 57, 2}), "(1, 57, 2)");
}

TEST(UtilTest, LogLines) {
  // Just make sure this code runs (not verifying the output).
  LogLines(tsl::INFO, "hello\n\nworld", __FILE__, __LINE__);
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

namespace {
template <typename T>
void TotalOrderHelper(T x, T y) {
  auto x_sm = ToSignMagnitude(x);
  bool x_sign = static_cast<bool>(Eigen::numext::signbit(x));
  bool y_sign = static_cast<bool>(Eigen::numext::signbit(y));
  auto y_sm = ToSignMagnitude(y);
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

void PackInt4(absl::Span<const char> input, absl::Span<char> output) {
  CHECK_EQ(output.size(), CeilOfRatio(input.size(), size_t{2}));
  for (size_t i = 0; i < input.size(); ++i) {
    // Mask out the high-order 4 bits in case they have extraneous data.
    char val = input[i] & 0xf;
    if (i % 2 == 0) {
      output[i / 2] = val << 4;
    } else {
      output[i / 2] |= val;
    }
  }
}

TEST(UtilTest, PackInt4) {
  std::vector<char> input(7);
  std::iota(input.begin(), input.end(), 0);

  std::vector<char> output_ref(CeilOfRatio<int64_t>(input.size(), 2));
  PackInt4(input, absl::MakeSpan(output_ref));

  std::vector<char> output_dut(CeilOfRatio<int64_t>(input.size(), 2));
  PackIntN(4, input, absl::MakeSpan(output_dut));
  for (size_t i = 0; i < output_dut.size(); ++i) {
    EXPECT_EQ(output_ref[i], output_dut[i]) << i;
  }

  std::vector<char> unpacked(input.size());
  UnpackIntN(4, output_ref, absl::MakeSpan(unpacked));
  for (size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(unpacked[i], input[i]) << i;
  }
}

TEST(UtilTest, MaybeOwningTestNull) {
  MaybeOwning<char> m(nullptr);
  EXPECT_EQ(m.get(), nullptr);
  EXPECT_EQ(m.get_mutable(), nullptr);
}

TEST(UtilTest, MaybeOwningTestOwning) {
  MaybeOwning<char> m(std::make_unique<char>());
  *m.get_mutable() = 'a';
  EXPECT_EQ(*m, 'a');
}

TEST(UtilTest, MaybeOwningTestShared) {
  auto owner = std::make_unique<char>();
  *owner = 'x';
  MaybeOwning<char> c1(owner.get());
  MaybeOwning<char> c2(owner.get());

  EXPECT_EQ(*c1, 'x');
  EXPECT_EQ(*c2, 'x');
  EXPECT_EQ(c1.get(), c2.get());
}

}  // namespace
}  // namespace xla
