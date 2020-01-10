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

#include "tensorflow/core/lib/strings/ordered_code.h"

#include <float.h>
#include <stddef.h>
#include <limits>
#include <vector>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace strings {
namespace {

string RandomString(random::SimplePhilox* rnd, size_t len) {
  string x;
  for (size_t i = 0; i < len; i++) {
    x += rnd->Uniform(256);
  }
  return x;
}

// ---------------------------------------------------------------------
// Utility template functions (they help templatize the tests below)

// Read/WriteIncreasing are defined for string, uint64, int64 below.
template <typename T>
void OCWriteIncreasing(string* dest, const T& val);
template <typename T>
bool OCReadIncreasing(StringPiece* src, T* result);

// Read/WriteIncreasing<string>
template <>
void OCWriteIncreasing<string>(string* dest, const string& val) {
  OrderedCode::WriteString(dest, val);
}
template <>
bool OCReadIncreasing<string>(StringPiece* src, string* result) {
  return OrderedCode::ReadString(src, result);
}

// Read/WriteIncreasing<uint64>
template <>
void OCWriteIncreasing<uint64>(string* dest, const uint64& val) {
  OrderedCode::WriteNumIncreasing(dest, val);
}
template <>
bool OCReadIncreasing<uint64>(StringPiece* src, uint64* result) {
  return OrderedCode::ReadNumIncreasing(src, result);
}

// Read/WriteIncreasing<int64>
template <>
void OCWriteIncreasing<int64>(string* dest, const int64& val) {
  OrderedCode::WriteSignedNumIncreasing(dest, val);
}
template <>
bool OCReadIncreasing<int64>(StringPiece* src, int64* result) {
  return OrderedCode::ReadSignedNumIncreasing(src, result);
}

template <typename T>
string OCWrite(T val) {
  string result;
  OCWriteIncreasing<T>(&result, val);
  return result;
}

template <typename T>
void OCWriteToString(string* result, T val) {
  OCWriteIncreasing<T>(result, val);
}

template <typename T>
bool OCRead(StringPiece* s, T* val) {
  return OCReadIncreasing<T>(s, val);
}

// ---------------------------------------------------------------------
// Numbers

template <typename T>
T TestRead(const string& a) {
  // gracefully reject any proper prefix of an encoding
  for (int i = 0; i < a.size() - 1; ++i) {
    StringPiece s(a.data(), i);
    CHECK(!OCRead<T>(&s, nullptr));
    CHECK_EQ(s, a.substr(0, i));
  }

  StringPiece s(a);
  T v;
  CHECK(OCRead<T>(&s, &v));
  CHECK(s.empty());
  return v;
}

template <typename T>
void TestWriteRead(T expected) {
  EXPECT_EQ(expected, TestRead<T>(OCWrite<T>(expected)));
}

// Verifies that the second Write* call appends a non-empty string to its
// output.
template <typename T, typename U>
void TestWriteAppends(T first, U second) {
  string encoded;
  OCWriteToString<T>(&encoded, first);
  string encoded_first_only = encoded;
  OCWriteToString<U>(&encoded, second);
  EXPECT_NE(encoded, encoded_first_only);
  EXPECT_TRUE(absl::StartsWith(encoded, encoded_first_only));
}

template <typename T>
void TestNumbers(T multiplier) {
  // first test powers of 2 (and nearby numbers)
  for (T x = std::numeric_limits<T>().max(); x != 0; x /= 2) {
    TestWriteRead(multiplier * (x - 1));
    TestWriteRead(multiplier * x);
    if (x != std::numeric_limits<T>::max()) {
      TestWriteRead(multiplier * (x + 1));
    } else if (multiplier < 0 && multiplier == -1) {
      TestWriteRead(-x - 1);
    }
  }

  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  for (int bits = 1; bits <= std::numeric_limits<T>().digits; ++bits) {
    // test random non-negative numbers with given number of significant bits
    const uint64 mask = (~0ULL) >> (64 - bits);
    for (int i = 0; i < 1000; i++) {
      T x = rnd.Rand64() & mask;
      TestWriteRead(multiplier * x);
      T y = rnd.Rand64() & mask;
      TestWriteAppends(multiplier * x, multiplier * y);
    }
  }
}

// Return true iff 'a' is "before" 'b'
bool CompareStrings(const string& a, const string& b) { return (a < b); }

template <typename T>
void TestNumberOrdering() {
  // first the negative numbers (if T is signed, otherwise no-op)
  string laststr = OCWrite<T>(std::numeric_limits<T>().min());
  for (T num = std::numeric_limits<T>().min() / 2; num != 0; num /= 2) {
    string strminus1 = OCWrite<T>(num - 1);
    string str = OCWrite<T>(num);
    string strplus1 = OCWrite<T>(num + 1);

    CHECK(CompareStrings(strminus1, str));
    CHECK(CompareStrings(str, strplus1));

    // Compare 'str' with 'laststr'.  When we approach 0, 'laststr' is
    // not necessarily before 'strminus1'.
    CHECK(CompareStrings(laststr, str));
    laststr = str;
  }

  // then the positive numbers
  laststr = OCWrite<T>(0);
  T num = 1;
  while (num < std::numeric_limits<T>().max() / 2) {
    num *= 2;
    string strminus1 = OCWrite<T>(num - 1);
    string str = OCWrite<T>(num);
    string strplus1 = OCWrite<T>(num + 1);

    CHECK(CompareStrings(strminus1, str));
    CHECK(CompareStrings(str, strplus1));

    // Compare 'str' with 'laststr'.
    CHECK(CompareStrings(laststr, str));
    laststr = str;
  }
}

// Helper routine for testing TEST_SkipToNextSpecialByte
size_t FindSpecial(const string& x) {
  const char* p = x.data();
  const char* limit = p + x.size();
  const char* result = OrderedCode::TEST_SkipToNextSpecialByte(p, limit);
  return result - p;
}

// Helper function template to create strings from string literals (excluding
// the terminal zero byte of the underlying character array).
template <size_t N>
string ByteSequence(const char (&arr)[N]) {
  return string(arr, N - 1);
}

TEST(OrderedCode, SkipToNextSpecialByte) {
  for (size_t len = 0; len < 256; len++) {
    random::PhiloxRandom philox(301, 17);
    random::SimplePhilox rnd(&philox);
    string x;
    while (x.size() < len) {
      char c = 1 + rnd.Uniform(254);
      ASSERT_NE(c, 0);
      ASSERT_NE(c, 255);
      x += c;  // No 0 bytes, no 255 bytes
    }
    EXPECT_EQ(FindSpecial(x), x.size());
    for (size_t special_pos = 0; special_pos < len; special_pos++) {
      for (size_t special_test = 0; special_test < 2; special_test++) {
        const char special_byte = (special_test == 0) ? 0 : 255;
        string y = x;
        y[special_pos] = special_byte;
        EXPECT_EQ(FindSpecial(y), special_pos);
        if (special_pos < 16) {
          // Add some special bytes after the one at special_pos to make sure
          // we still return the earliest special byte in the string
          for (size_t rest = special_pos + 1; rest < len; rest++) {
            if (rnd.OneIn(3)) {
              y[rest] = rnd.OneIn(2) ? 0 : 255;
              EXPECT_EQ(FindSpecial(y), special_pos);
            }
          }
        }
      }
    }
  }
}

TEST(OrderedCode, ExhaustiveFindSpecial) {
  char buf[16];
  char* limit = buf + sizeof(buf);
  int count = 0;
  for (int start_offset = 0; start_offset <= 5; start_offset += 5) {
    // We test exhaustively with all combinations of 3 bytes starting
    // at offset 0 and offset 5 (so as to test with the bytes at both
    // ends of a 64-bit word).
    for (size_t i = 0; i < sizeof(buf); i++) {
      buf[i] = 'a';  // Not a special byte
    }
    for (int b0 = 0; b0 < 256; b0++) {
      for (int b1 = 0; b1 < 256; b1++) {
        for (int b2 = 0; b2 < 256; b2++) {
          buf[start_offset + 0] = b0;
          buf[start_offset + 1] = b1;
          buf[start_offset + 2] = b2;
          char* expected;
          if (b0 == 0 || b0 == 255) {
            expected = &buf[start_offset];
          } else if (b1 == 0 || b1 == 255) {
            expected = &buf[start_offset + 1];
          } else if (b2 == 0 || b2 == 255) {
            expected = &buf[start_offset + 2];
          } else {
            expected = limit;
          }
          count++;
          EXPECT_EQ(expected,
                    OrderedCode::TEST_SkipToNextSpecialByte(buf, limit));
        }
      }
    }
  }
  EXPECT_EQ(count, 256 * 256 * 256 * 2);
}

TEST(Uint64, EncodeDecode) { TestNumbers<uint64>(1); }

TEST(Uint64, Ordering) { TestNumberOrdering<uint64>(); }

TEST(Int64, EncodeDecode) {
  TestNumbers<int64>(1);
  TestNumbers<int64>(-1);
}

TEST(Int64, Ordering) { TestNumberOrdering<int64>(); }

// Returns the bitwise complement of s.
inline string StrNot(const string& s) {
  string result;
  for (string::const_iterator it = s.begin(); it != s.end(); ++it)
    result.push_back(~*it);
  return result;
}

template <typename T>
void TestInvalidEncoding(const string& s) {
  StringPiece p(s);
  EXPECT_FALSE(OCRead<T>(&p, nullptr));
  EXPECT_EQ(s, p);
}

TEST(OrderedCodeInvalidEncodingsTest, Overflow) {
  // 1U << 64, increasing and decreasing
  const string k2xx64U = "\x09\x01" + string(8, 0);
  TestInvalidEncoding<uint64>(k2xx64U);

  // 1 << 63 and ~(1 << 63), increasing and decreasing
  const string k2xx63 = "\xff\xc0\x80" + string(7, 0);
  TestInvalidEncoding<int64>(k2xx63);
  TestInvalidEncoding<int64>(StrNot(k2xx63));
}

TEST(OrderedCodeInvalidEncodingsDeathTest, NonCanonical) {
  // Test "ambiguous"/"non-canonical" encodings.
  // These are non-minimal (but otherwise "valid") encodings that
  // differ from the minimal encoding chosen by OrderedCode::WriteXXX
  // and thus should be avoided to not mess up the string ordering of
  // encodings.

  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);

  for (int n = 2; n <= 9; ++n) {
    // The zero in non_minimal[1] is "redundant".
    string non_minimal =
        string(1, n - 1) + string(1, 0) + RandomString(&rnd, n - 2);
    EXPECT_EQ(n, non_minimal.length());

    EXPECT_NE(OCWrite<uint64>(0), non_minimal);
#ifndef NDEBUG
    StringPiece s(non_minimal);
    EXPECT_DEATH(OrderedCode::ReadNumIncreasing(&s, nullptr),
                 "invalid encoding");
#else
    TestRead<uint64>(non_minimal);
#endif
  }

  for (int n = 2; n <= 10; ++n) {
    // Header with 1 sign bit and n-1 size bits.
    string header = string(n / 8, 0xff) + string(1, 0xff << (8 - (n % 8)));
    // There are more than 7 zero bits between header bits and "payload".
    string non_minimal = header +
                         string(1, rnd.Uniform(256) & ~*header.rbegin()) +
                         RandomString(&rnd, n - header.length() - 1);
    EXPECT_EQ(n, non_minimal.length());

    EXPECT_NE(OCWrite<int64>(0), non_minimal);
#ifndef NDEBUG
    StringPiece s(non_minimal);
    EXPECT_DEATH(OrderedCode::ReadSignedNumIncreasing(&s, nullptr),
                 "invalid encoding")
        << n;
#else
    TestRead<int64>(non_minimal);
#endif
  }
}

// Returns random number with specified number of bits,
// i.e., in the range [2^(bits-1),2^bits).
uint64 NextBits(random::SimplePhilox* rnd, int bits) {
  return (bits != 0)
             ? (rnd->Rand64() % (1LL << (bits - 1))) + (1LL << (bits - 1))
             : 0;
}

template <typename T>
void BM_WriteNum(int n, T multiplier) {
  constexpr int kValues = 64;
  T values[kValues];
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  // Use enough distinct values to confuse the branch predictor
  for (int i = 0; i < kValues; i++) {
    values[i] = NextBits(&rnd, n % 64) * multiplier;
  }
  string result;
  int index = 0;
  while (n-- > 0) {
    result.clear();
    OCWriteToString<T>(&result, values[index % kValues]);
    index++;
  }
}

template <typename T>
void BM_ReadNum(int n, T multiplier) {
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  // Use enough distinct values to confuse the branch predictor
  constexpr int kValues = 64;
  string values[kValues];
  for (int i = 0; i < kValues; i++) {
    T val = NextBits(&rnd, i % 64) * multiplier;
    values[i] = OCWrite<T>(val);
  }
  uint32 index = 0;
  while (n-- > 0) {
    T val;
    StringPiece s = values[index++ % kValues];
    OCRead<T>(&s, &val);
  }
}

#define BENCHMARK_NUM(name, T, multiplier)                      \
  void BM_Write##name(int n) { BM_WriteNum<T>(n, multiplier); } \
  BENCHMARK(BM_Write##name);                                    \
  void BM_Read##name(int n) { BM_ReadNum<T>(n, multiplier); }   \
  BENCHMARK(BM_Read##name)

BENCHMARK_NUM(NumIncreasing, uint64, 1);
BENCHMARK_NUM(SignedNum, int64, 1);
BENCHMARK_NUM(SignedNumNegative, int64, -1);

#undef BENCHMARK_NUM

// ---------------------------------------------------------------------
// Strings

TEST(String, EncodeDecode) {
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);

  for (int len = 0; len < 256; len++) {
    const string a = RandomString(&rnd, len);
    TestWriteRead(a);
    for (int len2 = 0; len2 < 64; len2++) {
      const string b = RandomString(&rnd, len2);

      TestWriteAppends(a, b);

      string out;
      OCWriteToString<string>(&out, a);
      OCWriteToString<string>(&out, b);

      string a2, b2, dummy;
      StringPiece s = out;
      StringPiece s2 = out;
      CHECK(OCRead<string>(&s, &a2));
      CHECK(OCRead<string>(&s2, nullptr));
      CHECK_EQ(s, s2);

      CHECK(OCRead<string>(&s, &b2));
      CHECK(OCRead<string>(&s2, nullptr));
      CHECK_EQ(s, s2);

      CHECK(!OCRead<string>(&s, &dummy));
      CHECK(!OCRead<string>(&s2, nullptr));
      CHECK_EQ(a, a2);
      CHECK_EQ(b, b2);
      CHECK(s.empty());
      CHECK(s2.empty());
    }
  }
}

// 'str' is a string literal that may contain '\0'.
#define STATIC_STR(str) StringPiece((str), sizeof(str) - 1)

string EncodeStringIncreasing(StringPiece value) {
  string encoded;
  OrderedCode::WriteString(&encoded, value);
  return encoded;
}

TEST(String, Increasing) {
  // Here are a series of strings in non-decreasing order, including
  // consecutive strings such that the second one is equal to, a proper
  // prefix of, or has the same length as the first one.  Most also contain
  // the special escaping characters '\x00' and '\xff'.
  ASSERT_EQ(EncodeStringIncreasing(STATIC_STR("")),
            EncodeStringIncreasing(STATIC_STR("")));

  ASSERT_LT(EncodeStringIncreasing(STATIC_STR("")),
            EncodeStringIncreasing(STATIC_STR("\x00")));

  ASSERT_EQ(EncodeStringIncreasing(STATIC_STR("\x00")),
            EncodeStringIncreasing(STATIC_STR("\x00")));

  ASSERT_LT(EncodeStringIncreasing(STATIC_STR("\x00")),
            EncodeStringIncreasing(STATIC_STR("\x01")));

  ASSERT_LT(EncodeStringIncreasing(STATIC_STR("\x01")),
            EncodeStringIncreasing(STATIC_STR("a")));

  ASSERT_EQ(EncodeStringIncreasing(STATIC_STR("a")),
            EncodeStringIncreasing(STATIC_STR("a")));

  ASSERT_LT(EncodeStringIncreasing(STATIC_STR("a")),
            EncodeStringIncreasing(STATIC_STR("aa")));

  ASSERT_LT(EncodeStringIncreasing(STATIC_STR("aa")),
            EncodeStringIncreasing(STATIC_STR("\xff")));

  ASSERT_LT(EncodeStringIncreasing(STATIC_STR("\xff")),
            EncodeStringIncreasing(STATIC_STR("\xff\x00")));

  ASSERT_LT(EncodeStringIncreasing(STATIC_STR("\xff\x00")),
            EncodeStringIncreasing(STATIC_STR("\xff\x01")));
}

TEST(EncodingIsExpected, String) {
  std::vector<std::pair<string, string>> data = {
      {"", string("\x00\x01", 2)},
      {"foo", string("foo\x00\x01", 5)},
      {"hello", string("hello\x00\x01", 7)},
      {string("\x00\x01\xff", 3), string("\x00\xff\x01\xff\x00\x00\x01", 7)},
  };
  for (const auto& t : data) {
    string result;
    OrderedCode::WriteString(&result, t.first);
    EXPECT_EQ(t.second, result);

    StringPiece in = result;
    string decoded;
    EXPECT_TRUE(OrderedCode::ReadString(&in, &decoded));
    EXPECT_EQ(t.first, decoded);
    EXPECT_EQ("", in);
  }
}

TEST(EncodingIsExpected, Unsigned) {
  std::vector<std::pair<uint64, string>> data = {
      {0x0ull, ByteSequence("\000")},
      {0x1ull, ByteSequence("\001\001")},
      {0x2ull, ByteSequence("\001\002")},
      {0x1ull, ByteSequence("\001\001")},
      {0x2ull, ByteSequence("\001\002")},
      {0x3ull, ByteSequence("\001\003")},
      {0x3ull, ByteSequence("\001\003")},
      {0x4ull, ByteSequence("\001\004")},
      {0x5ull, ByteSequence("\001\005")},
      {0x7ull, ByteSequence("\001\007")},
      {0x8ull, ByteSequence("\001\010")},
      {0x9ull, ByteSequence("\001\t")},
      {0xfull, ByteSequence("\001\017")},
      {0x10ull, ByteSequence("\001\020")},
      {0x11ull, ByteSequence("\001\021")},
      {0x1full, ByteSequence("\001\037")},
      {0x20ull, ByteSequence("\001 ")},
      {0x21ull, ByteSequence("\001!")},
      {0x3full, ByteSequence("\001?")},
      {0x40ull, ByteSequence("\001@")},
      {0x41ull, ByteSequence("\001A")},
      {0x7full, ByteSequence("\001\177")},
      {0x80ull, ByteSequence("\001\200")},
      {0x81ull, ByteSequence("\001\201")},
      {0xffull, ByteSequence("\001\377")},
      {0x100ull, ByteSequence("\002\001\000")},
      {0x101ull, ByteSequence("\002\001\001")},
      {0x1ffull, ByteSequence("\002\001\377")},
      {0x200ull, ByteSequence("\002\002\000")},
      {0x201ull, ByteSequence("\002\002\001")},
      {0x3ffull, ByteSequence("\002\003\377")},
      {0x400ull, ByteSequence("\002\004\000")},
      {0x401ull, ByteSequence("\002\004\001")},
      {0x7ffull, ByteSequence("\002\007\377")},
      {0x800ull, ByteSequence("\002\010\000")},
      {0x801ull, ByteSequence("\002\010\001")},
      {0xfffull, ByteSequence("\002\017\377")},
      {0x1000ull, ByteSequence("\002\020\000")},
      {0x1001ull, ByteSequence("\002\020\001")},
      {0x1fffull, ByteSequence("\002\037\377")},
      {0x2000ull, ByteSequence("\002 \000")},
      {0x2001ull, ByteSequence("\002 \001")},
      {0x3fffull, ByteSequence("\002?\377")},
      {0x4000ull, ByteSequence("\002@\000")},
      {0x4001ull, ByteSequence("\002@\001")},
      {0x7fffull, ByteSequence("\002\177\377")},
      {0x8000ull, ByteSequence("\002\200\000")},
      {0x8001ull, ByteSequence("\002\200\001")},
      {0xffffull, ByteSequence("\002\377\377")},
      {0x10000ull, ByteSequence("\003\001\000\000")},
      {0x10001ull, ByteSequence("\003\001\000\001")},
      {0x1ffffull, ByteSequence("\003\001\377\377")},
      {0x20000ull, ByteSequence("\003\002\000\000")},
      {0x20001ull, ByteSequence("\003\002\000\001")},
      {0x3ffffull, ByteSequence("\003\003\377\377")},
      {0x40000ull, ByteSequence("\003\004\000\000")},
      {0x40001ull, ByteSequence("\003\004\000\001")},
      {0x7ffffull, ByteSequence("\003\007\377\377")},
      {0x80000ull, ByteSequence("\003\010\000\000")},
      {0x80001ull, ByteSequence("\003\010\000\001")},
      {0xfffffull, ByteSequence("\003\017\377\377")},
      {0x100000ull, ByteSequence("\003\020\000\000")},
      {0x100001ull, ByteSequence("\003\020\000\001")},
      {0x1fffffull, ByteSequence("\003\037\377\377")},
      {0x200000ull, ByteSequence("\003 \000\000")},
      {0x200001ull, ByteSequence("\003 \000\001")},
      {0x3fffffull, ByteSequence("\003?\377\377")},
      {0x400000ull, ByteSequence("\003@\000\000")},
      {0x400001ull, ByteSequence("\003@\000\001")},
      {0x7fffffull, ByteSequence("\003\177\377\377")},
      {0x800000ull, ByteSequence("\003\200\000\000")},
      {0x800001ull, ByteSequence("\003\200\000\001")},
      {0xffffffull, ByteSequence("\003\377\377\377")},
      {0x1000000ull, ByteSequence("\004\001\000\000\000")},
      {0x1000001ull, ByteSequence("\004\001\000\000\001")},
      {0x1ffffffull, ByteSequence("\004\001\377\377\377")},
      {0x2000000ull, ByteSequence("\004\002\000\000\000")},
      {0x2000001ull, ByteSequence("\004\002\000\000\001")},
      {0x3ffffffull, ByteSequence("\004\003\377\377\377")},
      {0x4000000ull, ByteSequence("\004\004\000\000\000")},
      {0x4000001ull, ByteSequence("\004\004\000\000\001")},
      {0x7ffffffull, ByteSequence("\004\007\377\377\377")},
      {0x8000000ull, ByteSequence("\004\010\000\000\000")},
      {0x8000001ull, ByteSequence("\004\010\000\000\001")},
      {0xfffffffull, ByteSequence("\004\017\377\377\377")},
      {0x10000000ull, ByteSequence("\004\020\000\000\000")},
      {0x10000001ull, ByteSequence("\004\020\000\000\001")},
      {0x1fffffffull, ByteSequence("\004\037\377\377\377")},
      {0x20000000ull, ByteSequence("\004 \000\000\000")},
      {0x20000001ull, ByteSequence("\004 \000\000\001")},
      {0x3fffffffull, ByteSequence("\004?\377\377\377")},
      {0x40000000ull, ByteSequence("\004@\000\000\000")},
      {0x40000001ull, ByteSequence("\004@\000\000\001")},
      {0x7fffffffull, ByteSequence("\004\177\377\377\377")},
      {0x80000000ull, ByteSequence("\004\200\000\000\000")},
      {0x80000001ull, ByteSequence("\004\200\000\000\001")},
      {0xffffffffull, ByteSequence("\004\377\377\377\377")},
      {0x100000000ull, ByteSequence("\005\001\000\000\000\000")},
      {0x100000001ull, ByteSequence("\005\001\000\000\000\001")},
      {0x1ffffffffull, ByteSequence("\005\001\377\377\377\377")},
      {0x200000000ull, ByteSequence("\005\002\000\000\000\000")},
      {0x200000001ull, ByteSequence("\005\002\000\000\000\001")},
      {0x3ffffffffull, ByteSequence("\005\003\377\377\377\377")},
      {0x400000000ull, ByteSequence("\005\004\000\000\000\000")},
      {0x400000001ull, ByteSequence("\005\004\000\000\000\001")},
      {0x7ffffffffull, ByteSequence("\005\007\377\377\377\377")},
      {0x800000000ull, ByteSequence("\005\010\000\000\000\000")},
      {0x800000001ull, ByteSequence("\005\010\000\000\000\001")},
      {0xfffffffffull, ByteSequence("\005\017\377\377\377\377")},
      {0x1000000000ull, ByteSequence("\005\020\000\000\000\000")},
      {0x1000000001ull, ByteSequence("\005\020\000\000\000\001")},
      {0x1fffffffffull, ByteSequence("\005\037\377\377\377\377")},
      {0x2000000000ull, ByteSequence("\005 \000\000\000\000")},
      {0x2000000001ull, ByteSequence("\005 \000\000\000\001")},
      {0x3fffffffffull, ByteSequence("\005?\377\377\377\377")},
      {0x4000000000ull, ByteSequence("\005@\000\000\000\000")},
      {0x4000000001ull, ByteSequence("\005@\000\000\000\001")},
      {0x7fffffffffull, ByteSequence("\005\177\377\377\377\377")},
      {0x8000000000ull, ByteSequence("\005\200\000\000\000\000")},
      {0x8000000001ull, ByteSequence("\005\200\000\000\000\001")},
      {0xffffffffffull, ByteSequence("\005\377\377\377\377\377")},
      {0x10000000000ull, ByteSequence("\006\001\000\000\000\000\000")},
      {0x10000000001ull, ByteSequence("\006\001\000\000\000\000\001")},
      {0x1ffffffffffull, ByteSequence("\006\001\377\377\377\377\377")},
      {0x20000000000ull, ByteSequence("\006\002\000\000\000\000\000")},
      {0x20000000001ull, ByteSequence("\006\002\000\000\000\000\001")},
      {0x3ffffffffffull, ByteSequence("\006\003\377\377\377\377\377")},
      {0x40000000000ull, ByteSequence("\006\004\000\000\000\000\000")},
      {0x40000000001ull, ByteSequence("\006\004\000\000\000\000\001")},
      {0x7ffffffffffull, ByteSequence("\006\007\377\377\377\377\377")},
      {0x80000000000ull, ByteSequence("\006\010\000\000\000\000\000")},
      {0x80000000001ull, ByteSequence("\006\010\000\000\000\000\001")},
      {0xfffffffffffull, ByteSequence("\006\017\377\377\377\377\377")},
      {0x100000000000ull, ByteSequence("\006\020\000\000\000\000\000")},
      {0x100000000001ull, ByteSequence("\006\020\000\000\000\000\001")},
      {0x1fffffffffffull, ByteSequence("\006\037\377\377\377\377\377")},
      {0x200000000000ull, ByteSequence("\006 \000\000\000\000\000")},
      {0x200000000001ull, ByteSequence("\006 \000\000\000\000\001")},
      {0x3fffffffffffull, ByteSequence("\006?\377\377\377\377\377")},
      {0x400000000000ull, ByteSequence("\006@\000\000\000\000\000")},
      {0x400000000001ull, ByteSequence("\006@\000\000\000\000\001")},
      {0x7fffffffffffull, ByteSequence("\006\177\377\377\377\377\377")},
      {0x800000000000ull, ByteSequence("\006\200\000\000\000\000\000")},
      {0x800000000001ull, ByteSequence("\006\200\000\000\000\000\001")},
      {0xffffffffffffull, ByteSequence("\006\377\377\377\377\377\377")},
      {0x1000000000000ull, ByteSequence("\007\001\000\000\000\000\000\000")},
      {0x1000000000001ull, ByteSequence("\007\001\000\000\000\000\000\001")},
      {0x1ffffffffffffull, ByteSequence("\007\001\377\377\377\377\377\377")},
      {0x2000000000000ull, ByteSequence("\007\002\000\000\000\000\000\000")},
      {0x2000000000001ull, ByteSequence("\007\002\000\000\000\000\000\001")},
      {0x3ffffffffffffull, ByteSequence("\007\003\377\377\377\377\377\377")},
      {0x4000000000000ull, ByteSequence("\007\004\000\000\000\000\000\000")},
      {0x4000000000001ull, ByteSequence("\007\004\000\000\000\000\000\001")},
      {0x7ffffffffffffull, ByteSequence("\007\007\377\377\377\377\377\377")},
      {0x8000000000000ull, ByteSequence("\007\010\000\000\000\000\000\000")},
      {0x8000000000001ull, ByteSequence("\007\010\000\000\000\000\000\001")},
      {0xfffffffffffffull, ByteSequence("\007\017\377\377\377\377\377\377")},
      {0x10000000000000ull, ByteSequence("\007\020\000\000\000\000\000\000")},
      {0x10000000000001ull, ByteSequence("\007\020\000\000\000\000\000\001")},
      {0x1fffffffffffffull, ByteSequence("\007\037\377\377\377\377\377\377")},
      {0x20000000000000ull, ByteSequence("\007 \000\000\000\000\000\000")},
      {0x20000000000001ull, ByteSequence("\007 \000\000\000\000\000\001")},
      {0x3fffffffffffffull, ByteSequence("\007?\377\377\377\377\377\377")},
      {0x40000000000000ull, ByteSequence("\007@\000\000\000\000\000\000")},
      {0x40000000000001ull, ByteSequence("\007@\000\000\000\000\000\001")},
      {0x7fffffffffffffull, ByteSequence("\007\177\377\377\377\377\377\377")},
      {0x80000000000000ull, ByteSequence("\007\200\000\000\000\000\000\000")},
      {0x80000000000001ull, ByteSequence("\007\200\000\000\000\000\000\001")},
      {0xffffffffffffffull, ByteSequence("\007\377\377\377\377\377\377\377")},
      {0x100000000000000ull,
       ByteSequence("\010\001\000\000\000\000\000\000\000")},
      {0x100000000000001ull,
       ByteSequence("\010\001\000\000\000\000\000\000\001")},
      {0x1ffffffffffffffull,
       ByteSequence("\010\001\377\377\377\377\377\377\377")},
      {0x200000000000000ull,
       ByteSequence("\010\002\000\000\000\000\000\000\000")},
      {0x200000000000001ull,
       ByteSequence("\010\002\000\000\000\000\000\000\001")},
      {0x3ffffffffffffffull,
       ByteSequence("\010\003\377\377\377\377\377\377\377")},
      {0x400000000000000ull,
       ByteSequence("\010\004\000\000\000\000\000\000\000")},
      {0x400000000000001ull,
       ByteSequence("\010\004\000\000\000\000\000\000\001")},
      {0x7ffffffffffffffull,
       ByteSequence("\010\007\377\377\377\377\377\377\377")},
      {0x800000000000000ull,
       ByteSequence("\010\010\000\000\000\000\000\000\000")},
      {0x800000000000001ull,
       ByteSequence("\010\010\000\000\000\000\000\000\001")},
      {0xfffffffffffffffull,
       ByteSequence("\010\017\377\377\377\377\377\377\377")},
      {0x1000000000000000ull,
       ByteSequence("\010\020\000\000\000\000\000\000\000")},
      {0x1000000000000001ull,
       ByteSequence("\010\020\000\000\000\000\000\000\001")},
      {0x1fffffffffffffffull,
       ByteSequence("\010\037\377\377\377\377\377\377\377")},
      {0x2000000000000000ull,
       ByteSequence("\010 \000\000\000\000\000\000\000")},
      {0x2000000000000001ull,
       ByteSequence("\010 \000\000\000\000\000\000\001")},
      {0x3fffffffffffffffull,
       ByteSequence("\010?\377\377\377\377\377\377\377")},
      {0x4000000000000000ull,
       ByteSequence("\010@\000\000\000\000\000\000\000")},
      {0x4000000000000001ull,
       ByteSequence("\010@\000\000\000\000\000\000\001")},
      {0x7fffffffffffffffull,
       ByteSequence("\010\177\377\377\377\377\377\377\377")},
      {0x8000000000000000ull,
       ByteSequence("\010\200\000\000\000\000\000\000\000")},
      {0x8000000000000001ull,
       ByteSequence("\010\200\000\000\000\000\000\000\001")},
  };
  for (const auto& t : data) {
    uint64 num = t.first;
    string result;
    OrderedCode::WriteNumIncreasing(&result, num);
    EXPECT_EQ(t.second, result) << std::hex << num;

    StringPiece in = result;
    uint64 decoded;
    EXPECT_TRUE(OrderedCode::ReadNumIncreasing(&in, &decoded));
    EXPECT_EQ(num, decoded);
    EXPECT_EQ("", in);
  }
}

TEST(EncodingIsExpected, Signed) {
  std::vector<std::pair<int64, string>> data = {
      {0ll, ByteSequence("\200")},
      {1ll, ByteSequence("\201")},
      {2ll, ByteSequence("\202")},
      {1ll, ByteSequence("\201")},
      {2ll, ByteSequence("\202")},
      {3ll, ByteSequence("\203")},
      {3ll, ByteSequence("\203")},
      {4ll, ByteSequence("\204")},
      {5ll, ByteSequence("\205")},
      {7ll, ByteSequence("\207")},
      {8ll, ByteSequence("\210")},
      {9ll, ByteSequence("\211")},
      {15ll, ByteSequence("\217")},
      {16ll, ByteSequence("\220")},
      {17ll, ByteSequence("\221")},
      {31ll, ByteSequence("\237")},
      {32ll, ByteSequence("\240")},
      {33ll, ByteSequence("\241")},
      {63ll, ByteSequence("\277")},
      {64ll, ByteSequence("\300@")},
      {65ll, ByteSequence("\300A")},
      {127ll, ByteSequence("\300\177")},
      {128ll, ByteSequence("\300\200")},
      {129ll, ByteSequence("\300\201")},
      {255ll, ByteSequence("\300\377")},
      {256ll, ByteSequence("\301\000")},
      {257ll, ByteSequence("\301\001")},
      {511ll, ByteSequence("\301\377")},
      {512ll, ByteSequence("\302\000")},
      {513ll, ByteSequence("\302\001")},
      {1023ll, ByteSequence("\303\377")},
      {1024ll, ByteSequence("\304\000")},
      {1025ll, ByteSequence("\304\001")},
      {2047ll, ByteSequence("\307\377")},
      {2048ll, ByteSequence("\310\000")},
      {2049ll, ByteSequence("\310\001")},
      {4095ll, ByteSequence("\317\377")},
      {4096ll, ByteSequence("\320\000")},
      {4097ll, ByteSequence("\320\001")},
      {8191ll, ByteSequence("\337\377")},
      {8192ll, ByteSequence("\340 \000")},
      {8193ll, ByteSequence("\340 \001")},
      {16383ll, ByteSequence("\340?\377")},
      {16384ll, ByteSequence("\340@\000")},
      {16385ll, ByteSequence("\340@\001")},
      {32767ll, ByteSequence("\340\177\377")},
      {32768ll, ByteSequence("\340\200\000")},
      {32769ll, ByteSequence("\340\200\001")},
      {65535ll, ByteSequence("\340\377\377")},
      {65536ll, ByteSequence("\341\000\000")},
      {65537ll, ByteSequence("\341\000\001")},
      {131071ll, ByteSequence("\341\377\377")},
      {131072ll, ByteSequence("\342\000\000")},
      {131073ll, ByteSequence("\342\000\001")},
      {262143ll, ByteSequence("\343\377\377")},
      {262144ll, ByteSequence("\344\000\000")},
      {262145ll, ByteSequence("\344\000\001")},
      {524287ll, ByteSequence("\347\377\377")},
      {524288ll, ByteSequence("\350\000\000")},
      {524289ll, ByteSequence("\350\000\001")},
      {1048575ll, ByteSequence("\357\377\377")},
      {1048576ll, ByteSequence("\360\020\000\000")},
      {1048577ll, ByteSequence("\360\020\000\001")},
      {2097151ll, ByteSequence("\360\037\377\377")},
      {2097152ll, ByteSequence("\360 \000\000")},
      {2097153ll, ByteSequence("\360 \000\001")},
      {4194303ll, ByteSequence("\360?\377\377")},
      {4194304ll, ByteSequence("\360@\000\000")},
      {4194305ll, ByteSequence("\360@\000\001")},
      {8388607ll, ByteSequence("\360\177\377\377")},
      {8388608ll, ByteSequence("\360\200\000\000")},
      {8388609ll, ByteSequence("\360\200\000\001")},
      {16777215ll, ByteSequence("\360\377\377\377")},
      {16777216ll, ByteSequence("\361\000\000\000")},
      {16777217ll, ByteSequence("\361\000\000\001")},
      {33554431ll, ByteSequence("\361\377\377\377")},
      {33554432ll, ByteSequence("\362\000\000\000")},
      {33554433ll, ByteSequence("\362\000\000\001")},
      {67108863ll, ByteSequence("\363\377\377\377")},
      {67108864ll, ByteSequence("\364\000\000\000")},
      {67108865ll, ByteSequence("\364\000\000\001")},
      {134217727ll, ByteSequence("\367\377\377\377")},
      {134217728ll, ByteSequence("\370\010\000\000\000")},
      {134217729ll, ByteSequence("\370\010\000\000\001")},
      {268435455ll, ByteSequence("\370\017\377\377\377")},
      {268435456ll, ByteSequence("\370\020\000\000\000")},
      {268435457ll, ByteSequence("\370\020\000\000\001")},
      {536870911ll, ByteSequence("\370\037\377\377\377")},
      {536870912ll, ByteSequence("\370 \000\000\000")},
      {536870913ll, ByteSequence("\370 \000\000\001")},
      {1073741823ll, ByteSequence("\370?\377\377\377")},
      {1073741824ll, ByteSequence("\370@\000\000\000")},
      {1073741825ll, ByteSequence("\370@\000\000\001")},
      {2147483647ll, ByteSequence("\370\177\377\377\377")},
      {2147483648ll, ByteSequence("\370\200\000\000\000")},
      {2147483649ll, ByteSequence("\370\200\000\000\001")},
      {4294967295ll, ByteSequence("\370\377\377\377\377")},
      {4294967296ll, ByteSequence("\371\000\000\000\000")},
      {4294967297ll, ByteSequence("\371\000\000\000\001")},
      {8589934591ll, ByteSequence("\371\377\377\377\377")},
      {8589934592ll, ByteSequence("\372\000\000\000\000")},
      {8589934593ll, ByteSequence("\372\000\000\000\001")},
      {17179869183ll, ByteSequence("\373\377\377\377\377")},
      {17179869184ll, ByteSequence("\374\004\000\000\000\000")},
      {17179869185ll, ByteSequence("\374\004\000\000\000\001")},
      {34359738367ll, ByteSequence("\374\007\377\377\377\377")},
      {34359738368ll, ByteSequence("\374\010\000\000\000\000")},
      {34359738369ll, ByteSequence("\374\010\000\000\000\001")},
      {68719476735ll, ByteSequence("\374\017\377\377\377\377")},
      {68719476736ll, ByteSequence("\374\020\000\000\000\000")},
      {68719476737ll, ByteSequence("\374\020\000\000\000\001")},
      {137438953471ll, ByteSequence("\374\037\377\377\377\377")},
      {137438953472ll, ByteSequence("\374 \000\000\000\000")},
      {137438953473ll, ByteSequence("\374 \000\000\000\001")},
      {274877906943ll, ByteSequence("\374?\377\377\377\377")},
      {274877906944ll, ByteSequence("\374@\000\000\000\000")},
      {274877906945ll, ByteSequence("\374@\000\000\000\001")},
      {549755813887ll, ByteSequence("\374\177\377\377\377\377")},
      {549755813888ll, ByteSequence("\374\200\000\000\000\000")},
      {549755813889ll, ByteSequence("\374\200\000\000\000\001")},
      {1099511627775ll, ByteSequence("\374\377\377\377\377\377")},
      {1099511627776ll, ByteSequence("\375\000\000\000\000\000")},
      {1099511627777ll, ByteSequence("\375\000\000\000\000\001")},
      {2199023255551ll, ByteSequence("\375\377\377\377\377\377")},
      {2199023255552ll, ByteSequence("\376\002\000\000\000\000\000")},
      {2199023255553ll, ByteSequence("\376\002\000\000\000\000\001")},
      {4398046511103ll, ByteSequence("\376\003\377\377\377\377\377")},
      {4398046511104ll, ByteSequence("\376\004\000\000\000\000\000")},
      {4398046511105ll, ByteSequence("\376\004\000\000\000\000\001")},
      {8796093022207ll, ByteSequence("\376\007\377\377\377\377\377")},
      {8796093022208ll, ByteSequence("\376\010\000\000\000\000\000")},
      {8796093022209ll, ByteSequence("\376\010\000\000\000\000\001")},
      {17592186044415ll, ByteSequence("\376\017\377\377\377\377\377")},
      {17592186044416ll, ByteSequence("\376\020\000\000\000\000\000")},
      {17592186044417ll, ByteSequence("\376\020\000\000\000\000\001")},
      {35184372088831ll, ByteSequence("\376\037\377\377\377\377\377")},
      {35184372088832ll, ByteSequence("\376 \000\000\000\000\000")},
      {35184372088833ll, ByteSequence("\376 \000\000\000\000\001")},
      {70368744177663ll, ByteSequence("\376?\377\377\377\377\377")},
      {70368744177664ll, ByteSequence("\376@\000\000\000\000\000")},
      {70368744177665ll, ByteSequence("\376@\000\000\000\000\001")},
      {140737488355327ll, ByteSequence("\376\177\377\377\377\377\377")},
      {140737488355328ll, ByteSequence("\376\200\000\000\000\000\000")},
      {140737488355329ll, ByteSequence("\376\200\000\000\000\000\001")},
      {281474976710655ll, ByteSequence("\376\377\377\377\377\377\377")},
      {281474976710656ll, ByteSequence("\377\001\000\000\000\000\000\000")},
      {281474976710657ll, ByteSequence("\377\001\000\000\000\000\000\001")},
      {562949953421311ll, ByteSequence("\377\001\377\377\377\377\377\377")},
      {562949953421312ll, ByteSequence("\377\002\000\000\000\000\000\000")},
      {562949953421313ll, ByteSequence("\377\002\000\000\000\000\000\001")},
      {1125899906842623ll, ByteSequence("\377\003\377\377\377\377\377\377")},
      {1125899906842624ll, ByteSequence("\377\004\000\000\000\000\000\000")},
      {1125899906842625ll, ByteSequence("\377\004\000\000\000\000\000\001")},
      {2251799813685247ll, ByteSequence("\377\007\377\377\377\377\377\377")},
      {2251799813685248ll, ByteSequence("\377\010\000\000\000\000\000\000")},
      {2251799813685249ll, ByteSequence("\377\010\000\000\000\000\000\001")},
      {4503599627370495ll, ByteSequence("\377\017\377\377\377\377\377\377")},
      {4503599627370496ll, ByteSequence("\377\020\000\000\000\000\000\000")},
      {4503599627370497ll, ByteSequence("\377\020\000\000\000\000\000\001")},
      {9007199254740991ll, ByteSequence("\377\037\377\377\377\377\377\377")},
      {9007199254740992ll, ByteSequence("\377 \000\000\000\000\000\000")},
      {9007199254740993ll, ByteSequence("\377 \000\000\000\000\000\001")},
      {18014398509481983ll, ByteSequence("\377?\377\377\377\377\377\377")},
      {18014398509481984ll, ByteSequence("\377@\000\000\000\000\000\000")},
      {18014398509481985ll, ByteSequence("\377@\000\000\000\000\000\001")},
      {36028797018963967ll, ByteSequence("\377\177\377\377\377\377\377\377")},
      {36028797018963968ll,
       ByteSequence("\377\200\200\000\000\000\000\000\000")},
      {36028797018963969ll,
       ByteSequence("\377\200\200\000\000\000\000\000\001")},
      {72057594037927935ll,
       ByteSequence("\377\200\377\377\377\377\377\377\377")},
      {72057594037927936ll,
       ByteSequence("\377\201\000\000\000\000\000\000\000")},
      {72057594037927937ll,
       ByteSequence("\377\201\000\000\000\000\000\000\001")},
      {144115188075855871ll,
       ByteSequence("\377\201\377\377\377\377\377\377\377")},
      {144115188075855872ll,
       ByteSequence("\377\202\000\000\000\000\000\000\000")},
      {144115188075855873ll,
       ByteSequence("\377\202\000\000\000\000\000\000\001")},
      {288230376151711743ll,
       ByteSequence("\377\203\377\377\377\377\377\377\377")},
      {288230376151711744ll,
       ByteSequence("\377\204\000\000\000\000\000\000\000")},
      {288230376151711745ll,
       ByteSequence("\377\204\000\000\000\000\000\000\001")},
      {576460752303423487ll,
       ByteSequence("\377\207\377\377\377\377\377\377\377")},
      {576460752303423488ll,
       ByteSequence("\377\210\000\000\000\000\000\000\000")},
      {576460752303423489ll,
       ByteSequence("\377\210\000\000\000\000\000\000\001")},
      {1152921504606846975ll,
       ByteSequence("\377\217\377\377\377\377\377\377\377")},
      {1152921504606846976ll,
       ByteSequence("\377\220\000\000\000\000\000\000\000")},
      {1152921504606846977ll,
       ByteSequence("\377\220\000\000\000\000\000\000\001")},
      {2305843009213693951ll,
       ByteSequence("\377\237\377\377\377\377\377\377\377")},
      {2305843009213693952ll,
       ByteSequence("\377\240\000\000\000\000\000\000\000")},
      {2305843009213693953ll,
       ByteSequence("\377\240\000\000\000\000\000\000\001")},
      {4611686018427387903ll,
       ByteSequence("\377\277\377\377\377\377\377\377\377")},
      {4611686018427387904ll,
       ByteSequence("\377\300@\000\000\000\000\000\000\000")},
      {4611686018427387905ll,
       ByteSequence("\377\300@\000\000\000\000\000\000\001")},
      {9223372036854775807ll,
       ByteSequence("\377\300\177\377\377\377\377\377\377\377")},
      {-9223372036854775807ll,
       ByteSequence("\000?\200\000\000\000\000\000\000\001")},
      {0ll, ByteSequence("\200")},
      {-1ll, ByteSequence("\177")},
      {-2ll, ByteSequence("~")},
      {-1ll, ByteSequence("\177")},
      {-2ll, ByteSequence("~")},
      {-3ll, ByteSequence("}")},
      {-3ll, ByteSequence("}")},
      {-4ll, ByteSequence("|")},
      {-5ll, ByteSequence("{")},
      {-7ll, ByteSequence("y")},
      {-8ll, ByteSequence("x")},
      {-9ll, ByteSequence("w")},
      {-15ll, ByteSequence("q")},
      {-16ll, ByteSequence("p")},
      {-17ll, ByteSequence("o")},
      {-31ll, ByteSequence("a")},
      {-32ll, ByteSequence("`")},
      {-33ll, ByteSequence("_")},
      {-63ll, ByteSequence("A")},
      {-64ll, ByteSequence("@")},
      {-65ll, ByteSequence("?\277")},
      {-127ll, ByteSequence("?\201")},
      {-128ll, ByteSequence("?\200")},
      {-129ll, ByteSequence("?\177")},
      {-255ll, ByteSequence("?\001")},
      {-256ll, ByteSequence("?\000")},
      {-257ll, ByteSequence(">\377")},
      {-511ll, ByteSequence(">\001")},
      {-512ll, ByteSequence(">\000")},
      {-513ll, ByteSequence("=\377")},
      {-1023ll, ByteSequence("<\001")},
      {-1024ll, ByteSequence("<\000")},
      {-1025ll, ByteSequence(";\377")},
      {-2047ll, ByteSequence("8\001")},
      {-2048ll, ByteSequence("8\000")},
      {-2049ll, ByteSequence("7\377")},
      {-4095ll, ByteSequence("0\001")},
      {-4096ll, ByteSequence("0\000")},
      {-4097ll, ByteSequence("/\377")},
      {-8191ll, ByteSequence(" \001")},
      {-8192ll, ByteSequence(" \000")},
      {-8193ll, ByteSequence("\037\337\377")},
      {-16383ll, ByteSequence("\037\300\001")},
      {-16384ll, ByteSequence("\037\300\000")},
      {-16385ll, ByteSequence("\037\277\377")},
      {-32767ll, ByteSequence("\037\200\001")},
      {-32768ll, ByteSequence("\037\200\000")},
      {-32769ll, ByteSequence("\037\177\377")},
      {-65535ll, ByteSequence("\037\000\001")},
      {-65536ll, ByteSequence("\037\000\000")},
      {-65537ll, ByteSequence("\036\377\377")},
      {-131071ll, ByteSequence("\036\000\001")},
      {-131072ll, ByteSequence("\036\000\000")},
      {-131073ll, ByteSequence("\035\377\377")},
      {-262143ll, ByteSequence("\034\000\001")},
      {-262144ll, ByteSequence("\034\000\000")},
      {-262145ll, ByteSequence("\033\377\377")},
      {-524287ll, ByteSequence("\030\000\001")},
      {-524288ll, ByteSequence("\030\000\000")},
      {-524289ll, ByteSequence("\027\377\377")},
      {-1048575ll, ByteSequence("\020\000\001")},
      {-1048576ll, ByteSequence("\020\000\000")},
      {-1048577ll, ByteSequence("\017\357\377\377")},
      {-2097151ll, ByteSequence("\017\340\000\001")},
      {-2097152ll, ByteSequence("\017\340\000\000")},
      {-2097153ll, ByteSequence("\017\337\377\377")},
      {-4194303ll, ByteSequence("\017\300\000\001")},
      {-4194304ll, ByteSequence("\017\300\000\000")},
      {-4194305ll, ByteSequence("\017\277\377\377")},
      {-8388607ll, ByteSequence("\017\200\000\001")},
      {-8388608ll, ByteSequence("\017\200\000\000")},
      {-8388609ll, ByteSequence("\017\177\377\377")},
      {-16777215ll, ByteSequence("\017\000\000\001")},
      {-16777216ll, ByteSequence("\017\000\000\000")},
      {-16777217ll, ByteSequence("\016\377\377\377")},
      {-33554431ll, ByteSequence("\016\000\000\001")},
      {-33554432ll, ByteSequence("\016\000\000\000")},
      {-33554433ll, ByteSequence("\r\377\377\377")},
      {-67108863ll, ByteSequence("\014\000\000\001")},
      {-67108864ll, ByteSequence("\014\000\000\000")},
      {-67108865ll, ByteSequence("\013\377\377\377")},
      {-134217727ll, ByteSequence("\010\000\000\001")},
      {-134217728ll, ByteSequence("\010\000\000\000")},
      {-134217729ll, ByteSequence("\007\367\377\377\377")},
      {-268435455ll, ByteSequence("\007\360\000\000\001")},
      {-268435456ll, ByteSequence("\007\360\000\000\000")},
      {-268435457ll, ByteSequence("\007\357\377\377\377")},
      {-536870911ll, ByteSequence("\007\340\000\000\001")},
      {-536870912ll, ByteSequence("\007\340\000\000\000")},
      {-536870913ll, ByteSequence("\007\337\377\377\377")},
      {-1073741823ll, ByteSequence("\007\300\000\000\001")},
      {-1073741824ll, ByteSequence("\007\300\000\000\000")},
      {-1073741825ll, ByteSequence("\007\277\377\377\377")},
      {-2147483647ll, ByteSequence("\007\200\000\000\001")},
      {-2147483648ll, ByteSequence("\007\200\000\000\000")},
      {-2147483649ll, ByteSequence("\007\177\377\377\377")},
      {-4294967295ll, ByteSequence("\007\000\000\000\001")},
      {-4294967296ll, ByteSequence("\007\000\000\000\000")},
      {-4294967297ll, ByteSequence("\006\377\377\377\377")},
      {-8589934591ll, ByteSequence("\006\000\000\000\001")},
      {-8589934592ll, ByteSequence("\006\000\000\000\000")},
      {-8589934593ll, ByteSequence("\005\377\377\377\377")},
      {-17179869183ll, ByteSequence("\004\000\000\000\001")},
      {-17179869184ll, ByteSequence("\004\000\000\000\000")},
      {-17179869185ll, ByteSequence("\003\373\377\377\377\377")},
      {-34359738367ll, ByteSequence("\003\370\000\000\000\001")},
      {-34359738368ll, ByteSequence("\003\370\000\000\000\000")},
      {-34359738369ll, ByteSequence("\003\367\377\377\377\377")},
      {-68719476735ll, ByteSequence("\003\360\000\000\000\001")},
      {-68719476736ll, ByteSequence("\003\360\000\000\000\000")},
      {-68719476737ll, ByteSequence("\003\357\377\377\377\377")},
      {-137438953471ll, ByteSequence("\003\340\000\000\000\001")},
      {-137438953472ll, ByteSequence("\003\340\000\000\000\000")},
      {-137438953473ll, ByteSequence("\003\337\377\377\377\377")},
      {-274877906943ll, ByteSequence("\003\300\000\000\000\001")},
      {-274877906944ll, ByteSequence("\003\300\000\000\000\000")},
      {-274877906945ll, ByteSequence("\003\277\377\377\377\377")},
      {-549755813887ll, ByteSequence("\003\200\000\000\000\001")},
      {-549755813888ll, ByteSequence("\003\200\000\000\000\000")},
      {-549755813889ll, ByteSequence("\003\177\377\377\377\377")},
      {-1099511627775ll, ByteSequence("\003\000\000\000\000\001")},
      {-1099511627776ll, ByteSequence("\003\000\000\000\000\000")},
      {-1099511627777ll, ByteSequence("\002\377\377\377\377\377")},
      {-2199023255551ll, ByteSequence("\002\000\000\000\000\001")},
      {-2199023255552ll, ByteSequence("\002\000\000\000\000\000")},
      {-2199023255553ll, ByteSequence("\001\375\377\377\377\377\377")},
      {-4398046511103ll, ByteSequence("\001\374\000\000\000\000\001")},
      {-4398046511104ll, ByteSequence("\001\374\000\000\000\000\000")},
      {-4398046511105ll, ByteSequence("\001\373\377\377\377\377\377")},
      {-8796093022207ll, ByteSequence("\001\370\000\000\000\000\001")},
      {-8796093022208ll, ByteSequence("\001\370\000\000\000\000\000")},
      {-8796093022209ll, ByteSequence("\001\367\377\377\377\377\377")},
      {-17592186044415ll, ByteSequence("\001\360\000\000\000\000\001")},
      {-17592186044416ll, ByteSequence("\001\360\000\000\000\000\000")},
      {-17592186044417ll, ByteSequence("\001\357\377\377\377\377\377")},
      {-35184372088831ll, ByteSequence("\001\340\000\000\000\000\001")},
      {-35184372088832ll, ByteSequence("\001\340\000\000\000\000\000")},
      {-35184372088833ll, ByteSequence("\001\337\377\377\377\377\377")},
      {-70368744177663ll, ByteSequence("\001\300\000\000\000\000\001")},
      {-70368744177664ll, ByteSequence("\001\300\000\000\000\000\000")},
      {-70368744177665ll, ByteSequence("\001\277\377\377\377\377\377")},
      {-140737488355327ll, ByteSequence("\001\200\000\000\000\000\001")},
      {-140737488355328ll, ByteSequence("\001\200\000\000\000\000\000")},
      {-140737488355329ll, ByteSequence("\001\177\377\377\377\377\377")},
      {-281474976710655ll, ByteSequence("\001\000\000\000\000\000\001")},
      {-281474976710656ll, ByteSequence("\001\000\000\000\000\000\000")},
      {-281474976710657ll, ByteSequence("\000\376\377\377\377\377\377\377")},
      {-562949953421311ll, ByteSequence("\000\376\000\000\000\000\000\001")},
      {-562949953421312ll, ByteSequence("\000\376\000\000\000\000\000\000")},
      {-562949953421313ll, ByteSequence("\000\375\377\377\377\377\377\377")},
      {-1125899906842623ll, ByteSequence("\000\374\000\000\000\000\000\001")},
      {-1125899906842624ll, ByteSequence("\000\374\000\000\000\000\000\000")},
      {-1125899906842625ll, ByteSequence("\000\373\377\377\377\377\377\377")},
      {-2251799813685247ll, ByteSequence("\000\370\000\000\000\000\000\001")},
      {-2251799813685248ll, ByteSequence("\000\370\000\000\000\000\000\000")},
      {-2251799813685249ll, ByteSequence("\000\367\377\377\377\377\377\377")},
      {-4503599627370495ll, ByteSequence("\000\360\000\000\000\000\000\001")},
      {-4503599627370496ll, ByteSequence("\000\360\000\000\000\000\000\000")},
      {-4503599627370497ll, ByteSequence("\000\357\377\377\377\377\377\377")},
      {-9007199254740991ll, ByteSequence("\000\340\000\000\000\000\000\001")},
      {-9007199254740992ll, ByteSequence("\000\340\000\000\000\000\000\000")},
      {-9007199254740993ll, ByteSequence("\000\337\377\377\377\377\377\377")},
      {-18014398509481983ll, ByteSequence("\000\300\000\000\000\000\000\001")},
      {-18014398509481984ll, ByteSequence("\000\300\000\000\000\000\000\000")},
      {-18014398509481985ll, ByteSequence("\000\277\377\377\377\377\377\377")},
      {-36028797018963967ll, ByteSequence("\000\200\000\000\000\000\000\001")},
      {-36028797018963968ll, ByteSequence("\000\200\000\000\000\000\000\000")},
      {-36028797018963969ll,
       ByteSequence("\000\177\177\377\377\377\377\377\377")},
      {-72057594037927935ll,
       ByteSequence("\000\177\000\000\000\000\000\000\001")},
      {-72057594037927936ll,
       ByteSequence("\000\177\000\000\000\000\000\000\000")},
      {-72057594037927937ll, ByteSequence("\000~\377\377\377\377\377\377\377")},
      {-144115188075855871ll,
       ByteSequence("\000~\000\000\000\000\000\000\001")},
      {-144115188075855872ll,
       ByteSequence("\000~\000\000\000\000\000\000\000")},
      {-144115188075855873ll,
       ByteSequence("\000}\377\377\377\377\377\377\377")},
      {-288230376151711743ll,
       ByteSequence("\000|\000\000\000\000\000\000\001")},
      {-288230376151711744ll,
       ByteSequence("\000|\000\000\000\000\000\000\000")},
      {-288230376151711745ll,
       ByteSequence("\000{\377\377\377\377\377\377\377")},
      {-576460752303423487ll,
       ByteSequence("\000x\000\000\000\000\000\000\001")},
      {-576460752303423488ll,
       ByteSequence("\000x\000\000\000\000\000\000\000")},
      {-576460752303423489ll,
       ByteSequence("\000w\377\377\377\377\377\377\377")},
      {-1152921504606846975ll,
       ByteSequence("\000p\000\000\000\000\000\000\001")},
      {-1152921504606846976ll,
       ByteSequence("\000p\000\000\000\000\000\000\000")},
      {-1152921504606846977ll,
       ByteSequence("\000o\377\377\377\377\377\377\377")},
      {-2305843009213693951ll,
       ByteSequence("\000`\000\000\000\000\000\000\001")},
      {-2305843009213693952ll,
       ByteSequence("\000`\000\000\000\000\000\000\000")},
      {-2305843009213693953ll,
       ByteSequence("\000_\377\377\377\377\377\377\377")},
      {-4611686018427387903ll,
       ByteSequence("\000@\000\000\000\000\000\000\001")},
      {-4611686018427387904ll,
       ByteSequence("\000@\000\000\000\000\000\000\000")},
      {-4611686018427387905ll,
       ByteSequence("\000?\277\377\377\377\377\377\377\377")},
      {-9223372036854775807ll,
       ByteSequence("\000?\200\000\000\000\000\000\000\001")},
      {9223372036854775807ll,
       ByteSequence("\377\300\177\377\377\377\377\377\377\377")},
  };
  for (const auto& t : data) {
    int64 num = t.first;
    string result;
    OrderedCode::WriteSignedNumIncreasing(&result, num);
    EXPECT_EQ(t.second, result) << std::hex << num;

    StringPiece in = result;
    int64 decoded;
    EXPECT_TRUE(OrderedCode::ReadSignedNumIncreasing(&in, &decoded));
    EXPECT_EQ(num, decoded);
    EXPECT_EQ("", in);
  }
}

void BM_WriteString(int n, int len) {
  testing::StopTiming();
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  string x;
  for (int i = 0; i < len; i++) {
    x += rnd.Uniform(256);
  }
  string y;

  testing::BytesProcessed(n * len);
  testing::StartTiming();
  while (n-- > 0) {
    y.clear();
    OCWriteToString<string>(&y, x);
  }
}

void BM_ReadString(int n, int len) {
  testing::StopTiming();
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  string x;
  for (int i = 0; i < len; i++) {
    x += rnd.Uniform(256);
  }
  string data;
  OCWriteToString<string>(&data, x);
  string result;

  testing::BytesProcessed(n * len);
  testing::StartTiming();
  while (n-- > 0) {
    result.clear();
    StringPiece s = data;
    OCRead<string>(&s, &result);
  }
}

void BM_WriteStringIncreasing(int n, int len) { BM_WriteString(n, len); }
void BM_ReadStringIncreasing(int n, int len) { BM_ReadString(n, len); }

BENCHMARK(BM_WriteStringIncreasing)->Range(0, 1024);
BENCHMARK(BM_ReadStringIncreasing)->Range(0, 1024);

}  // namespace
}  // namespace strings
}  // namespace tensorflow
