#include "tensorflow/core/lib/strings/ordered_code.h"

#include <float.h>
#include <stddef.h>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace strings {

static string RandomString(random::SimplePhilox* rnd, int len) {
  string x;
  for (int i = 0; i < len; i++) {
    x += rnd->Uniform(256);
  }
  return x;
}

// ---------------------------------------------------------------------
// Utility template functions (they help templatize the tests below)

// Read/WriteIncreasing are defined for string, uint64, int64 below.
template <typename T>
static void OCWriteIncreasing(string* dest, const T& val);
template <typename T>
static bool OCReadIncreasing(StringPiece* src, T* result);

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
static T TestRead(const string& a) {
  // gracefully reject any proper prefix of an encoding
  for (int i = 0; i < a.size() - 1; ++i) {
    StringPiece s(a.data(), i);
    CHECK(!OCRead<T>(&s, NULL));
    CHECK_EQ(s, a.substr(0, i));
  }

  StringPiece s(a);
  T v;
  CHECK(OCRead<T>(&s, &v));
  CHECK(s.empty());
  return v;
}

template <typename T>
static void TestWriteRead(T expected) {
  EXPECT_EQ(expected, TestRead<T>(OCWrite<T>(expected)));
}

// Verifies that the second Write* call appends a non-empty string to its
// output.
template <typename T, typename U>
static void TestWriteAppends(T first, U second) {
  string encoded;
  OCWriteToString<T>(&encoded, first);
  string encoded_first_only = encoded;
  OCWriteToString<U>(&encoded, second);
  EXPECT_NE(encoded, encoded_first_only);
  EXPECT_TRUE(StringPiece(encoded).starts_with(encoded_first_only));
}

template <typename T>
static void TestNumbers(T multiplier) {
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
static bool CompareStrings(const string& a, const string& b) { return (a < b); }

template <typename T>
static void TestNumberOrdering() {
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
static int FindSpecial(const string& x) {
  const char* p = x.data();
  const char* limit = p + x.size();
  const char* result = OrderedCode::TEST_SkipToNextSpecialByte(p, limit);
  return result - p;
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
static inline string StrNot(const string& s) {
  string result;
  for (string::const_iterator it = s.begin(); it != s.end(); ++it)
    result.push_back(~*it);
  return result;
}

template <typename T>
static void TestInvalidEncoding(const string& s) {
  StringPiece p(s);
  EXPECT_FALSE(OCRead<T>(&p, static_cast<T*>(NULL)));
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
    EXPECT_DEATH(OrderedCode::ReadNumIncreasing(&s, NULL), "invalid encoding");
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
    EXPECT_DEATH(OrderedCode::ReadSignedNumIncreasing(&s, NULL),
                 "invalid encoding")
        << n;
#else
    TestRead<int64>(non_minimal);
#endif
  }
}

// Returns random number with specified number of bits,
// i.e., in the range [2^(bits-1),2^bits).
static uint64 NextBits(random::SimplePhilox* rnd, int bits) {
  return (bits != 0)
             ? (rnd->Rand64() % (1LL << (bits - 1))) + (1LL << (bits - 1))
             : 0;
}

template <typename T>
static void BM_WriteNum(int n, T multiplier) {
  static const int kValues = 64;
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
static void BM_ReadNum(int n, T multiplier) {
  string x;
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  // Use enough distinct values to confuse the branch predictor
  static const int kValues = 64;
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

#define BENCHMARK_NUM(name, T, multiplier)                             \
  static void BM_Write##name(int n) { BM_WriteNum<T>(n, multiplier); } \
  BENCHMARK(BM_Write##name);                                           \
  static void BM_Read##name(int n) { BM_ReadNum<T>(n, multiplier); }   \
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
      CHECK(OCRead<string>(&s2, NULL));
      CHECK_EQ(s, s2);

      CHECK(OCRead<string>(&s, &b2));
      CHECK(OCRead<string>(&s2, NULL));
      CHECK_EQ(s, s2);

      CHECK(!OCRead<string>(&s, &dummy));
      CHECK(!OCRead<string>(&s2, NULL));
      CHECK_EQ(a, a2);
      CHECK_EQ(b, b2);
      CHECK(s.empty());
      CHECK(s2.empty());
    }
  }
}

// 'str' is a static C-style string that may contain '\0'
#define STATIC_STR(str) StringPiece((str), sizeof(str) - 1)

static string EncodeStringIncreasing(StringPiece value) {
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
      {0x0ull, string("\000", 1)},
      {0x1ull, string("\001\001", 2)},
      {0x2ull, string("\001\002", 2)},
      {0x1ull, string("\001\001", 2)},
      {0x2ull, string("\001\002", 2)},
      {0x3ull, string("\001\003", 2)},
      {0x3ull, string("\001\003", 2)},
      {0x4ull, string("\001\004", 2)},
      {0x5ull, string("\001\005", 2)},
      {0x7ull, string("\001\007", 2)},
      {0x8ull, string("\001\010", 2)},
      {0x9ull, string("\001\t", 2)},
      {0xfull, string("\001\017", 2)},
      {0x10ull, string("\001\020", 2)},
      {0x11ull, string("\001\021", 2)},
      {0x1full, string("\001\037", 2)},
      {0x20ull, string("\001 ", 2)},
      {0x21ull, string("\001!", 2)},
      {0x3full, string("\001?", 2)},
      {0x40ull, string("\001@", 2)},
      {0x41ull, string("\001A", 2)},
      {0x7full, string("\001\177", 2)},
      {0x80ull, string("\001\200", 2)},
      {0x81ull, string("\001\201", 2)},
      {0xffull, string("\001\377", 2)},
      {0x100ull, string("\002\001\000", 3)},
      {0x101ull, string("\002\001\001", 3)},
      {0x1ffull, string("\002\001\377", 3)},
      {0x200ull, string("\002\002\000", 3)},
      {0x201ull, string("\002\002\001", 3)},
      {0x3ffull, string("\002\003\377", 3)},
      {0x400ull, string("\002\004\000", 3)},
      {0x401ull, string("\002\004\001", 3)},
      {0x7ffull, string("\002\007\377", 3)},
      {0x800ull, string("\002\010\000", 3)},
      {0x801ull, string("\002\010\001", 3)},
      {0xfffull, string("\002\017\377", 3)},
      {0x1000ull, string("\002\020\000", 3)},
      {0x1001ull, string("\002\020\001", 3)},
      {0x1fffull, string("\002\037\377", 3)},
      {0x2000ull, string("\002 \000", 3)},
      {0x2001ull, string("\002 \001", 3)},
      {0x3fffull, string("\002?\377", 3)},
      {0x4000ull, string("\002@\000", 3)},
      {0x4001ull, string("\002@\001", 3)},
      {0x7fffull, string("\002\177\377", 3)},
      {0x8000ull, string("\002\200\000", 3)},
      {0x8001ull, string("\002\200\001", 3)},
      {0xffffull, string("\002\377\377", 3)},
      {0x10000ull, string("\003\001\000\000", 4)},
      {0x10001ull, string("\003\001\000\001", 4)},
      {0x1ffffull, string("\003\001\377\377", 4)},
      {0x20000ull, string("\003\002\000\000", 4)},
      {0x20001ull, string("\003\002\000\001", 4)},
      {0x3ffffull, string("\003\003\377\377", 4)},
      {0x40000ull, string("\003\004\000\000", 4)},
      {0x40001ull, string("\003\004\000\001", 4)},
      {0x7ffffull, string("\003\007\377\377", 4)},
      {0x80000ull, string("\003\010\000\000", 4)},
      {0x80001ull, string("\003\010\000\001", 4)},
      {0xfffffull, string("\003\017\377\377", 4)},
      {0x100000ull, string("\003\020\000\000", 4)},
      {0x100001ull, string("\003\020\000\001", 4)},
      {0x1fffffull, string("\003\037\377\377", 4)},
      {0x200000ull, string("\003 \000\000", 4)},
      {0x200001ull, string("\003 \000\001", 4)},
      {0x3fffffull, string("\003?\377\377", 4)},
      {0x400000ull, string("\003@\000\000", 4)},
      {0x400001ull, string("\003@\000\001", 4)},
      {0x7fffffull, string("\003\177\377\377", 4)},
      {0x800000ull, string("\003\200\000\000", 4)},
      {0x800001ull, string("\003\200\000\001", 4)},
      {0xffffffull, string("\003\377\377\377", 4)},
      {0x1000000ull, string("\004\001\000\000\000", 5)},
      {0x1000001ull, string("\004\001\000\000\001", 5)},
      {0x1ffffffull, string("\004\001\377\377\377", 5)},
      {0x2000000ull, string("\004\002\000\000\000", 5)},
      {0x2000001ull, string("\004\002\000\000\001", 5)},
      {0x3ffffffull, string("\004\003\377\377\377", 5)},
      {0x4000000ull, string("\004\004\000\000\000", 5)},
      {0x4000001ull, string("\004\004\000\000\001", 5)},
      {0x7ffffffull, string("\004\007\377\377\377", 5)},
      {0x8000000ull, string("\004\010\000\000\000", 5)},
      {0x8000001ull, string("\004\010\000\000\001", 5)},
      {0xfffffffull, string("\004\017\377\377\377", 5)},
      {0x10000000ull, string("\004\020\000\000\000", 5)},
      {0x10000001ull, string("\004\020\000\000\001", 5)},
      {0x1fffffffull, string("\004\037\377\377\377", 5)},
      {0x20000000ull, string("\004 \000\000\000", 5)},
      {0x20000001ull, string("\004 \000\000\001", 5)},
      {0x3fffffffull, string("\004?\377\377\377", 5)},
      {0x40000000ull, string("\004@\000\000\000", 5)},
      {0x40000001ull, string("\004@\000\000\001", 5)},
      {0x7fffffffull, string("\004\177\377\377\377", 5)},
      {0x80000000ull, string("\004\200\000\000\000", 5)},
      {0x80000001ull, string("\004\200\000\000\001", 5)},
      {0xffffffffull, string("\004\377\377\377\377", 5)},
      {0x100000000ull, string("\005\001\000\000\000\000", 6)},
      {0x100000001ull, string("\005\001\000\000\000\001", 6)},
      {0x1ffffffffull, string("\005\001\377\377\377\377", 6)},
      {0x200000000ull, string("\005\002\000\000\000\000", 6)},
      {0x200000001ull, string("\005\002\000\000\000\001", 6)},
      {0x3ffffffffull, string("\005\003\377\377\377\377", 6)},
      {0x400000000ull, string("\005\004\000\000\000\000", 6)},
      {0x400000001ull, string("\005\004\000\000\000\001", 6)},
      {0x7ffffffffull, string("\005\007\377\377\377\377", 6)},
      {0x800000000ull, string("\005\010\000\000\000\000", 6)},
      {0x800000001ull, string("\005\010\000\000\000\001", 6)},
      {0xfffffffffull, string("\005\017\377\377\377\377", 6)},
      {0x1000000000ull, string("\005\020\000\000\000\000", 6)},
      {0x1000000001ull, string("\005\020\000\000\000\001", 6)},
      {0x1fffffffffull, string("\005\037\377\377\377\377", 6)},
      {0x2000000000ull, string("\005 \000\000\000\000", 6)},
      {0x2000000001ull, string("\005 \000\000\000\001", 6)},
      {0x3fffffffffull, string("\005?\377\377\377\377", 6)},
      {0x4000000000ull, string("\005@\000\000\000\000", 6)},
      {0x4000000001ull, string("\005@\000\000\000\001", 6)},
      {0x7fffffffffull, string("\005\177\377\377\377\377", 6)},
      {0x8000000000ull, string("\005\200\000\000\000\000", 6)},
      {0x8000000001ull, string("\005\200\000\000\000\001", 6)},
      {0xffffffffffull, string("\005\377\377\377\377\377", 6)},
      {0x10000000000ull, string("\006\001\000\000\000\000\000", 7)},
      {0x10000000001ull, string("\006\001\000\000\000\000\001", 7)},
      {0x1ffffffffffull, string("\006\001\377\377\377\377\377", 7)},
      {0x20000000000ull, string("\006\002\000\000\000\000\000", 7)},
      {0x20000000001ull, string("\006\002\000\000\000\000\001", 7)},
      {0x3ffffffffffull, string("\006\003\377\377\377\377\377", 7)},
      {0x40000000000ull, string("\006\004\000\000\000\000\000", 7)},
      {0x40000000001ull, string("\006\004\000\000\000\000\001", 7)},
      {0x7ffffffffffull, string("\006\007\377\377\377\377\377", 7)},
      {0x80000000000ull, string("\006\010\000\000\000\000\000", 7)},
      {0x80000000001ull, string("\006\010\000\000\000\000\001", 7)},
      {0xfffffffffffull, string("\006\017\377\377\377\377\377", 7)},
      {0x100000000000ull, string("\006\020\000\000\000\000\000", 7)},
      {0x100000000001ull, string("\006\020\000\000\000\000\001", 7)},
      {0x1fffffffffffull, string("\006\037\377\377\377\377\377", 7)},
      {0x200000000000ull, string("\006 \000\000\000\000\000", 7)},
      {0x200000000001ull, string("\006 \000\000\000\000\001", 7)},
      {0x3fffffffffffull, string("\006?\377\377\377\377\377", 7)},
      {0x400000000000ull, string("\006@\000\000\000\000\000", 7)},
      {0x400000000001ull, string("\006@\000\000\000\000\001", 7)},
      {0x7fffffffffffull, string("\006\177\377\377\377\377\377", 7)},
      {0x800000000000ull, string("\006\200\000\000\000\000\000", 7)},
      {0x800000000001ull, string("\006\200\000\000\000\000\001", 7)},
      {0xffffffffffffull, string("\006\377\377\377\377\377\377", 7)},
      {0x1000000000000ull, string("\007\001\000\000\000\000\000\000", 8)},
      {0x1000000000001ull, string("\007\001\000\000\000\000\000\001", 8)},
      {0x1ffffffffffffull, string("\007\001\377\377\377\377\377\377", 8)},
      {0x2000000000000ull, string("\007\002\000\000\000\000\000\000", 8)},
      {0x2000000000001ull, string("\007\002\000\000\000\000\000\001", 8)},
      {0x3ffffffffffffull, string("\007\003\377\377\377\377\377\377", 8)},
      {0x4000000000000ull, string("\007\004\000\000\000\000\000\000", 8)},
      {0x4000000000001ull, string("\007\004\000\000\000\000\000\001", 8)},
      {0x7ffffffffffffull, string("\007\007\377\377\377\377\377\377", 8)},
      {0x8000000000000ull, string("\007\010\000\000\000\000\000\000", 8)},
      {0x8000000000001ull, string("\007\010\000\000\000\000\000\001", 8)},
      {0xfffffffffffffull, string("\007\017\377\377\377\377\377\377", 8)},
      {0x10000000000000ull, string("\007\020\000\000\000\000\000\000", 8)},
      {0x10000000000001ull, string("\007\020\000\000\000\000\000\001", 8)},
      {0x1fffffffffffffull, string("\007\037\377\377\377\377\377\377", 8)},
      {0x20000000000000ull, string("\007 \000\000\000\000\000\000", 8)},
      {0x20000000000001ull, string("\007 \000\000\000\000\000\001", 8)},
      {0x3fffffffffffffull, string("\007?\377\377\377\377\377\377", 8)},
      {0x40000000000000ull, string("\007@\000\000\000\000\000\000", 8)},
      {0x40000000000001ull, string("\007@\000\000\000\000\000\001", 8)},
      {0x7fffffffffffffull, string("\007\177\377\377\377\377\377\377", 8)},
      {0x80000000000000ull, string("\007\200\000\000\000\000\000\000", 8)},
      {0x80000000000001ull, string("\007\200\000\000\000\000\000\001", 8)},
      {0xffffffffffffffull, string("\007\377\377\377\377\377\377\377", 8)},
      {0x100000000000000ull, string("\010\001\000\000\000\000\000\000\000", 9)},
      {0x100000000000001ull, string("\010\001\000\000\000\000\000\000\001", 9)},
      {0x1ffffffffffffffull, string("\010\001\377\377\377\377\377\377\377", 9)},
      {0x200000000000000ull, string("\010\002\000\000\000\000\000\000\000", 9)},
      {0x200000000000001ull, string("\010\002\000\000\000\000\000\000\001", 9)},
      {0x3ffffffffffffffull, string("\010\003\377\377\377\377\377\377\377", 9)},
      {0x400000000000000ull, string("\010\004\000\000\000\000\000\000\000", 9)},
      {0x400000000000001ull, string("\010\004\000\000\000\000\000\000\001", 9)},
      {0x7ffffffffffffffull, string("\010\007\377\377\377\377\377\377\377", 9)},
      {0x800000000000000ull, string("\010\010\000\000\000\000\000\000\000", 9)},
      {0x800000000000001ull, string("\010\010\000\000\000\000\000\000\001", 9)},
      {0xfffffffffffffffull, string("\010\017\377\377\377\377\377\377\377", 9)},
      {0x1000000000000000ull,
       string("\010\020\000\000\000\000\000\000\000", 9)},
      {0x1000000000000001ull,
       string("\010\020\000\000\000\000\000\000\001", 9)},
      {0x1fffffffffffffffull,
       string("\010\037\377\377\377\377\377\377\377", 9)},
      {0x2000000000000000ull, string("\010 \000\000\000\000\000\000\000", 9)},
      {0x2000000000000001ull, string("\010 \000\000\000\000\000\000\001", 9)},
      {0x3fffffffffffffffull, string("\010?\377\377\377\377\377\377\377", 9)},
      {0x4000000000000000ull, string("\010@\000\000\000\000\000\000\000", 9)},
      {0x4000000000000001ull, string("\010@\000\000\000\000\000\000\001", 9)},
      {0x7fffffffffffffffull,
       string("\010\177\377\377\377\377\377\377\377", 9)},
      {0x8000000000000000ull,
       string("\010\200\000\000\000\000\000\000\000", 9)},
      {0x8000000000000001ull,
       string("\010\200\000\000\000\000\000\000\001", 9)},
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
      {0ll, string("\200", 1)},
      {1ll, string("\201", 1)},
      {2ll, string("\202", 1)},
      {1ll, string("\201", 1)},
      {2ll, string("\202", 1)},
      {3ll, string("\203", 1)},
      {3ll, string("\203", 1)},
      {4ll, string("\204", 1)},
      {5ll, string("\205", 1)},
      {7ll, string("\207", 1)},
      {8ll, string("\210", 1)},
      {9ll, string("\211", 1)},
      {15ll, string("\217", 1)},
      {16ll, string("\220", 1)},
      {17ll, string("\221", 1)},
      {31ll, string("\237", 1)},
      {32ll, string("\240", 1)},
      {33ll, string("\241", 1)},
      {63ll, string("\277", 1)},
      {64ll, string("\300@", 2)},
      {65ll, string("\300A", 2)},
      {127ll, string("\300\177", 2)},
      {128ll, string("\300\200", 2)},
      {129ll, string("\300\201", 2)},
      {255ll, string("\300\377", 2)},
      {256ll, string("\301\000", 2)},
      {257ll, string("\301\001", 2)},
      {511ll, string("\301\377", 2)},
      {512ll, string("\302\000", 2)},
      {513ll, string("\302\001", 2)},
      {1023ll, string("\303\377", 2)},
      {1024ll, string("\304\000", 2)},
      {1025ll, string("\304\001", 2)},
      {2047ll, string("\307\377", 2)},
      {2048ll, string("\310\000", 2)},
      {2049ll, string("\310\001", 2)},
      {4095ll, string("\317\377", 2)},
      {4096ll, string("\320\000", 2)},
      {4097ll, string("\320\001", 2)},
      {8191ll, string("\337\377", 2)},
      {8192ll, string("\340 \000", 3)},
      {8193ll, string("\340 \001", 3)},
      {16383ll, string("\340?\377", 3)},
      {16384ll, string("\340@\000", 3)},
      {16385ll, string("\340@\001", 3)},
      {32767ll, string("\340\177\377", 3)},
      {32768ll, string("\340\200\000", 3)},
      {32769ll, string("\340\200\001", 3)},
      {65535ll, string("\340\377\377", 3)},
      {65536ll, string("\341\000\000", 3)},
      {65537ll, string("\341\000\001", 3)},
      {131071ll, string("\341\377\377", 3)},
      {131072ll, string("\342\000\000", 3)},
      {131073ll, string("\342\000\001", 3)},
      {262143ll, string("\343\377\377", 3)},
      {262144ll, string("\344\000\000", 3)},
      {262145ll, string("\344\000\001", 3)},
      {524287ll, string("\347\377\377", 3)},
      {524288ll, string("\350\000\000", 3)},
      {524289ll, string("\350\000\001", 3)},
      {1048575ll, string("\357\377\377", 3)},
      {1048576ll, string("\360\020\000\000", 4)},
      {1048577ll, string("\360\020\000\001", 4)},
      {2097151ll, string("\360\037\377\377", 4)},
      {2097152ll, string("\360 \000\000", 4)},
      {2097153ll, string("\360 \000\001", 4)},
      {4194303ll, string("\360?\377\377", 4)},
      {4194304ll, string("\360@\000\000", 4)},
      {4194305ll, string("\360@\000\001", 4)},
      {8388607ll, string("\360\177\377\377", 4)},
      {8388608ll, string("\360\200\000\000", 4)},
      {8388609ll, string("\360\200\000\001", 4)},
      {16777215ll, string("\360\377\377\377", 4)},
      {16777216ll, string("\361\000\000\000", 4)},
      {16777217ll, string("\361\000\000\001", 4)},
      {33554431ll, string("\361\377\377\377", 4)},
      {33554432ll, string("\362\000\000\000", 4)},
      {33554433ll, string("\362\000\000\001", 4)},
      {67108863ll, string("\363\377\377\377", 4)},
      {67108864ll, string("\364\000\000\000", 4)},
      {67108865ll, string("\364\000\000\001", 4)},
      {134217727ll, string("\367\377\377\377", 4)},
      {134217728ll, string("\370\010\000\000\000", 5)},
      {134217729ll, string("\370\010\000\000\001", 5)},
      {268435455ll, string("\370\017\377\377\377", 5)},
      {268435456ll, string("\370\020\000\000\000", 5)},
      {268435457ll, string("\370\020\000\000\001", 5)},
      {536870911ll, string("\370\037\377\377\377", 5)},
      {536870912ll, string("\370 \000\000\000", 5)},
      {536870913ll, string("\370 \000\000\001", 5)},
      {1073741823ll, string("\370?\377\377\377", 5)},
      {1073741824ll, string("\370@\000\000\000", 5)},
      {1073741825ll, string("\370@\000\000\001", 5)},
      {2147483647ll, string("\370\177\377\377\377", 5)},
      {2147483648ll, string("\370\200\000\000\000", 5)},
      {2147483649ll, string("\370\200\000\000\001", 5)},
      {4294967295ll, string("\370\377\377\377\377", 5)},
      {4294967296ll, string("\371\000\000\000\000", 5)},
      {4294967297ll, string("\371\000\000\000\001", 5)},
      {8589934591ll, string("\371\377\377\377\377", 5)},
      {8589934592ll, string("\372\000\000\000\000", 5)},
      {8589934593ll, string("\372\000\000\000\001", 5)},
      {17179869183ll, string("\373\377\377\377\377", 5)},
      {17179869184ll, string("\374\004\000\000\000\000", 6)},
      {17179869185ll, string("\374\004\000\000\000\001", 6)},
      {34359738367ll, string("\374\007\377\377\377\377", 6)},
      {34359738368ll, string("\374\010\000\000\000\000", 6)},
      {34359738369ll, string("\374\010\000\000\000\001", 6)},
      {68719476735ll, string("\374\017\377\377\377\377", 6)},
      {68719476736ll, string("\374\020\000\000\000\000", 6)},
      {68719476737ll, string("\374\020\000\000\000\001", 6)},
      {137438953471ll, string("\374\037\377\377\377\377", 6)},
      {137438953472ll, string("\374 \000\000\000\000", 6)},
      {137438953473ll, string("\374 \000\000\000\001", 6)},
      {274877906943ll, string("\374?\377\377\377\377", 6)},
      {274877906944ll, string("\374@\000\000\000\000", 6)},
      {274877906945ll, string("\374@\000\000\000\001", 6)},
      {549755813887ll, string("\374\177\377\377\377\377", 6)},
      {549755813888ll, string("\374\200\000\000\000\000", 6)},
      {549755813889ll, string("\374\200\000\000\000\001", 6)},
      {1099511627775ll, string("\374\377\377\377\377\377", 6)},
      {1099511627776ll, string("\375\000\000\000\000\000", 6)},
      {1099511627777ll, string("\375\000\000\000\000\001", 6)},
      {2199023255551ll, string("\375\377\377\377\377\377", 6)},
      {2199023255552ll, string("\376\002\000\000\000\000\000", 7)},
      {2199023255553ll, string("\376\002\000\000\000\000\001", 7)},
      {4398046511103ll, string("\376\003\377\377\377\377\377", 7)},
      {4398046511104ll, string("\376\004\000\000\000\000\000", 7)},
      {4398046511105ll, string("\376\004\000\000\000\000\001", 7)},
      {8796093022207ll, string("\376\007\377\377\377\377\377", 7)},
      {8796093022208ll, string("\376\010\000\000\000\000\000", 7)},
      {8796093022209ll, string("\376\010\000\000\000\000\001", 7)},
      {17592186044415ll, string("\376\017\377\377\377\377\377", 7)},
      {17592186044416ll, string("\376\020\000\000\000\000\000", 7)},
      {17592186044417ll, string("\376\020\000\000\000\000\001", 7)},
      {35184372088831ll, string("\376\037\377\377\377\377\377", 7)},
      {35184372088832ll, string("\376 \000\000\000\000\000", 7)},
      {35184372088833ll, string("\376 \000\000\000\000\001", 7)},
      {70368744177663ll, string("\376?\377\377\377\377\377", 7)},
      {70368744177664ll, string("\376@\000\000\000\000\000", 7)},
      {70368744177665ll, string("\376@\000\000\000\000\001", 7)},
      {140737488355327ll, string("\376\177\377\377\377\377\377", 7)},
      {140737488355328ll, string("\376\200\000\000\000\000\000", 7)},
      {140737488355329ll, string("\376\200\000\000\000\000\001", 7)},
      {281474976710655ll, string("\376\377\377\377\377\377\377", 7)},
      {281474976710656ll, string("\377\001\000\000\000\000\000\000", 8)},
      {281474976710657ll, string("\377\001\000\000\000\000\000\001", 8)},
      {562949953421311ll, string("\377\001\377\377\377\377\377\377", 8)},
      {562949953421312ll, string("\377\002\000\000\000\000\000\000", 8)},
      {562949953421313ll, string("\377\002\000\000\000\000\000\001", 8)},
      {1125899906842623ll, string("\377\003\377\377\377\377\377\377", 8)},
      {1125899906842624ll, string("\377\004\000\000\000\000\000\000", 8)},
      {1125899906842625ll, string("\377\004\000\000\000\000\000\001", 8)},
      {2251799813685247ll, string("\377\007\377\377\377\377\377\377", 8)},
      {2251799813685248ll, string("\377\010\000\000\000\000\000\000", 8)},
      {2251799813685249ll, string("\377\010\000\000\000\000\000\001", 8)},
      {4503599627370495ll, string("\377\017\377\377\377\377\377\377", 8)},
      {4503599627370496ll, string("\377\020\000\000\000\000\000\000", 8)},
      {4503599627370497ll, string("\377\020\000\000\000\000\000\001", 8)},
      {9007199254740991ll, string("\377\037\377\377\377\377\377\377", 8)},
      {9007199254740992ll, string("\377 \000\000\000\000\000\000", 8)},
      {9007199254740993ll, string("\377 \000\000\000\000\000\001", 8)},
      {18014398509481983ll, string("\377?\377\377\377\377\377\377", 8)},
      {18014398509481984ll, string("\377@\000\000\000\000\000\000", 8)},
      {18014398509481985ll, string("\377@\000\000\000\000\000\001", 8)},
      {36028797018963967ll, string("\377\177\377\377\377\377\377\377", 8)},
      {36028797018963968ll, string("\377\200\200\000\000\000\000\000\000", 9)},
      {36028797018963969ll, string("\377\200\200\000\000\000\000\000\001", 9)},
      {72057594037927935ll, string("\377\200\377\377\377\377\377\377\377", 9)},
      {72057594037927936ll, string("\377\201\000\000\000\000\000\000\000", 9)},
      {72057594037927937ll, string("\377\201\000\000\000\000\000\000\001", 9)},
      {144115188075855871ll, string("\377\201\377\377\377\377\377\377\377", 9)},
      {144115188075855872ll, string("\377\202\000\000\000\000\000\000\000", 9)},
      {144115188075855873ll, string("\377\202\000\000\000\000\000\000\001", 9)},
      {288230376151711743ll, string("\377\203\377\377\377\377\377\377\377", 9)},
      {288230376151711744ll, string("\377\204\000\000\000\000\000\000\000", 9)},
      {288230376151711745ll, string("\377\204\000\000\000\000\000\000\001", 9)},
      {576460752303423487ll, string("\377\207\377\377\377\377\377\377\377", 9)},
      {576460752303423488ll, string("\377\210\000\000\000\000\000\000\000", 9)},
      {576460752303423489ll, string("\377\210\000\000\000\000\000\000\001", 9)},
      {1152921504606846975ll,
       string("\377\217\377\377\377\377\377\377\377", 9)},
      {1152921504606846976ll,
       string("\377\220\000\000\000\000\000\000\000", 9)},
      {1152921504606846977ll,
       string("\377\220\000\000\000\000\000\000\001", 9)},
      {2305843009213693951ll,
       string("\377\237\377\377\377\377\377\377\377", 9)},
      {2305843009213693952ll,
       string("\377\240\000\000\000\000\000\000\000", 9)},
      {2305843009213693953ll,
       string("\377\240\000\000\000\000\000\000\001", 9)},
      {4611686018427387903ll,
       string("\377\277\377\377\377\377\377\377\377", 9)},
      {4611686018427387904ll,
       string("\377\300@\000\000\000\000\000\000\000", 10)},
      {4611686018427387905ll,
       string("\377\300@\000\000\000\000\000\000\001", 10)},
      {9223372036854775807ll,
       string("\377\300\177\377\377\377\377\377\377\377", 10)},
      {-9223372036854775807ll,
       string("\000?\200\000\000\000\000\000\000\001", 10)},
      {0ll, string("\200", 1)},
      {-1ll, string("\177", 1)},
      {-2ll, string("~", 1)},
      {-1ll, string("\177", 1)},
      {-2ll, string("~", 1)},
      {-3ll, string("}", 1)},
      {-3ll, string("}", 1)},
      {-4ll, string("|", 1)},
      {-5ll, string("{", 1)},
      {-7ll, string("y", 1)},
      {-8ll, string("x", 1)},
      {-9ll, string("w", 1)},
      {-15ll, string("q", 1)},
      {-16ll, string("p", 1)},
      {-17ll, string("o", 1)},
      {-31ll, string("a", 1)},
      {-32ll, string("`", 1)},
      {-33ll, string("_", 1)},
      {-63ll, string("A", 1)},
      {-64ll, string("@", 1)},
      {-65ll, string("?\277", 2)},
      {-127ll, string("?\201", 2)},
      {-128ll, string("?\200", 2)},
      {-129ll, string("?\177", 2)},
      {-255ll, string("?\001", 2)},
      {-256ll, string("?\000", 2)},
      {-257ll, string(">\377", 2)},
      {-511ll, string(">\001", 2)},
      {-512ll, string(">\000", 2)},
      {-513ll, string("=\377", 2)},
      {-1023ll, string("<\001", 2)},
      {-1024ll, string("<\000", 2)},
      {-1025ll, string(";\377", 2)},
      {-2047ll, string("8\001", 2)},
      {-2048ll, string("8\000", 2)},
      {-2049ll, string("7\377", 2)},
      {-4095ll, string("0\001", 2)},
      {-4096ll, string("0\000", 2)},
      {-4097ll, string("/\377", 2)},
      {-8191ll, string(" \001", 2)},
      {-8192ll, string(" \000", 2)},
      {-8193ll, string("\037\337\377", 3)},
      {-16383ll, string("\037\300\001", 3)},
      {-16384ll, string("\037\300\000", 3)},
      {-16385ll, string("\037\277\377", 3)},
      {-32767ll, string("\037\200\001", 3)},
      {-32768ll, string("\037\200\000", 3)},
      {-32769ll, string("\037\177\377", 3)},
      {-65535ll, string("\037\000\001", 3)},
      {-65536ll, string("\037\000\000", 3)},
      {-65537ll, string("\036\377\377", 3)},
      {-131071ll, string("\036\000\001", 3)},
      {-131072ll, string("\036\000\000", 3)},
      {-131073ll, string("\035\377\377", 3)},
      {-262143ll, string("\034\000\001", 3)},
      {-262144ll, string("\034\000\000", 3)},
      {-262145ll, string("\033\377\377", 3)},
      {-524287ll, string("\030\000\001", 3)},
      {-524288ll, string("\030\000\000", 3)},
      {-524289ll, string("\027\377\377", 3)},
      {-1048575ll, string("\020\000\001", 3)},
      {-1048576ll, string("\020\000\000", 3)},
      {-1048577ll, string("\017\357\377\377", 4)},
      {-2097151ll, string("\017\340\000\001", 4)},
      {-2097152ll, string("\017\340\000\000", 4)},
      {-2097153ll, string("\017\337\377\377", 4)},
      {-4194303ll, string("\017\300\000\001", 4)},
      {-4194304ll, string("\017\300\000\000", 4)},
      {-4194305ll, string("\017\277\377\377", 4)},
      {-8388607ll, string("\017\200\000\001", 4)},
      {-8388608ll, string("\017\200\000\000", 4)},
      {-8388609ll, string("\017\177\377\377", 4)},
      {-16777215ll, string("\017\000\000\001", 4)},
      {-16777216ll, string("\017\000\000\000", 4)},
      {-16777217ll, string("\016\377\377\377", 4)},
      {-33554431ll, string("\016\000\000\001", 4)},
      {-33554432ll, string("\016\000\000\000", 4)},
      {-33554433ll, string("\r\377\377\377", 4)},
      {-67108863ll, string("\014\000\000\001", 4)},
      {-67108864ll, string("\014\000\000\000", 4)},
      {-67108865ll, string("\013\377\377\377", 4)},
      {-134217727ll, string("\010\000\000\001", 4)},
      {-134217728ll, string("\010\000\000\000", 4)},
      {-134217729ll, string("\007\367\377\377\377", 5)},
      {-268435455ll, string("\007\360\000\000\001", 5)},
      {-268435456ll, string("\007\360\000\000\000", 5)},
      {-268435457ll, string("\007\357\377\377\377", 5)},
      {-536870911ll, string("\007\340\000\000\001", 5)},
      {-536870912ll, string("\007\340\000\000\000", 5)},
      {-536870913ll, string("\007\337\377\377\377", 5)},
      {-1073741823ll, string("\007\300\000\000\001", 5)},
      {-1073741824ll, string("\007\300\000\000\000", 5)},
      {-1073741825ll, string("\007\277\377\377\377", 5)},
      {-2147483647ll, string("\007\200\000\000\001", 5)},
      {-2147483648ll, string("\007\200\000\000\000", 5)},
      {-2147483649ll, string("\007\177\377\377\377", 5)},
      {-4294967295ll, string("\007\000\000\000\001", 5)},
      {-4294967296ll, string("\007\000\000\000\000", 5)},
      {-4294967297ll, string("\006\377\377\377\377", 5)},
      {-8589934591ll, string("\006\000\000\000\001", 5)},
      {-8589934592ll, string("\006\000\000\000\000", 5)},
      {-8589934593ll, string("\005\377\377\377\377", 5)},
      {-17179869183ll, string("\004\000\000\000\001", 5)},
      {-17179869184ll, string("\004\000\000\000\000", 5)},
      {-17179869185ll, string("\003\373\377\377\377\377", 6)},
      {-34359738367ll, string("\003\370\000\000\000\001", 6)},
      {-34359738368ll, string("\003\370\000\000\000\000", 6)},
      {-34359738369ll, string("\003\367\377\377\377\377", 6)},
      {-68719476735ll, string("\003\360\000\000\000\001", 6)},
      {-68719476736ll, string("\003\360\000\000\000\000", 6)},
      {-68719476737ll, string("\003\357\377\377\377\377", 6)},
      {-137438953471ll, string("\003\340\000\000\000\001", 6)},
      {-137438953472ll, string("\003\340\000\000\000\000", 6)},
      {-137438953473ll, string("\003\337\377\377\377\377", 6)},
      {-274877906943ll, string("\003\300\000\000\000\001", 6)},
      {-274877906944ll, string("\003\300\000\000\000\000", 6)},
      {-274877906945ll, string("\003\277\377\377\377\377", 6)},
      {-549755813887ll, string("\003\200\000\000\000\001", 6)},
      {-549755813888ll, string("\003\200\000\000\000\000", 6)},
      {-549755813889ll, string("\003\177\377\377\377\377", 6)},
      {-1099511627775ll, string("\003\000\000\000\000\001", 6)},
      {-1099511627776ll, string("\003\000\000\000\000\000", 6)},
      {-1099511627777ll, string("\002\377\377\377\377\377", 6)},
      {-2199023255551ll, string("\002\000\000\000\000\001", 6)},
      {-2199023255552ll, string("\002\000\000\000\000\000", 6)},
      {-2199023255553ll, string("\001\375\377\377\377\377\377", 7)},
      {-4398046511103ll, string("\001\374\000\000\000\000\001", 7)},
      {-4398046511104ll, string("\001\374\000\000\000\000\000", 7)},
      {-4398046511105ll, string("\001\373\377\377\377\377\377", 7)},
      {-8796093022207ll, string("\001\370\000\000\000\000\001", 7)},
      {-8796093022208ll, string("\001\370\000\000\000\000\000", 7)},
      {-8796093022209ll, string("\001\367\377\377\377\377\377", 7)},
      {-17592186044415ll, string("\001\360\000\000\000\000\001", 7)},
      {-17592186044416ll, string("\001\360\000\000\000\000\000", 7)},
      {-17592186044417ll, string("\001\357\377\377\377\377\377", 7)},
      {-35184372088831ll, string("\001\340\000\000\000\000\001", 7)},
      {-35184372088832ll, string("\001\340\000\000\000\000\000", 7)},
      {-35184372088833ll, string("\001\337\377\377\377\377\377", 7)},
      {-70368744177663ll, string("\001\300\000\000\000\000\001", 7)},
      {-70368744177664ll, string("\001\300\000\000\000\000\000", 7)},
      {-70368744177665ll, string("\001\277\377\377\377\377\377", 7)},
      {-140737488355327ll, string("\001\200\000\000\000\000\001", 7)},
      {-140737488355328ll, string("\001\200\000\000\000\000\000", 7)},
      {-140737488355329ll, string("\001\177\377\377\377\377\377", 7)},
      {-281474976710655ll, string("\001\000\000\000\000\000\001", 7)},
      {-281474976710656ll, string("\001\000\000\000\000\000\000", 7)},
      {-281474976710657ll, string("\000\376\377\377\377\377\377\377", 8)},
      {-562949953421311ll, string("\000\376\000\000\000\000\000\001", 8)},
      {-562949953421312ll, string("\000\376\000\000\000\000\000\000", 8)},
      {-562949953421313ll, string("\000\375\377\377\377\377\377\377", 8)},
      {-1125899906842623ll, string("\000\374\000\000\000\000\000\001", 8)},
      {-1125899906842624ll, string("\000\374\000\000\000\000\000\000", 8)},
      {-1125899906842625ll, string("\000\373\377\377\377\377\377\377", 8)},
      {-2251799813685247ll, string("\000\370\000\000\000\000\000\001", 8)},
      {-2251799813685248ll, string("\000\370\000\000\000\000\000\000", 8)},
      {-2251799813685249ll, string("\000\367\377\377\377\377\377\377", 8)},
      {-4503599627370495ll, string("\000\360\000\000\000\000\000\001", 8)},
      {-4503599627370496ll, string("\000\360\000\000\000\000\000\000", 8)},
      {-4503599627370497ll, string("\000\357\377\377\377\377\377\377", 8)},
      {-9007199254740991ll, string("\000\340\000\000\000\000\000\001", 8)},
      {-9007199254740992ll, string("\000\340\000\000\000\000\000\000", 8)},
      {-9007199254740993ll, string("\000\337\377\377\377\377\377\377", 8)},
      {-18014398509481983ll, string("\000\300\000\000\000\000\000\001", 8)},
      {-18014398509481984ll, string("\000\300\000\000\000\000\000\000", 8)},
      {-18014398509481985ll, string("\000\277\377\377\377\377\377\377", 8)},
      {-36028797018963967ll, string("\000\200\000\000\000\000\000\001", 8)},
      {-36028797018963968ll, string("\000\200\000\000\000\000\000\000", 8)},
      {-36028797018963969ll, string("\000\177\177\377\377\377\377\377\377", 9)},
      {-72057594037927935ll, string("\000\177\000\000\000\000\000\000\001", 9)},
      {-72057594037927936ll, string("\000\177\000\000\000\000\000\000\000", 9)},
      {-72057594037927937ll, string("\000~\377\377\377\377\377\377\377", 9)},
      {-144115188075855871ll, string("\000~\000\000\000\000\000\000\001", 9)},
      {-144115188075855872ll, string("\000~\000\000\000\000\000\000\000", 9)},
      {-144115188075855873ll, string("\000}\377\377\377\377\377\377\377", 9)},
      {-288230376151711743ll, string("\000|\000\000\000\000\000\000\001", 9)},
      {-288230376151711744ll, string("\000|\000\000\000\000\000\000\000", 9)},
      {-288230376151711745ll, string("\000{\377\377\377\377\377\377\377", 9)},
      {-576460752303423487ll, string("\000x\000\000\000\000\000\000\001", 9)},
      {-576460752303423488ll, string("\000x\000\000\000\000\000\000\000", 9)},
      {-576460752303423489ll, string("\000w\377\377\377\377\377\377\377", 9)},
      {-1152921504606846975ll, string("\000p\000\000\000\000\000\000\001", 9)},
      {-1152921504606846976ll, string("\000p\000\000\000\000\000\000\000", 9)},
      {-1152921504606846977ll, string("\000o\377\377\377\377\377\377\377", 9)},
      {-2305843009213693951ll, string("\000`\000\000\000\000\000\000\001", 9)},
      {-2305843009213693952ll, string("\000`\000\000\000\000\000\000\000", 9)},
      {-2305843009213693953ll, string("\000_\377\377\377\377\377\377\377", 9)},
      {-4611686018427387903ll, string("\000@\000\000\000\000\000\000\001", 9)},
      {-4611686018427387904ll, string("\000@\000\000\000\000\000\000\000", 9)},
      {-4611686018427387905ll,
       string("\000?\277\377\377\377\377\377\377\377", 10)},
      {-9223372036854775807ll,
       string("\000?\200\000\000\000\000\000\000\001", 10)},
      {9223372036854775807ll,
       string("\377\300\177\377\377\377\377\377\377\377", 10)},
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

static void BM_WriteString(int n, int len) {
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

static void BM_ReadString(int n, int len) {
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

static void BM_WriteStringIncreasing(int n, int len) { BM_WriteString(n, len); }
static void BM_ReadStringIncreasing(int n, int len) { BM_ReadString(n, len); }

BENCHMARK(BM_WriteStringIncreasing)->Range(0, 1024);
BENCHMARK(BM_ReadStringIncreasing)->Range(0, 1024);

}  // namespace strings
}  // namespace tensorflow
