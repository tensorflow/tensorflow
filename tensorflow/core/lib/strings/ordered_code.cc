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

#include <assert.h>
#include <stddef.h>

#include <cstdint>

#include "absl/log/check.h"
#include "xla/tsl/lib/core/bits.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace tensorflow {
namespace strings {

// We encode a string in different ways depending on whether the item
// should be in lexicographically increasing or decreasing order.
//
//
// Lexicographically increasing order
//
// We want a string-to-string mapping F(x) such that for any two strings
//
//      x < y   =>   F(x) < F(y)
//
// In addition to the normal characters '\x00' through '\xff', we want to
// encode a few extra symbols in strings:
//
//      <sep>           Separator between items
//      <infinity>      Infinite string
//
// Therefore we need an alphabet with at least 258 symbols.  Each
// character '\1' through '\xfe' is mapped to itself.  The other four are
// encoded into two-letter sequences starting with '\0' and '\xff':
//
//      <sep>           encoded as =>           \0\1
//      \0              encoded as =>           \0\xff
//      \xff            encoded as =>           \xff\x00
//      <infinity>      encoded as =>           \xff\xff
//
// The remaining two-letter sequences starting with '\0' and '\xff' are
// currently unused.
//
// F(<infinity>) is defined above.  For any finite string x, F(x) is the
// the encodings of x's characters followed by the encoding for <sep>.  The
// ordering of two finite strings is the same as the ordering of the
// respective characters at the first position where they differ, which in
// turn is the same as the ordering of the encodings of those two
// characters.  Moreover, for every finite string x, F(x) < F(<infinity>).
//
//
// Lexicographically decreasing order
//
// We want a string-to-string mapping G(x) such that for any two strings,
// whether finite or not,
//
//      x < y   =>   G(x) > G(y)
//
// To achieve this, define G(x) to be the inversion of F(x): I(F(x)).  In
// other words, invert every bit in F(x) to get G(x). For example,
//
//        x  = \x00\x13\xff
//      F(x) = \x00\xff\x13\xff\x00\x00\x01  escape \0, \xff, append F(<sep>)
//      G(x) = \xff\x00\xec\x00\xff\xff\xfe  invert every bit in F(x)
//
//        x  = <infinity>
//      F(x) = \xff\xff
//      G(x) = \x00\x00
//
// Another example is
//
//        x            F(x)        G(x) = I(F(x))
//        -            ----        --------------
//        <infinity>   \xff\xff    \x00\x00
//        "foo"        foo\0\1     \x99\x90\x90\xff\xfe
//        "aaa"        aaa\0\1     \x9e\x9e\x9e\xff\xfe
//        "aa"         aa\0\1      \x9e\x9e\xff\xfe
//        ""           \0\1        \xff\xfe
//
// More generally and rigorously, if for any two strings x and y
//
//      F(x) < F(y)   =>   I(F(x)) > I(F(y))                      (1)
//
// it would follow that x < y => G(x) > G(y) because
//
//      x < y   =>   F(x) < F(y)   =>   G(x) = I(F(x)) > I(F(y)) = G(y)
//
// We now show why (1) is true, in two parts.  Notice that for any two
// strings x < y, F(x) is *not* a proper prefix of F(y).  Suppose x is a
// proper prefix of y (say, x="abc" < y="abcd").  F(x) and F(y) diverge at
// the F(<sep>) in F(x) (v. F('d') in the example).  Suppose x is not a
// proper prefix of y (say, x="abce" < y="abd"), F(x) and F(y) diverge at
// their respective encodings of the characters where x and y diverge
// (F('c') v. F('d')).  Finally, if y=<infinity>, we can see that
// F(y)=\xff\xff is not the prefix of F(x) for any finite string x, simply
// by considering all the possible first characters of F(x).
//
// Given that F(x) is not a proper prefix F(y), the order of F(x) and F(y)
// is determined by the byte where F(x) and F(y) diverge.  For example, the
// order of F(x)="eefh" and F(y)="eeg" is determined by their third
// characters.  I(p) inverts each byte in p, which effectively subtracts
// each byte from 0xff.  So, in this example, I('f') > I('g'), and thus
// I(F(x)) > I(F(y)).
//
//
// Implementation
//
// To implement G(x) efficiently, we use C++ template to instantiate two
// versions of the code to produce F(x), one for normal encoding (giving us
// F(x)) and one for inverted encoding (giving us G(x) = I(F(x))).

static const char kEscape1 = '\000';
static const char kNullCharacter = '\xff';  // Combined with kEscape1
static const char kSeparator = '\001';      // Combined with kEscape1

static const char kEscape2 = '\xff';
static const char kFFCharacter = '\000';  // Combined with kEscape2

static const char kEscape1_Separator[2] = {kEscape1, kSeparator};

// Append to "*dest" the "len" bytes starting from "*src".
inline static void AppendBytes(string* dest, const char* src, size_t len) {
  dest->append(src, len);
}

inline bool IsSpecialByte(char c) {
  return (static_cast<unsigned char>(c + 1)) < 2;
}

// Return a pointer to the first byte in the range "[start..limit)"
// whose value is 0 or 255 (kEscape1 or kEscape2).  If no such byte
// exists in the range, returns "limit".
inline const char* SkipToNextSpecialByte(const char* start, const char* limit) {
  // If these constants were ever changed, this routine needs to change
  DCHECK_EQ(kEscape1, 0);
  DCHECK_EQ(kEscape2 & 0xffu, 255u);
  const char* p = start;
  while (p < limit && !IsSpecialByte(*p)) {
    p++;
  }
  return p;
}

// Expose SkipToNextSpecialByte for testing purposes
const char* OrderedCode::TEST_SkipToNextSpecialByte(const char* start,
                                                    const char* limit) {
  return SkipToNextSpecialByte(start, limit);
}

// Helper routine to encode "s" and append to "*dest", escaping special
// characters.
inline static void EncodeStringFragment(string* dest, absl::string_view s) {
  const char* p = s.data();
  const char* limit = p + s.size();
  const char* copy_start = p;
  while (true) {
    p = SkipToNextSpecialByte(p, limit);
    if (p >= limit) break;  // No more special characters that need escaping
    char c = *(p++);
    DCHECK(IsSpecialByte(c));
    if (c == kEscape1) {
      AppendBytes(dest, copy_start, p - copy_start - 1);
      dest->push_back(kEscape1);
      dest->push_back(kNullCharacter);
      copy_start = p;
    } else {
      assert(c == kEscape2);
      AppendBytes(dest, copy_start, p - copy_start - 1);
      dest->push_back(kEscape2);
      dest->push_back(kFFCharacter);
      copy_start = p;
    }
  }
  if (p > copy_start) {
    AppendBytes(dest, copy_start, p - copy_start);
  }
}

void OrderedCode::WriteString(string* dest, absl::string_view s) {
  EncodeStringFragment(dest, s);
  AppendBytes(dest, kEscape1_Separator, 2);
}

void OrderedCode::WriteNumIncreasing(string* dest, uint64 val) {
  // Values are encoded with a single byte length prefix, followed
  // by the actual value in big-endian format with leading 0 bytes
  // dropped.
  unsigned char buf[9];  // 8 bytes for value plus one byte for length
  int len = 0;
  while (val > 0) {
    len++;
    buf[9 - len] = (val & 0xff);
    val >>= 8;
  }
  buf[9 - len - 1] = len;
  len++;
  AppendBytes(dest, reinterpret_cast<const char*>(buf + 9 - len), len);
}

// Parse the encoding of a previously encoded string.
// If parse succeeds, return true, consume encoding from
// "*src", and if result != NULL append the decoded string to "*result".
// Otherwise, return false and leave both undefined.
inline static bool ReadStringInternal(absl::string_view* src, string* result) {
  const char* start = src->data();
  const char* string_limit = src->data() + src->size();

  // We only scan up to "limit-2" since a valid string must end with
  // a two character terminator: 'kEscape1 kSeparator'
  const char* limit = string_limit - 1;
  const char* copy_start = start;
  while (true) {
    start = SkipToNextSpecialByte(start, limit);
    if (start >= limit) break;  // No terminator sequence found
    const char c = *(start++);
    // If inversion is required, instead of inverting 'c', we invert the
    // character constants to which 'c' is compared.  We get the same
    // behavior but save the runtime cost of inverting 'c'.
    DCHECK(IsSpecialByte(c));
    if (c == kEscape1) {
      if (result) {
        AppendBytes(result, copy_start, start - copy_start - 1);
      }
      // kEscape1 kSeparator ends component
      // kEscape1 kNullCharacter represents '\0'
      const char next = *(start++);
      if (next == kSeparator) {
        src->remove_prefix(start - src->data());
        return true;
      } else if (next == kNullCharacter) {
        if (result) {
          *result += '\0';
        }
      } else {
        return false;
      }
      copy_start = start;
    } else {
      assert(c == kEscape2);
      if (result) {
        AppendBytes(result, copy_start, start - copy_start - 1);
      }
      // kEscape2 kFFCharacter represents '\xff'
      // kEscape2 kInfinity is an error
      const char next = *(start++);
      if (next == kFFCharacter) {
        if (result) {
          *result += '\xff';
        }
      } else {
        return false;
      }
      copy_start = start;
    }
  }
  return false;
}

bool OrderedCode::ReadString(absl::string_view* src, string* result) {
  return ReadStringInternal(src, result);
}

bool OrderedCode::ReadNumIncreasing(absl::string_view* src, uint64* result) {
  if (src->empty()) {
    return false;  // Not enough bytes
  }

  // Decode length byte
  const size_t len = static_cast<unsigned char>((*src)[0]);

  // If len > 0 and src is longer than 1, the first byte of "payload"
  // must be non-zero (otherwise the encoding is not minimal).
  // In opt mode, we don't enforce that encodings must be minimal.
  DCHECK(0 == len || src->size() == 1 || (*src)[1] != '\0')
      << "invalid encoding";

  if (len + 1 > src->size() || len > 8) {
    return false;  // Not enough bytes or too many bytes
  }

  if (result) {
    uint64 tmp = 0;
    for (size_t i = 0; i < len; i++) {
      tmp <<= 8;
      tmp |= static_cast<unsigned char>((*src)[1 + i]);
    }
    *result = tmp;
  }
  src->remove_prefix(len + 1);
  return true;
}

void OrderedCode::TEST_Corrupt(string* str, int k) {
  int seen_seps = 0;
  for (size_t i = 0; i + 1 < str->size(); i++) {
    if ((*str)[i] == kEscape1 && (*str)[i + 1] == kSeparator) {
      seen_seps++;
      if (seen_seps == k) {
        (*str)[i + 1] = kSeparator + 1;
        return;
      }
    }
  }
}

// Signed number encoding/decoding /////////////////////////////////////
//
// The format is as follows:
//
// The first bit (the most significant bit of the first byte)
// represents the sign, 0 if the number is negative and
// 1 if the number is >= 0.
//
// Any unbroken sequence of successive bits with the same value as the sign
// bit, up to 9 (the 8th and 9th are the most significant bits of the next
// byte), are size bits that count the number of bytes after the first byte.
// That is, the total length is between 1 and 10 bytes.
//
// The value occupies the bits after the sign bit and the "size bits"
// till the end of the string, in network byte order.  If the number
// is negative, the bits are in 2-complement.
//
//
// Example 1: number 0x424242 -> 4 byte big-endian hex string 0xf0424242:
//
// +---------------+---------------+---------------+---------------+
//  1 1 1 1 0 0 0 0 0 1 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 0
// +---------------+---------------+---------------+---------------+
//  ^ ^ ^ ^   ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
//  | | | |   | | | | | | | | | | | | | | | | | | | | | | | | | | |
//  | | | |   payload: the remaining bits after the sign and size bits
//  | | | |            and the delimiter bit, the value is 0x424242
//  | | | |
//  | size bits: 3 successive bits with the same value as the sign bit
//  |            (followed by a delimiter bit with the opposite value)
//  |            mean that there are 3 bytes after the first byte, 4 total
//  |
//  sign bit: 1 means that the number is non-negative
//
// Example 2: negative number -0x800 -> 2 byte big-endian hex string 0x3800:
//
// +---------------+---------------+
//  0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0
// +---------------+---------------+
//  ^ ^   ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
//  | |   | | | | | | | | | | | | | | | | | | | | | | | | | | |
//  | |   payload: the remaining bits after the sign and size bits and the
//  | |            delimiter bit, 2-complement because of the negative sign,
//  | |            value is ~0x7ff, represents the value -0x800
//  | |
//  | size bits: 1 bit with the same value as the sign bit
//  |            (followed by a delimiter bit with the opposite value)
//  |            means that there is 1 byte after the first byte, 2 total
//  |
//  sign bit: 0 means that the number is negative
//
//
// Compared with the simpler unsigned format used for uint64 numbers,
// this format is more compact for small numbers, namely one byte encodes
// numbers in the range [-64,64), two bytes cover the range [-2^13,2^13), etc.
// In general, n bytes encode numbers in the range [-2^(n*7-1),2^(n*7-1)).
// (The cross-over point for compactness of representation is 8 bytes,
// where this format only covers the range [-2^55,2^55),
// whereas an encoding with sign bit and length in the first byte and
// payload in all following bytes would cover [-2^56,2^56).)

static const int kMaxSigned64Length = 10;

// This array maps encoding length to header bits in the first two bytes.
static const char kLengthToHeaderBits[1 + kMaxSigned64Length][2] = {
    {0, 0},      {'\x80', 0},      {'\xc0', 0},     {'\xe0', 0},
    {'\xf0', 0}, {'\xf8', 0},      {'\xfc', 0},     {'\xfe', 0},
    {'\xff', 0}, {'\xff', '\x80'}, {'\xff', '\xc0'}};

// This array maps encoding lengths to the header bits that overlap with
// the payload and need fixing when reading.
static const uint64 kLengthToMask[1 + kMaxSigned64Length] = {
    0ULL,
    0x80ULL,
    0xc000ULL,
    0xe00000ULL,
    0xf0000000ULL,
    0xf800000000ULL,
    0xfc0000000000ULL,
    0xfe000000000000ULL,
    0xff00000000000000ULL,
    0x8000000000000000ULL,
    0ULL};

// This array maps the number of bits in a number to the encoding
// length produced by WriteSignedNumIncreasing.
// For positive numbers, the number of bits is 1 plus the most significant
// bit position (the highest bit position in a positive int64 is 63).
// For a negative number n, we count the bits in ~n.
// That is, length = kBitsToLength[tsl::Log2Floor64(n < 0 ? ~n : n) + 1].
static const int8 kBitsToLength[1 + 63] = {
    1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4,
    4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7,
    7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10};

// Calculates the encoding length in bytes of the signed number n.
static inline int SignedEncodingLength(int64_t n) {
  return kBitsToLength[tsl::Log2Floor64(n < 0 ? ~n : n) + 1];
}

static void StoreBigEndian64(char* dst, uint64 v) {
  for (int i = 0; i < 8; i++) {
    dst[i] = (v >> (56 - 8 * i)) & 0xff;
  }
}

static uint64 LoadBigEndian64(const char* src) {
  uint64 result = 0;
  for (int i = 0; i < 8; i++) {
    unsigned char c = static_cast<unsigned char>(src[i]);
    result |= static_cast<uint64>(c) << (56 - 8 * i);
  }
  return result;
}

void OrderedCode::WriteSignedNumIncreasing(string* dest, int64_t val) {
  const uint64 x = val < 0 ? ~val : val;
  if (x < 64) {  // fast path for encoding length == 1
    *dest += kLengthToHeaderBits[1][0] ^ val;
    return;
  }
  // buf = val in network byte order, sign extended to 10 bytes
  const char sign_byte = val < 0 ? '\xff' : '\0';
  char buf[10] = {
      sign_byte,
      sign_byte,
  };
  StoreBigEndian64(buf + 2, val);
  static_assert(sizeof(buf) == kMaxSigned64Length, "max length size mismatch");
  const int len = SignedEncodingLength(x);
  DCHECK_GE(len, 2);
  char* const begin = buf + sizeof(buf) - len;
  begin[0] ^= kLengthToHeaderBits[len][0];
  begin[1] ^= kLengthToHeaderBits[len][1];  // ok because len >= 2
  dest->append(begin, len);
}

bool OrderedCode::ReadSignedNumIncreasing(absl::string_view* src,
                                          int64_t* result) {
  if (src->empty()) return false;
  const uint64 xor_mask = (!((*src)[0] & 0x80)) ? ~0ULL : 0ULL;
  const unsigned char first_byte = (*src)[0] ^ (xor_mask & 0xff);

  // now calculate and test length, and set x to raw (unmasked) result
  int len;
  uint64 x;
  if (first_byte != 0xff) {
    len = 7 - tsl::Log2Floor64(first_byte ^ 0xff);
    if (src->size() < static_cast<size_t>(len)) return false;
    x = xor_mask;  // sign extend using xor_mask
    for (int i = 0; i < len; ++i)
      x = (x << 8) | static_cast<unsigned char>((*src)[i]);
  } else {
    len = 8;
    if (src->size() < static_cast<size_t>(len)) return false;
    const unsigned char second_byte = (*src)[1] ^ (xor_mask & 0xff);
    if (second_byte >= 0x80) {
      if (second_byte < 0xc0) {
        len = 9;
      } else {
        const unsigned char third_byte = (*src)[2] ^ (xor_mask & 0xff);
        if (second_byte == 0xc0 && third_byte < 0x80) {
          len = 10;
        } else {
          return false;  // either len > 10 or len == 10 and #bits > 63
        }
      }
      if (src->size() < static_cast<size_t>(len)) return false;
    }
    x = LoadBigEndian64(src->data() + len - 8);
  }

  x ^= kLengthToMask[len];  // remove spurious header bits

  DCHECK_EQ(len, SignedEncodingLength(x)) << "invalid encoding";

  if (result) *result = x;
  src->remove_prefix(len);
  return true;
}

}  // namespace strings
}  // namespace tensorflow
