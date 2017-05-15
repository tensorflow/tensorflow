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

#include "tensorflow/core/lib/strings/strcat.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace strings {

AlphaNum::AlphaNum(const Eigen::half &f)
    : piece_(digits_, strlen(FloatToBuffer(static_cast<float>(f), digits_))) {}

AlphaNum::AlphaNum(Hex hex) {
  char *const end = &digits_[kFastToBufferSize];
  char *writer = end;
  uint64 value = hex.value;
  uint64 width = hex.spec;
  // We accomplish minimum width by OR'ing in 0x10000 to the user's value,
  // where 0x10000 is the smallest hex number that is as wide as the user
  // asked for.
  uint64 mask = ((static_cast<uint64>(1) << (width - 1) * 4)) | value;
  static const char hexdigits[] = "0123456789abcdef";
  do {
    *--writer = hexdigits[value & 0xF];
    value >>= 4;
    mask >>= 4;
  } while (mask != 0);
  piece_.set(writer, end - writer);
}

// ----------------------------------------------------------------------
// StrCat()
//    This merges the given strings or integers, with no delimiter.  This
//    is designed to be the fastest possible way to construct a string out
//    of a mix of raw C strings, StringPieces, strings, and integer values.
// ----------------------------------------------------------------------

// Append is merely a version of memcpy that returns the address of the byte
// after the area just overwritten.  It comes in multiple flavors to minimize
// call overhead.
static char *Append1(char *out, const AlphaNum &x) {
  memcpy(out, x.data(), x.size());
  return out + x.size();
}

static char *Append2(char *out, const AlphaNum &x1, const AlphaNum &x2) {
  memcpy(out, x1.data(), x1.size());
  out += x1.size();

  memcpy(out, x2.data(), x2.size());
  return out + x2.size();
}

static char *Append4(char *out, const AlphaNum &x1, const AlphaNum &x2,
                     const AlphaNum &x3, const AlphaNum &x4) {
  memcpy(out, x1.data(), x1.size());
  out += x1.size();

  memcpy(out, x2.data(), x2.size());
  out += x2.size();

  memcpy(out, x3.data(), x3.size());
  out += x3.size();

  memcpy(out, x4.data(), x4.size());
  return out + x4.size();
}

string StrCat(const AlphaNum &a) { return string(a.data(), a.size()); }

string StrCat(const AlphaNum &a, const AlphaNum &b) {
  string result;
  gtl::STLStringResizeUninitialized(&result, a.size() + b.size());
  char *const begin = &*result.begin();
  char *out = Append2(begin, a, b);
  DCHECK_EQ(out, begin + result.size());
  return result;
}

string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c) {
  string result;
  gtl::STLStringResizeUninitialized(&result, a.size() + b.size() + c.size());
  char *const begin = &*result.begin();
  char *out = Append2(begin, a, b);
  out = Append1(out, c);
  DCHECK_EQ(out, begin + result.size());
  return result;
}

string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c,
              const AlphaNum &d) {
  string result;
  gtl::STLStringResizeUninitialized(&result,
                                    a.size() + b.size() + c.size() + d.size());
  char *const begin = &*result.begin();
  char *out = Append4(begin, a, b, c, d);
  DCHECK_EQ(out, begin + result.size());
  return result;
}

namespace internal {

// Do not call directly - these are not part of the public API.
string CatPieces(std::initializer_list<StringPiece> pieces) {
  string result;
  size_t total_size = 0;
  for (const StringPiece piece : pieces) total_size += piece.size();
  gtl::STLStringResizeUninitialized(&result, total_size);

  char *const begin = &*result.begin();
  char *out = begin;
  for (const StringPiece piece : pieces) {
    const size_t this_size = piece.size();
    memcpy(out, piece.data(), this_size);
    out += this_size;
  }
  DCHECK_EQ(out, begin + result.size());
  return result;
}

// It's possible to call StrAppend with a StringPiece that is itself a fragment
// of the string we're appending to.  However the results of this are random.
// Therefore, check for this in debug mode.  Use unsigned math so we only have
// to do one comparison.
#define DCHECK_NO_OVERLAP(dest, src) \
  DCHECK_GE(uintptr_t((src).data() - (dest).data()), uintptr_t((dest).size()))

void AppendPieces(string *result, std::initializer_list<StringPiece> pieces) {
  size_t old_size = result->size();
  size_t total_size = old_size;
  for (const StringPiece piece : pieces) {
    DCHECK_NO_OVERLAP(*result, piece);
    total_size += piece.size();
  }
  gtl::STLStringResizeUninitialized(result, total_size);

  char *const begin = &*result->begin();
  char *out = begin + old_size;
  for (const StringPiece piece : pieces) {
    const size_t this_size = piece.size();
    memcpy(out, piece.data(), this_size);
    out += this_size;
  }
  DCHECK_EQ(out, begin + result->size());
}

}  // namespace internal

void StrAppend(string *result, const AlphaNum &a) {
  DCHECK_NO_OVERLAP(*result, a);
  result->append(a.data(), a.size());
}

void StrAppend(string *result, const AlphaNum &a, const AlphaNum &b) {
  DCHECK_NO_OVERLAP(*result, a);
  DCHECK_NO_OVERLAP(*result, b);
  string::size_type old_size = result->size();
  gtl::STLStringResizeUninitialized(result, old_size + a.size() + b.size());
  char *const begin = &*result->begin();
  char *out = Append2(begin + old_size, a, b);
  DCHECK_EQ(out, begin + result->size());
}

void StrAppend(string *result, const AlphaNum &a, const AlphaNum &b,
               const AlphaNum &c) {
  DCHECK_NO_OVERLAP(*result, a);
  DCHECK_NO_OVERLAP(*result, b);
  DCHECK_NO_OVERLAP(*result, c);
  string::size_type old_size = result->size();
  gtl::STLStringResizeUninitialized(result,
                                    old_size + a.size() + b.size() + c.size());
  char *const begin = &*result->begin();
  char *out = Append2(begin + old_size, a, b);
  out = Append1(out, c);
  DCHECK_EQ(out, begin + result->size());
}

void StrAppend(string *result, const AlphaNum &a, const AlphaNum &b,
               const AlphaNum &c, const AlphaNum &d) {
  DCHECK_NO_OVERLAP(*result, a);
  DCHECK_NO_OVERLAP(*result, b);
  DCHECK_NO_OVERLAP(*result, c);
  DCHECK_NO_OVERLAP(*result, d);
  string::size_type old_size = result->size();
  gtl::STLStringResizeUninitialized(
      result, old_size + a.size() + b.size() + c.size() + d.size());
  char *const begin = &*result->begin();
  char *out = Append4(begin + old_size, a, b, c, d);
  DCHECK_EQ(out, begin + result->size());
}

}  // namespace strings
}  // namespace tensorflow
