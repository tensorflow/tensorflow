/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace strings {

AlphaNum gEmptyAlphaNum("");

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
static void Append1(string *out, const AlphaNum &x) {
  out->append(x.data(), x.size());
}

static void Append2(string *out, const AlphaNum &x1, const AlphaNum &x2) {
  out->append(x1.data(), x1.size());
  out->append(x2.data(), x2.size());
}

static void Append3(string *out, const AlphaNum &x1, const AlphaNum &x2,
                     const AlphaNum &x3) {
  out->append(x1.data(), x1.size());
  out->append(x2.data(), x2.size());
  out->append(x3.data(), x3.size());
}

static void Append4(string *out, const AlphaNum &x1, const AlphaNum &x2,
    const AlphaNum &x3, const AlphaNum &x4)
{
  out->append(x1.data(), x1.size());
  out->append(x2.data(), x2.size());
  out->append(x3.data(), x3.size());
  out->append(x4.data(), x4.size());
}

string StrCat(const AlphaNum &a, const AlphaNum &b) {
  string result;
  gtl::STLStringResizeUninitialized(&result, a.size() + b.size());
  Append2(&result, a, b);
  return result;
}

string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c) {
  string result;
  gtl::STLStringResizeUninitialized(&result, a.size() + b.size() + c.size());
  Append3(&result, a, b, c);
  return result;
}

string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c,
              const AlphaNum &d) {
  string result;
  gtl::STLStringResizeUninitialized(&result,
                                    a.size() + b.size() + c.size() + d.size());
  Append4(&result, a, b, c, d);
  return result;
}

namespace internal {

// Do not call directly - these are not part of the public API.
string CatPieces(std::initializer_list<StringPiece> pieces) {
  string result;
  size_t total_size = 0;
  for (const StringPiece piece : pieces) total_size += piece.size();
  gtl::STLStringResizeUninitialized(&result, total_size);

  for (const StringPiece piece : pieces) {
    result.append(piece.data(), piece.size());
  }
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

  for (const StringPiece piece : pieces) {
    result->append(piece.data(), piece.size());
  }
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
  Append2(result, a, b);
}

void StrAppend(string *result, const AlphaNum &a, const AlphaNum &b,
               const AlphaNum &c) {
  DCHECK_NO_OVERLAP(*result, a);
  DCHECK_NO_OVERLAP(*result, b);
  DCHECK_NO_OVERLAP(*result, c);
  string::size_type old_size = result->size();
  gtl::STLStringResizeUninitialized(result,
                                    old_size + a.size() + b.size() + c.size());
  Append3(result, a, b, c);
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
  Append4(result, a, b, c, d);
}

}  // namespace strings
}  // namespace tensorflow
