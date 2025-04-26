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

// #status: RECOMMENDED
// #category: operations on strings
// #summary: Merges strings or numbers with no delimiter.
//
#ifndef TENSORFLOW_TSL_PLATFORM_STRCAT_H_
#define TENSORFLOW_TSL_PLATFORM_STRCAT_H_

#include <string>

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/stringpiece.h"

// The AlphaNum type was designed to be used as the parameter type for StrCat().
// Any routine accepting either a string or a number may accept it.
// The basic idea is that by accepting a "const AlphaNum &" as an argument
// to your function, your callers will automatically convert bools, integers,
// and floating point values to strings for you.
//
// NOTE: Use of AlphaNum outside of the "strings" package is unsupported except
// for the specific case of function parameters of type "AlphaNum" or "const
// AlphaNum &". In particular, instantiating AlphaNum directly as a stack
// variable is not supported.
//
// Conversion from 8-bit values is not accepted because if it were, then an
// attempt to pass ':' instead of ":" might result in a 58 ending up in your
// result.
//
// Bools convert to "0" or "1".
//
// Floating point values are converted to a string which, if passed to strtod(),
// would produce the exact same original double (except in case of NaN; all NaNs
// are considered the same value). We try to keep the string short but it's not
// guaranteed to be as short as possible.
//
// You can convert to Hexadecimal output rather than Decimal output using Hex.
// To do this, pass strings::Hex(my_int) as a parameter to StrCat. You may
// specify a minimum field width using a separate parameter, so the equivalent
// of Printf("%04x", my_int) is StrCat(Hex(my_int, absl::kZeroPad4))
//
// This class has implicit constructors.
namespace tsl {
namespace strings {

using PadSpec ABSL_DEPRECATE_AND_INLINE() = absl::PadSpec;
using absl::kNoPad;
using absl::kZeroPad10;
using absl::kZeroPad11;
using absl::kZeroPad12;
using absl::kZeroPad13;
using absl::kZeroPad14;
using absl::kZeroPad15;
using absl::kZeroPad16;
using absl::kZeroPad2;
using absl::kZeroPad3;
using absl::kZeroPad4;
using absl::kZeroPad5;
using absl::kZeroPad6;
using absl::kZeroPad7;
using absl::kZeroPad8;
using absl::kZeroPad9;
using Hex ABSL_DEPRECATE_AND_INLINE() = absl::Hex;

class AlphaNum {
  // NOLINTBEGIN(google-explicit-constructor)
 public:
  // No bool ctor -- bools convert to an integral type.
  // A bool ctor would also convert incoming pointers (bletch).
  AlphaNum(int i32)  // NOLINT(runtime/explicit)
      : piece_(digits_, FastInt32ToBufferLeft(i32, digits_)) {}
  AlphaNum(unsigned int u32)  // NOLINT(runtime/explicit)
      : piece_(digits_, FastUInt32ToBufferLeft(u32, digits_)) {}
  AlphaNum(long x)  // NOLINT(runtime/explicit)
      : piece_(digits_, FastInt64ToBufferLeft(x, digits_)) {}
  AlphaNum(unsigned long x)  // NOLINT(runtime/explicit)
      : piece_(digits_, FastUInt64ToBufferLeft(x, digits_)) {}
  AlphaNum(long long int i64)  // NOLINT(runtime/explicit)
      : piece_(digits_, FastInt64ToBufferLeft(i64, digits_)) {}
  AlphaNum(unsigned long long int u64)  // NOLINT(runtime/explicit)
      : piece_(digits_, FastUInt64ToBufferLeft(u64, digits_)) {}

  AlphaNum(float f)  // NOLINT(runtime/explicit)
      : piece_(digits_, FloatToBuffer(f, digits_)) {}
  AlphaNum(double f)  // NOLINT(runtime/explicit)
      : piece_(digits_, DoubleToBuffer(f, digits_)) {}
  AlphaNum(bfloat16 bf)  // NOLINT(runtime/explicit)
      : piece_(digits_, FloatToBuffer(static_cast<float>(bf), digits_)) {}

  AlphaNum(Hex hex);  // NOLINT(runtime/explicit)

  AlphaNum(const char *c_str) : piece_(c_str) {}  // NOLINT(runtime/explicit)
  AlphaNum(const absl::string_view &pc)
      : piece_(pc) {}               // NOLINT(runtime/explicit)
  AlphaNum(const std::string &str)  // NOLINT(runtime/explicit)
      : piece_(str) {}
  AlphaNum(const tstring &str)  // NOLINT(runtime/explicit)
      : piece_(str) {}
  template <typename A>
  AlphaNum(const std::basic_string<char, std::char_traits<char>, A> &str)
      : piece_(str) {}  // NOLINT(runtime/explicit)

  absl::string_view::size_type size() const { return piece_.size(); }
  const char *data() const { return piece_.data(); }
  absl::string_view Piece() const { return piece_; }

 private:
  absl::string_view piece_;
  char digits_[kFastToBufferSize];

  // Use ":" not ':'
  AlphaNum(char c);  // NOLINT(runtime/explicit)

  // NOLINTEND(google-explicit-constructor)
  AlphaNum(const AlphaNum &) = delete;
  void operator=(const AlphaNum &) = delete;
};

// ----------------------------------------------------------------------
// StrCat()
//    This merges the given strings or numbers, with no delimiter.  This
//    is designed to be the fastest possible way to construct a string out
//    of a mix of raw C strings, StringPieces, strings, bool values,
//    and numeric values.
//
//    Don't use this for user-visible strings.  The localization process
//    works poorly on strings built up out of fragments.
//
//    For clarity and performance, don't use StrCat when appending to a
//    string.  In particular, avoid using any of these (anti-)patterns:
//      str.append(StrCat(...))
//      str += StrCat(...)
//      str = StrCat(str, ...)
//    where the last is the worse, with the potential to change a loop
//    from a linear time operation with O(1) dynamic allocations into a
//    quadratic time operation with O(n) dynamic allocations.  StrAppend
//    is a better choice than any of the above, subject to the restriction
//    of StrAppend(&str, a, b, c, ...) that none of the a, b, c, ... may
//    be a reference into str.
// ----------------------------------------------------------------------

// For performance reasons, we have specializations for <= 4 args.
std::string StrCat(const AlphaNum &a) TF_MUST_USE_RESULT;
std::string StrCat(const AlphaNum &a, const AlphaNum &b) TF_MUST_USE_RESULT;
std::string StrCat(const AlphaNum &a, const AlphaNum &b,
                   const AlphaNum &c) TF_MUST_USE_RESULT;
std::string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c,
                   const AlphaNum &d) TF_MUST_USE_RESULT;

namespace internal {

// Do not call directly - this is not part of the public API.
std::string CatPieces(std::initializer_list<absl::string_view> pieces);
void AppendPieces(std::string *dest,
                  std::initializer_list<absl::string_view> pieces);

}  // namespace internal

// Support 5 or more arguments
template <typename... AV>
std::string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c,
                   const AlphaNum &d, const AlphaNum &e,
                   const AV &...args) TF_MUST_USE_RESULT;

template <typename... AV>
std::string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c,
                   const AlphaNum &d, const AlphaNum &e, const AV &...args) {
  return internal::CatPieces({a.Piece(), b.Piece(), c.Piece(), d.Piece(),
                              e.Piece(),
                              static_cast<const AlphaNum &>(args).Piece()...});
}

// ----------------------------------------------------------------------
// StrAppend()
//    Same as above, but adds the output to the given string.
//    WARNING: For speed, StrAppend does not try to check each of its input
//    arguments to be sure that they are not a subset of the string being
//    appended to.  That is, while this will work:
//
//    string s = "foo";
//    s += s;
//
//    This will not (necessarily) work:
//
//    string s = "foo";
//    StrAppend(&s, s);
//
//    Note: while StrCat supports appending up to 26 arguments, StrAppend
//    is currently limited to 9.  That's rarely an issue except when
//    automatically transforming StrCat to StrAppend, and can easily be
//    worked around as consecutive calls to StrAppend are quite efficient.
// ----------------------------------------------------------------------

void StrAppend(std::string *dest, const AlphaNum &a);
void StrAppend(std::string *dest, const AlphaNum &a, const AlphaNum &b);
void StrAppend(std::string *dest, const AlphaNum &a, const AlphaNum &b,
               const AlphaNum &c);
void StrAppend(std::string *dest, const AlphaNum &a, const AlphaNum &b,
               const AlphaNum &c, const AlphaNum &d);

// Support 5 or more arguments
template <typename... AV>
inline void StrAppend(std::string *dest, const AlphaNum &a, const AlphaNum &b,
                      const AlphaNum &c, const AlphaNum &d, const AlphaNum &e,
                      const AV &...args) {
  internal::AppendPieces(dest,
                         {a.Piece(), b.Piece(), c.Piece(), d.Piece(), e.Piece(),
                          static_cast<const AlphaNum &>(args).Piece()...});
}

}  // namespace strings
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_STRCAT_H_
