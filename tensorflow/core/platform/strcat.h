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
#ifndef TENSORFLOW_CORE_PLATFORM_STRCAT_H_
#define TENSORFLOW_CORE_PLATFORM_STRCAT_H_

#include <string>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

// The AlphaNum type was designed to be used as the parameter type for StrCat().
// Any routine accepting either a string or a number may accept it.
// The basic idea is that by accepting a "const AlphaNum &" as an argument
// to your function, your callers will automatically convert bools, integers,
// and floating point values to strings for you.
//
// NOTE: Use of AlphaNum outside of the //strings package is unsupported except
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
// of Printf("%04x", my_int) is StrCat(Hex(my_int, strings::kZeroPad4))
//
// This class has implicit constructors.
namespace tensorflow {
namespace strings {

enum PadSpec {
  kNoPad = 1,
  kZeroPad2,
  kZeroPad3,
  kZeroPad4,
  kZeroPad5,
  kZeroPad6,
  kZeroPad7,
  kZeroPad8,
  kZeroPad9,
  kZeroPad10,
  kZeroPad11,
  kZeroPad12,
  kZeroPad13,
  kZeroPad14,
  kZeroPad15,
  kZeroPad16
};

struct Hex {
  uint64 value;
  enum PadSpec spec;
  template <class Int>
  explicit Hex(Int v, PadSpec s = kNoPad) : spec(s) {
    // Prevent sign-extension by casting integers to
    // their unsigned counterparts.
    static_assert(
        sizeof(v) == 1 || sizeof(v) == 2 || sizeof(v) == 4 || sizeof(v) == 8,
        "Unknown integer type");
    value = sizeof(v) == 1
                ? static_cast<uint8>(v)
                : sizeof(v) == 2 ? static_cast<uint16>(v)
                                 : sizeof(v) == 4 ? static_cast<uint32>(v)
                                                  : static_cast<uint64>(v);
  }
};

class AlphaNum {
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

  AlphaNum(Hex hex);               // NOLINT(runtime/explicit)

  AlphaNum(const char *c_str) : piece_(c_str) {}   // NOLINT(runtime/explicit)
  AlphaNum(const StringPiece &pc) : piece_(pc) {}  // NOLINT(runtime/explicit)
  AlphaNum(const tensorflow::string &str)          // NOLINT(runtime/explicit)
      : piece_(str) {}
  AlphaNum(const tensorflow::tstring &str)  // NOLINT(runtime/explicit)
      : piece_(str) {}
  template <typename A>
  AlphaNum(const std::basic_string<char, std::char_traits<char>, A> &str)
      : piece_(str) {}  // NOLINT(runtime/explicit)

  StringPiece::size_type size() const { return piece_.size(); }
  const char *data() const { return piece_.data(); }
  StringPiece Piece() const { return piece_; }

 private:
  StringPiece piece_;
  char digits_[kFastToBufferSize];

  // Use ":" not ':'
  AlphaNum(char c);  // NOLINT(runtime/explicit)

  TF_DISALLOW_COPY_AND_ASSIGN(AlphaNum);
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
string StrCat(const AlphaNum &a) TF_MUST_USE_RESULT;
string StrCat(const AlphaNum &a, const AlphaNum &b) TF_MUST_USE_RESULT;
string StrCat(const AlphaNum &a, const AlphaNum &b,
              const AlphaNum &c) TF_MUST_USE_RESULT;
string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c,
              const AlphaNum &d) TF_MUST_USE_RESULT;

namespace internal {

// Do not call directly - this is not part of the public API.
string CatPieces(std::initializer_list<StringPiece> pieces);
void AppendPieces(string *dest, std::initializer_list<StringPiece> pieces);

}  // namespace internal

// Support 5 or more arguments
template <typename... AV>
string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c,
              const AlphaNum &d, const AlphaNum &e,
              const AV &... args) TF_MUST_USE_RESULT;

template <typename... AV>
string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c,
              const AlphaNum &d, const AlphaNum &e, const AV &... args) {
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

void StrAppend(string *dest, const AlphaNum &a);
void StrAppend(string *dest, const AlphaNum &a, const AlphaNum &b);
void StrAppend(string *dest, const AlphaNum &a, const AlphaNum &b,
               const AlphaNum &c);
void StrAppend(string *dest, const AlphaNum &a, const AlphaNum &b,
               const AlphaNum &c, const AlphaNum &d);

// Support 5 or more arguments
template <typename... AV>
inline void StrAppend(string *dest, const AlphaNum &a, const AlphaNum &b,
                      const AlphaNum &c, const AlphaNum &d, const AlphaNum &e,
                      const AV &... args) {
  internal::AppendPieces(dest,
                         {a.Piece(), b.Piece(), c.Piece(), d.Piece(), e.Piece(),
                          static_cast<const AlphaNum &>(args).Piece()...});
}

}  // namespace strings
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STRCAT_H_
