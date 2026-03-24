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
#include "tsl/platform/numbers.h"
#include "tsl/platform/stringpiece.h"

// The AlphaNum type was designed to be used as the parameter type for StrCat().
// Any routine accepting either a string or a number may accept it.
// The basic idea is that by accepting a "const absl::AlphaNum& " as an argument
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
using AlphaNum ABSL_DEPRECATE_AND_INLINE() = absl::AlphaNum;

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
ABSL_DEPRECATE_AND_INLINE()
inline std::string StrCat(const absl::AlphaNum& a) { return absl::StrCat(a); }
ABSL_DEPRECATE_AND_INLINE()
inline std::string StrCat(const absl::AlphaNum& a, const absl::AlphaNum& b) {
  return absl::StrCat(a, b);
}
ABSL_DEPRECATE_AND_INLINE()
inline std::string StrCat(const absl::AlphaNum& a, const absl::AlphaNum& b,
                          const absl::AlphaNum& c) {
  return absl::StrCat(a, b, c);
}
ABSL_DEPRECATE_AND_INLINE()
inline std::string StrCat(const absl::AlphaNum& a, const absl::AlphaNum& b,
                          const absl::AlphaNum& c, const absl::AlphaNum& d) {
  return absl::StrCat(a, b, c, d);
}

// Support 5 or more arguments
template <typename... AV>
ABSL_DEPRECATED("Use absl::StrCat() instead.")
std::string StrCat(const absl::AlphaNum& a, const absl::AlphaNum& b,
                   const absl::AlphaNum& c, const absl::AlphaNum& d,
                   const absl::AlphaNum& e, const AV&... args) {
  return absl::StrCat(a, b, c, d, e, args...);
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

ABSL_DEPRECATE_AND_INLINE()
inline void StrAppend(std::string* dest, const absl::AlphaNum& a) {
  absl::StrAppend(dest, a);
}
ABSL_DEPRECATE_AND_INLINE()
inline void StrAppend(std::string* dest, const absl::AlphaNum& a,
                      const absl::AlphaNum& b) {
  absl::StrAppend(dest, a, b);
}
ABSL_DEPRECATE_AND_INLINE()
inline void StrAppend(std::string* dest, const absl::AlphaNum& a,
                      const absl::AlphaNum& b, const absl::AlphaNum& c) {
  absl::StrAppend(dest, a, b, c);
}
ABSL_DEPRECATE_AND_INLINE()
inline void StrAppend(std::string* dest, const absl::AlphaNum& a,
                      const absl::AlphaNum& b, const absl::AlphaNum& c,
                      const absl::AlphaNum& d) {
  absl::StrAppend(dest, a, b, c, d);
}

// Support 5 or more arguments
template <typename... AV>
ABSL_DEPRECATED("Use absl::StrAppend() instead.")
inline void StrAppend(std::string* dest, const absl::AlphaNum& a,
                      const absl::AlphaNum& b, const absl::AlphaNum& c,
                      const absl::AlphaNum& d, const absl::AlphaNum& e,
                      const AV&... args) {
  absl::StrAppend(dest, a, b, c, d, e, args...);
}

}  // namespace strings
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_STRCAT_H_
