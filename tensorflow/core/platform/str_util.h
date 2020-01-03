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

#ifndef TENSORFLOW_CORE_PLATFORM_STR_UTIL_H_
#define TENSORFLOW_CORE_PLATFORM_STR_UTIL_H_

#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

// Basic string utility routines
namespace tensorflow {
namespace str_util {

// Returns a version of 'src' where unprintable characters have been
// escaped using C-style escape sequences.
string CEscape(StringPiece src);

// Copies "source" to "dest", rewriting C-style escape sequences --
// '\n', '\r', '\\', '\ooo', etc -- to their ASCII equivalents.
//
// Errors: Sets the description of the first encountered error in
// 'error'. To disable error reporting, set 'error' to NULL.
//
// NOTE: Does not support \u or \U!
bool CUnescape(StringPiece source, string* dest, string* error);

// Removes any trailing whitespace from "*s".
void StripTrailingWhitespace(string* s);

// Removes leading ascii_isspace() characters.
// Returns number of characters removed.
size_t RemoveLeadingWhitespace(StringPiece* text);

// Removes trailing ascii_isspace() characters.
// Returns number of characters removed.
size_t RemoveTrailingWhitespace(StringPiece* text);

// Removes leading and trailing ascii_isspace() chars.
// Returns number of chars removed.
size_t RemoveWhitespaceContext(StringPiece* text);

// Consume a leading positive integer value.  If any digits were
// found, store the value of the leading unsigned number in "*val",
// advance "*s" past the consumed number, and return true.  If
// overflow occurred, returns false.  Otherwise, returns false.
bool ConsumeLeadingDigits(StringPiece* s, uint64* val);

// Consume a leading token composed of non-whitespace characters only.
// If *s starts with a non-zero number of non-whitespace characters, store
// them in *val, advance *s past them, and return true.  Else return false.
bool ConsumeNonWhitespace(StringPiece* s, StringPiece* val);

// If "*s" starts with "expected", consume it and return true.
// Otherwise, return false.
bool ConsumePrefix(StringPiece* s, StringPiece expected);

// If "*s" ends with "expected", remove it and return true.
// Otherwise, return false.
bool ConsumeSuffix(StringPiece* s, StringPiece expected);

// If "s" starts with "expected", return a view into "s" after "expected" but
// keep "s" unchanged.
// Otherwise, return the original "s".
TF_MUST_USE_RESULT StringPiece StripPrefix(StringPiece s, StringPiece expected);

// If "s" ends with "expected", return a view into "s" until "expected" but
// keep "s" unchanged.
// Otherwise, return the original "s".
TF_MUST_USE_RESULT StringPiece StripSuffix(StringPiece s, StringPiece expected);

// Return lower-cased version of s.
string Lowercase(StringPiece s);

// Return upper-cased version of s.
string Uppercase(StringPiece s);

// Capitalize first character of each word in "*s".  "delimiters" is a
// set of characters that can be used as word boundaries.
void TitlecaseString(string* s, StringPiece delimiters);

// Replaces the first occurrence (if replace_all is false) or all occurrences
// (if replace_all is true) of oldsub in s with newsub.
string StringReplace(StringPiece s, StringPiece oldsub, StringPiece newsub,
                     bool replace_all);

// Join functionality
template <typename T>
string Join(const T& s, const char* sep) {
  return absl::StrJoin(s, sep);
}

// A variant of Join where for each element of "s", f(&dest_string, elem)
// is invoked (f is often constructed with a lambda of the form:
//   [](string* result, ElemType elem)
template <typename T, typename Formatter>
string Join(const T& s, const char* sep, Formatter f) {
  return absl::StrJoin(s, sep, f);
}

struct AllowEmpty {
  bool operator()(StringPiece sp) const { return true; }
};
struct SkipEmpty {
  bool operator()(StringPiece sp) const { return !sp.empty(); }
};
struct SkipWhitespace {
  bool operator()(StringPiece sp) const {
    return !absl::StripTrailingAsciiWhitespace(sp).empty();
  }
};

// Split strings using any of the supplied delimiters. For example:
// Split("a,b.c,d", ".,") would return {"a", "b", "c", "d"}.
inline std::vector<string> Split(StringPiece text, StringPiece delims) {
  return text.empty() ? std::vector<string>()
                      : absl::StrSplit(text, absl::ByAnyChar(delims));
}

template <typename Predicate>
std::vector<string> Split(StringPiece text, StringPiece delims, Predicate p) {
  return text.empty() ? std::vector<string>()
                      : absl::StrSplit(text, absl::ByAnyChar(delims), p);
}

inline std::vector<string> Split(StringPiece text, char delim) {
  return text.empty() ? std::vector<string>() : absl::StrSplit(text, delim);
}

template <typename Predicate>
std::vector<string> Split(StringPiece text, char delim, Predicate p) {
  return text.empty() ? std::vector<string>() : absl::StrSplit(text, delim, p);
}

// StartsWith()
//
// Returns whether a given string `text` begins with `prefix`.
bool StartsWith(StringPiece text, StringPiece prefix);

// EndsWith()
//
// Returns whether a given string `text` ends with `suffix`.
bool EndsWith(StringPiece text, StringPiece suffix);

// StrContains()
//
// Returns whether a given string `haystack` contains the substring `needle`.
bool StrContains(StringPiece haystack, StringPiece needle);

// Returns the length of the given null-terminated byte string 'str'.
// Returns 'string_max_len' if the null character was not found in the first
// 'string_max_len' bytes of 'str'.
size_t Strnlen(const char* str, const size_t string_max_len);

//   ----- NON STANDARD, TF SPECIFIC METHOD -----
// Converts "^2ILoveYou!" to "i_love_you_". More specifically:
// - converts all non-alphanumeric characters to underscores
// - replaces each occurrence of a capital letter (except the very
//   first character and if there is already an '_' before it) with '_'
//   followed by this letter in lower case
// - Skips leading non-alpha characters
// This method is useful for producing strings matching "[a-z][a-z0-9_]*"
// as required by OpDef.ArgDef.name. The resulting string is either empty or
// matches this regex.
string ArgDefCase(StringPiece s);

}  // namespace str_util
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STR_UTIL_H_
