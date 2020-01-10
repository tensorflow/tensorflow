#ifndef TENSORFLOW_LIB_STRINGS_STR_UTIL_H_
#define TENSORFLOW_LIB_STRINGS_STR_UTIL_H_

#include <string>
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/strcat.h"

// Basic string utility routines
namespace tensorflow {
namespace str_util {

// Returns a version of 'src' where unprintable characters have been
// escaped using C-style escape sequences.
string CEscape(const string& src);

// Copies "source" to "dest", rewriting C-style escape sequences --
// '\n', '\r', '\\', '\ooo', etc -- to their ASCII equivalents.
//
// Errors: Sets the description of the first encountered error in
// 'error'. To disable error reporting, set 'error' to NULL.
//
// NOTE: Does not support \u or \U!
bool CUnescape(StringPiece source, string* dest, string* error);

// If "text" can be successfully parsed as the ASCII representation of
// an integer, sets "*val" to the value and returns true.  Otherwise,
// returns false.
bool NumericParse32(const string& text, int32* val);

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

// If "*s" starts with "expected", consume it and return true.
// Otherwise, return false.
bool ConsumePrefix(StringPiece* s, StringPiece expected);

// Return lower-cased version of s.
string Lowercase(StringPiece s);

// Return upper-cased version of s.
string Uppercase(StringPiece s);

// Capitalize first character of each word in "*s".  "delimiters" is a
// set of characters that can be used as word boundaries.
void TitlecaseString(string* s, StringPiece delimiters);

// Join functionality
template <typename T>
string Join(const std::vector<T>& s, const char* sep);
template <typename T>
string Join(const gtl::ArraySlice<T>& s, const char* sep);

struct AllowEmpty {
  bool operator()(StringPiece sp) const { return true; }
};
struct SkipEmpty {
  bool operator()(StringPiece sp) const { return !sp.empty(); }
};
struct SkipWhitespace {
  bool operator()(StringPiece sp) const {
    RemoveTrailingWhitespace(&sp);
    return !sp.empty();
  }
};

std::vector<string> Split(StringPiece text, char delim);
template <typename Predicate>
std::vector<string> Split(StringPiece text, char delim, Predicate p);

// Split "text" at "delim" characters, and parse each component as
// an integer.  If successful, adds the individual numbers in order
// to "*result" and returns true.  Otherwise returns false.
bool SplitAndParseAsInts(StringPiece text, char delim,
                         std::vector<int32>* result);

// ------------------------------------------------------------------
// Implementation details below
namespace internal {
template <typename T>
string JoinHelper(typename gtl::ArraySlice<T>::const_iterator begin,
                  typename gtl::ArraySlice<T>::const_iterator end,
                  const char* sep) {
  string result;
  bool first = true;
  for (typename gtl::ArraySlice<T>::const_iterator it = begin; it != end;
       ++it) {
    tensorflow::strings::StrAppend(&result, (first ? "" : sep), *it);
    first = false;
  }
  return result;
}
}  // namespace internal

template <typename T>
string Join(const std::vector<T>& s, const char* sep) {
  return Join<T>(gtl::ArraySlice<T>(s), sep);
}

template <typename T>
string Join(const gtl::ArraySlice<T>& s, const char* sep) {
  return internal::JoinHelper<T>(s.begin(), s.end(), sep);
}

inline std::vector<string> Split(StringPiece text, char delim) {
  return Split(text, delim, AllowEmpty());
}

template <typename Predicate>
std::vector<string> Split(StringPiece text, char delim, Predicate p) {
  std::vector<string> result;
  int token_start = 0;
  if (!text.empty()) {
    for (int i = 0; i < text.size() + 1; i++) {
      if ((i == text.size()) || (text[i] == delim)) {
        StringPiece token(text.data() + token_start, i - token_start);
        if (p(token)) {
          result.push_back(token.ToString());
        }
        token_start = i + 1;
      }
    }
  }
  return result;
}

}  // namespace str_util
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_STRINGS_STR_UTIL_H_
