/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLATBUFFERS_UTIL_H_
#define FLATBUFFERS_UTIL_H_

#include "flatbuffers/base.h"

#include <errno.h>

#ifndef FLATBUFFERS_PREFER_PRINTF
#  include <sstream>
#else  // FLATBUFFERS_PREFER_PRINTF
#  include <float.h>
#  include <stdio.h>
#endif  // FLATBUFFERS_PREFER_PRINTF

#include <iomanip>
#include <string>

namespace flatbuffers {

// @locale-independent functions for ASCII characters set.

// Fast checking that character lies in closed range: [a <= x <= b]
// using one compare (conditional branch) operator.
inline bool check_ascii_range(char x, char a, char b) {
  FLATBUFFERS_ASSERT(a <= b);
  // (Hacker's Delight): `a <= x <= b` <=> `(x-a) <={u} (b-a)`.
  // The x, a, b will be promoted to int and subtracted without overflow.
  return static_cast<unsigned int>(x - a) <= static_cast<unsigned int>(b - a);
}

// Case-insensitive isalpha
inline bool is_alpha(char c) {
  // ASCII only: alpha to upper case => reset bit 0x20 (~0x20 = 0xDF).
  return check_ascii_range(c & 0xDF, 'a' & 0xDF, 'z' & 0xDF);
}

// Check (case-insensitive) that `c` is equal to alpha.
inline bool is_alpha_char(char c, char alpha) {
  FLATBUFFERS_ASSERT(is_alpha(alpha));
  // ASCII only: alpha to upper case => reset bit 0x20 (~0x20 = 0xDF).
  return ((c & 0xDF) == (alpha & 0xDF));
}

// https://en.cppreference.com/w/cpp/string/byte/isxdigit
// isdigit and isxdigit are the only standard narrow character classification
// functions that are not affected by the currently installed C locale. although
// some implementations (e.g. Microsoft in 1252 codepage) may classify
// additional single-byte characters as digits.
inline bool is_digit(char c) { return check_ascii_range(c, '0', '9'); }

inline bool is_xdigit(char c) {
  // Replace by look-up table.
  return is_digit(c) || check_ascii_range(c & 0xDF, 'a' & 0xDF, 'f' & 0xDF);
}

// Case-insensitive isalnum
inline bool is_alnum(char c) { return is_alpha(c) || is_digit(c); }

// @end-locale-independent functions for ASCII character set

#ifdef FLATBUFFERS_PREFER_PRINTF
template<typename T> size_t IntToDigitCount(T t) {
  size_t digit_count = 0;
  // Count the sign for negative numbers
  if (t < 0) digit_count++;
  // Count a single 0 left of the dot for fractional numbers
  if (-1 < t && t < 1) digit_count++;
  // Count digits until fractional part
  T eps = std::numeric_limits<float>::epsilon();
  while (t <= (-1 + eps) || (1 - eps) <= t) {
    t /= 10;
    digit_count++;
  }
  return digit_count;
}

template<typename T> size_t NumToStringWidth(T t, int precision = 0) {
  size_t string_width = IntToDigitCount(t);
  // Count the dot for floating point numbers
  if (precision) string_width += (precision + 1);
  return string_width;
}

template<typename T>
std::string NumToStringImplWrapper(T t, const char *fmt, int precision = 0) {
  size_t string_width = NumToStringWidth(t, precision);
  std::string s(string_width, 0x00);
  // Allow snprintf to use std::string trailing null to detect buffer overflow
  snprintf(const_cast<char *>(s.data()), (s.size() + 1), fmt, string_width, t);
  return s;
}
#endif  // FLATBUFFERS_PREFER_PRINTF

// Convert an integer or floating point value to a string.
// In contrast to std::stringstream, "char" values are
// converted to a string of digits, and we don't use scientific notation.
template<typename T> std::string NumToString(T t) {
  // clang-format off

  #ifndef FLATBUFFERS_PREFER_PRINTF
    std::stringstream ss;
    ss << t;
    return ss.str();
  #else // FLATBUFFERS_PREFER_PRINTF
    auto v = static_cast<long long>(t);
    return NumToStringImplWrapper(v, "%.*lld");
  #endif // FLATBUFFERS_PREFER_PRINTF
  // clang-format on
}
// Avoid char types used as character data.
template<> inline std::string NumToString<signed char>(signed char t) {
  return NumToString(static_cast<int>(t));
}
template<> inline std::string NumToString<unsigned char>(unsigned char t) {
  return NumToString(static_cast<int>(t));
}
template<> inline std::string NumToString<char>(char t) {
  return NumToString(static_cast<int>(t));
}
#if defined(FLATBUFFERS_CPP98_STL)
template<> inline std::string NumToString<long long>(long long t) {
  char buf[21];  // (log((1 << 63) - 1) / log(10)) + 2
  snprintf(buf, sizeof(buf), "%lld", t);
  return std::string(buf);
}

template<>
inline std::string NumToString<unsigned long long>(unsigned long long t) {
  char buf[22];  // (log((1 << 63) - 1) / log(10)) + 1
  snprintf(buf, sizeof(buf), "%llu", t);
  return std::string(buf);
}
#endif  // defined(FLATBUFFERS_CPP98_STL)

// Special versions for floats/doubles.
template<typename T> std::string FloatToString(T t, int precision) {
  // clang-format off

  #ifndef FLATBUFFERS_PREFER_PRINTF
    // to_string() prints different numbers of digits for floats depending on
    // platform and isn't available on Android, so we use stringstream
    std::stringstream ss;
    // Use std::fixed to suppress scientific notation.
    ss << std::fixed;
    // Default precision is 6, we want that to be higher for doubles.
    ss << std::setprecision(precision);
    ss << t;
    auto s = ss.str();
  #else // FLATBUFFERS_PREFER_PRINTF
    auto v = static_cast<double>(t);
    auto s = NumToStringImplWrapper(v, "%0.*f", precision);
  #endif // FLATBUFFERS_PREFER_PRINTF
  // clang-format on
  // Sadly, std::fixed turns "1" into "1.00000", so here we undo that.
  auto p = s.find_last_not_of('0');
  if (p != std::string::npos) {
    // Strip trailing zeroes. If it is a whole number, keep one zero.
    s.resize(p + (s[p] == '.' ? 2 : 1));
  }
  return s;
}

template<> inline std::string NumToString<double>(double t) {
  return FloatToString(t, 12);
}
template<> inline std::string NumToString<float>(float t) {
  return FloatToString(t, 6);
}

// Convert an integer value to a hexadecimal string.
// The returned string length is always xdigits long, prefixed by 0 digits.
// For example, IntToStringHex(0x23, 8) returns the string "00000023".
inline std::string IntToStringHex(int i, int xdigits) {
  FLATBUFFERS_ASSERT(i >= 0);
  // clang-format off

  #ifndef FLATBUFFERS_PREFER_PRINTF
    std::stringstream ss;
    ss << std::setw(xdigits) << std::setfill('0') << std::hex << std::uppercase
       << i;
    return ss.str();
  #else // FLATBUFFERS_PREFER_PRINTF
    return NumToStringImplWrapper(i, "%.*X", xdigits);
  #endif // FLATBUFFERS_PREFER_PRINTF
  // clang-format on
}

// clang-format off
// Use locale independent functions {strtod_l, strtof_l, strtoll_l, strtoull_l}.
#if defined(FLATBUFFERS_LOCALE_INDEPENDENT) && (FLATBUFFERS_LOCALE_INDEPENDENT > 0)
  class ClassicLocale {
    #ifdef _MSC_VER
      typedef _locale_t locale_type;
    #else
      typedef locale_t locale_type;  // POSIX.1-2008 locale_t type
    #endif
    ClassicLocale();
    ~ClassicLocale();
    locale_type locale_;
    static ClassicLocale instance_;
  public:
    static locale_type Get() { return instance_.locale_; }
  };

  #ifdef _MSC_VER
    #define __strtoull_impl(s, pe, b) _strtoui64_l(s, pe, b, ClassicLocale::Get())
    #define __strtoll_impl(s, pe, b) _strtoi64_l(s, pe, b, ClassicLocale::Get())
    #define __strtod_impl(s, pe) _strtod_l(s, pe, ClassicLocale::Get())
    #define __strtof_impl(s, pe) _strtof_l(s, pe, ClassicLocale::Get())
  #else
    #define __strtoull_impl(s, pe, b) strtoull_l(s, pe, b, ClassicLocale::Get())
    #define __strtoll_impl(s, pe, b) strtoll_l(s, pe, b, ClassicLocale::Get())
    #define __strtod_impl(s, pe) strtod_l(s, pe, ClassicLocale::Get())
    #define __strtof_impl(s, pe) strtof_l(s, pe, ClassicLocale::Get())
  #endif
#else
  #define __strtod_impl(s, pe) strtod(s, pe)
  #define __strtof_impl(s, pe) static_cast<float>(strtod(s, pe))
  #ifdef _MSC_VER
    #define __strtoull_impl(s, pe, b) _strtoui64(s, pe, b)
    #define __strtoll_impl(s, pe, b) _strtoi64(s, pe, b)
  #else
    #define __strtoull_impl(s, pe, b) strtoull(s, pe, b)
    #define __strtoll_impl(s, pe, b) strtoll(s, pe, b)
  #endif
#endif

inline void strtoval_impl(int64_t *val, const char *str, char **endptr,
                                 int base) {
    *val = __strtoll_impl(str, endptr, base);
}

inline void strtoval_impl(uint64_t *val, const char *str, char **endptr,
                                 int base) {
  *val = __strtoull_impl(str, endptr, base);
}

inline void strtoval_impl(double *val, const char *str, char **endptr) {
  *val = __strtod_impl(str, endptr);
}

// UBSAN: double to float is safe if numeric_limits<float>::is_iec559 is true.
__supress_ubsan__("float-cast-overflow")
inline void strtoval_impl(float *val, const char *str, char **endptr) {
  *val = __strtof_impl(str, endptr);
}
#undef __strtoull_impl
#undef __strtoll_impl
#undef __strtod_impl
#undef __strtof_impl
// clang-format on

// Adaptor for strtoull()/strtoll().
// Flatbuffers accepts numbers with any count of leading zeros (-009 is -9),
// while strtoll with base=0 interprets first leading zero as octal prefix.
// In future, it is possible to add prefixed 0b0101.
// 1) Checks errno code for overflow condition (out of range).
// 2) If base <= 0, function try to detect base of number by prefix.
//
// Return value (like strtoull and strtoll, but reject partial result):
// - If successful, an integer value corresponding to the str is returned.
// - If full string conversion can't be performed, 0 is returned.
// - If the converted value falls out of range of corresponding return type, a
// range error occurs. In this case value MAX(T)/MIN(T) is returned.
template<typename T>
inline bool StringToIntegerImpl(T *val, const char *const str,
                                const int base = 0,
                                const bool check_errno = true) {
  // T is int64_t or uint64_T
  FLATBUFFERS_ASSERT(str);
  if (base <= 0) {
    auto s = str;
    while (*s && !is_digit(*s)) s++;
    if (s[0] == '0' && is_alpha_char(s[1], 'X'))
      return StringToIntegerImpl(val, str, 16, check_errno);
    // if a prefix not match, try base=10
    return StringToIntegerImpl(val, str, 10, check_errno);
  } else {
    if (check_errno) errno = 0;  // clear thread-local errno
    auto endptr = str;
    strtoval_impl(val, str, const_cast<char **>(&endptr), base);
    if ((*endptr != '\0') || (endptr == str)) {
      *val = 0;      // erase partial result
      return false;  // invalid string
    }
    // errno is out-of-range, return MAX/MIN
    if (check_errno && errno) return false;
    return true;
  }
}

template<typename T>
inline bool StringToFloatImpl(T *val, const char *const str) {
  // Type T must be either float or double.
  FLATBUFFERS_ASSERT(str && val);
  auto end = str;
  strtoval_impl(val, str, const_cast<char **>(&end));
  auto done = (end != str) && (*end == '\0');
  if (!done) *val = 0;  // erase partial result
  return done;
}

// Convert a string to an instance of T.
// Return value (matched with StringToInteger64Impl and strtod):
// - If successful, a numeric value corresponding to the str is returned.
// - If full string conversion can't be performed, 0 is returned.
// - If the converted value falls out of range of corresponding return type, a
// range error occurs. In this case value MAX(T)/MIN(T) is returned.
template<typename T> inline bool StringToNumber(const char *s, T *val) {
  FLATBUFFERS_ASSERT(s && val);
  int64_t i64;
  // The errno check isn't needed, will return MAX/MIN on overflow.
  if (StringToIntegerImpl(&i64, s, 0, false)) {
    const int64_t max = (flatbuffers::numeric_limits<T>::max)();
    const int64_t min = flatbuffers::numeric_limits<T>::lowest();
    if (i64 > max) {
      *val = static_cast<T>(max);
      return false;
    }
    if (i64 < min) {
      // For unsigned types return max to distinguish from
      // "no conversion can be performed" when 0 is returned.
      *val = static_cast<T>(flatbuffers::is_unsigned<T>::value ? max : min);
      return false;
    }
    *val = static_cast<T>(i64);
    return true;
  }
  *val = 0;
  return false;
}

template<> inline bool StringToNumber<int64_t>(const char *str, int64_t *val) {
  return StringToIntegerImpl(val, str);
}

template<>
inline bool StringToNumber<uint64_t>(const char *str, uint64_t *val) {
  if (!StringToIntegerImpl(val, str)) return false;
  // The strtoull accepts negative numbers:
  // If the minus sign was part of the input sequence, the numeric value
  // calculated from the sequence of digits is negated as if by unary minus
  // in the result type, which applies unsigned integer wraparound rules.
  // Fix this behaviour (except -0).
  if (*val) {
    auto s = str;
    while (*s && !is_digit(*s)) s++;
    s = (s > str) ? (s - 1) : s;  // step back to one symbol
    if (*s == '-') {
      // For unsigned types return the max to distinguish from
      // "no conversion can be performed".
      *val = (flatbuffers::numeric_limits<uint64_t>::max)();
      return false;
    }
  }
  return true;
}

template<> inline bool StringToNumber(const char *s, float *val) {
  return StringToFloatImpl(val, s);
}

template<> inline bool StringToNumber(const char *s, double *val) {
  return StringToFloatImpl(val, s);
}

inline int64_t StringToInt(const char *s, int base = 10) {
  int64_t val;
  return StringToIntegerImpl(&val, s, base) ? val : 0;
}

inline uint64_t StringToUInt(const char *s, int base = 10) {
  uint64_t val;
  return StringToIntegerImpl(&val, s, base) ? val : 0;
}

typedef bool (*LoadFileFunction)(const char *filename, bool binary,
                                 std::string *dest);
typedef bool (*FileExistsFunction)(const char *filename);

LoadFileFunction SetLoadFileFunction(LoadFileFunction load_file_function);

FileExistsFunction SetFileExistsFunction(
    FileExistsFunction file_exists_function);

// Check if file "name" exists.
bool FileExists(const char *name);

// Check if "name" exists and it is also a directory.
bool DirExists(const char *name);

// Load file "name" into "buf" returning true if successful
// false otherwise.  If "binary" is false data is read
// using ifstream's text mode, otherwise data is read with
// no transcoding.
bool LoadFile(const char *name, bool binary, std::string *buf);

// Save data "buf" of length "len" bytes into a file
// "name" returning true if successful, false otherwise.
// If "binary" is false data is written using ifstream's
// text mode, otherwise data is written with no
// transcoding.
bool SaveFile(const char *name, const char *buf, size_t len, bool binary);

// Save data "buf" into file "name" returning true if
// successful, false otherwise.  If "binary" is false
// data is written using ifstream's text mode, otherwise
// data is written with no transcoding.
inline bool SaveFile(const char *name, const std::string &buf, bool binary) {
  return SaveFile(name, buf.c_str(), buf.size(), binary);
}

// Functionality for minimalistic portable path handling.

// The functions below behave correctly regardless of whether posix ('/') or
// Windows ('/' or '\\') separators are used.

// Any new separators inserted are always posix.
FLATBUFFERS_CONSTEXPR char kPathSeparator = '/';

// Returns the path with the extension, if any, removed.
std::string StripExtension(const std::string &filepath);

// Returns the extension, if any.
std::string GetExtension(const std::string &filepath);

// Return the last component of the path, after the last separator.
std::string StripPath(const std::string &filepath);

// Strip the last component of the path + separator.
std::string StripFileName(const std::string &filepath);

// Concatenates a path with a filename, regardless of wether the path
// ends in a separator or not.
std::string ConCatPathFileName(const std::string &path,
                               const std::string &filename);

// Replaces any '\\' separators with '/'
std::string PosixPath(const char *path);

// This function ensure a directory exists, by recursively
// creating dirs for any parts of the path that don't exist yet.
void EnsureDirExists(const std::string &filepath);

// Obtains the absolute path from any other path.
// Returns the input path if the absolute path couldn't be resolved.
std::string AbsolutePath(const std::string &filepath);

// To and from UTF-8 unicode conversion functions

// Convert a unicode code point into a UTF-8 representation by appending it
// to a string. Returns the number of bytes generated.
inline int ToUTF8(uint32_t ucc, std::string *out) {
  FLATBUFFERS_ASSERT(!(ucc & 0x80000000));  // Top bit can't be set.
  // 6 possible encodings: http://en.wikipedia.org/wiki/UTF-8
  for (int i = 0; i < 6; i++) {
    // Max bits this encoding can represent.
    uint32_t max_bits = 6 + i * 5 + static_cast<int>(!i);
    if (ucc < (1u << max_bits)) {  // does it fit?
      // Remaining bits not encoded in the first byte, store 6 bits each
      uint32_t remain_bits = i * 6;
      // Store first byte:
      (*out) += static_cast<char>((0xFE << (max_bits - remain_bits)) |
                                  (ucc >> remain_bits));
      // Store remaining bytes:
      for (int j = i - 1; j >= 0; j--) {
        (*out) += static_cast<char>(((ucc >> (j * 6)) & 0x3F) | 0x80);
      }
      return i + 1;  // Return the number of bytes added.
    }
  }
  FLATBUFFERS_ASSERT(0);  // Impossible to arrive here.
  return -1;
}

// Converts whatever prefix of the incoming string corresponds to a valid
// UTF-8 sequence into a unicode code. The incoming pointer will have been
// advanced past all bytes parsed.
// returns -1 upon corrupt UTF-8 encoding (ignore the incoming pointer in
// this case).
inline int FromUTF8(const char **in) {
  int len = 0;
  // Count leading 1 bits.
  for (int mask = 0x80; mask >= 0x04; mask >>= 1) {
    if (**in & mask) {
      len++;
    } else {
      break;
    }
  }
  if ((static_cast<unsigned char>(**in) << len) & 0x80)
    return -1;  // Bit after leading 1's must be 0.
  if (!len) return *(*in)++;
  // UTF-8 encoded values with a length are between 2 and 4 bytes.
  if (len < 2 || len > 4) { return -1; }
  // Grab initial bits of the code.
  int ucc = *(*in)++ & ((1 << (7 - len)) - 1);
  for (int i = 0; i < len - 1; i++) {
    if ((**in & 0xC0) != 0x80) return -1;  // Upper bits must 1 0.
    ucc <<= 6;
    ucc |= *(*in)++ & 0x3F;  // Grab 6 more bits of the code.
  }
  // UTF-8 cannot encode values between 0xD800 and 0xDFFF (reserved for
  // UTF-16 surrogate pairs).
  if (ucc >= 0xD800 && ucc <= 0xDFFF) { return -1; }
  // UTF-8 must represent code points in their shortest possible encoding.
  switch (len) {
    case 2:
      // Two bytes of UTF-8 can represent code points from U+0080 to U+07FF.
      if (ucc < 0x0080 || ucc > 0x07FF) { return -1; }
      break;
    case 3:
      // Three bytes of UTF-8 can represent code points from U+0800 to U+FFFF.
      if (ucc < 0x0800 || ucc > 0xFFFF) { return -1; }
      break;
    case 4:
      // Four bytes of UTF-8 can represent code points from U+10000 to U+10FFFF.
      if (ucc < 0x10000 || ucc > 0x10FFFF) { return -1; }
      break;
  }
  return ucc;
}

#ifndef FLATBUFFERS_PREFER_PRINTF
// Wraps a string to a maximum length, inserting new lines where necessary. Any
// existing whitespace will be collapsed down to a single space. A prefix or
// suffix can be provided, which will be inserted before or after a wrapped
// line, respectively.
inline std::string WordWrap(const std::string in, size_t max_length,
                            const std::string wrapped_line_prefix,
                            const std::string wrapped_line_suffix) {
  std::istringstream in_stream(in);
  std::string wrapped, line, word;

  in_stream >> word;
  line = word;

  while (in_stream >> word) {
    if ((line.length() + 1 + word.length() + wrapped_line_suffix.length()) <
        max_length) {
      line += " " + word;
    } else {
      wrapped += line + wrapped_line_suffix + "\n";
      line = wrapped_line_prefix + word;
    }
  }
  wrapped += line;

  return wrapped;
}
#endif  // !FLATBUFFERS_PREFER_PRINTF

inline bool EscapeString(const char *s, size_t length, std::string *_text,
                         bool allow_non_utf8, bool natural_utf8) {
  std::string &text = *_text;
  text += "\"";
  for (uoffset_t i = 0; i < length; i++) {
    char c = s[i];
    switch (c) {
      case '\n': text += "\\n"; break;
      case '\t': text += "\\t"; break;
      case '\r': text += "\\r"; break;
      case '\b': text += "\\b"; break;
      case '\f': text += "\\f"; break;
      case '\"': text += "\\\""; break;
      case '\\': text += "\\\\"; break;
      default:
        if (c >= ' ' && c <= '~') {
          text += c;
        } else {
          // Not printable ASCII data. Let's see if it's valid UTF-8 first:
          const char *utf8 = s + i;
          int ucc = FromUTF8(&utf8);
          if (ucc < 0) {
            if (allow_non_utf8) {
              text += "\\x";
              text += IntToStringHex(static_cast<uint8_t>(c), 2);
            } else {
              // There are two cases here:
              //
              // 1) We reached here by parsing an IDL file. In that case,
              // we previously checked for non-UTF-8, so we shouldn't reach
              // here.
              //
              // 2) We reached here by someone calling GenerateText()
              // on a previously-serialized flatbuffer. The data might have
              // non-UTF-8 Strings, or might be corrupt.
              //
              // In both cases, we have to give up and inform the caller
              // they have no JSON.
              return false;
            }
          } else {
            if (natural_utf8) {
              // utf8 points to past all utf-8 bytes parsed
              text.append(s + i, static_cast<size_t>(utf8 - s - i));
            } else if (ucc <= 0xFFFF) {
              // Parses as Unicode within JSON's \uXXXX range, so use that.
              text += "\\u";
              text += IntToStringHex(ucc, 4);
            } else if (ucc <= 0x10FFFF) {
              // Encode Unicode SMP values to a surrogate pair using two \u
              // escapes.
              uint32_t base = ucc - 0x10000;
              auto high_surrogate = (base >> 10) + 0xD800;
              auto low_surrogate = (base & 0x03FF) + 0xDC00;
              text += "\\u";
              text += IntToStringHex(high_surrogate, 4);
              text += "\\u";
              text += IntToStringHex(low_surrogate, 4);
            }
            // Skip past characters recognized.
            i = static_cast<uoffset_t>(utf8 - s - 1);
          }
        }
        break;
    }
  }
  text += "\"";
  return true;
}

// Remove paired quotes in a string: "text"|'text' -> text.
std::string RemoveStringQuotes(const std::string &s);

// Change th global C-locale to locale with name <locale_name>.
// Returns an actual locale name in <_value>, useful if locale_name is "" or
// null.
bool SetGlobalTestLocale(const char *locale_name,
                         std::string *_value = nullptr);

// Read (or test) a value of environment variable.
bool ReadEnvironmentVariable(const char *var_name,
                             std::string *_value = nullptr);

// MSVC specific: Send all assert reports to STDOUT to prevent CI hangs.
void SetupDefaultCRTReportMode();

}  // namespace flatbuffers

#endif  // FLATBUFFERS_UTIL_H_
