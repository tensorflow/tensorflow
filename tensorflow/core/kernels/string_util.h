/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_STRING_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_STRING_UTIL_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

// Enumeration for unicode encodings.  Used by ops such as
// tf.strings.unicode_encode and tf.strings.unicode_decode.
enum class UnicodeEncoding { UTF8, UTF16BE, UTF32BE };

// Enumeration for character units.  Used by string such as
// tf.strings.length and tf.substr.
// TODO(edloper): Add support for: UTF32_CHAR, etc.
enum class CharUnit { BYTE, UTF8_CHAR };

// Whether or not the given byte is the trailing byte of a UTF-8/16/32 char.
inline bool IsTrailByte(char x) { return static_cast<signed char>(x) < -0x40; }

// Sets `encoding` based on `str`.
absl::Status ParseUnicodeEncoding(const string& str, UnicodeEncoding* encoding);

// Sets `unit` value based on `str`.
absl::Status ParseCharUnit(const string& str, CharUnit* unit);

// Returns the number of Unicode characters in a UTF-8 string.
// Result may be incorrect if the input string is not valid UTF-8.
int32 UTF8StrLen(const string& str);

// Get the next UTF8 character position starting at the given position and
// skipping the given number of characters. Position is a byte offset, and
// should never be `null`. The function return true if successful. However, if
// the end of the string is reached before the requested characters, then the
// position will point to the end of string and this function will return false.
template <typename T>
bool ForwardNUTF8CharPositions(const StringPiece in,
                               const T num_utf8_chars_to_shift, T* pos) {
  const size_t size = in.size();
  T utf8_chars_counted = 0;
  while (utf8_chars_counted < num_utf8_chars_to_shift && *pos < size) {
    // move forward one utf-8 character
    do {
      ++*pos;
    } while (*pos < size && IsTrailByte(in[*pos]));
    ++utf8_chars_counted;
  }
  return utf8_chars_counted == num_utf8_chars_to_shift;
}

// Get the previous UTF8 character position starting at the given position and
// skipping the given number of characters. Position is a byte offset with a
// positive value, relative to the beginning of the string, and should never be
// `null`. The function return true if successful. However, if the beginning of
// the string is reached before the requested character, then the position will
// point to the beginning of the string and this function will return false.
template <typename T>
bool BackNUTF8CharPositions(const StringPiece in,
                            const T num_utf8_chars_to_shift, T* pos) {
  const size_t start = 0;
  T utf8_chars_counted = 0;
  while (utf8_chars_counted < num_utf8_chars_to_shift && (*pos > start)) {
    // move back one utf-8 character
    do {
      --*pos;
    } while (IsTrailByte(in[*pos]) && *pos > start);
    ++utf8_chars_counted;
  }
  return utf8_chars_counted == num_utf8_chars_to_shift;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STRING_UTIL_H_
