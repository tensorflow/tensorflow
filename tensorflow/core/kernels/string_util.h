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

namespace tensorflow {

// Enumeration for unicode encodings.  Used by ops such as
// tf.strings.unicode_encode and tf.strings.unicode_decode.
// TODO(edloper): Add support for:
// UTF16, UTF32, UTF16BE, UTF32BE, UTF16LE, UTF32LE
enum class UnicodeEncoding { UTF8 };

// Enumeration for character units.  Used by string such as
// tf.strings.length and tf.substr.
// TODO(edloper): Add support for: UTF32_CHAR, etc.
enum class CharUnit { BYTE, UTF8_CHAR };

// Sets `encoding` based on `str`.
Status ParseUnicodeEncoding(const string& str, UnicodeEncoding* encoding);

// Sets `unit` value based on `str`.
Status ParseCharUnit(const string& str, CharUnit* unit);

// Returns the number of Unicode characters in a UTF-8 string.
// Result may be incorrect if the input string is not valid UTF-8.
int32 UTF8StrLen(const string& string);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STRING_UTIL_H_
