/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_BASE64_H_
#define TENSORFLOW_TSL_PLATFORM_BASE64_H_

#include <string>

#include "absl/status/status.h"
#include "tsl/platform/status.h"
#include "tsl/platform/stringpiece.h"

namespace tsl {

/// \brief Converts data into web-safe base64 encoding.
///
/// See https://en.wikipedia.org/wiki/Base64
template <typename T>
absl::Status Base64Encode(absl::string_view source, bool with_padding,
                          T* encoded);
template <typename T>
absl::Status Base64Encode(absl::string_view source,
                          T* encoded);  // with_padding=false.

/// \brief Converts data from web-safe base64 encoding.
///
/// See https://en.wikipedia.org/wiki/Base64
template <typename T>
absl::Status Base64Decode(absl::string_view data, T* decoded);

// Explicit instantiations defined in base64.cc.
extern template Status Base64Decode<std::string>(StringPiece data,
                                                 std::string* decoded);
extern template Status Base64Encode<std::string>(StringPiece source,
                                                 std::string* encoded);
extern template Status Base64Encode<std::string>(StringPiece source,
                                                 bool with_padding,
                                                 std::string* encoded);

extern template Status Base64Decode<tstring>(StringPiece data,
                                             tstring* decoded);
extern template Status Base64Encode<tstring>(StringPiece source,
                                             tstring* encoded);
extern template Status Base64Encode<tstring>(StringPiece source,
                                             bool with_padding,
                                             tstring* encoded);

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_BASE64_H_
