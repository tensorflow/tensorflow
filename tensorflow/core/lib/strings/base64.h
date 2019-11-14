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

#ifndef TENSORFLOW_CORE_LIB_STRINGS_BASE64_H_
#define TENSORFLOW_CORE_LIB_STRINGS_BASE64_H_

#include <string>
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

/// \brief Converts data into web-safe base64 encoding.
///
/// See https://en.wikipedia.org/wiki/Base64
template <typename T>
Status Base64Encode(StringPiece source, bool with_padding, T* encoded);
template <typename T>
Status Base64Encode(StringPiece source,
                    T* encoded);  // with_padding=false.

/// \brief Converts data from web-safe base64 encoding.
///
/// See https://en.wikipedia.org/wiki/Base64
template <typename T>
Status Base64Decode(StringPiece data, T* decoded);

// Explicit instantiations defined in base64.cc.
extern template Status Base64Decode<string>(StringPiece data, string* decoded);
extern template Status Base64Encode<string>(StringPiece source,
                                            string* encoded);
extern template Status Base64Encode<string>(StringPiece source,
                                            bool with_padding, string* encoded);

#ifdef USE_TSTRING
extern template Status Base64Decode<tstring>(StringPiece data,
                                             tstring* decoded);
extern template Status Base64Encode<tstring>(StringPiece source,
                                             tstring* encoded);
extern template Status Base64Encode<tstring>(StringPiece source,
                                             bool with_padding,
                                             tstring* encoded);
#endif  // USE_TSTRING

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_STRINGS_BASE64_H_
