/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_PLATFORM_REGEXP_H_
#define TENSORFLOW_PLATFORM_REGEXP_H_

#include "tensorflow/core/platform/port.h"

#if defined(PLATFORM_GOOGLE) || defined(PLATFORM_GOOGLE_ANDROID)
#include "third_party/re2/re2.h"
namespace tensorflow {
typedef ::StringPiece RegexpStringPiece;
}  // namespace tensorflow

#else

#include "external/re2/re2/re2.h"
namespace tensorflow {
typedef re2::StringPiece RegexpStringPiece;
}  // namespace tensorflow

#endif

namespace tensorflow {

// Conversion to/from the appropriate StringPiece type for using in RE2
inline RegexpStringPiece ToRegexpStringPiece(tensorflow::StringPiece sp) {
  return RegexpStringPiece(sp.data(), sp.size());
}
inline tensorflow::StringPiece FromRegexpStringPiece(RegexpStringPiece sp) {
  return tensorflow::StringPiece(sp.data(), sp.size());
}

}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_REGEXP_H_
