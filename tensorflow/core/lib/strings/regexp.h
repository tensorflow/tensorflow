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

#ifndef TENSORFLOW_CORE_LIB_STRINGS_REGEXP_H_
#define TENSORFLOW_CORE_LIB_STRINGS_REGEXP_H_

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {

// Conversion to/from the appropriate StringPiece type for using in RE2
inline RegexpStringPiece ToRegexpStringPiece(tensorflow::StringPiece sp) {
  return RegexpStringPiece(sp.data(), sp.size());
}
inline tensorflow::StringPiece FromRegexpStringPiece(RegexpStringPiece sp) {
  return tensorflow::StringPiece(sp.data(), sp.size());
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_STRINGS_REGEXP_H_
