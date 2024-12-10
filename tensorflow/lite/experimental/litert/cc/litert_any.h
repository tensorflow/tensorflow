// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ANY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ANY_H_

#include <any>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"

namespace litert {

inline std::any ToStdAny(LiteRtAny litert_any) {
  std::any res;
  switch (litert_any.type) {
    case kLiteRtAnyTypeNone:
      break;
    case kLiteRtAnyTypeBool:
      res = litert_any.bool_value;
      break;
    case kLiteRtAnyTypeInt:
      res = litert_any.int_value;
      break;
    case kLiteRtAnyTypeReal:
      res = litert_any.real_value;
      break;
    case kLiteRtAnyTypeString:
      res = litert_any.str_value;
      break;
    case kLiteRtAnyTypeVoidPtr:
      res = litert_any.ptr_value;
      break;
  }
  return res;
}

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_ANY_H_
