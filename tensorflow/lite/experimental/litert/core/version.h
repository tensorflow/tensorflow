// Copyright 2025 Google LLC.
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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_VERSION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_VERSION_H_

#include "tensorflow/lite/experimental/litert/c/litert_common.h"

namespace litert::internal {

// Return true if two API versions are the same.
inline bool IsSameVersion(const LiteRtApiVersion& v1,
                          const LiteRtApiVersion& v2) {
  return (v1.major == v2.major) && (v1.minor == v2.minor) &&
         (v1.patch == v2.patch);
}

// Return true if a given API version is the same as the current runtime.
inline bool IsSameVersionAsRuntime(const LiteRtApiVersion& v) {
  return IsSameVersion(v, {LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
                           LITERT_API_VERSION_PATCH});
}

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_VERSION_H_
