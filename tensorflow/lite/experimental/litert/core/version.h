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

#include <string>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"

namespace litert::internal {

// Return true if a given API version is compatible with the current runtime.
inline bool CheckVersionComatibility(const LiteRtApiVersion& v,
                                     const std::string& api_name) {
  // Two versions are compatible only if they have the same major number because
  // major number changes indicate backward compatibility breakages.
  bool result = (v.major == LITERT_API_VERSION_MAJOR);
  if (!result) {
    LITERT_LOG(LITERT_ERROR,
               "Unsupported %s API version, found version %d.%d.%d and "
               "expected version %d.%d.%d",
               api_name.c_str(), v.major, v.minor, v.patch,
               LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
               LITERT_API_VERSION_PATCH);
  }
  return result;
}

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_VERSION_H_
