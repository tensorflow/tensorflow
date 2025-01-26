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

#include "tensorflow/lite/experimental/litert/c/litert_common.h"

extern "C" {

int LiteRtCompareApiVersion(LiteRtApiVersion v1, LiteRtApiVersion v2) {
  if (v1.major > v2.major) {
    return 1;
  } else if (v1.major == v2.major) {
    if (v1.minor > v2.minor) {
      return 1;
    } else if (v1.minor == v2.minor) {
      if (v1.patch > v2.patch) {
        return 1;
      } else if (v1.patch == v2.patch) {
        return 0;
      }
    }
  }
  return -1;
}

}  // extern "C"
