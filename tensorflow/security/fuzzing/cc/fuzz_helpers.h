/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_SECURITY_FUZZING_CC_FUZZ_HELPERS_H_
#define TENSORFLOW_SECURITY_FUZZING_CC_FUZZ_HELPERS_H_

#include <cstdint>

#include "tensorflow/core/platform/status.h"

namespace helper {

inline tensorflow::error::Code BuildRandomErrorCode(uint32_t code) {
  // We cannot build a `Status` with error_code of 0 and a message, so force
  // error code to be non-zero.
  if (code == 0) {
    return tensorflow::error::UNKNOWN;
  }

  return static_cast<tensorflow::error::Code>(code);
}

}  // namespace helper

#endif  // TENSORFLOW_SECURITY_FUZZING_CC_FUZZ_HELPERS_H_
