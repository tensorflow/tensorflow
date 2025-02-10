/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_REDUCED_PRECISION_SUPPORT_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_REDUCED_PRECISION_SUPPORT_H_

#include <string>

#include "tensorflow/compiler/mlir/lite/tools/optimize/reduced_precision_metadata.h"

namespace tflite {
namespace optimize {

inline bool ReadInferenceType(const std::string& metadata, size_t* idx,
                              ReducedPrecisionSupport* mask) {
  if (metadata.substr(*idx, 4) == kTfLiteFloat16String) {
    *idx += 4;
    *mask = *mask | ReducedPrecisionSupport::Float16Inference;
    return true;
  } else if (metadata.substr(*idx, 4) == kTfLiteBfloat16String) {
    *idx += 4;
    *mask = *mask | ReducedPrecisionSupport::Bfloat16Inference;
    return true;
  }
  return false;
}

inline bool ReadAccumulationType(const std::string& metadata, size_t* idx,
                                 ReducedPrecisionSupport* mask) {
  if (metadata.substr(*idx, 4) == kTfLiteFloat16String) {
    *idx += 4;
    *mask = *mask | ReducedPrecisionSupport::Float16Accumulation;
    return true;
  } else if (metadata.substr(*idx, 4) == kTfLiteFloat32String) {
    *idx += 4;
    *mask = *mask | ReducedPrecisionSupport::Float32Accumulation;
    return true;
  }
  return false;
}

// If the string is valid, set the given mask to indicate the state in
// string and return true. If the string is invalid, return false.
// A valid string is:
// >= 1 valid inference types + accumulation token + 1 valid accumulation type.
// Valid examples would be: "fp16accfp16", "bf16accfp32"
inline bool SetMaskFromReducedPrecisionMetadata(const std::string& metadata,
                                                ReducedPrecisionSupport* mask) {
  bool check = true;
  size_t idx = 0;
  ReducedPrecisionSupport rsp = ReducedPrecisionSupport::None;
  do {
    check = ReadInferenceType(metadata, &idx, &rsp);
  } while (check);
  // Ensure we read at least 1 inference type.
  if (idx == 0) {
    return false;
  }
  // Next read the accumulation token.
  if (metadata.substr(idx, 3) != kTfLiteAccumulationString) {
    return false;
  }
  idx += std::string(kTfLiteAccumulationString).size();
  // Next read a valid accumulation type.
  if (!ReadAccumulationType(metadata, &idx, &rsp)) {
    return false;
  }
  // This should be the end of string.
  if (idx != metadata.length()) {
    return false;
  }
  // The string is a valid mask description. Set the value and return.
  *mask = rsp;
  return true;
}

}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_REDUCED_PRECISION_SUPPORT_H_
