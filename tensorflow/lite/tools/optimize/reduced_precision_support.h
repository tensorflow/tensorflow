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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_REDUCED_PRECISION_SUPPORT_H
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_REDUCED_PRECISION_SUPPORT_H

#include <string>

#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace optimize {

static constexpr char kTfLiteReducedPrecisionKey[] =
    "reduced_precision_support";

static constexpr char kTfLiteFloat16String[] = "fp16";
static constexpr char kTfLiteBfloat16String[] = "bf16";
static constexpr char kTfLiteFloat32String[] = "fp32";
static constexpr char kTfLiteAccumulationString[] = "acc";

enum class ReducedPrecisionSupport : std::uint8_t {
  None = 0,
  Float16Inference = 0x1,
  Bfloat16Inference = 0x2,
  Float16Accumulation = 0x4,
  Float32Accumulation = 0x8,
};

inline ReducedPrecisionSupport operator|(ReducedPrecisionSupport a,
                                         ReducedPrecisionSupport b) {
  return static_cast<ReducedPrecisionSupport>(static_cast<std::uint32_t>(a) |
                                              static_cast<std::uint32_t>(b));
}

inline ReducedPrecisionSupport& operator|=(ReducedPrecisionSupport& a,
                                           ReducedPrecisionSupport b) {
  return a = static_cast<ReducedPrecisionSupport>(
             static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

inline ReducedPrecisionSupport operator&(ReducedPrecisionSupport a,
                                         ReducedPrecisionSupport b) {
  return static_cast<ReducedPrecisionSupport>(static_cast<std::uint32_t>(a) &
                                              static_cast<std::uint32_t>(b));
}

inline ReducedPrecisionSupport& operator&=(ReducedPrecisionSupport& a,
                                           ReducedPrecisionSupport b) {
  return a = static_cast<ReducedPrecisionSupport>(
             static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b));
}

inline bool SupportsFP16Inference(const ReducedPrecisionSupport& mask) {
  return static_cast<bool>(mask & ReducedPrecisionSupport::Float16Inference);
}

inline bool SupportsBfloat16Inference(const ReducedPrecisionSupport& mask) {
  return static_cast<bool>(mask & ReducedPrecisionSupport::Bfloat16Inference);
}

inline bool SupportsFP16Accumulation(const ReducedPrecisionSupport& mask) {
  return static_cast<bool>(mask & ReducedPrecisionSupport::Float16Accumulation);
}

inline bool SupportsFP32Accumulation(const ReducedPrecisionSupport& mask) {
  return static_cast<bool>(mask & ReducedPrecisionSupport::Float32Accumulation);
}

inline bool SupportsReducedPrecisionInference(
    const ReducedPrecisionSupport& mask) {
  return SupportsFP16Inference(mask) || SupportsBfloat16Inference(mask);
}

inline bool SupportsEitherFP16OrFP32Accumulation(
    const ReducedPrecisionSupport& mask) {
  return SupportsFP16Accumulation(mask) != SupportsFP32Accumulation(mask);
}

// Return the key-value pair for reduced precision support metadata.
// Example: mask = Float16Inference | Bfloat16Inference | Float32Accumulation;
// Returned value would be <"reduced_precision_support", "fp16bf16accfp32">.
inline std::pair<std::string, std::string> MetadataForReducedPrecisionSupport(
    const ReducedPrecisionSupport& mask) {
  TFLITE_DCHECK(SupportsReducedPrecisionInference(mask));
  TFLITE_DCHECK(SupportsEitherFP16OrFP32Accumulation(mask));
  std::string value = "";
  if (SupportsFP16Inference(mask)) {
    value += kTfLiteFloat16String;
  }
  if (SupportsBfloat16Inference(mask)) {
    value += kTfLiteBfloat16String;
  }
  value += kTfLiteAccumulationString;
  if (SupportsFP16Accumulation(mask)) {
    value += kTfLiteFloat16String;
  } else if (SupportsFP32Accumulation(mask)) {
    value += kTfLiteFloat32String;
  }
  return std::make_pair(std::string(kTfLiteReducedPrecisionKey), value);
}

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

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_REDUCED_PRECISION_SUPPORT_H
