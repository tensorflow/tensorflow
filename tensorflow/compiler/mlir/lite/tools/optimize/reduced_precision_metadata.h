/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TOOLS_OPTIMIZE_REDUCED_PRECISION_METADATA_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TOOLS_OPTIMIZE_REDUCED_PRECISION_METADATA_H_

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>

#include "tensorflow/compiler/mlir/lite/kernels/internal/compatibility_macros.h"

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

}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TOOLS_OPTIMIZE_REDUCED_PRECISION_METADATA_H_
