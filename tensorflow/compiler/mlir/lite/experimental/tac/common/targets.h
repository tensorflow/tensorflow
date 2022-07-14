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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_TARGETS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_TARGETS_H_

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace tac {

// Device attribute string on the TFL dialect.
constexpr char kDevice[] = "tac.device";

// Inference type.
constexpr char kInferenceType[] = "tac.inference_type";

// TODO(renjieliu): Add more inference types.
enum InferenceType {
  UNKNOWN = 0,
  FLOAT = 1,
  QUANTIZED_INT8 = 2,
  QUANTIZED_UINT8 = 3,
  HYBRID = 4
};

inline InferenceType GetInferenceTypeEnum(llvm::StringRef inference_type_str) {
  if (inference_type_str == "FLOAT") {
    return FLOAT;
  } else if (inference_type_str == "QUANTIZED_INT8") {
    return QUANTIZED_INT8;
  } else if (inference_type_str == "QUANTIZED_UINT8") {
    return QUANTIZED_UINT8;
  } else if (inference_type_str == "HYBRID") {
    return HYBRID;
  } else {
    return UNKNOWN;
  }
}

inline std::string GetInferenceString(InferenceType inference_type) {
  if (inference_type == FLOAT) {
    return "FLOAT";
  } else if (inference_type == QUANTIZED_INT8) {
    return "QUANTIZED_INT8";
  } else if (inference_type == QUANTIZED_UINT8) {
    return "QUANTIZED_UINT8";
  } else if (inference_type == HYBRID) {
    return "HYBRID";
  } else {
    return "UNKNOWN";
  }
}

// Returns canonical representation for hardware name (All uppercase).
// TODO(b/177376459): Remove this in favor of the string defined by hardwares
// MyHardware::kId.
inline std::string GetCanonicalHardwareName(const std::string& hardware_name) {
  std::string name = hardware_name;
  std::transform(
      name.begin(), name.end(), name.begin(),
      [](unsigned char c) -> unsigned char { return std::toupper(c); });
  return name;
}

// Get the target annotation form the op.
inline llvm::Optional<std::string> GetTargetAnnotation(Operation* op) {
  auto device = op->getAttrOfType<StringAttr>(kDevice);
  if (device == nullptr || device.getValue().empty()) return llvm::None;

  return GetCanonicalHardwareName(device.getValue().str());
}

// Get inference type attribute from the operation if available.
inline llvm::Optional<InferenceType> GetInferenceTypeAnnotation(Operation* op) {
  auto inference_type = op->getAttrOfType<StringAttr>(kInferenceType);
  if (inference_type == nullptr) return llvm::None;

  llvm::StringRef device_name_str = inference_type.getValue();
  return GetInferenceTypeEnum(device_name_str);
}

// InferenceDeviceType is a combination of the hardware with inference type.
struct InferenceDeviceType {
  std::string hardware;
  InferenceType inference_type;

  bool operator==(const InferenceDeviceType& other) const {
    return (hardware == other.hardware) &&
           (inference_type == other.inference_type);
  }

  bool operator!=(const InferenceDeviceType& other) const {
    return !(*this == other);
  }

  struct inference_device_type_hash {
    size_t operator()(const InferenceDeviceType& p) const {
      auto hash1 = std::hash<std::string>{}(p.hardware);
      auto hash2 = std::hash<InferenceType>{}(p.inference_type);
      return hash1 ^ hash2;
    }
  };
};

// Get InferenceDeviceType attribute from the operation if available.
inline llvm::Optional<InferenceDeviceType> GetInferenceDeviceTypeForOp(
    Operation* op) {
  auto hardware = GetTargetAnnotation(op);
  if (!hardware.has_value()) return llvm::None;

  auto inference_type = GetInferenceTypeAnnotation(op);
  if (!inference_type.has_value()) return llvm::None;

  InferenceDeviceType inference_device_type;
  inference_device_type.hardware = hardware.getValue();
  inference_device_type.inference_type = inference_type.getValue();
  return inference_device_type;
}

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_COMMON_TARGETS_H_
