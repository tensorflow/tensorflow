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

#include "tensorflow/lite/tools/delegates/compatibility/gpu/gpu_delegate_compatibility_checker.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "tensorflow/lite/tools/versioning/gpu_compatibility.h"
#include "tensorflow/lite/tools/versioning/op_signature.h"

namespace tflite {
namespace tools {

namespace {

void convertToValidationFailureType(absl::Status status,
                                    proto::OpCompatibilityResult* op_result) {
  auto compatibility_failure = op_result->add_compatibility_failures();
  compatibility_failure->set_description(std::string(status.message()));
  switch (status.code()) {
    case absl::StatusCode::kInvalidArgument:
      compatibility_failure->set_failure_type(
          proto::CompatibilityFailureType::DCC_INVALID_ARGUMENT);
      break;
    case absl::StatusCode::kUnimplemented:
      compatibility_failure->set_failure_type(
          proto::CompatibilityFailureType::DCC_UNIMPLEMENTED_ERROR);
      break;
    case absl::StatusCode::kInternal:
      compatibility_failure->set_failure_type(
          proto::CompatibilityFailureType::DCC_INTERNAL_ERROR);
      break;
    case absl::StatusCode::kOutOfRange:
      compatibility_failure->set_failure_type(
          proto::CompatibilityFailureType::DCC_OUT_OF_RANGE);
      break;
    default:
      compatibility_failure->set_failure_type(
          proto::CompatibilityFailureType::DCC_INTERNAL_ERROR);
      compatibility_failure->set_description(
          "Unknown validation failure type.");
  }
}

}  // namespace

std::unordered_map<std::string, std::string>
tools::GpuDelegateCompatibilityChecker::getDccConfigurations() {
  return {};
}

absl::Status tools::GpuDelegateCompatibilityChecker::setDccConfigurations(
    const std::unordered_map<std::string, std::string>& dcc_configs) {
  return absl::OkStatus();
}

absl::Status
tools::GpuDelegateCompatibilityChecker::checkModelCompatibilityOnline(
    tflite::FlatBufferModel* model_buffer,
    tflite::proto::CompatibilityResult* result) {
  return absl::UnimplementedError(
      "Online mode is not supported on GPU delegate compatibility checker.");
}

absl::Status tools::GpuDelegateCompatibilityChecker::checkOpSigCompatibility(
    const OpSignature& op_sig,
    tflite::proto::OpCompatibilityResult* op_result) {
  auto status = CheckGpuDelegateCompatibility(op_sig);
  if (!status.ok()) {
    convertToValidationFailureType(status, op_result);
    op_result->set_is_supported(false);
  } else {
    op_result->set_is_supported(true);
  }
  return absl::OkStatus();
}

}  // namespace tools
}  // namespace tflite
