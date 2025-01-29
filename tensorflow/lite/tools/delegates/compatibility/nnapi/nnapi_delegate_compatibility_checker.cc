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

#include "tensorflow/lite/tools/delegates/compatibility/nnapi/nnapi_delegate_compatibility_checker.h"

#include <cctype>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_kernel.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/tools/delegates/compatibility/common/delegate_compatibility_checker_util.h"
#include "tensorflow/lite/tools/delegates/compatibility/common/online_helper_delegate.h"
#include "tensorflow/lite/tools/delegates/compatibility/protos/compatibility_result.pb.h"
#include "tensorflow/lite/tools/versioning/op_signature.h"

namespace tflite {
namespace tools {

namespace {

// Gets canonical feature level enum value from the number. In case the
// feature level is not between 1 and 8, the default value (8) would be
// assigned.
void getCanonicalFeatureLevel(int runtime_feature_level,
                              int& canonical_feature_level) {
  switch (runtime_feature_level) {
    case 1:
      canonical_feature_level = ANEURALNETWORKS_FEATURE_LEVEL_1;
      break;
    case 2:
      canonical_feature_level = ANEURALNETWORKS_FEATURE_LEVEL_2;
      break;
    case 3:
      canonical_feature_level = ANEURALNETWORKS_FEATURE_LEVEL_3;
      break;
    case 4:
      canonical_feature_level = ANEURALNETWORKS_FEATURE_LEVEL_4;
      break;
    case 5:
      canonical_feature_level = ANEURALNETWORKS_FEATURE_LEVEL_5;
      break;
    case 6:
      canonical_feature_level = ANEURALNETWORKS_FEATURE_LEVEL_6;
      break;
    case 7:
      canonical_feature_level = ANEURALNETWORKS_FEATURE_LEVEL_7;
      break;
    case 8:
      canonical_feature_level = ANEURALNETWORKS_FEATURE_LEVEL_8;
      break;
    default:
      canonical_feature_level = ANEURALNETWORKS_FEATURE_LEVEL_8;
  }
}

// Checks if the input string is a valid feature level. To be a valid feature
// level, the string passed as a parameter should contain only one digit
// between 1 and 8.
absl::Status IsValidFeatureLevelInt(const std::string& s) {
  if (s.size() == 1 && std::isdigit(s[0]) && s[0] > '0' && s[0] < '9') {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError("Invalid runtime feature level.");
}

// Gets the runtime feature level from the configurations and convert its
// value to an integer.
absl::Status extractRuntimeFeatureLevel(
    const std::unordered_map<std::string, std::string>& dcc_configs,
    int& runtime_feature_level) {
  std::string str_runtime_feature_level;
  if (dcc_configs.find("nnapi-runtime_feature_level") == dcc_configs.end()) {
    for (const auto& dcc_config : dcc_configs) {
      // If an NNAPI parameter is set, but spelled incorrectly, return an error.
      if (absl::StrContains(dcc_config.first, "nnapi")) {
        return absl::InvalidArgumentError(
            "The correct flag name is 'nnapi-runtime_feature_level");
      }
    }
    // Use default as no NNAPI parameter is set.
    str_runtime_feature_level =
        std::to_string(tools::kDefaultRuntimeFeatureLevel);
  } else {
    str_runtime_feature_level = dcc_configs.at("nnapi-runtime_feature_level");
    RETURN_IF_ERROR(IsValidFeatureLevelInt(str_runtime_feature_level));
  }
  runtime_feature_level = std::stoi(str_runtime_feature_level);
  return absl::OkStatus();
}

// Converts NnapiValidationFailureType to compatibilityFailureType to store
// the error in op_result.
absl::Status convertToCompatibilityFailureType(
    std::vector<delegate::nnapi::NNAPIValidationFailure> map_failures,
    proto::OpCompatibilityResult* op_result) {
  for (const auto& status : map_failures) {
    auto compatibility_failure = op_result->add_compatibility_failures();
    compatibility_failure->set_description(status.message);
    switch (status.type) {
      case delegate::nnapi::NNAPIValidationFailureType::kUnsupportedOperator:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::DCC_UNSUPPORTED_OPERATOR);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::
          kUnsupportedAndroidVersion:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::DCC_UNSUPPORTED_VERSION);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::
          kUnsupportedOperatorVersion:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::DCC_UNSUPPORTED_OPERATOR_VERSION);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::kUnsupportedInputType:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::DCC_UNSUPPORTED_INPUT_TYPE);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::
          kNotRestrictedScaleCompliant:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::
                DCC_NOT_RESTRICTED_SCALE_COMPLIANT);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::kUnsupportedOutputType:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::DCC_UNSUPPORTED_OUTPUT_TYPE);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::kUnsupportedOperandSize:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::DCC_UNSUPPORTED_OPERAND_SIZE);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::
          kUnsupportedOperandValue:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::DCC_UNSUPPORTED_OPERAND_VALUE);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::
          kUnsupportedHybridOperator:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::DCC_UNSUPPORTED_HYBRID_OPERATOR);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::
          kUnsupportedQuantizationType:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::DCC_UNSUPPORTED_QUANTIZATION_TYPE);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::kMissingRequiredOperand:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::DCC_MISSING_REQUIRED_OPERAND);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::kUnsupportedOperandRank:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::DCC_UNSUPPORTED_OPERAND_RANK);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::
          kInputTensorShouldHaveConstantShape:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::
                DCC_INPUT_TENSOR_SHOULD_HAVE_CONSTANT_SHAPE);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::
          kUnsupportedOperatorVariant:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::DCC_UNSUPPORTED_OPERATOR_VARIANT);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::kNoActivationExpected:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::DCC_NO_ACTIVATION_EXPECTED);
        break;
      case delegate::nnapi::NNAPIValidationFailureType::
          kUnsupportedQuantizationParameters:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::
                DCC_UNSUPPORTED_QUANTIZATION_PARAMETERS);
        break;
      default:
        compatibility_failure->set_failure_type(
            proto::CompatibilityFailureType::DCC_INTERNAL_ERROR);
        compatibility_failure->set_description(
            "Unknown validation failure type.");
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status
tools::NnapiDelegateCompatibilityChecker::checkOpCompatibilityOnline(
    TfLiteContext* context, const TfLiteNode* node,
    const TfLiteRegistration* registration,
    std::unordered_map<std::string, std::string> dcc_configs,
    tflite::proto::OpCompatibilityResult* op_result) {
  std::vector<delegate::nnapi::NNAPIValidationFailure> map_failures;
  int runtime_feature_level;
  RETURN_IF_ERROR(
      extractRuntimeFeatureLevel(dcc_configs, runtime_feature_level));
  getCanonicalFeatureLevel(runtime_feature_level, runtime_feature_level);
  if (NNAPIDelegateKernel::Validate(
          context, registration, runtime_feature_level, node,
          /* is_accelerator_specified= */ true,
          /* vendor_plugin= */ nullptr, &map_failures)) {
    op_result->set_is_supported(true);
  } else {
    RETURN_IF_ERROR(convertToCompatibilityFailureType(map_failures, op_result));
    op_result->set_is_supported(false);
  }
  return absl::OkStatus();
}

std::unordered_map<std::string, std::string>
tools::NnapiDelegateCompatibilityChecker::getDccConfigurations() {
  std::unordered_map<std::string, std::string> dcc_configs;
  dcc_configs["nnapi-runtime_feature_level"] =
      std::to_string(runtime_feature_level_);
  return dcc_configs;
}

absl::Status tools::NnapiDelegateCompatibilityChecker::setDccConfigurations(
    const std::unordered_map<std::string, std::string>& dcc_configs) {
  RETURN_IF_ERROR(
      extractRuntimeFeatureLevel(dcc_configs, runtime_feature_level_));
  return absl::OkStatus();
}

absl::Status
tools::NnapiDelegateCompatibilityChecker::checkModelCompatibilityOnline(
    tflite::FlatBufferModel* model_buffer,
    tflite::proto::CompatibilityResult* result) {
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder interpreter_builder(*model_buffer, resolver);
  auto dcc_configs = getDccConfigurations();
  std::function<absl::Status(TfLiteContext*, const TfLiteNode*,
                             const TfLiteRegistration*,
                             std::unordered_map<std::string, std::string>,
                             proto::OpCompatibilityResult*)>
      check_op_func_ptr = &checkOpCompatibilityOnline;
  OnlineHelperDelegate delegate(dcc_configs, check_op_func_ptr, result);
  interpreter_builder.AddDelegate(&delegate);
  interpreter_builder(&interpreter);
  return absl::OkStatus();
}

// TODO(b/243489631): Implement the function.
absl::Status tools::NnapiDelegateCompatibilityChecker::checkOpSigCompatibility(
    const OpSignature& op_sig,
    tflite::proto::OpCompatibilityResult* op_result) {
  return absl::UnimplementedError(
      "Offline mode is not yet supported on NNAPI delegate compatibility "
      "checker.");
}

}  // namespace tools
}  // namespace tflite
