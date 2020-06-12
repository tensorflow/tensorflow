/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/optimize/calibration/logging_op_resolver.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace optimize {
namespace calibration {

LoggingOpResolver::LoggingOpResolver(
    const BuiltinOpsSet& builtin_ops_to_replace,
    const CustomOpsSet& custom_ops_to_replace, const OpResolver& base_resolver,
    KernelEvalFuncPtr logging_eval_fn) {
  std::vector<std::string> unresolved_builtin_ops;
  std::vector<std::string> unresolved_custom_ops;

  for (const auto& op_and_version : builtin_ops_to_replace) {
    const TfLiteRegistration* base_registration =
        base_resolver.FindOp(op_and_version.first, op_and_version.second);
    if (!base_registration) {
      unresolved_builtin_ops.push_back(
          EnumNameBuiltinOperator(op_and_version.first));
      continue;
    }
    BuiltinOperatorKey key = op_and_version;
    builtin_op_evalfn_map_[key] = base_registration->invoke;
    auto logging_registration =
        absl::make_unique<TfLiteRegistration>(*base_registration);
    logging_registration->invoke = logging_eval_fn;
    builtin_op_registration_map_[key] = std::move(logging_registration);
  }
  for (const auto& op_and_version : custom_ops_to_replace) {
    const TfLiteRegistration* base_registration = base_resolver.FindOp(
        op_and_version.first.c_str(), op_and_version.second);
    if (!base_registration) {
      if (!IsFlexOp(op_and_version.first.c_str()))
        unresolved_custom_ops.push_back(op_and_version.first.c_str());
      continue;
    }
    CustomOperatorKey key = op_and_version;
    custom_op_evalfn_map_[key] = base_registration->invoke;
    auto logging_registration =
        absl::make_unique<TfLiteRegistration>(*base_registration);
    logging_registration->invoke = logging_eval_fn;
    custom_op_registration_map_[key] = std::move(logging_registration);
  }

  if (!unresolved_builtin_ops.empty() || !unresolved_custom_ops.empty()) {
    std::string error_message =
        "Failed to initialize op resolver for calibration:";
    if (!unresolved_builtin_ops.empty())
      absl::StrAppend(&error_message, "\nThere are unresolved builtin ops: [",
                      absl::StrJoin(unresolved_builtin_ops, ", "), "]");
    if (!unresolved_custom_ops.empty()) {
      absl::StrAppend(&error_message, "\nThere are unresolved custom ops: [",
                      absl::StrJoin(unresolved_builtin_ops, ", "), "]");
    }
    LOG(ERROR) << error_message;
  }
}

const TfLiteRegistration* LoggingOpResolver::FindOp(BuiltinOperator op,
                                                    int version) const {
  BuiltinOperatorKey key = {op, version};
  if (builtin_op_registration_map_.find(key) !=
      builtin_op_registration_map_.end()) {
    return builtin_op_registration_map_.at(key).get();
  }

  return nullptr;
}

KernelEvalFuncPtr LoggingOpResolver::GetWrappedKernelInvoke(BuiltinOperator op,
                                                            int version) const {
  return builtin_op_evalfn_map_.at({op, version});
}

const TfLiteRegistration* LoggingOpResolver::FindOp(const char* op,
                                                    int version) const {
  CustomOperatorKey key = {op, version};
  if (custom_op_registration_map_.find(key) !=
      custom_op_registration_map_.end()) {
    return custom_op_registration_map_.at(key).get();
  }

  return nullptr;
}

KernelEvalFuncPtr LoggingOpResolver::GetWrappedKernelInvoke(const char* op,
                                                            int version) const {
  return custom_op_evalfn_map_.at({op, version});
}

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
