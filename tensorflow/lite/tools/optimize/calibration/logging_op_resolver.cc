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

namespace tflite {
namespace optimize {
namespace calibration {

LoggingOpResolver::LoggingOpResolver(const BuiltinOpsSet& ops_to_replace,
                                     const OpResolver& base_resolver,
                                     KernelEvalFuncPtr logging_eval_fn) {
  for (const auto& op_and_version : ops_to_replace) {
    const TfLiteRegistration* base_registration =
        base_resolver.FindOp(op_and_version.first, op_and_version.second);
    BuiltinOperatorKey key = op_and_version;
    builtin_op_evalfn_map_[key] = base_registration->invoke;
    std::unique_ptr<TfLiteRegistration> logging_registation =
        absl::make_unique<TfLiteRegistration>(*base_registration);
    logging_registation->invoke = logging_eval_fn;
    builtin_op_registration_map_[key] = std::move(logging_registation);
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
  // TODO(b/121374947): Support custom ops as well.
  return nullptr;
}
}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
