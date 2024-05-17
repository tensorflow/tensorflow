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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_LOGGING_OP_RESOLVER_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_LOGGING_OP_RESOLVER_H_

#include <set>
#include <unordered_map>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_common.h"

namespace tflite {
namespace optimize {
namespace calibration {

// A resolver that replaces the kernel invocations with a wrapper
// eval function.
class LoggingOpResolver : public OpResolver {
 public:
  // Creates an instance of |LoggingOpResolver|.
  // All |TfLiteRegistration.invoke| functions are replaced by
  // |logging_eval_fn|.
  // TODO(shashishekhar): This interface needs to change for
  // BuiltinOps that need special logging implementations.
  LoggingOpResolver(const BuiltinOpsSet& builtin_ops_to_replace,
                    const CustomOpsSet& custom_ops_to_replace,
                    const OpResolver& base_resolver,
                    KernelEvalFuncPtr logging_eval_fn,
                    ErrorReporter* error_reporter);

  const TfLiteRegistration* FindOp(BuiltinOperator op,
                                   int version) const override;
  KernelEvalFuncPtr GetWrappedKernelInvoke(BuiltinOperator op,
                                           int version) const;

  const TfLiteRegistration* FindOp(const char* op, int version) const override;
  KernelEvalFuncPtr GetWrappedKernelInvoke(const char* op, int version) const;

 private:
  BuiltinOpsMap<std::unique_ptr<TfLiteRegistration>>
      builtin_op_registration_map_;
  BuiltinOpsMap<KernelEvalFuncPtr> builtin_op_evalfn_map_;
  CustomOpsMap<std::unique_ptr<TfLiteRegistration>> custom_op_registration_map_;
  CustomOpsMap<KernelEvalFuncPtr> custom_op_evalfn_map_;
};

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_LOGGING_OP_RESOLVER_H_
