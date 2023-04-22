/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_TF_SIGNATURE_DEF_FUNCTION_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_TF_SIGNATURE_DEF_FUNCTION_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/flat_tensor_function.h"
#include "tensorflow/c/experimental/saved_model/core/signature_def_function.h"
#include "tensorflow/c/experimental/saved_model/core/signature_def_function_metadata.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow {

// This is the TF eager runtime implementation of SignatureDefFunction (separate
// from the TFRT implementation). The user-facing API of SignatureDefFunctions
// and their semantic differences from ConcreteFunction are described here:
// https://github.com/tensorflow/tensorflow/blob/e2db60c9d9598ebae0b7741587ce6f5d473584d9/tensorflow/cc/saved_model/experimental/public/signature_def_function.h#L30-L59
// Additional implementation notes are available here:
// https://github.com/tensorflow/tensorflow/blob/e2db60c9d9598ebae0b7741587ce6f5d473584d9/tensorflow/c/experimental/saved_model/core/signature_def_function.h#L31-L48
class TFSignatureDefFunction : public SignatureDefFunction {
 public:
  // Factory function for creating a TFSignatureDefFunction.
  //
  // Params:
  //  function_def - The function_def associated with the created
  //                 TFSignatureDefFunction. TFSignatureDefFunction will
  //                 register this function_def with `ctx` on creation, and
  //                 de-register it on destruction. function_def must be
  //                 non-null, but otherwise has no lifetime requirements.
  //  captures - The captured TensorHandles associated with this
  //             TFConcreteFunction.
  //  metadata - FunctionMetadata associated with this TFSignatureDefFunction.
  //  ctx      - A handle to the Tensorflow runtime. This MUST be non-null and
  //             outlive TFSignatureDefFunction.
  //  out      - The output TFSignatureDefFunction.
  static Status Create(const FunctionDef* function_def,
                       std::vector<ImmediateExecutionTensorHandle*> captures,
                       SignatureDefFunctionMetadata metadata,
                       ImmediateExecutionContext* ctx,
                       std::unique_ptr<TFSignatureDefFunction>* out);

  // This method creates a "Call" Op used to execute the function.
  Status MakeCallOp(absl::Span<AbstractTensorHandle* const> inputs,
                    ImmediateOpPtr* out) const override;

  const SignatureDefFunctionMetadata& GetFunctionMetadata() const override;

  ~TFSignatureDefFunction() override = default;

 private:
  TFSignatureDefFunction(std::unique_ptr<FlatTensorFunction> func,
                         SignatureDefFunctionMetadata metadata);

  TFSignatureDefFunction(const TFSignatureDefFunction&) = delete;
  TFSignatureDefFunction& operator=(const TFSignatureDefFunction&) = delete;

  std::unique_ptr<FlatTensorFunction> func_;
  SignatureDefFunctionMetadata metadata_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_TF_SIGNATURE_DEF_FUNCTION_H_
