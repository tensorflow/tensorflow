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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_FLAT_TENSOR_FUNCTION_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_FLAT_TENSOR_FUNCTION_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow {

// FlatTensorFunction models a TF2 eager runtime view of a callable function,
// taking + returning flat lists of tensors, including any captures.
// Effectively, it is a thin wrapper around a FunctionDef owned by the
// EagerContext, and any TensorHandle captures associated with the function. The
// MakeCallOp method handles the logic of marshaling captures after the user
// provided inputs automatically.
// Note(bmzhao): This class is mainly intended to house low-level reusable
// function logic between SignatureDefFunction and ConcreteFunction, which
// present higher level interfaces. This type does *not* hold any "function
// metadata".
class FlatTensorFunction {
 public:
  // Factory for creating a FlatTensorFunction.
  //
  // Params:
  //  function_def - The function_def associated with the created
  //                 FlatTensorFunction. FlatTensorFunction will register this
  //                 function_def with `ctx` on creation, and de-register it on
  //                 destruction. function_def must be non-null, but
  //                 otherwise has no lifetime requirements.
  //  captures - The captured TensorHandles associated with this
  //             FlatTensorFunction. FlatTensorFunction will participate in
  //             ownership of the handles (it explicitly increments the refcount
  //             of each handle, and will decrement them on destruction).
  //  ctx      - A handle to the Tensorflow runtime. This MUST be non-null and
  //             outlive TFConcreteFunction.
  //  out      - The output FlatTensorFunction.
  static Status Create(const FunctionDef* function_def,
                       std::vector<ImmediateExecutionTensorHandle*> captures,
                       ImmediateExecutionContext* ctx,
                       std::unique_ptr<FlatTensorFunction>* out);

  // This method creates a "Call" Op used to execute the function.
  Status MakeCallOp(absl::Span<AbstractTensorHandle* const> inputs,
                    ImmediateOpPtr* out) const;

  ~FlatTensorFunction();

 private:
  FlatTensorFunction(const std::string& name,
                     std::vector<ImmediateTensorHandlePtr> captures,
                     ImmediateExecutionContext* ctx);

  FlatTensorFunction(const FlatTensorFunction&) = delete;
  FlatTensorFunction& operator=(const FlatTensorFunction&) = delete;

  // Name of the FunctionDef corresponding to this TFConcreteFunction
  std::string name_;
  std::vector<ImmediateTensorHandlePtr> captures_;
  ImmediateExecutionContext* ctx_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_TYPES_FLAT_TENSOR_FUNCTION_H_
