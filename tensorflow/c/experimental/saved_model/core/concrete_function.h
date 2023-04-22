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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_CONCRETE_FUNCTION_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_CONCRETE_FUNCTION_H_

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/function_metadata.h"

namespace tensorflow {

// ConcreteFunctions correspond to an instance of a tf.function with a known set
// of inputs (either through get_concrete_function) or an input_signature.
// ConcreteFunction attempts to preserve the user-facing semantics of the
// tf.function python API and can take a limited set of types as arguments
// (to be modeled in tensorflow::Value), not just Tensors.
//
// SavedModelAPI's ConcreteFunctions' lifetimes are bound to the SavedModel they
// are loaded from, since they retain pointers to the TensorHandles owned by the
// SavedModel, and the FunctionDef of the SavedModel.
//
// Note(bmzhao): This class is only TEMPORARILY virtual, as a way to unblock
// TFRT integration with TF Serving. Do not add more virtual implementations of
// this class. Eventually we want to remove this virtual base class indirection
// and have only a single implementation.
class ConcreteFunction {
 public:
  virtual ~ConcreteFunction() = default;

  // This method returns the "Call" Op used to execute the function.
  virtual Status MakeCallOp(absl::Span<AbstractTensorHandle* const> inputs,
                            ImmediateOpPtr* out) const = 0;

  virtual const FunctionMetadata& GetFunctionMetadata() const = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_CONCRETE_FUNCTION_H_
