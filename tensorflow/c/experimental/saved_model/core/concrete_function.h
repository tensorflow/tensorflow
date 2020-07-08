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

#include <vector>

#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/function_metadata.h"
#include "tensorflow/core/framework/function.pb.h"

namespace tensorflow {

// Note that ConcreteFunctions's lifetimes are effectively bound
// to the SavedModel they are loaded from, since they retain pointers
// to the TensorHandles owned by the SavedModel, and the FunctionDef
// of the SavedModel.
// Note(bmzhao): This class is only TEMPORARILY virtual, as a way to unblock
// TFRT integration with TF Serving. Do not add more virtual implementations of
// this class. Eventually we want to remove this virtual base class indirection
// and have only a single implementation.
class ConcreteFunction {
 public:
  virtual ~ConcreteFunction() = 0;

  // This method returns the "Call" Op used to execute the function.
  virtual ImmediateExecutionOperation* GetCallOp() = 0;

  const std::vector<tensorflow::ImmediateExecutionTensorHandle*>& GetCaptures()
      const;
  const FunctionMetadata& GetFunctionMetadata() const;

 private:
  FunctionMetadata metadata_;
  std::vector<tensorflow::ImmediateExecutionTensorHandle*> captures_;
  FunctionDef* function_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_CONCRETE_FUNCTION_H_
