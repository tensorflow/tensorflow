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

#ifndef TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_INTERNAL_H_
#define TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_INTERNAL_H_

#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/conversion_macros.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Represents the results of the execution of an operation.
struct OutputList {
  std::vector<AbstractTensorHandle*> outputs;
  int expected_num_outputs = -1;
};

namespace tracing {

// =============================================================================
// Implementation detail for the unified execution APIs for Eager and tracing
// backends (graph/MLIR).
//
// This defines a set of abstract classes that are intended to provide the
// functionality of the opaque C types exposed in the public APIs defined in the
// `c_api_unified_experimental.h` header.
// =============================================================================

// Represents either a MlirTensor or a GraphTensor.
// This base class does not expose any public methods other than to distinguish
// which subclass it actually is. The user is responsible to use the right
// type of AbstractTensor in their context (do not pass an MlirTensor to a
// GraphContext and vice-versa).
class TracingTensorHandle : public AbstractTensorHandle {
 protected:
  explicit TracingTensorHandle(AbstractTensorHandleKind kind)
      : AbstractTensorHandle(kind) {}

 public:
  // For LLVM style RTTI.
  static bool classof(const AbstractTensorHandle* ptr) {
    return ptr->getKind() == kGraph || ptr->getKind() == kMlir;
  }
};

// An abstract operation describes an operation by its type, name, and
// attributes. It can be "executed" by the context with some input tensors.
// It is allowed to reusing the same abstract operation for multiple execution
// on a given context, with the same or different input tensors.
class TracingOperation : public AbstractOperation {
 protected:
  explicit TracingOperation(AbstractOperationKind kind)
      : AbstractOperation(kind) {}

 public:
  // Sets the name of the operation: this is an optional identifier that is
  // not intended to carry semantics and preserved/propagated without
  // guarantees.
  virtual Status SetOpName(const char* op_name) = 0;

  // For LLVM style RTTI.
  static bool classof(const AbstractOperation* ptr) {
    return ptr->getKind() == kGraph || ptr->getKind() == kMlir;
  }
};

namespace internal {
struct TracingOperationDeleter {
  void operator()(TracingOperation* p) const {
    if (p != nullptr) {
      p->Release();
    }
  }
};
}  // namespace internal

using TracingOperationPtr =
    std::unique_ptr<TracingOperation, internal::TracingOperationDeleter>;

// This holds the context for the execution: dispatching operations either to an
// MLIR implementation or to a graph implementation.
class TracingContext : public AbstractContext {
 protected:
  explicit TracingContext(AbstractContextKind kind) : AbstractContext(kind) {}

 public:
  // Add a function parameter and return the corresponding tensor.
  virtual Status AddParameter(DataType dtype, const PartialTensorShape& shape,
                              TracingTensorHandle**) = 0;

  // Finalize this context and make a function out of it. The context is in a
  // invalid state after this call and must be destroyed.
  virtual Status Finalize(OutputList* outputs, AbstractFunction**) = 0;

  // For LLVM style RTTI.
  static bool classof(const AbstractContext* ptr) {
    return ptr->getKind() == kGraph || ptr->getKind() == kMlir;
  }
};

typedef TracingContext* (*FactoryFunction)(const char* fn_name, TF_Status*);
Status SetDefaultTracingEngine(const char* name);
void RegisterTracingEngineFactory(const ::tensorflow::string& name,
                                  FactoryFunction factory);
}  // namespace tracing

DEFINE_CONVERSION_FUNCTIONS(AbstractContext, TF_ExecutionContext)
DEFINE_CONVERSION_FUNCTIONS(AbstractTensorHandle, TF_AbstractTensor)
DEFINE_CONVERSION_FUNCTIONS(AbstractFunction, TF_AbstractFunction)
DEFINE_CONVERSION_FUNCTIONS(AbstractOperation, TF_AbstractOp)
DEFINE_CONVERSION_FUNCTIONS(OutputList, TF_OutputList)
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_INTERNAL_H_
