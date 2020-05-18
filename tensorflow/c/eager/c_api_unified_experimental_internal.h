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
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace internal {

// =============================================================================
// Implementation detail for the unified execution APIs for Eager and tracing
// backends (graph/MLIR).
//
// This defines a set of abstract classes that are intended to provide the
// functionality of the opaque C types exposed in the public APIs defined in the
// `c_api_unified_experimental.h` header.
// =============================================================================

// We can't depend on C++ rtti, but we still want to be able to have a safe
// dynamic_cast to provide diagnostics to the user when the API is misused.
// Instead we model RTTI by listing all the possible subclasses for each
// abstract base. Each subclass initializes the base class with the right
// `kind`, which allows an equivalent to `std::dynamic_cast` provided by this
// utility.
template <typename T, typename S>
T* dyncast(S source) {
  if (source->getKind() != T::kKind) {
    return nullptr;
  }
  return tensorflow::down_cast<T*>(source);
}

// Represents either an EagerTensor or a GraphTensor.
// This base class does not expose any public methods other than to distinguish
// which subclass it actually is. The user is responsible to use the right
// type of AbstractTensor in their context (do not pass an EagerTensor to a
// GraphContext and vice-versa).
class AbstractTensor {
 protected:
  enum AbstractTensorKind { kGraphTensor, kEagerTensor, kMLIRTensor };
  explicit AbstractTensor(AbstractTensorKind kind) : kind_(kind) {}

 public:
  // Returns which subclass is this instance of.
  AbstractTensorKind getKind() const { return kind_; }
  virtual ~AbstractTensor() = default;

 private:
  const AbstractTensorKind kind_;
};

// Represents the results of the execution of an operation.
struct OutputList {
  std::vector<AbstractTensor*> outputs;
  int expected_num_outputs = -1;
};

// Holds the result of tracing a function.
class AbstractFunction {
 protected:
  enum AbstractFunctionKind { kGraphFunc };
  explicit AbstractFunction(AbstractFunctionKind kind) : kind_(kind) {}

 public:
  // Returns which subclass is this instance of.
  AbstractFunctionKind getKind() const { return kind_; }
  virtual ~AbstractFunction() = default;

  // Temporary API till we figure the right abstraction for AbstractFunction.
  // At the moment both Eager and Graph needs access to a "TF_Function" object.
  virtual TF_Function* GetTfFunction(TF_Status* s) = 0;

 private:
  const AbstractFunctionKind kind_;
};

// An abstract operation describes an operation by its type, name, and
// attributes. It can be "executed" by the context with some input tensors.
// It is allowed to reusing the same abstract operation for multiple execution
// on a given context, with the same or different input tensors.
class AbstractOp {
 protected:
  enum AbstractOpKind { kGraphOp, kEagerOp };
  explicit AbstractOp(AbstractOpKind kind) : kind_(kind) {}

 public:
  // Returns which subclass is this instance of.
  AbstractOpKind getKind() const { return kind_; }
  virtual ~AbstractOp() = default;

  // Sets the type of the operation (for example `AddV2`).
  virtual void SetOpType(const char* op_type, TF_Status* s) = 0;

  // Sets the name of the operation: this is an optional identifier that is
  // not intended to carry semantics and preserved/propagated without
  // guarantees.
  virtual void SetOpName(const char* op_name, TF_Status* s) = 0;

  // Add a `TypeAttribute` on the operation.
  virtual void SetAttrType(const char* attr_name, TF_DataType value,
                           TF_Status* s) = 0;

 private:
  const AbstractOpKind kind_;
};

// This holds the context for the execution: dispatching operations either to an
// eager implementation or to a graph implementation.
struct ExecutionContext {
 protected:
  enum ExecutionContextKind { kGraphContext, kEagerContext };
  explicit ExecutionContext(ExecutionContextKind kind) : k(kind) {}

 public:
  // Returns which subclass is this instance of.
  ExecutionContextKind getKind() const { return k; }
  virtual ~ExecutionContext() = default;

  // Executes the operation on the provided inputs and populate the OutputList
  // with the results. The input tensors must match the current context.
  // The effect of "executing" an operation depends on the context: in an Eager
  // context it will dispatch it to the runtime for execution, while in a
  // tracing context it will add the operation to the current function.
  virtual void ExecuteOperation(AbstractOp* op, int num_inputs,
                                AbstractTensor* const* inputs, OutputList* o,
                                TF_Status* s) = 0;

  // Creates an empty AbstractOperation suitable to use with this context.
  virtual AbstractOp* CreateOperation() = 0;

  // Add a function parameter and return the corresponding tensor.
  // This is only valid with an ExecutionContext obtained from a TracingContext,
  // it'll always error out with an eager context.
  virtual AbstractTensor* AddParameter(TF_DataType dtype, TF_Status* s) = 0;

  // Finalize this context and make a function out of it. The context is in a
  // invalid state after this call and must be destroyed.
  // This is only valid with an ExecutionContext obtained from a TracingContext,
  // it'll always error out with an eager context.
  virtual AbstractFunction* Finalize(OutputList* outputs, TF_Status* s) = 0;

  // Registers a functions with this context, after this the function is
  // available to be called/referenced by its name in this context.
  virtual void RegisterFunction(AbstractFunction* func, TF_Status* s) = 0;

 private:
  const ExecutionContextKind k;
};

typedef ExecutionContext* (*FactoryFunction)(const char* fn_name, TF_Status*);
void SetDefaultTracingEngine(const char* name);
void RegisterTracingEngineFactory(const ::tensorflow::string& name,
                                  FactoryFunction factory);

// Create utilities to wrap/unwrap: this convert from the C opaque types to the
// C++ implementation, and back.
#define MAKE_WRAP_UNWRAP(C_TYPEDEF, CPP_CLASS)                              \
  static inline CPP_CLASS* const& unwrap(C_TYPEDEF* const& o) {             \
    return reinterpret_cast<CPP_CLASS* const&>(o);                          \
  }                                                                         \
  static inline const CPP_CLASS* const& unwrap(const C_TYPEDEF* const& o) { \
    return reinterpret_cast<const CPP_CLASS* const&>(o);                    \
  }                                                                         \
  static inline C_TYPEDEF* const& wrap(CPP_CLASS* const& o) {               \
    return reinterpret_cast<C_TYPEDEF* const&>(o);                          \
  }                                                                         \
  static inline const C_TYPEDEF* const& wrap(const CPP_CLASS* const& o) {   \
    return reinterpret_cast<const C_TYPEDEF* const&>(o);                    \
  }

MAKE_WRAP_UNWRAP(TF_ExecutionContext, ExecutionContext)
MAKE_WRAP_UNWRAP(TF_AbstractFunction, AbstractFunction)
MAKE_WRAP_UNWRAP(TF_AbstractTensor, AbstractTensor)
MAKE_WRAP_UNWRAP(TF_AbstractOp, AbstractOp)
MAKE_WRAP_UNWRAP(TF_OutputList, OutputList)

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_INTERNAL_H_
