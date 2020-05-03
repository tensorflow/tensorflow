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

#ifndef TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_PRIVATE_H_
#define TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_PRIVATE_H_

#include <vector>

#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/core/platform/casts.h"

namespace tensorflow {
namespace internal {

// =============================================================================
// Unified Execution APIs for Eager and tracing backends.
// =============================================================================

struct AbstractTensor {
  enum AbstractTensorKind { kGraphTensor, kEagerTensor, kMLIRTensor };
  explicit AbstractTensor(AbstractTensorKind kind) : k(kind) {}
  AbstractTensorKind getKind() const { return k; }
  virtual ~AbstractTensor() = default;

 private:
  const AbstractTensorKind k;
};

struct OutputList {
  std::vector<AbstractTensor*> outputs;
  int expected_num_outputs = -1;
};

struct AbstractFunction {
  enum AbstractFunctionKind { kGraphFunc };
  explicit AbstractFunction(AbstractFunctionKind kind) : k(kind) {}
  AbstractFunctionKind getKind() const { return k; }
  virtual ~AbstractFunction() = default;

  // Temporary API till we figure the right abstraction for AbstractFunction
  virtual TF_Function* GetTfFunction(TF_Status* s) = 0;

 private:
  const AbstractFunctionKind k;
};

struct AbstractOp {
  // Needed to implement our own version of RTTI since dynamic_cast is not
  // supported in mobile builds.
  enum AbstractOpKind { kGraphOp, kEagerOp };
  explicit AbstractOp(AbstractOpKind kind) : k(kind) {}
  AbstractOpKind getKind() const { return k; }
  virtual void SetOpType(const char* const op_type, TF_Status* s) = 0;
  virtual void SetOpName(const char* const op_name, TF_Status* s) = 0;
  virtual void SetAttrType(const char* const attr_name, TF_DataType value,
                           TF_Status* s) = 0;
  virtual ~AbstractOp() {}

 private:
  const AbstractOpKind k;
};

struct ExecutionContext {
  // Needed to implement our own version of RTTI since dynamic_cast is not
  // supported in mobile builds.
  enum ExecutionContextKind { kGraphContext, kEagerContext };
  explicit ExecutionContext(ExecutionContextKind kind) : k(kind) {}
  ExecutionContextKind getKind() const { return k; }

  virtual void ExecuteOperation(AbstractOp* op, int num_inputs,
                                AbstractTensor* const* inputs, OutputList* o,
                                TF_Status* s) = 0;
  virtual AbstractOp* CreateOperation() = 0;
  virtual void RegisterFunction(AbstractFunction* func, TF_Status* s) = 0;
  virtual ~ExecutionContext() = default;

 private:
  const ExecutionContextKind k;
};

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

template <typename T, typename S>
T* dynamic_cast_helper(S source) {
  if (source->getKind() != T::kKind) {
    return nullptr;
  }
  return tensorflow::down_cast<T*>(source);
}

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_C_API_UNIFIED_EXPERIMENTAL_PRIVATE_H_
