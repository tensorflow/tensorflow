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

#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/core/platform/casts.h"

namespace tensorflow {
namespace internal {

// =============================================================================
// Unified Execution APIs for Eager and tracing backends.
// =============================================================================

struct ExecutionContext {
  // Needed to implement our own version of RTTI since dynamic_cast is not
  // supported in mobile builds.
  enum ExecutionContextKind { kGraphContext, kEagerContext };
  explicit ExecutionContext(ExecutionContextKind kind) : k(kind) {}
  ExecutionContextKind getKind() const { return k; }

  virtual void ExecuteOperation(TF_AbstractOp* op, int num_inputs,
                                TF_AbstractTensor* const* inputs,
                                TF_OutputList* o, TF_Status* s) = 0;
  virtual TF_AbstractOp* CreateOperation() = 0;
  virtual void RegisterFunction(TF_AbstractFunction* func, TF_Status* s) = 0;
  virtual ~ExecutionContext() = default;

 private:
  const ExecutionContextKind k;
};

static inline ExecutionContext* unwrap(TF_ExecutionContext* ctx) {
  return reinterpret_cast<ExecutionContext*>(ctx);
}
static inline const ExecutionContext* unwrap(const TF_ExecutionContext* ctx) {
  return reinterpret_cast<const ExecutionContext*>(ctx);
}
static inline TF_ExecutionContext* wrap(ExecutionContext* ctx) {
  return reinterpret_cast<TF_ExecutionContext*>(ctx);
}
static inline const TF_ExecutionContext* wrap(const ExecutionContext* ctx) {
  return reinterpret_cast<const TF_ExecutionContext*>(ctx);
}

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
