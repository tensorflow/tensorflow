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

#ifndef TENSORFLOW_C_EAGER_ABSTRACT_CONTEXT_H_
#define TENSORFLOW_C_EAGER_ABSTRACT_CONTEXT_H_

#include <memory>
#include <string>

#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "absl/status/status.h"

namespace tensorflow {

// Represents an abstract execution context.
//
// This serves as a factory for creating `AbstractOperation` instances and
// for registering traced functions. Operations created within a context must
// only be executed in that context.
//
// Implementations of this class may encapsulate state, such as an execution
// environment or a traced representation.
class AbstractContext {
 public:
  // Represents the kind of context.
  enum class ContextKind { kGraph, kMlir, kEager, kTfrt, kTape, kOpHandler };

  explicit AbstractContext(ContextKind kind) : kind_(kind) {}
  virtual ~AbstractContext() = default;

  // Returns the type of the context.
  ContextKind GetKind() const { return kind_; }

  // Releases all underlying resources, including the interface object.
  //
  // Note: The destructor is protected to prevent direct destruction of the
  // object, as the lifetime of this object may be managed through reference
  // counting. Clients must call `Release()` to destroy an instance of this
  // class.
  virtual void Release() = 0;

  // Creates and returns a new operation tied to this context.
  //
  // The returned object can be used to set operation attributes, add inputs,
  // and execute the operation either immediately or lazily (e.g., during
  // tracing).
  virtual AbstractOperation* CreateOperation() = 0;

  // Registers a function with this context. Once registered, the function
  // becomes available for use within this context by its name.
  virtual absl::Status RegisterFunction(AbstractFunction* function) = 0;

  // Removes a previously registered function from this context.
  //
  // `func_name` should match the `FunctionDef` signature's name.
  virtual absl::Status RemoveFunction(const std::string& func_name) = 0;

 protected:
  const ContextKind kind_;
};

namespace internal {

// Custom deleter for `AbstractContext` to ensure proper resource cleanup.
struct AbstractContextDeleter {
  void operator()(AbstractContext* context) const {
    if (context != nullptr) {
      context->Release();
    }
  }
};

}  // namespace internal

// A smart pointer type for managing `AbstractContext` instances.
using AbstractContextPtr =
    std::unique_ptr<AbstractContext, internal::AbstractContextDeleter>;

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_ABSTRACT_CONTEXT_H_




