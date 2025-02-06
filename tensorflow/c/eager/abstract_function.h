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
#ifndef TENSORFLOW_C_EAGER_ABSTRACT_FUNCTION_H_
#define TENSORFLOW_C_EAGER_ABSTRACT_FUNCTION_H_

#include "absl/status/statusor.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/intrusive_ptr.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

class FunctionRecord;

// A traced function: this hides the complexity of converting the serialized
// representation between various supported formats e.g. FunctionDef and Mlir
// function.
class AbstractFunction : public core::RefCounted {
 protected:
  enum AbstractFunctionKind { kGraph, kMlir };
  explicit AbstractFunction(AbstractFunctionKind kind) : kind_(kind) {}

 public:
  // Returns which subclass is this instance of.
  AbstractFunctionKind getKind() const { return kind_; }

  // Returns the AbstractFunction as a FunctionDef.
  virtual absl::Status GetFunctionDef(const FunctionDef**) = 0;

  // Returns a shared reference to the wrapped function.
  virtual absl::StatusOr<core::RefCountPtr<FunctionRecord>>
  GetFunctionRecord() = 0;

 private:
  const AbstractFunctionKind kind_;
};

using AbstractFunctionPtr =
    tensorflow::core::IntrusivePtr<tensorflow::AbstractFunction>;

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_ABSTRACT_FUNCTION_H_
