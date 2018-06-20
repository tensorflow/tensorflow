/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_RESOURCE_VAR_H_
#define TENSORFLOW_CORE_FRAMEWORK_RESOURCE_VAR_H_

#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

// Resource stored by variables in the resource manager
// (new, resource-style version).
class Var : public ResourceBase {
 public:
  explicit Var(DataType dtype) : tensor_(dtype) {}
  // Not copyable or movable.
  Var(const Var&) = delete;
  Var& operator=(const Var&) = delete;

  // TODO(ebrevdo): Use LockSet instead of exposing mu.
  mutex* mu() { return &mu_; }
  Tensor* tensor() { return &tensor_; }

  string DebugString() override {
    return strings::StrCat(DataTypeString(tensor_.dtype()), "/",
                           tensor_.shape().DebugString());
  }

  // Only used in the resource variable path. In resource variables,
  // tensor.IsInitialized() can be true (i.e. have memory allocated to it) while
  // there is not a good value there due to a race condition, and it's possible
  // to stumble upon this during variable.initialized_value(). So it's best to
  // just store directly whether the variable is initialized.
  bool is_initialized = false;  // GUARDED_BY(mu_) but annotalysis doesn't like
                                // it.

 private:
  mutex mu_;
  Tensor tensor_;

  ~Var() override {}
};

}  //  end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_RESOURCE_VAR_H_
