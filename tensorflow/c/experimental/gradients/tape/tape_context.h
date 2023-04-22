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
#ifndef TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_TAPE_TAPE_CONTEXT_H_
#define TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_TAPE_TAPE_CONTEXT_H_

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/experimental/gradients/tape/tape_operation.h"

namespace tensorflow {
namespace gradients {
class TapeContext : public AbstractContext {
 public:
  explicit TapeContext(AbstractContext*, Tape*, const GradientRegistry&);
  void Release() override;
  TapeOperation* CreateOperation() override;
  Status RegisterFunction(AbstractFunction*) override;
  Status RemoveFunction(const string& func) override;
  // For LLVM style RTTI.
  static bool classof(const AbstractContext* ptr) {
    return ptr->getKind() == kTape;
  }
  ~TapeContext() override;

 private:
  AbstractContext* parent_ctx_;  // Not owned.
  Tape* tape_;
  const GradientRegistry& registry_;
};
}  // namespace gradients
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_GRADIENTS_TAPE_TAPE_CONTEXT_H_
