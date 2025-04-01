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
#include "tensorflow/c/experimental/gradients/tape/tape_context.h"

#include "absl/status/status.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/c/eager/gradients.h"
#include "tensorflow/c/experimental/gradients/tape/tape_operation.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gradients {
TapeContext::TapeContext(AbstractContext* c, Tape* tape,
                         const GradientRegistry& registry)
    : AbstractContext(kTape), parent_ctx_(c), tape_(tape), registry_(registry) {
  // TODO(srbs): Make AbstractContext ref counted.
  // parent_ctx_->Ref();
}
void TapeContext::Release() {
  // TODO(srbs): Change to Unref()
  delete this;
}
TapeContext::~TapeContext() {
  // TODO(srbs): Make AbstractContext ref counted.
  // parent_ctx_->Unref();
}
TapeOperation* TapeContext::CreateOperation() {
  return new TapeOperation(parent_ctx_->CreateOperation(), tape_, registry_);
}
absl::Status TapeContext::RegisterFunction(AbstractFunction* f) {
  return parent_ctx_->RegisterFunction(f);
}
absl::Status TapeContext::RemoveFunction(const string& func) {
  return parent_ctx_->RemoveFunction(func);
}

}  // namespace gradients
}  // namespace tensorflow
