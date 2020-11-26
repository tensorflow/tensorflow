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

#ifndef TENSORFLOW_C_EXPERIMENTAL_OP_HANDLER_INTERNAL_CC_
#define TENSORFLOW_C_EXPERIMENTAL_OP_HANDLER_INTERNAL_CC_

#include "tensorflow/c/experimental/op_handler/internal.h"

#include "tensorflow/c/conversion_macros.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/experimental/op_handler/wrapper_operation.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

OpHandlerContext::OpHandlerContext(AbstractContext* parent_ctx)
    : AbstractContext(kOpHandler), parent_ctx_(parent_ctx) {}
OpHandlerContext::~OpHandlerContext() {}
void OpHandlerContext::Release() { delete this; }
Status OpHandlerContext::RegisterFunction(AbstractFunction* function) {
  return parent_ctx_->RegisterFunction(function);
}

Status OpHandlerContext::RemoveFunction(const string& function) {
  return parent_ctx_->RemoveFunction(function);
}

void OpHandlerContext::set_default_handler(OpHandler* handler) {
  handler->Ref();
  default_handler_.reset(handler);
}

OpHandlerOperation* OpHandlerContext::CreateOperation() {
  OpHandlerOperation* result =
      new OpHandlerOperation(parent_ctx_->CreateOperation());
  if (default_handler_ != nullptr) {
    result->set_handler(default_handler_.get());
  }
  return result;
}

OpHandlerOperation::OpHandlerOperation(AbstractOperation* parent_op)
    : WrapperOperation(parent_op, kOpHandler) {}

OpHandler* OpHandlerOperation::get_handler() { return handler_.get(); }

void OpHandlerOperation::set_handler(OpHandler* handler) {
  if (handler != nullptr) {
    handler->Ref();
  }
  handler_.reset(handler);
}

Status OpHandlerOperation::Execute(absl::Span<AbstractTensorHandle*> retvals,
                                   int* num_retvals) {
  if (handler_ == nullptr) {
    return WrapperOperation::Execute(retvals, num_retvals);
  } else {
    return handler_->Execute(this, retvals, num_retvals);
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OP_HANDLER_INTERNAL_H_
