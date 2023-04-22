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

#include "tensorflow/c/experimental/op_handler/internal.h"

#include "absl/types/span.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class TestOpHandler : public OpHandler {
 public:
  TestOpHandler() : last_operation_(new std::string("")) {}
  Status Execute(OpHandlerOperation* operation,
                 absl::Span<AbstractTensorHandle*> retvals,
                 int* num_retvals) override {
    CHECK(operation->get_handler() == this);
    *last_operation_ = operation->Name();
    operation->set_handler(next_handler_.get());
    return operation->Execute(retvals, num_retvals);
  }
  Status Merge(OpHandler* next_handler,
               core::RefCountPtr<OpHandler>& merged_handler) override {
    merged_handler.reset(new TestOpHandler(next_handler, last_operation_));
    return Status::OK();
  }

  core::RefCountPtr<OpHandler> next_handler_ = nullptr;
  // Shared between merged handlers of this type.
  std::shared_ptr<std::string> last_operation_;

 private:
  TestOpHandler(OpHandler* next_handler,
                std::shared_ptr<std::string> last_operation)
      : next_handler_(next_handler), last_operation_(last_operation) {
    next_handler->Ref();
  }
};

TEST(INTERNAL_TEST, UseOpHandler) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TF_ExecutionContext, decltype(&TF_DeleteExecutionContext)>
      c_ctx(TF_NewEagerExecutionContext(opts.get(), status.get()),
            TF_DeleteExecutionContext);
  OpHandlerContext ctx(unwrap(c_ctx.get()));
  core::RefCountPtr<TestOpHandler> outer_handler(new TestOpHandler());
  core::RefCountPtr<TestOpHandler> inner_handler(new TestOpHandler());
  ctx.set_default_handler(outer_handler.get());
  OpHandlerOperationPtr op(ctx.CreateOperation());
  Status s = op->Reset("NoOp", "");
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  std::vector<AbstractTensorHandle*> retvals;
  int num_retvals = 0;
  EXPECT_EQ("", *outer_handler->last_operation_);
  s = op->Execute(absl::Span<AbstractTensorHandle*>(retvals), &num_retvals);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();

  EXPECT_EQ("NoOp", *outer_handler->last_operation_);
  *outer_handler->last_operation_ = "";
  EXPECT_EQ("", *inner_handler->last_operation_);

  // This op executes on both handlers, changing the state of `inner_handler`
  // since the handler has decided to preserve that state across merges.
  core::RefCountPtr<OpHandler> merged;
  s = inner_handler->Merge(outer_handler.get(), merged);
  ctx.set_default_handler(merged.get());
  op.reset(ctx.CreateOperation());
  s = op->Reset("NoOp", "");
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  s = op->Execute(absl::Span<AbstractTensorHandle*>(retvals), &num_retvals);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  EXPECT_EQ("NoOp", *inner_handler->last_operation_);
  EXPECT_EQ("NoOp", *outer_handler->last_operation_);

  inner_handler.reset();
  outer_handler.reset();
  op.reset(ctx.CreateOperation());
  s = op->Reset("NoOp", "");
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
  s = op->Execute(absl::Span<AbstractTensorHandle*>(retvals), &num_retvals);
  ASSERT_EQ(errors::OK, s.code()) << s.error_message();
}

}  // namespace tensorflow
