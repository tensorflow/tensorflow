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

#ifndef TENSORFLOW_C_EXPERIMENTAL_OP_HANDLER_INTERNAL_H_
#define TENSORFLOW_C_EXPERIMENTAL_OP_HANDLER_INTERNAL_H_

#include "tensorflow/c/conversion_macros.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/experimental/op_handler/wrapper_operation.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class OpHandlerOperation;

// Op handlers are a convenient way to intercept and transform computation.
//
// The implementation is currently experimental and incomplete, but aims
// eventually to support tracing and replay of function bodies, gradients
// through copy operations, and a variety of hooks for things like debug
// strings. A public C API for op handlers is planned.
class OpHandler : public core::RefCounted {
 public:
  // Called on operation->Execute when operation->get_handler() == this.
  //
  // Allows the handler to customize or inspect `operation`'s execution.
  virtual Status Execute(OpHandlerOperation* operation,
                         absl::Span<AbstractTensorHandle*> retvals,
                         int* num_retvals) = 0;
  // Creates a new handler by merging this handler with `next_handler`.
  //
  // The new handler is expected to transform operations first with this handler
  // and then execute the resulting operations on `next_handler` (by calling
  // `OpHandlerOperation::set_handler` and passing `next_handler`). If this is
  // not possible then the merge operation should fail.
  virtual Status Merge(OpHandler* next_handler,
                       core::RefCountPtr<OpHandler>& merged_handler) = 0;
};

// Keeps some handler-specific metadata, but otherwise wraps a single
// AbstractOperation in the underlying context. The operation is created, its
// attributes set, etc., and at execution time it is presented to its handler,
// which may choose to execute it or simply inspect it and do something else.
//
// This is somewhat different than the Context approach, where the operation's
// construction is streamed through each layered Context. The streaming approach
// would require a much larger op handler public API, one function pointer per
// attribute type, and there is some ambiguity before an op is finalized about
// whether it should be presented as-is to handlers (regular operations) or
// replayed (function calls and control flow operations).
class OpHandlerOperation : public WrapperOperation {
 public:
  explicit OpHandlerOperation(AbstractOperation*);
  OpHandler* get_handler();
  void set_handler(OpHandler* handler);
  Status Execute(absl::Span<AbstractTensorHandle*> retvals,
                 int* num_retvals) override;

 protected:
  core::RefCountPtr<OpHandler> handler_;
};

// A context which allows a default handler to be set for new operations. It
// otherwise defers to the context it wraps.
//
// TODO(allenl): A stack of contexts and a stack of handlers look pretty similar
// in some ways. Having each handler be its own context seems almost doable,
// with things like copy operations and function/control flow replay being
// somewhat tricky (since they should be generated at the top of the handler
// stack and "caught" at the bottom). After handlers have evolved for a bit we
// should re-evaluate whether the handler+context concepts can be merged.
class OpHandlerContext : public AbstractContext {
 public:
  explicit OpHandlerContext(AbstractContext*);
  void Release() override;
  OpHandlerOperation* CreateOperation() override;
  Status RegisterFunction(AbstractFunction*) override;
  Status RemoveFunction(const string&) override;
  // For LLVM style RTTI.
  static bool classof(const AbstractContext* ptr) {
    return ptr->getKind() == kOpHandler;
  }
  ~OpHandlerContext() override;

  void set_default_handler(OpHandler* handler);

 private:
  AbstractContext* parent_ctx_;  // Not owned.
  core::RefCountPtr<OpHandler> default_handler_;
};

class ReleaseOpHandlerOperation {
 public:
  void operator()(OpHandlerOperation* operation) { operation->Release(); }
};

typedef std::unique_ptr<OpHandlerOperation, ReleaseOpHandlerOperation>
    OpHandlerOperationPtr;

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OP_HANDLER_INTERNAL_H_
