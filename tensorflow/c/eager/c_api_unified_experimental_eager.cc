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

#include <vector>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::string;

namespace tensorflow {
namespace internal {

// Simple wrapper over a TFE_TensorHandle
struct EagerTensor : public AbstractTensor {
  TFE_TensorHandle* t = nullptr;
  EagerTensor() : AbstractTensor(kKind) {}
  explicit EagerTensor(TFE_TensorHandle* t) : AbstractTensor(kKind), t(t) {}
  ~EagerTensor() override { TFE_DeleteTensorHandle(t); }
  static constexpr AbstractTensorKind kKind = kEagerTensor;
};

// Simple wrapper over a TFE_Op
class EagerOp : public AbstractOp {
 public:
  explicit EagerOp(TFE_Context* ctx) : AbstractOp(kKind), ctx_(ctx) {}
  void SetOpType(const char* const op_type, TF_Status* s) override {
    op_ = TFE_NewOp(ctx_, op_type, s);
  }
  void SetOpName(const char* const op_name, TF_Status* s) override {
    // Name is ignored in eager mode.
  }
  void SetAttrType(const char* const attr_name, TF_DataType value,
                   TF_Status* s) override {
    if (op_ == nullptr) {
      TF_SetStatus(s, TF_FAILED_PRECONDITION,
                   "op_type must be specified before specifying attrs.");
      return;
    }
    TFE_OpSetAttrType(op_, attr_name, value);
  }

  ~EagerOp() override { TFE_DeleteOp(op_); }
  static constexpr AbstractOpKind kKind = kEagerOp;

 private:
  friend class EagerContext;  // For access to op_.
  TFE_Op* op_ = nullptr;
  TFE_Context* ctx_;
};

// Wraps a TFE_Context and dispatch EagerOp with EagerTensor inputs.
class EagerContext : public ExecutionContext {
 public:
  EagerContext() : ExecutionContext(kKind) {}

  void Build(TFE_ContextOptions* options, TF_Status* status) {
    eager_ctx_ = TFE_NewContext(options, status);
  }

  AbstractOp* CreateOperation() override {
    // TODO(srbs): Should the lifetime of this op be tied to the context.
    return new EagerOp(eager_ctx_);
  }

  void ExecuteOperation(AbstractOp* op, int num_inputs,
                        AbstractTensor* const* inputs, OutputList* o,
                        TF_Status* s) override {
    auto* eager_op = dyncast<EagerOp>(op);
    if (eager_op == nullptr) {
      TF_SetStatus(s, TF_INVALID_ARGUMENT,
                   "Unable to cast AbstractOp to TF_EagerOp.");
      return;
    }
    auto* tfe_op = eager_op->op_;
    if (TF_GetCode(s) != TF_OK) return;
    for (int i = 0; i < num_inputs; ++i) {
      auto* eager_tensor = dyncast<const EagerTensor>(inputs[i]);
      if (!eager_tensor) {
        TF_SetStatus(s, TF_INVALID_ARGUMENT, "Not an eager tensor.");
        return;
      }
      TFE_OpAddInput(tfe_op, eager_tensor->t, s);
      if (TF_GetCode(s) != TF_OK) return;
    }
    if (o->expected_num_outputs == -1) {
      string msg =
          "The number of outputs must be provided in eager mode. Use "
          "TF_OutputListSetNumOutputs.";
      TF_SetStatus(s, TF_INVALID_ARGUMENT, msg.c_str());
      return;
    }
    tensorflow::gtl::InlinedVector<TFE_TensorHandle*, 2> retvals;
    int num_retvals = o->expected_num_outputs;
    retvals.resize(num_retvals);
    TFE_Execute(tfe_op, retvals.data(), &num_retvals, s);
    if (TF_GetCode(s) != TF_OK) {
      return;
    }
    o->outputs.clear();
    o->outputs.reserve(num_retvals);
    for (int i = 0; i < num_retvals; ++i) {
      o->outputs.push_back(new EagerTensor(retvals[i]));
    }
  }

  AbstractTensor* AddParameter(TF_DataType dtype, TF_Status* s) override {
    TF_SetStatus(s, TF_INVALID_ARGUMENT,
                 "Can't add function parameter on an eager context.");
    return nullptr;
  }
  AbstractFunction* Finalize(OutputList* outputs, TF_Status* s) override {
    TF_SetStatus(s, TF_INVALID_ARGUMENT,
                 "Can't use finalize function on an eager context.");
    return nullptr;
  }

  void RegisterFunction(AbstractFunction* afunc, TF_Status* s) override {
    auto* func = afunc->GetTfFunction(s);
    if (!func) {
      return;
    }
    TFE_ContextAddFunction(eager_ctx_, func, s);
  }

  ~EagerContext() override { TFE_DeleteContext(eager_ctx_); }

  static constexpr ExecutionContextKind kKind = kEagerContext;

 private:
  friend TFE_Context* ::TF_ExecutionContextGetTFEContext(
      TF_ExecutionContext* ctx);
  TFE_Context* eager_ctx_;
};

}  // namespace internal
}  // namespace tensorflow

// =============================================================================
// Public C API entry points
// These are only the entry points specific to the Eager API.
// =============================================================================

using tensorflow::internal::dyncast;
using tensorflow::internal::unwrap;

TF_ExecutionContext* TF_NewEagerExecutionContext(TFE_ContextOptions* options,
                                                 TF_Status* s) {
  auto* ctx = new tensorflow::internal::EagerContext();
  ctx->Build(options, s);
  return wrap(ctx);
}

TF_AbstractTensor* TF_CreateAbstractTensorFromEagerTensor(TFE_TensorHandle* t,
                                                          TF_Status* s) {
  return wrap(new tensorflow::internal::EagerTensor(t));
}

TFE_TensorHandle* TF_AbstractTensorGetEagerTensor(TF_AbstractTensor* at,
                                                  TF_Status* s) {
  auto* eager_tensor = dyncast<tensorflow::internal::EagerTensor>(unwrap(at));
  if (!eager_tensor) {
    string msg = tensorflow::strings::StrCat("Not an eager tensor handle.",
                                             reinterpret_cast<uintptr_t>(at));
    TF_SetStatus(s, TF_INVALID_ARGUMENT, msg.c_str());
    return nullptr;
  }
  return eager_tensor->t;
}

TFE_Context* TF_ExecutionContextGetTFEContext(TF_ExecutionContext* ctx) {
  auto* eager_ctx = dyncast<tensorflow::internal::EagerContext>(unwrap(ctx));
  if (!eager_ctx) return nullptr;
  return eager_ctx->eager_ctx_;
}
