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

#include "absl/types/variant.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_unified_experimental_private.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/strcat.h"

using tensorflow::string;
using tensorflow::internal::AbstractFunction;
using tensorflow::internal::AbstractOp;
using tensorflow::internal::AbstractTensor;
using tensorflow::internal::dynamic_cast_helper;
using tensorflow::internal::ExecutionContext;
using tensorflow::internal::OutputList;
using tensorflow::internal::unwrap;
using tensorflow::internal::wrap;

class TF_EagerContext;

struct EagerTensor : public AbstractTensor {
  TFE_TensorHandle* t = nullptr;
  EagerTensor() : AbstractTensor(kKind) {}
  explicit EagerTensor(TFE_TensorHandle* t) : AbstractTensor(kKind), t(t) {}
  ~EagerTensor() override { TFE_DeleteTensorHandle(t); }
  static constexpr AbstractTensorKind kKind = kEagerTensor;
};

class TF_EagerOp : public AbstractOp {
 public:
  explicit TF_EagerOp(TFE_Context* ctx) : AbstractOp(kKind), ctx_(ctx) {}
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

  ~TF_EagerOp() override { TFE_DeleteOp(op_); }
  static constexpr AbstractOpKind kKind = kEagerOp;

 private:
  friend class TF_EagerContext;  // For access to op_.
  TFE_Op* op_ = nullptr;
  TFE_Context* ctx_;
};

class TF_EagerContext : public ExecutionContext {
 public:
  TF_EagerContext() : ExecutionContext(kKind) {}

  void Build(TFE_ContextOptions* options, TF_Status* status) {
    eager_ctx_ = TFE_NewContext(options, status);
  }

  AbstractOp* CreateOperation() override {
    // TODO(srbs): Should the lifetime of this op be tied to the context.
    return new TF_EagerOp(eager_ctx_);
  }

  void ExecuteOperation(AbstractOp* op, int num_inputs,
                        AbstractTensor* const* inputs, OutputList* o,
                        TF_Status* s) override {
    auto* eager_op = dynamic_cast_helper<TF_EagerOp>(op);
    if (eager_op == nullptr) {
      TF_SetStatus(s, TF_INVALID_ARGUMENT,
                   "Unable to cast AbstractOp to TF_EagerOp.");
      return;
    }
    auto* tfe_op = eager_op->op_;
    if (TF_GetCode(s) != TF_OK) return;
    for (int i = 0; i < num_inputs; ++i) {
      auto* eager_tensor = dynamic_cast_helper<const EagerTensor>(inputs[i]);
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

  void RegisterFunction(AbstractFunction* afunc, TF_Status* s) override {
    auto* func = afunc->GetTfFunction(s);
    if (!func) {
      return;
    }
    TFE_ContextAddFunction(eager_ctx_, func, s);
  }

  ~TF_EagerContext() override { TFE_DeleteContext(eager_ctx_); }

  static constexpr ExecutionContextKind kKind = kEagerContext;

 private:
  friend TFE_Context* TF_ExecutionContextGetTFEContext(
      TF_ExecutionContext* ctx);
  TFE_Context* eager_ctx_;
};

TF_ExecutionContext* TF_NewEagerExecutionContext(TFE_ContextOptions* options,
                                                 TF_Status* s) {
  auto* ctx = new TF_EagerContext();
  ctx->Build(options, s);
  return wrap(ctx);
}

TF_AbstractTensor* TF_CreateAbstractTensorFromEagerTensor(TFE_TensorHandle* t,
                                                          TF_Status* s) {
  return wrap(new EagerTensor(t));
}

TFE_TensorHandle* TF_AbstractTensorGetEagerTensor(TF_AbstractTensor* at,
                                                  TF_Status* s) {
  auto* eager_tensor = dynamic_cast_helper<EagerTensor>(unwrap(at));
  if (!eager_tensor) {
    string msg = absl::StrCat("Not an eager tensor handle.",
                              reinterpret_cast<uintptr_t>(at));
    TF_SetStatus(s, TF_INVALID_ARGUMENT, msg.c_str());
    return nullptr;
  }
  return eager_tensor->t;
}

TFE_Context* TF_ExecutionContextGetTFEContext(TF_ExecutionContext* ctx) {
  return dynamic_cast_helper<TF_EagerContext>(unwrap(ctx))->eager_ctx_;
}
