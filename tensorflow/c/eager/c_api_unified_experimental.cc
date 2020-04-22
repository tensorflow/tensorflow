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

#include "tensorflow/c/eager/c_api_unified_experimental.h"

#include "absl/types/variant.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/strcat.h"

using tensorflow::string;

// =============================================================================
// Unified Execution APIs for Eager and tracing backends.
// =============================================================================

typedef void (*ExecuteOperation)(TF_AbstractOp* op, int num_inputs,
                                 TF_AbstractTensor* const* inputs,
                                 TF_OutputList* o, TF_ExecutionContext* ctx,
                                 TF_Status* s);

struct TF_ExecutionContext {
  // Needed to implement our own version of RTTI since dynamic_cast is not
  // supported in mobile builds.
  enum ExecutionContextKind { GraphContext, EagerContext };
  explicit TF_ExecutionContext(ExecutionContextKind kind) : k(kind) {}
  ExecutionContextKind getKind() const { return k; }

  virtual void ExecuteOperation(TF_AbstractOp* op, int num_inputs,
                                TF_AbstractTensor* const* inputs,
                                TF_OutputList* o, TF_Status* s) = 0;
  virtual TF_AbstractOp* CreateOperation() = 0;
  virtual void RegisterFunction(TF_AbstractFunction* func, TF_Status* s) = 0;
  virtual ~TF_ExecutionContext() {}

 private:
  const ExecutionContextKind k;
};

void TF_DeleteExecutionContext(TF_ExecutionContext* c) { delete c; }

template <typename T, typename S>
T* dynamic_cast_helper(S source) {
  if (source->getKind() != T::kKind) {
    return nullptr;
  }
  return tensorflow::down_cast<T*>(source);
}

class TF_GraphContext;
class TF_EagerContext;

struct TF_GraphTensor {
  TF_Output output;
  TF_GraphContext* ctx;
};

struct TF_AbstractTensor {
  absl::variant<TFE_TensorHandle*, TF_GraphTensor*> t;

  ~TF_AbstractTensor() {
    if (absl::holds_alternative<TFE_TensorHandle*>(t)) {
      TFE_DeleteTensorHandle(absl::get<TFE_TensorHandle*>(t));
    } else if (absl::holds_alternative<TF_GraphTensor*>(t)) {
      delete absl::get<TF_GraphTensor*>(t);
    }
  }
};

struct TF_AbstractOp {
  // Needed to implement our own version of RTTI since dynamic_cast is not
  // supported in mobile builds.
  enum AbstractOpKind { GraphOp, EagerOp };
  explicit TF_AbstractOp(AbstractOpKind kind) : k(kind) {}
  AbstractOpKind getKind() const { return k; }
  virtual void SetOpType(const char* const op_type, TF_Status* s) = 0;
  virtual void SetOpName(const char* const op_name, TF_Status* s) = 0;
  virtual void SetAttrType(const char* const attr_name, TF_DataType value,
                           TF_Status* s) = 0;
  virtual ~TF_AbstractOp() {}

 private:
  const AbstractOpKind k;
};

TF_AbstractOp* TF_NewAbstractOp(TF_ExecutionContext* c) {
  return c->CreateOperation();
}

void TF_DeleteAbstractOp(TF_AbstractOp* op) { delete op; }

class TF_GraphOp : public TF_AbstractOp {
 public:
  explicit TF_GraphOp(TF_Graph* g) : TF_AbstractOp(kKind), g_(g) {}
  void SetOpType(const char* const op_type, TF_Status* s) override {
    if (op_) {
      TF_SetStatus(
          s, TF_FAILED_PRECONDITION,
          absl::StrCat("SetOpType called on already built op.").c_str());
      return;
    }
    if (op_name_ != nullptr) {
      op_.reset(TF_NewOperation(g_, op_type, op_name_));
      op_name_ = nullptr;
    } else {
      op_type_ = op_type;
    }
  }
  void SetOpName(const char* const op_name, TF_Status* s) override {
    if (op_) {
      TF_SetStatus(
          s, TF_FAILED_PRECONDITION,
          absl::StrCat("SetOpName called on already built op.").c_str());
      return;
    }
    if (op_type_ != nullptr) {
      op_.reset(TF_NewOperation(g_, op_type_, op_name));
      op_type_ = nullptr;
    } else {
      op_name_ = op_name;
    }
  }
  void SetAttrType(const char* const attr_name, TF_DataType value,
                   TF_Status* s) override {
    if (!op_) {
      TF_SetStatus(
          s, TF_FAILED_PRECONDITION,
          "op_type and op_name must be specified before specifying attrs.");
      return;
    }
    TF_SetAttrType(op_.get(), attr_name, value);
  }
  ~TF_GraphOp() override {}

  static constexpr AbstractOpKind kKind = GraphOp;

 private:
  friend class TF_GraphContext;  // For access to op_.
  TF_Graph* g_;
  std::unique_ptr<TF_OperationDescription> op_;
  // Hold `op_type` and `op_name` till both are available since we need both
  // to build a graph operation.
  const char* op_type_ = nullptr;
  const char* op_name_ = nullptr;
};

class TF_EagerOp : public TF_AbstractOp {
 public:
  explicit TF_EagerOp(TFE_Context* ctx) : TF_AbstractOp(kKind), ctx_(ctx) {}
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
  static constexpr AbstractOpKind kKind = EagerOp;

 private:
  friend class TF_EagerContext;  // For access to op_.
  TFE_Op* op_ = nullptr;
  TFE_Context* ctx_;
};

bool IsEagerTensor(const TF_AbstractTensor* const t) {
  return absl::holds_alternative<TFE_TensorHandle*>(t->t);
}

struct TF_OutputList {
  std::vector<TF_AbstractTensor*> outputs;
  int expected_num_outputs = -1;
};

struct TF_AbstractFunction {
  TF_Function* func = nullptr;

  ~TF_AbstractFunction() { TF_DeleteFunction(func); }
};

class TF_EagerContext : public TF_ExecutionContext {
 public:
  TF_EagerContext() : TF_ExecutionContext(kKind) {}

  void Build(TFE_ContextOptions* options, TF_Status* status) {
    eager_ctx_ = TFE_NewContext(options, status);
  }

  TF_AbstractOp* CreateOperation() override {
    // TODO(srbs): Should the lifetime of this op be tied to the context.
    return new TF_EagerOp(eager_ctx_);
  }

  void ExecuteOperation(TF_AbstractOp* op, int num_inputs,
                        TF_AbstractTensor* const* inputs, TF_OutputList* o,
                        TF_Status* s) override {
    auto* eager_op = dynamic_cast_helper<TF_EagerOp>(op);
    if (eager_op == nullptr) {
      TF_SetStatus(s, TF_INVALID_ARGUMENT,
                   "Unable to cast TF_AbstractOp to TF_EagerOp.");
      return;
    }
    auto* tfe_op = eager_op->op_;
    if (TF_GetCode(s) != TF_OK) return;
    for (int i = 0; i < num_inputs; ++i) {
      if (!IsEagerTensor(inputs[i])) {
        TF_SetStatus(s, TF_INVALID_ARGUMENT, "Not an eager tensor.");
        return;
      }
      TFE_OpAddInput(tfe_op, absl::get<TFE_TensorHandle*>(inputs[i]->t), s);
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
      auto* t = new TF_AbstractTensor();
      t->t = retvals[i];
      o->outputs.push_back(t);
    }
  }

  void RegisterFunction(TF_AbstractFunction* func, TF_Status* s) override {
    TFE_ContextAddFunction(eager_ctx_, func->func, s);
  }

  ~TF_EagerContext() override { TFE_DeleteContext(eager_ctx_); }

  static constexpr ExecutionContextKind kKind = EagerContext;

 private:
  friend TFE_Context* TF_ExecutionContextGetTFEContext(
      TF_ExecutionContext* ctx);
  TFE_Context* eager_ctx_;
};

void TF_DeleteAbstractTensor(TF_AbstractTensor* t) { delete t; }

TF_GraphContext* GetGraphContext(TF_AbstractTensor const* t) {
  return absl::get<TF_GraphTensor*>(t->t)->ctx;
}

class TF_GraphContext : public TF_ExecutionContext {
 public:
  TF_GraphContext()
      : TF_ExecutionContext(kKind), graph_(new TF_Graph(), TF_DeleteGraph) {}

  TF_AbstractOp* CreateOperation() override {
    // TODO(srbs): Should the lifetime of this op be tied to the context.
    return new TF_GraphOp(graph_.get());
  }

  void ExecuteOperation(TF_AbstractOp* op, int num_inputs,
                        TF_AbstractTensor* const* inputs, TF_OutputList* o,
                        TF_Status* s) override {
    auto* graph_op = dynamic_cast_helper<TF_GraphOp>(op);
    if (graph_op == nullptr) {
      TF_SetStatus(s, TF_INVALID_ARGUMENT,
                   "Unable to cast TF_AbstractOp to TF_GraphOp.");
      return;
    }
    auto* tf_opdesc = graph_op->op_.release();
    for (int i = 0; i < num_inputs; ++i) {
      auto* input = inputs[i];
      if (IsEagerTensor(input)) {
        TF_SetStatus(s, TF_INVALID_ARGUMENT,
                     "Capturing eager tensors is not supported yet.");
        return;
      } else {
        if (GetGraphContext(input) != this) {
          TF_SetStatus(
              s, TF_INVALID_ARGUMENT,
              "Capturing tensors from other graphs is not supported yet.");
          return;
        }
        TF_AddInput(tf_opdesc, absl::get<TF_GraphTensor*>(input->t)->output);
      }
    }
    auto* operation = TF_FinishOperation(tf_opdesc, s);
    // TF_FinishOperation deletes `tf_opdesc` so clear its reference.
    graph_op->op_ = nullptr;
    if (TF_GetCode(s) != TF_OK) return;
    int num_outputs = TF_OperationNumOutputs(operation);
    o->outputs.clear();
    o->outputs.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      auto* t = new TF_AbstractTensor;
      TF_GraphTensor* graph_t = new TF_GraphTensor;
      graph_t->ctx = this;
      graph_t->output = {operation, i};
      t->t = graph_t;
      o->outputs.push_back(t);
    }
  }

  TF_Function* ToFunction(const char* fn_name, int num_inputs,
                          const TF_AbstractTensor* inputs, int num_outputs,
                          const TF_AbstractTensor* outputs,
                          TF_Status* status) const {
    std::vector<TF_Output> graph_inputs;
    graph_inputs.resize(num_inputs);
    std::vector<TF_Output> graph_outputs;
    graph_outputs.resize(num_outputs);
    for (int i = 0; i < num_inputs; i++) {
      graph_inputs[i] = absl::get<TF_GraphTensor*>(inputs[i].t)->output;
    }
    for (int i = 0; i < num_outputs; i++) {
      graph_outputs[i] = absl::get<TF_GraphTensor*>(outputs[i].t)->output;
    }

    return TF_GraphToFunction(graph_.get(), fn_name, 0, -1, nullptr,
                              graph_inputs.size(), graph_inputs.data(),
                              graph_outputs.size(), graph_outputs.data(),
                              nullptr, nullptr, fn_name, status);
  }

  void RegisterFunction(TF_AbstractFunction* func, TF_Status* s) override {
    TF_SetStatus(s, TF_UNIMPLEMENTED,
                 "Registering graph functions has not been implemented yet.");
  }

  ~TF_GraphContext() override {}

  static constexpr ExecutionContextKind kKind = GraphContext;

 private:
  std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)> graph_;
};

struct TF_GraphContextOptions {};
struct TF_EagerContextOptions {
  explicit TF_EagerContextOptions(TFE_ContextOptions* options)
      : options(options) {}
  TFE_ContextOptions* options;  // Not owned.
};

struct TF_ExecutionContextOptions {
  absl::variant<TF_GraphContextOptions*, TF_EagerContextOptions*> options;
  ~TF_ExecutionContextOptions() {
    if (absl::holds_alternative<TF_GraphContextOptions*>(options)) {
      delete absl::get<TF_GraphContextOptions*>(options);
    } else if (absl::holds_alternative<TF_EagerContextOptions*>(options)) {
      delete absl::get<TF_EagerContextOptions*>(options);
    }
  }
};

TF_ExecutionContextOptions* TF_NewGraphContextOptions() {
  auto* options = new TF_ExecutionContextOptions();
  options->options = new TF_GraphContextOptions();
  return options;
}

void TF_DeleteExecutionContextOptions(TF_ExecutionContextOptions* options) {
  delete options;
}

TF_ExecutionContextOptions* TF_NewEagerContextOptions(
    TFE_ContextOptions* tfe_options) {
  auto* options = new TF_ExecutionContextOptions();
  options->options = new TF_EagerContextOptions(tfe_options);
  return options;
}

TF_ExecutionContext* TF_NewExecutionContext(TF_ExecutionContextOptions* options,
                                            TF_Status* s) {
  if (absl::holds_alternative<TF_EagerContextOptions*>(options->options)) {
    auto* ctx = new TF_EagerContext();
    ctx->Build(absl::get<TF_EagerContextOptions*>(options->options)->options,
               s);
    return ctx;
  } else {
    return new TF_GraphContext();
  }
}

TF_OutputList* TF_NewOutputList() { return new TF_OutputList; }
void TF_DeleteOutputList(TF_OutputList* o) { delete o; }
void TF_OutputListSetNumOutputs(TF_OutputList* o, int num_outputs,
                                TF_Status* s) {
  o->expected_num_outputs = num_outputs;
}
int TF_OutputListNumOutputs(TF_OutputList* o) { return o->outputs.size(); }
TF_AbstractTensor* TF_OutputListGet(TF_OutputList* o, int i) {
  return o->outputs[i];
}

void TF_AbstractOpSetOpType(TF_AbstractOp* op, const char* const op_type,
                            TF_Status* s) {
  op->SetOpType(op_type, s);
}

void TF_AbstractOpSetOpName(TF_AbstractOp* op, const char* const op_name,
                            TF_Status* s) {
  op->SetOpName(op_name, s);
}

void TF_AbstractOpSetAttrType(TF_AbstractOp* op, const char* const attr_name,
                              TF_DataType value, TF_Status* s) {
  op->SetAttrType(attr_name, value, s);
}

void TF_ExecuteOperation(TF_AbstractOp* op, int num_inputs,
                         TF_AbstractTensor* const* inputs, TF_OutputList* o,
                         TF_ExecutionContext* ctx, TF_Status* s) {
  ctx->ExecuteOperation(op, num_inputs, inputs, o, s);
}

TF_AbstractFunction* TF_ExecutionContextToFunction(
    const TF_ExecutionContext* fn_body, const char* fn_name, int num_inputs,
    const TF_AbstractTensor* inputs, int num_outputs,
    const TF_AbstractTensor* outputs, TF_Status* status) {
  auto* graph_ctx = dynamic_cast_helper<const TF_GraphContext>(fn_body);
  if (graph_ctx == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "fn_body is not a TF_GraphContext.");
    return nullptr;
  }
  TF_AbstractFunction* func = new TF_AbstractFunction;
  func->func = graph_ctx->ToFunction(fn_name, num_inputs, inputs, num_outputs,
                                     outputs, status);
  return func;
}

void TF_DeleteAbstractFunction(TF_AbstractFunction* func) { delete func; }

void TF_ExecutionContextRegisterFunction(TF_ExecutionContext* ctx,
                                         TF_AbstractFunction* func,
                                         TF_Status* s) {
  ctx->RegisterFunction(func, s);
}

// Temporary APIs till we figure out how to create scalar valued Eager
// tensors and how to get value out of eager abstract tensors.
TF_AbstractTensor* TF_NewAbstractTensor() {
  TF_AbstractTensor* t = new TF_AbstractTensor;
  return t;
}

void TF_AbstractTensorSetEagerTensor(TF_AbstractTensor* at, TFE_TensorHandle* t,
                                     TF_Status* s) {
  at->t = t;
}

TFE_TensorHandle* TF_AbstractTensorGetEagerTensor(TF_AbstractTensor* at,
                                                  TF_Status* s) {
  if (!absl::holds_alternative<TFE_TensorHandle*>(at->t)) {
    string msg = absl::StrCat("Not an eager tensor handle.",
                              reinterpret_cast<uintptr_t>(at));
    TF_SetStatus(s, TF_INVALID_ARGUMENT, msg.c_str());
    return nullptr;
  }
  return absl::get<TFE_TensorHandle*>(at->t);
}

TFE_Context* TF_ExecutionContextGetTFEContext(TF_ExecutionContext* ctx) {
  return dynamic_cast_helper<TF_EagerContext>(ctx)->eager_ctx_;
}
