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

void TF_DeleteExecutionContext(TF_ExecutionContext* c) { delete unwrap(c); }

class TF_GraphContext;
class TF_EagerContext;

struct EagerTensor : public AbstractTensor {
  TFE_TensorHandle* t = nullptr;
  EagerTensor() : AbstractTensor(kKind) {}
  explicit EagerTensor(TFE_TensorHandle* t) : AbstractTensor(kKind), t(t) {}
  ~EagerTensor() override { TFE_DeleteTensorHandle(t); }
  static constexpr AbstractTensorKind kKind = kEagerTensor;
};

struct GraphTensor : public AbstractTensor {
  TF_Output output{};
  TF_GraphContext* ctx = nullptr;
  GraphTensor() : AbstractTensor(kKind) {}
  GraphTensor(TF_Output output, TF_GraphContext* ctx)
      : AbstractTensor(kKind), output(output), ctx(ctx) {}
  static constexpr AbstractTensorKind kKind = kGraphTensor;
};

TF_AbstractOp* TF_NewAbstractOp(TF_ExecutionContext* c) {
  return wrap(unwrap(c)->CreateOperation());
}

void TF_DeleteAbstractOp(TF_AbstractOp* op) { delete unwrap(op); }

class TF_GraphOp : public AbstractOp {
 public:
  explicit TF_GraphOp(TF_Graph* g) : AbstractOp(kKind), g_(g) {}
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

  static constexpr AbstractOpKind kKind = kGraphOp;

 private:
  friend class TF_GraphContext;  // For access to op_.
  TF_Graph* g_;
  std::unique_ptr<TF_OperationDescription> op_;
  // Hold `op_type` and `op_name` till both are available since we need both
  // to build a graph operation.
  const char* op_type_ = nullptr;
  const char* op_name_ = nullptr;
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

struct GraphFunction : public AbstractFunction {
  TF_Function* func = nullptr;
  GraphFunction() : AbstractFunction(kKind) {}
  explicit GraphFunction(TF_Function* func)
      : AbstractFunction(kKind), func(func) {}
  ~GraphFunction() override {
    if (func) TF_DeleteFunction(func);
  }

  TF_Function* GetTfFunction(TF_Status* s) override { return func; }

  static constexpr AbstractFunctionKind kKind = kGraphFunc;
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

void TF_DeleteAbstractTensor(TF_AbstractTensor* t) { delete unwrap(t); }

class TF_GraphContext : public ExecutionContext {
 public:
  TF_GraphContext()
      : ExecutionContext(kKind), graph_(new TF_Graph(), TF_DeleteGraph) {}

  AbstractOp* CreateOperation() override {
    // TODO(srbs): Should the lifetime of this op be tied to the context.
    return new TF_GraphOp(graph_.get());
  }

  void ExecuteOperation(AbstractOp* op, int num_inputs,
                        AbstractTensor* const* inputs, OutputList* o,
                        TF_Status* s) override {
    auto* graph_op = dynamic_cast_helper<TF_GraphOp>(op);
    if (graph_op == nullptr) {
      TF_SetStatus(s, TF_INVALID_ARGUMENT,
                   "Unable to cast AbstractOp to TF_GraphOp.");
      return;
    }
    auto* tf_opdesc = graph_op->op_.release();
    for (int i = 0; i < num_inputs; ++i) {
      auto* graph_tensor = dynamic_cast_helper<GraphTensor>(inputs[i]);
      if (!graph_tensor) {
        TF_SetStatus(s, TF_INVALID_ARGUMENT,
                     "Capturing eager tensors is not supported yet.");
        return;
      } else {
        if (graph_tensor->ctx != this) {
          TF_SetStatus(
              s, TF_INVALID_ARGUMENT,
              "Capturing tensors from other graphs is not supported yet.");
          return;
        }
        TF_AddInput(tf_opdesc, graph_tensor->output);
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
      o->outputs.push_back(new GraphTensor({operation, i}, this));
    }
  }

  TF_Function* ToFunction(const char* fn_name, int num_inputs,
                          const GraphTensor* inputs, int num_outputs,
                          const GraphTensor* outputs, TF_Status* status) const {
    std::vector<TF_Output> graph_inputs;
    graph_inputs.resize(num_inputs);
    std::vector<TF_Output> graph_outputs;
    graph_outputs.resize(num_outputs);
    for (int i = 0; i < num_inputs; i++) {
      graph_inputs[i] = inputs[i].output;
    }
    for (int i = 0; i < num_outputs; i++) {
      graph_outputs[i] = outputs[i].output;
    }

    return TF_GraphToFunction(graph_.get(), fn_name, 0, -1, nullptr,
                              graph_inputs.size(), graph_inputs.data(),
                              graph_outputs.size(), graph_outputs.data(),
                              nullptr, nullptr, fn_name, status);
  }

  void RegisterFunction(AbstractFunction* func, TF_Status* s) override {
    TF_SetStatus(s, TF_UNIMPLEMENTED,
                 "Registering graph functions has not been implemented yet.");
  }

  ~TF_GraphContext() override {}

  static constexpr ExecutionContextKind kKind = kGraphContext;

 private:
  std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)> graph_;
};

TF_ExecutionContext* TF_NewGraphExecutionContext(TF_Status* s) {
  return wrap(new TF_GraphContext());
}
TF_ExecutionContext* TF_NewEagerExecutionContext(TFE_ContextOptions* options,
                                                 TF_Status* s) {
  auto* ctx = new TF_EagerContext();
  ctx->Build(options, s);
  return wrap(ctx);
}

TF_OutputList* TF_NewOutputList() { return wrap(new OutputList); }
void TF_DeleteOutputList(TF_OutputList* o) { delete unwrap(o); }
void TF_OutputListSetNumOutputs(TF_OutputList* o, int num_outputs,
                                TF_Status* s) {
  unwrap(o)->expected_num_outputs = num_outputs;
}
int TF_OutputListNumOutputs(TF_OutputList* o) {
  return unwrap(o)->outputs.size();
}
TF_AbstractTensor* TF_OutputListGet(TF_OutputList* o, int i) {
  return wrap(unwrap(o)->outputs[i]);
}

void TF_AbstractOpSetOpType(TF_AbstractOp* op, const char* const op_type,
                            TF_Status* s) {
  unwrap(op)->SetOpType(op_type, s);
}

void TF_AbstractOpSetOpName(TF_AbstractOp* op, const char* const op_name,
                            TF_Status* s) {
  unwrap(op)->SetOpName(op_name, s);
}

void TF_AbstractOpSetAttrType(TF_AbstractOp* op, const char* const attr_name,
                              TF_DataType value, TF_Status* s) {
  unwrap(op)->SetAttrType(attr_name, value, s);
}

void TF_ExecuteOperation(TF_AbstractOp* op, int num_inputs,
                         TF_AbstractTensor* const* inputs, TF_OutputList* o,
                         TF_ExecutionContext* ctx, TF_Status* s) {
  unwrap(ctx)->ExecuteOperation(unwrap(op), num_inputs, &unwrap(*inputs),
                                unwrap(o), s);
}

TF_AbstractFunction* TF_ExecutionContextToFunction(
    const TF_ExecutionContext* fn_body, const char* fn_name, int num_inputs,
    const TF_AbstractTensor* inputs, int num_outputs,
    const TF_AbstractTensor* outputs, TF_Status* status) {
  auto* graph_ctx = dynamic_cast_helper<const TF_GraphContext>(unwrap(fn_body));
  if (graph_ctx == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "fn_body is not a TF_GraphContext.");
    return nullptr;
  }
  auto* graph_inputs = dynamic_cast_helper<const GraphTensor>(unwrap(inputs));
  if (!graph_inputs) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "inputs aren't GraphTensors.");
    return nullptr;
  }
  auto* graph_outputs = dynamic_cast_helper<const GraphTensor>(unwrap(outputs));
  if (!graph_outputs) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "outputs aren't GraphTensors.");
    return nullptr;
  }
  GraphFunction* func = new GraphFunction;
  func->func = graph_ctx->ToFunction(fn_name, num_inputs, graph_inputs,
                                     num_outputs, graph_outputs, status);
  return wrap(func);
}

void TF_DeleteAbstractFunction(TF_AbstractFunction* func) {
  delete unwrap(func);
}

void TF_ExecutionContextRegisterFunction(TF_ExecutionContext* ctx,
                                         TF_AbstractFunction* func,
                                         TF_Status* s) {
  unwrap(ctx)->RegisterFunction(unwrap(func), s);
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
