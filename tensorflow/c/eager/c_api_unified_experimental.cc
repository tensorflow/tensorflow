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
  explicit TF_ExecutionContext() {}
  absl::variant<TFE_Context*, TF_GraphContext*> ctx;
  ExecuteOperation execution_callback;
};

struct TF_AbstractTensor {
  absl::variant<TFE_TensorHandle*, TF_GraphTensor*> t;
};

struct TF_AbstractOp {
  string op_type;
  string op_name;
};

TF_ExecutionContext* TF_NewExecutionContext() {
  return new TF_ExecutionContext();
}

void TF_DeleteExecutionContext(TF_ExecutionContext* c) { delete c; }

TF_AbstractOp* TF_NewAbstractOp() {
  TF_AbstractOp* op = new TF_AbstractOp;
  return op;
}

void TF_DeleteAbstractOp(TF_AbstractOp* op) { delete op; }

TF_AbstractTensor* TF_NewAbstractTensor() {
  TF_AbstractTensor* t = new TF_AbstractTensor;
  return t;
}

void TF_DeleteAbstractTensor(TF_AbstractTensor* t) { delete t; }

struct TF_GraphContext {
  TF_Graph* graph;
  // TODO(srbs): Handle captures.
};

TF_GraphContext* TF_NewGraphContext(TF_Graph* g) {
  auto ctx = new TF_GraphContext;
  ctx->graph = g;
  return ctx;
}

void TF_DeleteGraphContext(TF_GraphContext* ctx) { delete ctx; }

struct TF_GraphTensor {
  TF_Output output;
  TF_GraphContext* ctx;
};
TF_GraphTensor* TF_NewGraphTensor(TF_GraphContext* ctx, TF_Output output,
                                  TF_Status* s) {
  TF_GraphTensor* t = new TF_GraphTensor;
  t->output = output;
  t->ctx = ctx;
  return t;
}
TF_Output TF_GraphTensorToOutput(const TF_GraphTensor* const t, TF_Status* s) {
  return t->output;
}
void TF_DeleteGraphTensor(TF_GraphTensor* t) { delete t; }
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
void TF_AbstractTensorSetGraphTensor(TF_AbstractTensor* at, TF_GraphTensor* t,
                                     TF_Status* s) {
  at->t = t;
}
TF_GraphTensor* TF_AbstractTensorGetGraphTensor(TF_AbstractTensor* at,
                                                TF_Status* s) {
  if (!absl::holds_alternative<TF_GraphTensor*>(at->t)) {
    string msg = absl::StrCat("Not an graph tensor handle.");
    TF_SetStatus(s, TF_INVALID_ARGUMENT, msg.c_str());
    return nullptr;
  }
  return absl::get<TF_GraphTensor*>(at->t);
}

bool IsEagerTensor(const TF_AbstractTensor* const t) {
  return absl::holds_alternative<TFE_TensorHandle*>(t->t);
}

struct TF_OutputList {
  std::vector<TF_AbstractTensor*> outputs;
  int expected_num_outputs = -1;
};

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

void ExecuteOperationEager(TF_AbstractOp* op, int num_inputs,
                           TF_AbstractTensor* const* inputs, TF_OutputList* o,
                           TF_ExecutionContext* ctx, TF_Status* s) {
  auto* tfe_op =
      TFE_NewOp(absl::get<TFE_Context*>(ctx->ctx), op->op_type.c_str(), s);
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
  TFE_DeleteOp(tfe_op);
  if (TF_GetCode(s) != TF_OK) {
    return;
  }
  o->outputs.clear();
  o->outputs.reserve(num_retvals);
  for (int i = 0; i < num_retvals; ++i) {
    auto* t = TF_NewAbstractTensor();
    t->t = retvals[i];
    o->outputs.push_back(t);
  }
}

TF_GraphContext* GetGraphContext(TF_AbstractTensor const* t) {
  return absl::get<TF_GraphTensor*>(t->t)->ctx;
}

void ExecuteOperationGraph(TF_AbstractOp* op, int num_inputs,
                           TF_AbstractTensor* const* inputs, TF_OutputList* o,
                           TF_ExecutionContext* ctx, TF_Status* s) {
  TF_GraphContext* graph_ctx = absl::get<TF_GraphContext*>(ctx->ctx);
  TF_Graph* g = graph_ctx->graph;
  auto* tf_opdesc =
      TF_NewOperation(g, op->op_type.c_str(), op->op_name.c_str());
  for (int i = 0; i < num_inputs; ++i) {
    auto* input = inputs[i];
    if (IsEagerTensor(input)) {
      TF_SetStatus(s, TF_INVALID_ARGUMENT,
                   "Capturing eager tensors is not supported yet.");
      return;
    } else {
      if (GetGraphContext(input) != graph_ctx) {
        TF_SetStatus(
            s, TF_INVALID_ARGUMENT,
            "Capturing tensors from other graphs is not supported yet.");
        return;
      }
      TF_AddInput(tf_opdesc, absl::get<TF_GraphTensor*>(input->t)->output);
    }
  }
  auto* operation = TF_FinishOperation(tf_opdesc, s);
  if (TF_GetCode(s) != TF_OK) return;
  int num_outputs = TF_OperationNumOutputs(operation);
  o->outputs.clear();
  o->outputs.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    auto* t = TF_NewAbstractTensor();
    TF_GraphTensor* output_t = TF_NewGraphTensor(graph_ctx, {operation, i}, s);
    if (TF_GetCode(s) != TF_OK) {
      return;
    }
    t->t = output_t;
    o->outputs.push_back(t);
  }
}

void TF_ExecutionContextSetEagerContext(TF_ExecutionContext* context,
                                        TFE_Context* eager_context,
                                        TF_Status* s) {
  context->ctx = eager_context;
  context->execution_callback = &ExecuteOperationEager;
}

void TF_ExecutionContextSetGraphContext(TF_ExecutionContext* context,
                                        TF_GraphContext* graph_context,
                                        TF_Status* s) {
  context->ctx = graph_context;
  context->execution_callback = &ExecuteOperationGraph;
}

void TF_AbstractOpSetOpType(TF_AbstractOp* op, const char* const op_type,
                            TF_Status* s) {
  op->op_type = op_type;
}

void TF_AbstractOpSetOpName(TF_AbstractOp* op, const char* const op_name,
                            TF_Status* s) {
  op->op_name = op_name;
}

void TF_ExecuteOperation(TF_AbstractOp* op, int num_inputs,
                         TF_AbstractTensor* const* inputs, TF_OutputList* o,
                         TF_ExecutionContext* ctx, TF_Status* s) {
  ctx->execution_callback(op, num_inputs, inputs, o, ctx, s);
}
