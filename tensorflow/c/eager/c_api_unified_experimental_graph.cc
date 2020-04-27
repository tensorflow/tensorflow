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

class TF_GraphContext;

struct GraphTensor : public AbstractTensor {
  TF_Output output{};
  TF_GraphContext* ctx = nullptr;
  GraphTensor() : AbstractTensor(kKind) {}
  GraphTensor(TF_Output output, TF_GraphContext* ctx)
      : AbstractTensor(kKind), output(output), ctx(ctx) {}
  static constexpr AbstractTensorKind kKind = kGraphTensor;
};

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
