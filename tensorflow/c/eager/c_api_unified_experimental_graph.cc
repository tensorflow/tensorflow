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

#include <memory>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::string;

namespace tensorflow {
namespace internal {

class GraphContext;

// GraphTensor wraps a `TF_Output`, i.e. a pointer to TF_Operation and the index
// into the list of outputs for the operation.
struct GraphTensor : public AbstractTensor {
  TF_Output output{};
  GraphContext* ctx = nullptr;
  GraphTensor() : AbstractTensor(kKind) {}
  GraphTensor(TF_Output output, GraphContext* ctx)
      : AbstractTensor(kKind), output(output), ctx(ctx) {}
  static constexpr AbstractTensorKind kKind = kGraphTensor;
};

// GraphOp wraps and populate a TF_OperationDescription.
class GraphOp : public AbstractOp {
 public:
  explicit GraphOp(TF_Graph* g) : AbstractOp(kKind), g_(g) {}
  void SetOpType(const char* const op_type, TF_Status* s) override {
    if (op_) {
      TF_SetStatus(
          s, TF_FAILED_PRECONDITION,
          strings::StrCat("SetOpType called on already built op.").c_str());
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
          strings::StrCat("SetOpName called on already built op.").c_str());
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
  ~GraphOp() override {}

  static constexpr AbstractOpKind kKind = kGraphOp;

 private:
  friend class GraphContext;  // For access to op_.
  TF_Graph* g_;
  std::unique_ptr<TF_OperationDescription> op_;
  // Hold `op_type` and `op_name` till both are available since we need both
  // to build a graph operation.
  const char* op_type_ = nullptr;
  const char* op_name_ = nullptr;
};

// GraphFunction is a thin wrapper over a TF_Function.
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

// GraphContext wraps a TF_Graph modeling a single function and manages the
// "execution" of operation, i.e. adding them to the function.
class GraphContext : public ExecutionContext {
 public:
  explicit GraphContext(const char* name)
      : ExecutionContext(kKind),
        graph_(new TF_Graph(), TF_DeleteGraph),
        name_(name) {}

  AbstractOp* CreateOperation() override {
    // TODO(srbs): Should the lifetime of this op be tied to the context.
    return new GraphOp(graph_.get());
  }

  void ExecuteOperation(AbstractOp* op, int num_inputs,
                        AbstractTensor* const* inputs, OutputList* o,
                        TF_Status* s) override {
    auto* graph_op = dyncast<GraphOp>(op);
    if (graph_op == nullptr) {
      TF_SetStatus(s, TF_INVALID_ARGUMENT,
                   "Unable to cast AbstractOp to TF_GraphOp.");
      return;
    }
    auto* tf_opdesc = graph_op->op_.release();
    if (tf_opdesc == nullptr) {
      TF_SetStatus(s, TF_INVALID_ARGUMENT, "AbstractOp is incomplete.");
      return;
    }
    for (int i = 0; i < num_inputs; ++i) {
      auto* graph_tensor = dyncast<GraphTensor>(inputs[i]);
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

  AbstractTensor* AddParameter(TF_DataType dtype, TF_Status* s) override {
    TF_OperationDescription* opdesc =
        TF_NewOperation(graph_.get(), "Placeholder",
                        absl::StrCat("_input_", inputs_.size()).c_str());
    TF_SetAttrType(opdesc, "dtype", dtype);
    auto* operation = TF_FinishOperation(opdesc, s);
    if (!s->status.ok()) return nullptr;

    inputs_.push_back(TF_Output{operation, 0});
    return new GraphTensor(inputs_.back(), this);
  }

  AbstractFunction* Finalize(OutputList* outputs, TF_Status* s) override {
    std::unique_ptr<GraphFunction> func(new GraphFunction);
    std::vector<TF_Output> graph_outputs;
    graph_outputs.reserve(outputs->outputs.size());
    for (AbstractTensor* abstract_output : outputs->outputs) {
      GraphTensor* output = dyncast<GraphTensor>(abstract_output);
      if (!output) {
        TF_SetStatus(s, TF_UNIMPLEMENTED,
                     "Returning a non-graph tensor from a function has not "
                     "been implemented yet.");
        return nullptr;
      }
      graph_outputs.push_back(output->output);
    }

    func->func = TF_GraphToFunction(
        graph_.get(), name_, 0, -1, nullptr, inputs_.size(), inputs_.data(),
        graph_outputs.size(), graph_outputs.data(), nullptr, nullptr, name_, s);
    if (TF_GetCode(s) != TF_OK) return nullptr;
    return func.release();
  }

  void RegisterFunction(AbstractFunction* func, TF_Status* s) override {
    TF_SetStatus(s, TF_UNIMPLEMENTED,
                 "Registering graph functions has not been implemented yet.");
  }

  ~GraphContext() override {}

  static constexpr ExecutionContextKind kKind = kGraphContext;

 private:
  std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)> graph_;
  std::vector<TF_Output> inputs_;
  const char* name_;
};

static ExecutionContext* GraphTracingFactory(const char* name, TF_Status* s) {
  return new GraphContext(name);
}

// Register the tracing implemented in this file as the default tracing engine.
static bool register_tracing = [] {
  RegisterTracingEngineFactory("graphdef", GraphTracingFactory);
  SetDefaultTracingEngine("graphdef");
  return true;
}();

}  // namespace internal
}  // namespace tensorflow
