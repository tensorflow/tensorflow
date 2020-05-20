/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/flex/kernel.h"

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/delegates/flex/delegate_data.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_type.h"

// Note: this is part of TF Lite's Flex delegation code which is to be
// completed soon.

// This is the TF Lite op that is created by the flex delegate to handle
// execution of a supported subgraph. The usual flow is that the delegate
// informs the interpreter of supported nodes in a graph, and each supported
// subgraph is replaced with one instance of this kernel.
//
// The kernel is initialized with TfLiteDelegateParams from which we retrieve
// the global EagerContext and BufferMap, as well as a list of inputs and
// outputs to the subgraph. Those are used to build the OpData, with a list of
// TensorFlow Ops that should be executed in order (which we call an OpNode).
//
// For each node included in the subgraph, we query the interpreter and
// retrieve the associated NodeDef, which is then used to configure the
// corresponding TensorFlow/Eager Op.

namespace tflite {
namespace flex {
namespace kernel {

struct OpNode;

// Represents the origin of a given tensor as a reference to the output
// of an upstream node.
struct TensorSource {
  OpNode* node;
  int node_output_index;
};

// A list of inputs of a given node of the TensorFlow/Eager graph.
class OpInputs {
 public:
  explicit OpInputs(const TfLiteIntArray* indexes) {
    for (int index : TfLiteIntArrayView(indexes)) {
      inputs_.push_back(index);
    }
    forwardable_.resize(inputs_.size());
  }
  ~OpInputs() {}

  int Size() const { return inputs_.size(); }

  int TfLiteIndex(int i) const { return inputs_[i]; }

  // Given a map relating tensors to the node that originates them, populate a
  // list of sources for the tensors in this class.
  void InitializeTensorSources(
      const std::map<int, TensorSource>& tflite_tensor_sources) {
    sources_.clear();
    for (int i : inputs_) {
      auto it = tflite_tensor_sources.find(i);
      if (it == tflite_tensor_sources.end()) {
        sources_.push_back({nullptr, 0});
      } else {
        sources_.push_back(it->second);
      }
    }
  }

  void SetForwardable(int i, bool v) { forwardable_[i] = v; }

  bool IsForwardable(int i) const { return forwardable_[i]; }

  TensorSource GetTensorSource(int i) const { return sources_[i]; }

 private:
  std::vector<int> inputs_;
  std::vector<TensorSource> sources_;

  // List of tensors that can be used by TF in its forwarding optimization.
  // Doing so allows an input tensor to be modified and used as the output
  // tensor. The delegate takes care of not holding any references to tensors
  // in this list while Eager is executing the corresponding op.
  std::vector<int> forwardable_;
};

// A list of outputs of a given node of the TensorFlow/Eager graph, along with
// the actual outputs of the EagerOperation.
class OpOutputs {
 public:
  explicit OpOutputs(const TfLiteIntArray* indexes) {
    for (int index : TfLiteIntArrayView(indexes)) {
      outputs_.push_back(index);
    }
    vector_.resize(outputs_.size());
  }
  ~OpOutputs() { ResetTensorHandles(); }

  // Stores information about which of the tensors in this class are also
  // outputs of the sugbraph.
  void InitializeGraphOutputs(const std::set<int>& subgraph_outputs) {
    subgraph_outputs_.clear();
    for (int i : outputs_) {
      subgraph_outputs_.push_back(subgraph_outputs.count(i) > 0);
    }
  }

  // Returns true if the tensor given by index 'i' is an output of the entire
  // subgraph.
  bool IsSubgraphOutput(int i) const { return subgraph_outputs_[i]; }

  // Returns a handle to a given tensor and, optionally, remove it from the
  // internal vector.
  tensorflow::TensorHandle* GetHandle(int i, bool remove) {
    auto* handle = vector_[i];
    if (!remove) {
      handle->Ref();
    } else {
      // Don't increase the ref-count. Instead, simply take it out of the
      // vector.
      vector_[i] = nullptr;
    }
    return handle;
  }

  int Size() const { return outputs_.size(); }

  int TfLiteIndex(int i) const { return outputs_[i]; }

  // Carefully unreference all the handles in the eager output vector.
  void ResetTensorHandles() {
    for (int i = 0; i < vector_.size(); ++i) {
      if (vector_[i]) {
        vector_[i]->Unref();
        vector_[i] = nullptr;
      }
    }
  }

  tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 2>*
  GetTensorHandles() {
    return &vector_;
  }

 private:
  std::vector<int> outputs_;
  std::vector<bool> subgraph_outputs_;
  tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 2> vector_;
};

// A single node within the larger 'op'. Note that this kernel executes many
// TensorFlow ops within a single TF Lite op.
class OpNode {
 public:
  OpNode(const TfLiteIntArray* inputs, const TfLiteIntArray* outputs)
      : inputs_(inputs), outputs_(outputs) {}
  ~OpNode() {
    if (op_) ClearEagerInputs();
  }

  const string& name() const { return name_; }
  void set_name(const string& name) { name_ = name; }

  int index() const { return index_; }
  void set_index(int index) { index_ = index; }

  const tensorflow::NodeDef& nodedef() const { return nodedef_; }

  const OpInputs& inputs() const { return inputs_; }
  OpInputs* mutable_inputs() { return &inputs_; }

  const OpOutputs& outputs() const { return outputs_; }
  OpOutputs* mutable_outputs() { return &outputs_; }

  int NumInputs() const { return inputs_.Size(); }
  int NumOutputs() const { return outputs_.Size(); }

  tensorflow::EagerOperation* op() { return op_.get(); }

  tensorflow::Status InitializeNodeDef(const void* custom_initial_data,
                                       int custom_initial_data_size) {
    if (!custom_initial_data) {
      return tensorflow::errors::Internal(
          "Cannot convert empty data into a valid NodeDef");
    }
    // The flexbuffer contains a vector where the first elements is the
    // op name and the second is a serialized NodeDef.
    const flexbuffers::Vector& v =
        flexbuffers::GetRoot(
            reinterpret_cast<const uint8_t*>(custom_initial_data),
            custom_initial_data_size)
            .AsVector();

    name_ = v[0].AsString().str();
    if (!nodedef_.ParseFromString(v[1].AsString().str())) {
      nodedef_.Clear();
      return tensorflow::errors::Internal(
          "Failed to parse data into a valid NodeDef");
    }

    // Fill NodeDef with defaults if it's a valid op.
    const tensorflow::OpRegistrationData* op_reg_data;
    TF_RETURN_IF_ERROR(
        tensorflow::OpRegistry::Global()->LookUp(nodedef_.op(), &op_reg_data));
    AddDefaultsToNodeDef(op_reg_data->op_def, &nodedef_);

    return tensorflow::Status::OK();
  }

  // Build thew new EagerOperation. In case of error, the returned 'op' is
  // guaranteed to be 'nullptr'.
  tensorflow::Status BuildEagerOp(tensorflow::EagerContext* eager_context) {
    op_.reset(new tensorflow::EagerOperation(eager_context));
    TF_RETURN_IF_ERROR(op_->Reset(name_.c_str(), nullptr, false, nullptr));
    if (op_->is_function()) {
      op_.reset();
      return tensorflow::errors::NotFound(
          "Operation '", name_,
          "' is not registered.  (while processing attributes of '", name_,
          "')");
    }

    op_->MutableAttrs()->NumInputs(inputs_.Size());
    for (const auto& attr : nodedef_.attr()) {
      op_->MutableAttrs()->Set(attr.first, attr.second);
    }

    // Precalculating a cache key saves about 10% of inference time for very
    // small models.
    op_->MutableAttrs()->CacheKey(op_->DeviceName());

    return tensorflow::Status::OK();
  }

  void ClearEagerInputs() {
    for (tensorflow::TensorHandle* h : *op_->MutableInputs()) {
      if (h) h->Unref();
    }
    op_->MutableInputs()->clear();
  }

  tensorflow::Status BuildEagerInputs(const BufferMap* buffer_map) {
    for (int i = 0; i < inputs_.Size(); ++i) {
      int input_index = inputs_.TfLiteIndex(i);
      TensorSource s = inputs_.GetTensorSource(i);
      if (!s.node) {
        // This input is not produced by this Eager subgraph (it could be a TF
        // Lite native buffer, or could be produced by a separater subgraph). We
        // need to fetch it from the delegate's buffer_map.
        if (!buffer_map->HasTensor(input_index)) {
          return tensorflow::errors::Internal(
              "Cannot read from invalid tensor index ", input_index);
        }
        tensorflow::TensorHandle* handle =
            tensorflow::TensorHandle::CreateLocalHandle(
                buffer_map->GetTensor(input_index));
        op_->MutableInputs()->push_back(handle);
      } else {
        // If this is a forwardable tensor, we will remove it from the previous
        // op's list, giving TF the opportunity to reuse its buffer.
        bool unref_handle = inputs_.IsForwardable(i);
        auto* handle =
            s.node->outputs_.GetHandle(s.node_output_index, unref_handle);
        op_->MutableInputs()->push_back(handle);
      }
    }
    return tensorflow::Status::OK();
  }

  tensorflow::Status PersistEagerOutputs(BufferMap* buffer_map) {
    auto* handles = outputs_.GetTensorHandles();
    for (int i = 0; i < outputs_.Size(); ++i) {
      if (outputs_.IsSubgraphOutput(i)) {
        const tensorflow::Tensor* tensor = nullptr;
        TF_RETURN_IF_ERROR(handles->at(i)->Tensor(&tensor));
        buffer_map->SetFromTensorFlow(outputs_.TfLiteIndex(i), *tensor);
      }
    }
    return tensorflow::Status::OK();
  }

 private:
  OpNode(const OpNode&) = delete;
  OpNode& operator=(const OpNode&) = delete;

  // The name of the TensorFlow op to execute.
  string name_;
  // Index of this node into TF Lite's operator list.
  int index_;
  // The corresponding NodeDef, containing the attributes for the op.
  tensorflow::NodeDef nodedef_;
  // List of inputs, as TF Lite tensor indices.
  OpInputs inputs_;
  // List of outputs, as TF Lite tensor indices.
  OpOutputs outputs_;

  std::unique_ptr<tensorflow::EagerOperation> op_;
};

// Executes the TensorFlow op given by 'op_name', with the attributes specified
// in 'nodedef'. Inputs and outputs are given as indices into the 'buffer_map'.
tensorflow::Status ExecuteFlexOp(TfLiteContext* context, BufferMap* buffer_map,
                                 OpNode* node_data) {
  TF_RETURN_WITH_CONTEXT_IF_ERROR(node_data->BuildEagerInputs(buffer_map),
                                  " (while executing '", node_data->name(),
                                  "' via Eager)");

  node_data->mutable_outputs()->ResetTensorHandles();
  int num_retvals = node_data->NumOutputs();
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      EagerExecute(node_data->op(),
                   node_data->mutable_outputs()->GetTensorHandles()->data(),
                   &num_retvals),
      " (while executing '", node_data->name(), "' via Eager)");

  if (num_retvals != node_data->NumOutputs()) {
    return tensorflow::errors::Internal(
        "Unexpected number of outputs from EagerExecute");
  }

  TF_RETURN_IF_ERROR(node_data->PersistEagerOutputs(buffer_map));

  node_data->ClearEagerInputs();

  return tensorflow::Status::OK();
}

// The larger 'op', which contains all the nodes in a supported subgraph.
struct OpData {
  tensorflow::EagerContext* eager_context;
  BufferMap* buffer_map;
  std::vector<std::unique_ptr<OpNode>> nodes;
  std::vector<int> subgraph_inputs;
  std::vector<int> subgraph_outputs;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData;

  const TfLiteDelegateParams* params =
      reinterpret_cast<const TfLiteDelegateParams*>(buffer);
  CHECK(params);
  CHECK(params->delegate);
  CHECK(params->delegate->data_);
  op_data->eager_context =
      reinterpret_cast<DelegateData*>(params->delegate->data_)
          ->GetEagerContext();
  op_data->buffer_map = reinterpret_cast<DelegateData*>(params->delegate->data_)
                            ->GetBufferMap(context);

  CHECK(params->output_tensors);
  std::set<int> output_set;
  for (auto tensor_index : TfLiteIntArrayView(params->output_tensors)) {
    op_data->subgraph_outputs.push_back(tensor_index);
    output_set.insert(tensor_index);
  }

  CHECK(params->input_tensors);
  for (auto tensor_index : TfLiteIntArrayView(params->input_tensors)) {
    op_data->subgraph_inputs.push_back(tensor_index);
  }

  op_data->nodes.reserve(params->nodes_to_replace->size);

  CHECK(params->nodes_to_replace);
  tensorflow::Status status;
  for (auto node_index : TfLiteIntArrayView(params->nodes_to_replace)) {
    TfLiteNode* node;
    TfLiteRegistration* reg;
    context->GetNodeAndRegistration(context, node_index, &node, &reg);

    op_data->nodes.emplace_back(new OpNode(node->inputs, node->outputs));
    OpNode& node_data = *op_data->nodes.back();

    node_data.set_index(node_index);
    node_data.set_name("");

    status = node_data.InitializeNodeDef(node->custom_initial_data,
                                         node->custom_initial_data_size);
    if (!status.ok()) break;
    status = node_data.BuildEagerOp(op_data->eager_context);
    if (!status.ok()) break;
  }

  if (ConvertStatus(context, status) != kTfLiteOk) {
    // We can't return an error from this function but ConvertStatus will
    // report them and we will stop processing in Prepare() if anything went
    // wrong.
    return op_data;
  }

  // Given a TfLite tensor index, return the OpNode that produces it,
  // along with it index into that OpNodes list of outputs.
  std::map<int, TensorSource> tflite_tensor_sources;

  // Find out how each tensor is produced. This does not account for
  // tensors that are not produce by eager ops.
  for (auto& node_data : op_data->nodes) {
    node_data->mutable_outputs()->InitializeGraphOutputs(output_set);
    for (int i = 0; i < node_data->outputs().Size(); ++i) {
      int output_index = node_data->outputs().TfLiteIndex(i);
      tflite_tensor_sources[output_index] = TensorSource{node_data.get(), i};
    }
  }

  // For each node, resolve the inputs, so we can keep pointers to the nodes
  // that produces them.
  for (auto& node_data : op_data->nodes) {
    node_data->mutable_inputs()->InitializeTensorSources(tflite_tensor_sources);
  }

  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const auto* op_data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE_MSG(
      context, op_data->eager_context != nullptr,
      "Failed to initialize eager context. This often happens when a CPU "
      "device has not been registered, presumably because some symbols from "
      "tensorflow/core:core_cpu_impl were not linked into the binary.");

  // We will keep track of the number of references to each tensor in the
  // graph, so we can make them "forwardable" if there is only one reference.
  std::map<int, int> tensor_ref_count;

  // Whenever we find a constant tensor, insert it in the buffer map.
  BufferMap* buffer_map = op_data->buffer_map;
  for (auto tensor_index : op_data->subgraph_inputs) {
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    if (IsConstantTensor(tensor)) {
      if (!buffer_map->HasTensor(tensor_index)) {
        buffer_map->SetFromTfLite(tensor_index, tensor);
      }
    }

    // Input tensors should never be forwarded so we increment their ref counts
    // twice: once for this graph and another for the possibility of them being
    // used by another subgraph, or being an output of the full graph.
    tensor_ref_count[tensor_index] += 2;
  }

  // All output tensors are allocated by TensorFlow/Eager, so we
  // mark them as kTfLiteDynamic.
  for (auto tensor_index : op_data->subgraph_outputs) {
    SetTensorToDynamic(&context->tensors[tensor_index]);
    ++tensor_ref_count[tensor_index];
  }

  for (const auto& node_data : op_data->nodes) {
    if (node_data->nodedef().op().empty()) {
      context->ReportError(context, "Invalid NodeDef in Flex op '%s'",
                           node_data->name().c_str());
      return kTfLiteError;
    }
    TF_LITE_ENSURE(context, node_data->op());

    for (int i = 0; i < node_data->inputs().Size(); ++i) {
      ++tensor_ref_count[node_data->inputs().TfLiteIndex(i)];
    }
  }

  // All tensors that are referenced exactly once are marked as "forwardable",
  // meaning that we will allow TensorFlow to reuse its buffer as the output of
  // an op.
  for (auto& node_data : op_data->nodes) {
    for (int i = 0; i < node_data->inputs().Size(); ++i) {
      bool f = (tensor_ref_count[node_data->inputs().TfLiteIndex(i)] == 1);
      node_data->mutable_inputs()->SetForwardable(i, f);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);
  BufferMap* buffer_map = op_data->buffer_map;

  // Insert a tensor in the buffer map for all inputs that are not constant.
  // Constants were handled in Prepare() already.
  for (auto tensor_index : op_data->subgraph_inputs) {
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    if (!IsConstantTensor(tensor)) {
      // If this tensor is part of an earlier TF subgraph we should not add it
      // to the BufferMap again, because TF already knows about it and its
      // contents are kept automatically up-to-date.
      if (!buffer_map->IsTensorFlowTensor(tensor_index)) {
        buffer_map->SetFromTfLite(tensor_index, tensor);
      }
    }
  }

  // Execute the TensorFlow Ops sequentially.
  for (auto& node_data : op_data->nodes) {
    TFLITE_SCOPED_DELEGATE_OPERATOR_PROFILE(
        reinterpret_cast<Profiler*>(context->profiler),
        node_data->name().c_str(), node_data->index());

    auto status = ExecuteFlexOp(context, buffer_map, node_data.get());
    TF_LITE_ENSURE_OK(context, ConvertStatus(context, status));
  }

  for (auto tensor_index : op_data->subgraph_outputs) {
    if (!buffer_map->HasTensor(tensor_index)) {
      context->ReportError(context, "Cannot write to invalid tensor index %d",
                           tensor_index);
      return kTfLiteError;
    }

    TfLiteTensor* tensor = &context->tensors[tensor_index];
    TF_LITE_ENSURE_OK(
        context,
        CopyShapeAndType(context, buffer_map->GetTensor(tensor_index), tensor));
    tensor->buffer_handle = tensor_index;
    tensor->data_is_stale = true;
  }

  return kTfLiteOk;
}

}  // namespace kernel

TfLiteRegistration GetKernel() {
  TfLiteRegistration registration{
      &kernel::Init,
      &kernel::Free,
      &kernel::Prepare,
      &kernel::Eval,
      nullptr,                 // .profiling_string
      kTfLiteBuiltinDelegate,  // .builtin_code
      "TfLiteFlexDelegate",    // .custom_name
      1,                       // .version
  };
  return registration;
}

}  // namespace flex
}  // namespace tflite
