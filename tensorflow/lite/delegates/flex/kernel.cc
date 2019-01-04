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

#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/flex/delegate_data.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string.h"

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

// Controls the lifetime of tensor handles in a vector.
class OpOutputs {
 public:
  explicit OpOutputs(int num_elements) : vector_(num_elements, nullptr) {}

  ~OpOutputs() {
    for (auto* handle : vector_) {
      if (handle) handle->Unref();
    }
  }

  tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 2>* GetVector() {
    return &vector_;
  }

  tensorflow::TensorHandle* GetHandle(int index) { return vector_[index]; }

 private:
  tensorflow::gtl::InlinedVector<tensorflow::TensorHandle*, 2> vector_;
};

// A single node within the larger 'op'. Note that this kernel executes many
// TensorFlow ops within a single TF Lite op.
class OpNode {
 public:
  OpNode() {}
  ~OpNode() {}

  const string& name() const { return name_; }
  void set_name(const string& name) { name_ = name; }

  int index() const { return index_; }
  void set_index(int index) { index_ = index; }

  const tensorflow::NodeDef& nodedef() const { return nodedef_; }

  const std::vector<int>& inputs() const { return inputs_; }
  std::vector<int>* mutable_inputs() { return &inputs_; }

  const std::vector<int>& outputs() const { return outputs_; }
  std::vector<int>* mutable_outputs() { return &outputs_; }

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

 private:
  // The name of the TensorFlow op to execute.
  string name_;
  // Index of this node into TF Lite's operator list.
  int index_;
  // The corresponding NodeDef, containing the attributes for the op.
  tensorflow::NodeDef nodedef_;
  // List of inputs, as TF Lite tensor indices.
  std::vector<int> inputs_;
  // List of outputs, as TF Lite tensor indices.
  std::vector<int> outputs_;
};

// Executes the TensorFlow op given by 'op_name', with the attributes specified
// in 'nodedef'. Inputs and outputs are given as indices into the 'buffer_map'.
tensorflow::Status ExecuteFlexOp(tensorflow::EagerContext* eager_context,
                                 BufferMap* buffer_map, const string& op_name,
                                 const tensorflow::NodeDef& nodedef,
                                 const std::vector<int>& inputs,
                                 const std::vector<int>& outputs) {
  const tensorflow::AttrTypeMap* attr_types;
  bool is_function = false;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      tensorflow::AttrTypeMapForOp(op_name.c_str(), &attr_types, &is_function),
      " (while processing attributes of '", op_name, "')");
  if (is_function) {
    return tensorflow::errors::NotFound(
        "Operation '", op_name,
        "' is not registered.  (while processing attributes of '", op_name,
        "')");
  }
  tensorflow::EagerOperation op(eager_context, op_name.c_str(),
                                /*is_function=*/false, attr_types);
  for (const auto& attr : nodedef.attr()) {
    op.MutableAttrs()->Set(attr.first, attr.second);
  }

  for (int input_index : inputs) {
    if (!buffer_map->HasTensor(input_index)) {
      return tensorflow::errors::Internal(
          "Cannot read from invalid tensor index ", input_index);
    }
    auto* handle = new tensorflow::TensorHandle(
        buffer_map->GetTensor(input_index), nullptr, nullptr, nullptr);
    op.AddInput(handle);
    handle->Unref();

    if (buffer_map->IsForwardable(input_index)) {
      // Take it out of the map, so Eager/TF can reuse the buffer for an output
      // tensor of the op.
      buffer_map->RemoveTensor(input_index);
    }
  }

  int num_retvals = outputs.size();
  OpOutputs retvals(num_retvals);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      EagerExecute(&op, retvals.GetVector(), &num_retvals),
      " (while executing '", op_name, "' via Eager)");

  if (num_retvals != outputs.size()) {
    return tensorflow::errors::Internal(
        "Unexpected number of outputs from EagerExecute");
  }

  for (int i = 0; i < num_retvals; ++i) {
    const tensorflow::Tensor* tensor = nullptr;
    TF_RETURN_IF_ERROR(retvals.GetHandle(i)->Tensor(&tensor));
    buffer_map->SetFromTensorFlow(outputs[i], *tensor);
  }

  return tensorflow::Status::OK();
}

// The larger 'op', which contains all the nodes in a supported subgraph.
struct OpData {
  tensorflow::EagerContext* eager_context;
  BufferMap* buffer_map;
  std::vector<OpNode> nodes;
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
  for (auto tensor_index : TfLiteIntArrayView(params->output_tensors)) {
    op_data->subgraph_outputs.push_back(tensor_index);
  }

  CHECK(params->input_tensors);
  for (auto tensor_index : TfLiteIntArrayView(params->input_tensors)) {
    op_data->subgraph_inputs.push_back(tensor_index);
  }

  CHECK(params->nodes_to_replace);
  tensorflow::Status status;
  for (auto node_index : TfLiteIntArrayView(params->nodes_to_replace)) {
    TfLiteNode* node;
    TfLiteRegistration* reg;
    context->GetNodeAndRegistration(context, node_index, &node, &reg);

    op_data->nodes.push_back(OpNode());
    OpNode& node_data = op_data->nodes.back();

    node_data.set_index(node_index);
    node_data.set_name("");

    status = node_data.InitializeNodeDef(node->custom_initial_data,
                                         node->custom_initial_data_size);
    if (!status.ok()) break;

    for (auto input_index : TfLiteIntArrayView(node->inputs)) {
      node_data.mutable_inputs()->push_back(input_index);
    }
    for (auto output_index : TfLiteIntArrayView(node->outputs)) {
      node_data.mutable_outputs()->push_back(output_index);
    }
  }

  if (ConvertStatus(context, status) != kTfLiteOk) {
    // We can't return an error from this function but ConvertStatus will
    // report them and we will stop processing in Prepare() if anything went
    // wrong.
    return op_data;
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
    ++tensor_ref_count[tensor_index];
  }

  // All output tensors are allocated by TensorFlow/Eager, so we
  // mark them as kTfLiteDynamic.
  for (auto tensor_index : op_data->subgraph_outputs) {
    SetTensorToDynamic(&context->tensors[tensor_index]);
    ++tensor_ref_count[tensor_index];
  }

  for (const auto& node_data : op_data->nodes) {
    if (node_data.nodedef().op().empty()) {
      context->ReportError(context, "Invalid NodeDef in Flex op '%s'",
                           node_data.name().c_str());
      return kTfLiteError;
    }

    for (int tensor_index : node_data.inputs()) {
      ++tensor_ref_count[tensor_index];
    }
  }

  for (const auto& x : tensor_ref_count) {
    if (x.second == 1) {
      // This tensor is referenced once by a single op. We can allow the TF
      // kernel to "forward" it to the output, meaning its buffer will be
      // reused and overwritten.
      buffer_map->SetForwardable(x.first);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* op_data = reinterpret_cast<OpData*>(node->user_data);
  BufferMap* buffer_map = op_data->buffer_map;
  tensorflow::EagerContext* eager_context = op_data->eager_context;

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
  for (const auto& node_data : op_data->nodes) {
    SCOPED_TAGGED_OPERATOR_PROFILE(
        reinterpret_cast<profiling::Profiler*>(context->profiler),
        node_data->name().c_str(), node_data->index());

    auto status = ExecuteFlexOp(eager_context, buffer_map, node_data.name(),
                                node_data.nodedef(), node_data.inputs(),
                                node_data.outputs());
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

  // We don't need to keep track of internal TF tensors any longer, so take
  // them out of the buffer_map, but make sure we keep all the one we might
  // need for other subgraphs, or as final output of inference.
  const auto& outputs = op_data->subgraph_outputs;
  std::set<int> keep(outputs.begin(), outputs.end());
  buffer_map->RemoveTensorsNotInSet(keep);

  return kTfLiteOk;
}

}  // namespace kernel

TfLiteRegistration GetKernel() {
  TfLiteRegistration registration{&kernel::Init,    &kernel::Free,
                                  &kernel::Prepare, &kernel::Eval,
                                  nullptr,          kTfLiteBuiltinDelegate};
  return registration;
}

}  // namespace flex
}  // namespace tflite
