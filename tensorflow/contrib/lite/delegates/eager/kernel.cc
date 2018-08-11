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
#include "tensorflow/contrib/lite/delegates/eager/kernel.h"

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/contrib/lite/builtin_ops.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/context_util.h"
#include "tensorflow/contrib/lite/delegates/eager/delegate_data.h"
#include "tensorflow/contrib/lite/delegates/eager/util.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/string.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/node_def.pb.h"

// Note: this is part of TF Lite's Eager delegation code which is to be
// completed soon.

// This is the TF Lite op that is created by the eager delegate to handle
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
namespace eager {
namespace kernel {

// Controls the lifetime of tensor handles in a vector.
class VectorOfHandles {
 public:
  explicit VectorOfHandles(int num_elements) : vector_(num_elements, nullptr) {}

  ~VectorOfHandles() {
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

// Executes the TensorFlow op given by 'op_name', with the attributes specified
// in 'nodedef'. Inputs and outputs are given as indices into the 'buffer_map'.
tensorflow::Status ExecuteEagerOp(tensorflow::EagerContext* eager_context,
                                  BufferMap* buffer_map, const string& op_name,
                                  const tensorflow::NodeDef& nodedef,
                                  const std::vector<int>& inputs,
                                  const std::vector<int>& outputs) {
  const tensorflow::AttrTypeMap* attr_types;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      tensorflow::AttrTypeMapForOp(op_name.c_str(), &attr_types),
      " (while processing attributes of '", op_name, "')");

  tensorflow::EagerOperation op(eager_context, op_name.c_str(), attr_types);
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
  }

  int num_retvals = outputs.size();
  VectorOfHandles retvals(num_retvals);
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

// A single node within the larger 'op'. Note that this kernel executes many
// TensorFlow ops within a single TF Lite op.
struct OpNode {
  // The name of the TensorFlow op to execute.
  string name;
  // The corresponding NodeDef, containing the attributes for the op.
  tensorflow::NodeDef nodedef;
  // List of inputs, as TF Lite tensor indices.
  std::vector<int> inputs;
  // List of outputs, as TF Lite tensor indices.
  std::vector<int> outputs;
};

// The Larger 'op', which contains all the nodes in a supported subgraph.
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
  op_data->buffer_map =
      reinterpret_cast<DelegateData*>(params->delegate->data_)->GetBufferMap();

  CHECK(params->output_tensors);
  for (auto tensor_index : TfLiteIntArrayView(params->output_tensors)) {
    op_data->subgraph_outputs.push_back(tensor_index);
  }

  CHECK(params->input_tensors);
  for (auto tensor_index : TfLiteIntArrayView(params->input_tensors)) {
    op_data->subgraph_inputs.push_back(tensor_index);
  }

  CHECK(params->nodes_to_replace);
  for (auto node_index : TfLiteIntArrayView(params->nodes_to_replace)) {
    TfLiteNode* node;
    TfLiteRegistration* reg;
    context->GetNodeAndRegistration(context, node_index, &node, &reg);

    op_data->nodes.push_back(OpNode());
    OpNode& node_data = op_data->nodes.back();

    node_data.name = "";
    if (node->custom_initial_data) {
      // The flexbuffer contains a vector where the first elements is the
      // op name and the second is a serialized NodeDef.
      const flexbuffers::Vector& v =
          flexbuffers::GetRoot(
              reinterpret_cast<const uint8_t*>(node->custom_initial_data),
              node->custom_initial_data_size)
              .AsVector();

      node_data.name = v[0].AsString().str();
      if (!node_data.nodedef.ParseFromString(v[1].AsString().str())) {
        // We will just leave the nodedef empty and error out in Eval().
        node_data.nodedef.Clear();
      }
    }

    for (auto input_index : TfLiteIntArrayView(node->inputs)) {
      node_data.inputs.push_back(input_index);
    }
    for (auto output_index : TfLiteIntArrayView(node->outputs)) {
      node_data.outputs.push_back(output_index);
    }
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

  // Whenever we find a constant tensor, insert it in the buffer map.
  BufferMap* buffer_map = op_data->buffer_map;
  for (auto tensor_index : op_data->subgraph_inputs) {
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    if (IsConstantTensor(tensor)) {
      if (!buffer_map->HasTensor(tensor_index)) {
        buffer_map->SetFromTfLite(tensor_index, tensor);
      }
    }
  }

  // All output tensors are allocated by TensorFlow/Eager, so we
  // mark them as kTfLiteDynamic.
  for (auto tensor_index : op_data->subgraph_outputs) {
    SetTensorToDynamic(&context->tensors[tensor_index]);
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
      buffer_map->SetFromTfLite(tensor_index, tensor);
    }
  }

  // Execute the TensorFlow Ops sequentially.
  for (const auto& node_data : op_data->nodes) {
    if (node_data.nodedef.op().empty()) {
      context->ReportError(context, "Invalid NodeDef in Eager op '%s'",
                           node_data.name.c_str());
      return kTfLiteError;
    }
    auto status =
        ExecuteEagerOp(eager_context, buffer_map, node_data.name,
                       node_data.nodedef, node_data.inputs, node_data.outputs);
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
        CopyShape(context, buffer_map->GetTensor(tensor_index), tensor));
    tensor->buffer_handle = tensor_index;
    tensor->data_is_stale = true;
  }

  return kTfLiteOk;
}

}  // namespace kernel

TfLiteRegistration GetKernel() {
  TfLiteRegistration registration{&kernel::Init,    &kernel::Free,
                                  &kernel::Prepare, &kernel::Eval,
                                  nullptr,          kTfLiteBuiltinDelegate};
  return registration;
}

}  // namespace eager
}  // namespace tflite
