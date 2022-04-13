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

#include <map>
#include <set>
#include <utility>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/delegates/flex/delegate_data.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"
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
// corresponding TensorFlow OpKernel.

using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeAndType;
using tensorflow::shape_inference::ShapeHandle;

namespace tflite {
namespace flex {

constexpr char kReadVariableOp[] = "ReadVariableOp";

struct OpNode;

// Represents the origin of a given tensor as a reference to the output
// of an upstream node.
struct TensorSource {
  OpNode* node;
  int node_output_index;
};

// A list of inputs of a given node of the TensorFlow graph.
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
  // in this list while the corresponding tensorflow::OpKernel is executed.
  std::vector<int> forwardable_;
};

// A list of outputs of a given node of the TensorFlow graph, along with
// the actual outputs of the tensorflow::OpKernel.
class OpOutputs {
 public:
  explicit OpOutputs(const TfLiteIntArray* indexes) {
    for (int index : TfLiteIntArrayView(indexes)) {
      outputs_.push_back(index);
    }
    vector_.resize(outputs_.size());
  }
  ~OpOutputs() = default;

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

  const tensorflow::Tensor& GetTensor(int i) const { return vector_[i]; }
  tensorflow::Tensor ReleaseTensor(int i) { return std::move(vector_[i]); }

  int Size() const { return outputs_.size(); }

  int TfLiteIndex(int i) const { return outputs_[i]; }

  tensorflow::gtl::InlinedVector<tensorflow::Tensor, 2>* GetTensors() {
    return &vector_;
  }

 private:
  std::vector<int> outputs_;
  std::vector<bool> subgraph_outputs_;
  tensorflow::gtl::InlinedVector<tensorflow::Tensor, 2> vector_;
};

// This struct holds information such as tensor lifecycle and BufferMap which
// needs to be shared between `OpNode` and DelegateKernel.
struct OpDataInfo {
  // Buffer map which stores the mapping between TfLiteTensor index to TF
  // tensor.
  BufferMap* buffer_map;
  // Mapping information between TfLiteTensor index to last node which uses the
  // tensor.
  std::map<int, int>* tensor_release_map;
  // For output tensors that don't need to be preserved in the BufferMap, we
  // copy them to TF Lite tensors and keep the tensor indexes in this set.
  std::set<int> already_transferred_outputs;
};

// A single node within the larger 'op'. Note that this kernel executes many
// TensorFlow ops within a single TF Lite op.
class OpNode {
 public:
  OpNode(const TfLiteIntArray* inputs, const TfLiteIntArray* outputs)
      : inputs_(inputs), outputs_(outputs) {}
  ~OpNode() = default;

  const string& name() const { return name_; }
  void set_name(const string& name) { name_ = name; }

  int index() const { return index_; }
  void set_index(int index) { index_ = index; }

  const tensorflow::NodeDef& nodedef() const { return nodedef_; }
  const tensorflow::OpRegistrationData* op_reg_data() const {
    return op_reg_data_;
  }

  const OpInputs& inputs() const { return inputs_; }
  OpInputs* mutable_inputs() { return &inputs_; }

  const OpOutputs& outputs() const { return outputs_; }
  OpOutputs* mutable_outputs() { return &outputs_; }

  int NumInputs() const { return inputs_.Size(); }
  int NumOutputs() const { return outputs_.Size(); }

  const tensorflow::tfrt_stub::OpKernelRunner& op_kernel_runner() const {
    return op_kernel_runner_;
  }

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
    TF_RETURN_IF_ERROR(
        tensorflow::OpRegistry::Global()->LookUp(nodedef_.op(), &op_reg_data_));
    AddDefaultsToNodeDef(op_reg_data_->op_def, &nodedef_);

    return tensorflow::Status::OK();
  }

  tensorflow::Status BuildOpKernelRunner(
      tensorflow::EagerContext* eager_context) {
    // Create tensorflow::OpKernel on host CPU.
    TF_ASSIGN_OR_RETURN(op_kernel_runner_,
                        tensorflow::tfrt_stub::OpKernelRunner::Create(
                            name_, inputs_.Size(), /*attr_builder=*/
                            [this](tensorflow::AttrValueMap* attr_value_map) {
                              *attr_value_map = nodedef_.attr();
                              return tensorflow::Status::OK();
                            },
                            *eager_context->pflr(),
                            eager_context->local_device_mgr()->HostCPU()));

    return tensorflow::Status::OK();
  }

  tensorflow::Status BuildOpKernelInputs(
      const BufferMap* buffer_map,
      tensorflow::tfrt_stub::OpKernelRunState* run_state) {
    run_state->input_tf_tensors.resize(inputs_.Size());
    run_state->input_tf_tensor_values.resize(inputs_.Size());

    for (int i = 0; i < inputs_.Size(); ++i) {
      int input_index = inputs_.TfLiteIndex(i);
      TensorSource s = inputs_.GetTensorSource(i);
      if (!s.node) {
        // This input is not produced by this TF subgraph (it could be a TF
        // Lite native buffer, or could be produced by a separater subgraph). We
        // need to fetch it from the delegate's buffer_map.
        if (!buffer_map->HasTensor(input_index)) {
          return tensorflow::errors::Internal(
              "Cannot read from invalid tensor index ", input_index);
        }
        run_state->input_tf_tensors[i] = buffer_map->GetTensor(input_index);
      } else {
        // If this is a forwardable tensor, we will remove it from the previous
        // op's list, giving TF the opportunity to reuse its buffer.
        if (inputs_.IsForwardable(i)) {
          run_state->input_tf_tensors[i] =
              s.node->outputs_.ReleaseTensor(s.node_output_index);
        } else {
          run_state->input_tf_tensors[i] =
              s.node->outputs_.GetTensor(s.node_output_index);
        }
      }
      run_state->input_tf_tensor_values[i].tensor =
          &run_state->input_tf_tensors[i];
    }
    return tensorflow::Status::OK();
  }

  // Returns whether an output tensor should be preserved in the buffer map by
  // checking its lifetime information.
  // The eager tensor doesn't need to be persisted in the buffer map if it has
  // no future uses in the graph.
  bool ShouldPersistTensorflowTensor(TfLiteContext* context,
                                     const OpDataInfo* shared_info,
                                     int tensor_index, int node_index) {
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    // Always persist variant|resource|string tensors since they have special
    // storage requirement.
    if (IsResourceOrVariant(tensor) || tensor->type == kTfLiteString) {
      return true;
    }

    auto it = shared_info->tensor_release_map->find(tensor_index);
    return it != shared_info->tensor_release_map->end() &&
           it->second > node_index;
  }

  // Copies the data of Tensorflow tensor into the corresponding TfLite tensor,
  // after copy is done release the original tensor so that memory could be
  // released by TF runtime.
  TfLiteStatus CopyToTfLiteTensor(TfLiteContext* context,
                                  OpDataInfo* shared_info, TfLiteTensor* tensor,
                                  tensorflow::Tensor* tf_tensor,
                                  int tensor_index) const {
    if (tensor->allocation_type == kTfLiteDynamic) {
      // For dynamic tensors, update the TfLite tensor's shape information from
      // the Tensorflow tensor.
      CopyShapeAndType(context, *tf_tensor, tensor);
    }
    tensorflow::StringPiece t_data = tf_tensor->tensor_data();
    if (tf_tensor->NumElements() != NumElements(tensor) ||
        tf_tensor->TotalBytes() != tensor->bytes) {
      TF_LITE_KERNEL_LOG(context,
                         "FlexDelegate: Tensor %s(%d) buffer size mismatch "
                         "%zu(%lld) != %ld(%ld)",
                         tensor->name, tensor_index, tf_tensor->TotalBytes(),
                         tf_tensor->NumElements(), tensor->bytes,
                         NumElements(tensor));
      return kTfLiteError;
    }
    // Copy TF tensor's data content into TfLiteTensor, and release the tensor.
    memcpy(tensor->data.raw, t_data.data(), t_data.size());
    *tf_tensor = {};
    shared_info->already_transferred_outputs.insert(tensor_index);
    return kTfLiteOk;
  }

  // TODO(b/204479285): Release tensors from BufferMap if it has no future
  // uses.
  tensorflow::Status MaybePersistTensorflowOutputs(TfLiteContext* context,
                                                   OpDataInfo* shared_info,
                                                   int node_index) {
    auto* tensors = outputs_.GetTensors();

    for (int i = 0; i < outputs_.Size(); ++i) {
      if (outputs_.IsSubgraphOutput(i)) {
        tensorflow::Tensor& tf_tensor = tensors->at(i);
        const int tflite_index = outputs_.TfLiteIndex(i);
        TfLiteTensor* tensor = &context->tensors[tflite_index];
        if (!ShouldPersistTensorflowTensor(context, shared_info, tflite_index,
                                           node_index)) {
          if (CopyToTfLiteTensor(context, shared_info, tensor, &tf_tensor,
                                 tflite_index) != kTfLiteOk) {
            return tensorflow::Status(tensorflow::error::INTERNAL,
                                      "failed to copy data from TF tensor");
          }
        } else {
          shared_info->buffer_map->SetFromTensorFlow(outputs_.TfLiteIndex(i),
                                                     tf_tensor);
        }
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
  // The corresponding OpRegistrationData pointer.
  const tensorflow::OpRegistrationData* op_reg_data_;
  // List of inputs, as TF Lite tensor indices.
  OpInputs inputs_;
  // List of outputs, as TF Lite tensor indices.
  OpOutputs outputs_;

  tensorflow::tfrt_stub::OpKernelRunner op_kernel_runner_;
};

// The larger 'op', which contains all the nodes in a supported subgraph.
struct OpData {
  tensorflow::EagerContext* eager_context;
  tensorflow::CancellationManager* cancellation_manager;
  std::vector<std::unique_ptr<OpNode>> nodes;
  std::vector<int> subgraph_inputs;
  std::vector<int> subgraph_outputs;
  OpDataInfo shared_info;
};

tensorflow::Status DelegateKernel::ExecuteOpKernelRunner(
    tensorflow::tfrt_stub::OpKernelRunState* run_state, TfLiteContext* context,
    OpNode* node_data) {
  const auto& op_kernel_runner = node_data->op_kernel_runner();

  if (op_kernel_runner.op_kernel()->num_outputs() != node_data->NumOutputs()) {
    return tensorflow::errors::Internal(
        "Unexpected number of outputs from tensorflow::OpKernel");
  }

  TF_RETURN_IF_ERROR(node_data->BuildOpKernelInputs(
      op_data_->shared_info.buffer_map, run_state));

  run_state->params.inputs = &run_state->input_tf_tensor_values;
  run_state->params.op_kernel = op_kernel_runner.op_kernel();
  run_state->params.input_alloc_attrs = &op_kernel_runner.input_alloc_attrs();
  run_state->params.output_attr_array =
      op_kernel_runner.output_alloc_attrs().data();
  run_state->params.function_library =
      op_kernel_runner.function_library_runtime();

  tensorflow::OpKernelContext tf_context(&run_state->params,
                                         node_data->NumOutputs());
  op_kernel_runner.Run(&tf_context);
  TF_RETURN_IF_ERROR(tf_context.status());

  auto& outputs = *node_data->mutable_outputs()->GetTensors();
  for (int i = 0; i < tf_context.num_outputs(); ++i) {
    outputs[i] = std::move(*tf_context.mutable_output(i));
  }

  return node_data->MaybePersistTensorflowOutputs(
      context, &(op_data_->shared_info), node_data->index());
}

DelegateKernel::DelegateKernel() : op_data_(new OpData) {}
DelegateKernel::~DelegateKernel() {}

TfLiteStatus DelegateKernel::Init(TfLiteContext* context,
                                  const TfLiteDelegateParams* params) {
  auto* flex_delegate_data =
      reinterpret_cast<FlexDelegate*>(params->delegate->data_)->mutable_data();
  op_data_->eager_context = flex_delegate_data->GetEagerContext();
  op_data_->cancellation_manager = flex_delegate_data->GetCancellationManager();
  op_data_->shared_info.buffer_map = flex_delegate_data->GetBufferMap(context);
  op_data_->shared_info.tensor_release_map =
      flex_delegate_data->GetTensorReleaseMap(context);

  CHECK(params->output_tensors);
  std::set<int> output_set;
  for (auto tensor_index : TfLiteIntArrayView(params->output_tensors)) {
    op_data_->subgraph_outputs.push_back(tensor_index);
    output_set.insert(tensor_index);
  }

  CHECK(params->input_tensors);
  for (auto tensor_index : TfLiteIntArrayView(params->input_tensors)) {
    op_data_->subgraph_inputs.push_back(tensor_index);
  }

  op_data_->nodes.reserve(params->nodes_to_replace->size);

  CHECK(params->nodes_to_replace);
  tensorflow::Status status;
  for (auto node_index : TfLiteIntArrayView(params->nodes_to_replace)) {
    TfLiteNode* node;
    TfLiteRegistration* reg;
    context->GetNodeAndRegistration(context, node_index, &node, &reg);

    // For each node handled by this delegate partition, record the mapping
    // information between each input tensor and the node index. The node index
    // is the index of the last node in execution order that uses this tensor.
    // So the tensor is no longer needed after this last node is executed.
    // Since we execute in order, then the maximum index is the index of the
    // last node that needs this tensor.
    for (auto tensor_index : TfLiteIntArrayView(node->inputs)) {
      int node_id = node_index;
      if (op_data_->shared_info.tensor_release_map->find(tensor_index) !=
          op_data_->shared_info.tensor_release_map->end()) {
        node_id =
            std::max(op_data_->shared_info.tensor_release_map->at(tensor_index),
                     node_index);
      }
      (*op_data_->shared_info.tensor_release_map)[tensor_index] = node_id;
    }

    op_data_->nodes.emplace_back(new OpNode(node->inputs, node->outputs));
    OpNode& node_data = *op_data_->nodes.back();

    node_data.set_index(node_index);
    node_data.set_name("");

    status = node_data.InitializeNodeDef(node->custom_initial_data,
                                         node->custom_initial_data_size);
    if (!status.ok()) break;
    status = node_data.BuildOpKernelRunner(op_data_->eager_context);
    if (!status.ok()) break;
  }

  TF_LITE_ENSURE_STATUS(ConvertStatus(context, status));

  // Given a TfLite tensor index, return the OpNode that produces it,
  // along with it index into that OpNodes list of outputs.
  std::map<int, TensorSource> tflite_tensor_sources;

  // Find out how each tensor is produced. This does not account for
  // tensors that are not produced by tensorflow::Opkernels.
  for (auto& node_data : op_data_->nodes) {
    node_data->mutable_outputs()->InitializeGraphOutputs(output_set);
    for (int i = 0; i < node_data->outputs().Size(); ++i) {
      int output_index = node_data->outputs().TfLiteIndex(i);
      tflite_tensor_sources[output_index] = TensorSource{node_data.get(), i};
    }
  }

  // For each node, resolve the inputs, so we can keep pointers to the nodes
  // that produces them.
  for (auto& node_data : op_data_->nodes) {
    node_data->mutable_inputs()->InitializeTensorSources(tflite_tensor_sources);
  }
  return kTfLiteOk;
}

TfLiteStatus DelegateKernel::Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_MSG(
      context, op_data_->eager_context != nullptr,
      "Failed to initialize eager context. This often happens when a CPU "
      "device has not been registered, presumably because some symbols from "
      "tensorflow/core:core_cpu_impl were not linked into the binary.");

  // We will keep track of the number of references to each tensor in the
  // graph, so we can make them "forwardable" if there is only one reference.
  std::map<int, int> tensor_ref_count;

  // Whenever we find a constant tensor, insert it in the buffer map.
  BufferMap* buffer_map = op_data_->shared_info.buffer_map;
  for (auto tensor_index : op_data_->subgraph_inputs) {
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    if (IsConstantTensor(tensor)) {
      if (!tensor->data_is_stale || !buffer_map->HasTensor(tensor_index)) {
        buffer_map->SetFromTfLite(tensor_index, tensor);
      }
    }

    // Input tensors should never be forwarded so we increment their ref counts
    // twice: once for this graph and another for the possibility of them being
    // used by another subgraph, or being an output of the full graph.
    tensor_ref_count[tensor_index] += 2;
  }

  const bool shapes_are_valid =
      (ValidateOutputTensorShapeConsistency(context) == kTfLiteOk);
  if (shapes_are_valid) {
    TFLITE_LOG(tflite::TFLITE_LOG_INFO,
               "FlexDelegate: All tensor shapes are consistent.");
  } else {
    TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
               "FlexDelegate: Some tensor shapes are inconsistent.");
  }

  // All output tensors are allocated by TensorFlow, so we mark them as
  // kTfLiteDynamic.
  for (auto tensor_index : op_data_->subgraph_outputs) {
    if (!shapes_are_valid) {
      SetTensorToDynamic(&context->tensors[tensor_index]);
    }
    ++tensor_ref_count[tensor_index];
  }

  for (const auto& node_data : op_data_->nodes) {
    if (node_data->nodedef().op().empty()) {
      TF_LITE_KERNEL_LOG(context, "Invalid NodeDef in Flex op '%s'",
                         node_data->name().c_str());
      return kTfLiteError;
    }
    TF_LITE_ENSURE(context, node_data->op_kernel_runner());

    for (int i = 0; i < node_data->inputs().Size(); ++i) {
      ++tensor_ref_count[node_data->inputs().TfLiteIndex(i)];
    }
  }

  // All tensors that are referenced exactly once are marked as "forwardable",
  // meaning that we will allow TensorFlow to reuse its buffer as the output of
  // an op.
  for (auto& node_data : op_data_->nodes) {
    for (int i = 0; i < node_data->inputs().Size(); ++i) {
      bool f = (tensor_ref_count[node_data->inputs().TfLiteIndex(i)] == 1);
      node_data->mutable_inputs()->SetForwardable(i, f);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus DelegateKernel::ValidateOutputTensorShapeConsistency(
    TfLiteContext* context) const {
  for (const auto& node_data : op_data_->nodes) {
    auto op_name = node_data->name().c_str();
    // Create an InferenceContext object.
    auto num_inputs = node_data->inputs().Size();
    std::vector<const tensorflow::Tensor*> input_tensors_vector(num_inputs,
                                                                nullptr);
    InferenceContext c(
        TF_GRAPH_DEF_VERSION, node_data->nodedef(),
        node_data->op_reg_data()->op_def, std::vector<ShapeHandle>(num_inputs),
        input_tensors_vector, {},
        std::vector<std::unique_ptr<std::vector<ShapeAndType>>>());

    // Set input_shapes for ShapeInferenceFn.
    for (int i = 0; i < num_inputs; ++i) {
      const auto input_tensor_index = node_data->inputs().TfLiteIndex(i);
      TfLiteTensor* tfl_tensor = &context->tensors[input_tensor_index];
      // Provide constant input tensors since some op ("RFFT") needs it to
      // calculate the output shape.
      if (IsConstantTensor(tfl_tensor)) {
        input_tensors_vector[i] =
            op_data_->shared_info.buffer_map->GetTensorPtr(input_tensor_index);
      }
      const auto dims_array = tfl_tensor->dims;
      std::vector<DimensionHandle> dims(dims_array->size);
      for (int j = 0; j < dims_array->size; ++j) {
        dims[j] = c.MakeDim(dims_array->data[j]);
      }
      c.SetInput(i, c.MakeShape(dims));
    }
    c.set_input_tensors(input_tensors_vector);

    tensorflow::Status status = c.construction_status();
    if (!status.ok()) {
      TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
                 "Shape construction failed for op '%s'", op_name);
      return kTfLiteError;
    }

    // Run ShapeInferenceFn to calculate output shapes.
    if (node_data->op_reg_data()->shape_inference_fn == nullptr) {
      TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
                 "No shape inference function exists for op '%s'", op_name);
      return kTfLiteError;
    }
    status = c.Run(node_data->op_reg_data()->shape_inference_fn);

    // Compare calculated output shapes with node_data->outputs
    auto num_outputs = node_data->outputs().Size();
    if (num_outputs != c.num_outputs()) {
      TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
                 "Number of output tensors are mismatched for op '%s' %d != %d",
                 op_name, num_outputs, c.num_outputs());
      return kTfLiteError;
    }
    for (int i = 0; i < num_outputs; ++i) {
      const auto output_tensor_index = node_data->outputs().TfLiteIndex(i);
      TfLiteTensor* tfl_tensor = &context->tensors[output_tensor_index];
      // tfl_tensor->dims only has valid information if the given model is
      // converted by the MLIR converter. Also when ResizeInputTensor() is
      // called the dims information becomes invalid.
      const std::string tfl_shape_string =
          GetShapeDebugString(tfl_tensor->dims);
      const std::string calculated_shape_string = c.DebugString(c.output(i));
      // Getting a shape string via c.DebugString() is the easiest way to get
      // the shape information of the given ShapeHandle for now.
      // TODO(b/169017408): Find a better approach without using debug string.
      if (tfl_shape_string != calculated_shape_string) {
        if ((strcmp(op_name, kReadVariableOp) == 0) &&
            (tfl_tensor->dims->size > 0)) {
          // If ReadVariableOp has an output with valid shape, use it since
          // ShapeInferenceFn of ReadVariableOp doesn't work well without having
          // a valid resource handle.
          continue;
        }

        TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
                   "op '%s' output%d tensor#%d shape mismatch for  %s != %s",
                   op_name, i, output_tensor_index, tfl_shape_string.c_str(),
                   calculated_shape_string.c_str());
        return kTfLiteError;
      }
    }
  }
  return kTfLiteOk;
}

static tensorflow::CancellationManager* GetDefaultCancellationManager() {
  static auto* const cancellation_manager = new tensorflow::CancellationManager;
  return cancellation_manager;
}

TfLiteStatus DelegateKernel::Eval(TfLiteContext* context, TfLiteNode* node) {
  BufferMap* buffer_map = op_data_->shared_info.buffer_map;

  // Insert a tensor in the buffer map for all inputs that are not constant.
  // Constants were handled in Prepare() already.
  for (auto tensor_index : op_data_->subgraph_inputs) {
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    if (!IsConstantTensor(tensor)) {
      // If this tensor is part of an earlier TF subgraph we should not add it
      // to the BufferMap again, because TF already knows about it and its
      // contents are kept automatically up-to-date.
      if (!tensor->data_is_stale || !buffer_map->HasTensor(tensor_index)) {
        buffer_map->SetFromTfLite(tensor_index, tensor);
      }
    }
  }

  auto& eager_context = *op_data_->eager_context;

  {
    tensorflow::tfrt_stub::OpKernelRunState run_state;

    run_state.params.step_container = eager_context.StepContainer();
    auto* device = eager_context.local_device_mgr()->HostCPU();
    run_state.params.device = device;
    run_state.params.resource_manager = device->resource_manager();
    run_state.params.runner = eager_context.runner();
    run_state.params.cancellation_manager =
        op_data_->cancellation_manager ? op_data_->cancellation_manager
                                       : GetDefaultCancellationManager();
    // TODO(b/179048776): Set up remaining params such as collective and
    // rendezvous.

    // Execute the TensorFlow Ops sequentially.
    for (auto& node_data : op_data_->nodes) {
      TFLITE_SCOPED_DELEGATE_OPERATOR_PROFILE(
          reinterpret_cast<Profiler*>(context->profiler),
          node_data->name().c_str(), node_data->index());

      if (op_data_->cancellation_manager != nullptr &&
          op_data_->cancellation_manager->IsCancelled()) {
        TF_LITE_KERNEL_LOG(
            context, "Client requested cancel during DelegateKernel::Eval");
        return kTfLiteError;
      }

      auto status = ExecuteOpKernelRunner(&run_state, context, node_data.get());
      TF_LITE_ENSURE_OK(context, ConvertStatus(context, status));
    }
  }

  for (auto tensor_index : op_data_->subgraph_outputs) {
    if (op_data_->shared_info.already_transferred_outputs.count(tensor_index) !=
        0) {
      // Skip if a tensor output has already been copied to a TfLiteTensor.
      continue;
    }
    if (!buffer_map->HasTensor(tensor_index)) {
      TF_LITE_KERNEL_LOG(context, "Cannot write to invalid tensor index %d",
                         tensor_index);
      return kTfLiteError;
    }

    // Copy TF tensor data to TFL allocated buffer for non dynamic tensors.
    // For dynamic tensors, copy shape and put buffer_handle for the later
    // CopyFromBufferHandle() call.
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    const tensorflow::Tensor& tf_tensor = buffer_map->GetTensor(tensor_index);
    if (tensor->allocation_type == kTfLiteDynamic) {
      TF_LITE_ENSURE_OK(context, CopyShapeAndType(context, tf_tensor, tensor));
      tensor->buffer_handle = tensor_index;
      tensor->data_is_stale = true;
      continue;
    }
    // If the tensor isn't dynamic, we can copy data directly to the buffer of
    // the tensor. Before copying the data, check if the target buffer has
    // expected size.
    if (tf_tensor.NumElements() != NumElements(tensor) ||
        tf_tensor.TotalBytes() != tensor->bytes) {
      TF_LITE_KERNEL_LOG(context,
                         "FlexDelegate: Tensor %s(%d) buffer size mismatch "
                         "%zu(%lld) != %ld(%ld)",
                         tensor->name, tensor_index, tf_tensor.TotalBytes(),
                         tf_tensor.NumElements(), tensor->bytes,
                         NumElements(tensor));
      return kTfLiteError;
    }
    tensorflow::StringPiece t_data = tf_tensor.tensor_data();
    memcpy(tensor->data.raw, t_data.data(), t_data.size());
  }

  return kTfLiteOk;
}

const std::map<int, int>& DelegateKernel::GetTensorReleaseMap() const {
  return *(op_data_->shared_info.tensor_release_map);
}

}  // namespace flex
}  // namespace tflite
