/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/arena_planner.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/nnapi_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace {
TfLiteStatus ReportOpError(TfLiteContext* context, const TfLiteNode& node,
                           const TfLiteRegistration& registration,
                           int node_index, const char* message) {
  context->ReportError(
      context, "Node number %d (%s) %s.\n", node_index,
      registration.custom_name
          ? registration.custom_name
          : EnumNameBuiltinOperator(
                static_cast<BuiltinOperator>(registration.builtin_code)),
      message);
  return kTfLiteError;
}

// Stub method which returns kTfLiteError when the function is forbidden.
// We're registrating this function to several different function to save
// compiled binary size. Please note the restrictions:
// * The type of first parameter have to be `TfLiteContext*`.
// * All paramteters must be trivailly destructible. (E.g. No C++ class)
TfLiteStatus ForbiddenContextFunction(TfLiteContext* context, ...) {
  context->ReportError(context,
                       "The function is forbidden if not calling in delegate.");
  return kTfLiteError;
}

// Set the ForbiddenContextFunction to a compatible function pointer.
template <typename FunctionType>
void SetForbiddenContextFunction(FunctionType* func) {
  *func = reinterpret_cast<FunctionType>(ForbiddenContextFunction);
}

// Returns true if at least one tensor in the given list is kTfLiteDynamic.
template <typename TensorIntArray>
bool HasDynamicTensorImpl(const TfLiteContext& context,
                          const TensorIntArray& int_array) {
  for (int i : int_array) {
    const TfLiteTensor& tensor = context.tensors[i];
    if (tensor.allocation_type == kTfLiteDynamic) {
      return true;
    }
  }
  return false;
}

bool HasDynamicTensor(const TfLiteContext& context,
                      const TfLiteIntArray* int_array) {
  return HasDynamicTensorImpl(context, TfLiteIntArrayView{int_array});
}

}  // namespace

// A trivial implementation of GraphInfo around the Interpreter.
// NOTE: this interpreter info represents the subset of the
// graph that is executed according to execution plan. Thus,
// the indices are execution plan indices rather than raw node
// indices.
class InterpreterInfo : public GraphInfo {
 public:
  explicit InterpreterInfo(Subgraph* subgraph) : subgraph_(subgraph) {}

  size_t num_tensors() const override { return subgraph_->tensors().size(); }
  TfLiteTensor* tensor(size_t index) override {
    return &subgraph_->tensors()[index];
  }
  size_t num_nodes() const override {
    return subgraph_->execution_plan().size();
  }
  const TfLiteNode& node(size_t index) const override {
    int node_index = subgraph_->execution_plan()[index];
    return subgraph_->nodes_and_registration()[node_index].first;
  }
  const std::vector<int>& inputs() const override {
    return subgraph_->inputs();
  }
  const std::vector<int>& outputs() const override {
    return subgraph_->outputs();
  }
  const std::vector<int>& variables() const override {
    return subgraph_->variables();
  }

 public:
  Subgraph* subgraph_;
};

Subgraph::Subgraph(ErrorReporter* error_reporter,
                   TfLiteExternalContext** external_contexts,
                   std::vector<std::unique_ptr<Subgraph>>* subgraphs)
    : context_(&owned_context_),
      error_reporter_(error_reporter),
      next_execution_plan_index_to_prepare_(0),
      external_contexts_(external_contexts),
      subgraphs_(subgraphs) {
  context_->impl_ = static_cast<void*>(this);
  context_->ResizeTensor = ResizeTensor;
  context_->ReportError = ReportErrorC;
  context_->AddTensors = AddTensors;
  context_->tensors = nullptr;
  context_->tensors_size = 0;
  context_->allow_fp32_relax_to_fp16 = false;
  context_->recommended_num_threads = -1;
  context_->GetExternalContext = GetExternalContext;
  context_->SetExternalContext = SetExternalContext;
  context_->profiler = nullptr;

  // Reserve some space for the tensors to avoid excessive resizing.
  tensors_.reserve(kTensorsReservedCapacity);
  nodes_and_registration().reserve(kTensorsReservedCapacity);
  // Invalid to call these these except from TfLiteDelegate
  SwitchToKernelContext();
}

Subgraph::~Subgraph() {
  for (auto& node_and_reg : nodes_and_registration_) {
    TfLiteNode& node = node_and_reg.first;
    TfLiteIntArrayFree(node.inputs);
    TfLiteIntArrayFree(node.outputs);
    TfLiteIntArrayFree(node.temporaries);
    if (node.builtin_data) free(node.builtin_data);
    OpFree(node_and_reg.second, node.user_data);
    node.builtin_data = nullptr;
  }

  for (size_t i = 0; i < context_->tensors_size; i++) {
    TfLiteTensor* tensor = &context_->tensors[i];
    if (tensor->buffer_handle != kTfLiteNullBufferHandle &&
        tensor->delegate->FreeBufferHandle != nullptr) {
      tensor->delegate->FreeBufferHandle(context_, tensor->delegate,
                                         &tensor->buffer_handle);
    }
    TfLiteTensorFree(tensor);
  }
}

TfLiteStatus Subgraph::ReplaceNodeSubsetsWithDelegateKernels(
    TfLiteContext* context, TfLiteRegistration registration,
    const TfLiteIntArray* nodes_to_replace, TfLiteDelegate* delegate) {
  return static_cast<Subgraph*>(context->impl_)
      ->ReplaceNodeSubsetsWithDelegateKernels(registration, nodes_to_replace,
                                              delegate);
}

namespace {

// Copy a std::vector<int> to an existing TfLiteIntArray.
// This is a low-level data manipulation function, and it's caller's
// responsibility to ensure TfLiteIntArray has enough size.
void CopyVectorToTfLiteIntArray(const std::vector<int>& vec,
                                TfLiteIntArray* arr) {
  arr->size = vec.size();
  memcpy(arr->data, vec.data(), sizeof(int) * arr->size);
}

// This function allocates a continuous memory space that contains a
// TfLiteDelegateParams followed by a several TfLiteIntArray.
// When calling `free` at TfLiteDelegateParams*, all the allocated space
// will be freed together.
//
// +-----------------------------------+
// | TfLiteDelegateParams              |
// | TfLiteDelegate* delegate;         |
// | TfLiteIntArray* nodes_to_replace; |--\
// | TfLiteIntArray* input_tensors;    |--+--\
// | TfLiteIntArray* output_tensors;   |--+--+--\
// +-----------------------------------+  |  |  |
// | TfLiteIntArray (variable size)    |<-/  |  |
// +-----------------------------------+     |  |
// | TfLiteIntArray (variable size)    |<----/  |
// +-----------------------------------+        |
// | TfLiteIntArray (variable size)    |<-------/
// +-----------------------------------+
TfLiteDelegateParams* CreateDelegateParams(TfLiteDelegate* delegate,
                                           const NodeSubset& node_subset) {
  // Step 1: Calculate the allocation size.
  int allocation_size = sizeof(TfLiteDelegateParams);

  int nodes_to_replace_size =
      TfLiteIntArrayGetSizeInBytes(node_subset.nodes.size());
  allocation_size += nodes_to_replace_size;

  int input_tensors_size =
      TfLiteIntArrayGetSizeInBytes(node_subset.input_tensors.size());
  allocation_size += input_tensors_size;

  int output_tensors_size =
      TfLiteIntArrayGetSizeInBytes(node_subset.output_tensors.size());
  allocation_size += output_tensors_size;

  // Step 2: Allocate the memory.
  // Use `char*` for conveniently step through the allocated space by bytes.
  char* allocation = reinterpret_cast<char*>(malloc(allocation_size));

  // Step 3: Fill all data structures structures.
  TfLiteDelegateParams* params =
      reinterpret_cast<TfLiteDelegateParams*>(allocation);
  params->delegate = delegate;
  allocation += sizeof(TfLiteDelegateParams);

  params->nodes_to_replace = reinterpret_cast<TfLiteIntArray*>(allocation);
  CopyVectorToTfLiteIntArray(node_subset.nodes, params->nodes_to_replace);
  allocation += nodes_to_replace_size;

  params->input_tensors = reinterpret_cast<TfLiteIntArray*>(allocation);
  CopyVectorToTfLiteIntArray(node_subset.input_tensors, params->input_tensors);
  allocation += input_tensors_size;

  params->output_tensors = reinterpret_cast<TfLiteIntArray*>(allocation);
  CopyVectorToTfLiteIntArray(node_subset.output_tensors,
                             params->output_tensors);
  allocation += output_tensors_size;

  return params;
}

}  // namespace

TfLiteStatus Subgraph::ReplaceNodeSubsetsWithDelegateKernels(
    TfLiteRegistration registration, const TfLiteIntArray* nodes_to_replace,
    TfLiteDelegate* delegate) {
  // Annotate the registration as DELEGATE op.
  registration.builtin_code = BuiltinOperator_DELEGATE;

  // Analyze the graph to find all independent node_subsets that are either
  // fully not-this-delegate or this-delegate computation.
  InterpreterInfo info(this);
  std::vector<NodeSubset> node_subsets;
  PartitionGraphIntoIndependentNodeSubsets(&info, nodes_to_replace,
                                           &node_subsets);

  execution_plan_.clear();

  for (auto& node_subset : node_subsets) {
    // Subsets calimed by the delegate should have a "macro" op created, the
    // other node_subsets (kTfNonPartition) just have their nodes added back to
    // the execution plan.
    switch (node_subset.type) {
      case NodeSubset::kTfNonPartition:
        for (auto it = node_subset.nodes.begin(); it != node_subset.nodes.end();
             ++it) {
          execution_plan_.push_back(*it);
        }
        break;
      case NodeSubset::kTfPartition: {
        int node_index;

        TfLiteDelegateParams* params =
            CreateDelegateParams(delegate, node_subset);
        TF_LITE_ENSURE_STATUS(AddNodeWithParameters(
            node_subset.input_tensors, node_subset.output_tensors, nullptr, 0,
            params, &registration, &node_index));

        // Initialize the output tensors's delegate-related fields.
        for (int tensor_index : node_subset.output_tensors) {
          TfLiteTensor* tensor = &tensors_[tensor_index];
          TF_LITE_ENSURE(context_, tensor->delegate == nullptr ||
                                       tensor->delegate == delegate);
          tensor->delegate = delegate;
        }

        // Associate the node with the delegate.
        TfLiteNode* node = &nodes_and_registration_[node_index].first;
        node->delegate = delegate;
      } break;
      case NodeSubset::kTfUnexplored:
        return kTfLiteError;
        break;
    }
  }
  return kTfLiteOk;
}

TfLiteExternalContext* Subgraph::GetExternalContext(
    TfLiteExternalContextType type) {
  if (type >= 0 && type < kTfLiteMaxExternalContexts) {
    return external_contexts_[type];
  }
  return nullptr;
}

TfLiteExternalContext* Subgraph::GetExternalContext(
    struct TfLiteContext* context, TfLiteExternalContextType type) {
  return static_cast<Subgraph*>(context->impl_)->GetExternalContext(type);
}

void Subgraph::SetExternalContext(TfLiteExternalContextType type,
                                  TfLiteExternalContext* ctx) {
  if (type >= 0 && type < kTfLiteMaxExternalContexts) {
    external_contexts_[type] = ctx;
  }
}

void Subgraph::SetExternalContext(struct TfLiteContext* context,
                                  TfLiteExternalContextType type,
                                  TfLiteExternalContext* ctx) {
  return static_cast<Subgraph*>(context->impl_)->SetExternalContext(type, ctx);
}

// Gets an TfLiteIntArray* representing the execution plan. The interpreter owns
// this memory and it is only guaranteed to exist during the invocation of the
// delegate prepare.
TfLiteStatus Subgraph::GetExecutionPlan(TfLiteIntArray** execution_plan) {
  // TODO(aselle): Do not make a copy here
  plan_cache_.reset(TfLiteIntArrayCreate(execution_plan_.size()));
  *execution_plan = plan_cache_.get();
  static_assert(sizeof(plan_cache_->data[0]) == sizeof(execution_plan_[0]),
                "TfLiteIntArray and execution_plan do not contain same type.");
  std::memcpy(plan_cache_->data, execution_plan_.data(),
              sizeof(plan_cache_->data[0]) * execution_plan_.size());
  return kTfLiteOk;
}

// WARNING: This is an experimental interface that is subject to change.
// Entry point for C node plugin API to get the execution plan
TfLiteStatus Subgraph::GetExecutionPlan(struct TfLiteContext* context,
                                        TfLiteIntArray** execution_plan) {
  return static_cast<Subgraph*>(context->impl_)
      ->GetExecutionPlan(execution_plan);
}

TfLiteStatus Subgraph::SetInputs(std::vector<int> inputs) {
  TF_LITE_ENSURE_OK(&context_,
                    CheckTensorIndices("inputs", inputs.data(), inputs.size()));
  inputs_ = std::move(inputs);
  return kTfLiteOk;
}

TfLiteStatus Subgraph::SetOutputs(std::vector<int> outputs) {
  TF_LITE_ENSURE_OK(
      &context_, CheckTensorIndices("outputs", outputs.data(), outputs.size()));
  outputs_ = std::move(outputs);
  return kTfLiteOk;
}

TfLiteStatus Subgraph::SetVariables(std::vector<int> variables) {
  TF_LITE_ENSURE_OK(&context_, CheckTensorIndices("variables", variables.data(),
                                                  variables.size()));
  variables_ = std::move(variables);
  return kTfLiteOk;
}

TfLiteStatus Subgraph::CheckTensorIndices(const char* label, const int* indices,
                                          int length) {
  // Making sure kOptionalTensor is not re-defined to something other than -1.
  static_assert(kOptionalTensor == -1, "kOptionalTensor should be defined -1");

  for (int i = 0; i < length; i++) {
    int index = indices[i];
    // Continue if index == kOptionalTensor before additional comparisons below,
    // size_t(-1) is always >= context_tensors_size.
    if (index == kOptionalTensor) {
      continue;
    }
    if (index < 0 || static_cast<size_t>(index) >= context_->tensors_size) {
      ReportError("Invalid tensor index %d in %s\n", index, label);
      consistent_ = false;
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::BytesRequired(TfLiteType type, const int* dims,
                                     size_t dims_size, size_t* bytes) {
  // TODO(aselle): Check for overflow here using overflow.h in TensorFlow
  // MultiplyWithoutOverflow.
  TF_LITE_ENSURE(context_, bytes != nullptr);
  size_t count = 1;
  for (int k = 0; k < dims_size; k++) count *= dims[k];
  switch (type) {
    case kTfLiteFloat32:
      *bytes = sizeof(float) * count;
      break;
    case kTfLiteInt16:
      *bytes = sizeof(int16_t) * count;
      break;
    case kTfLiteInt32:
      *bytes = sizeof(int32_t) * count;
      break;
    case kTfLiteUInt8:
      *bytes = sizeof(uint8_t) * count;
      break;
    case kTfLiteInt64:
      *bytes = sizeof(int64_t) * count;
      break;
    case kTfLiteBool:
      *bytes = sizeof(bool) * count;
      break;
    case kTfLiteComplex64:
      *bytes = sizeof(std::complex<float>) * count;
      break;
    case kTfLiteInt8:
      *bytes = sizeof(int8_t) * count;
      break;
    default:
      ReportError(
          "Only float32, int8, int16, int32, int64, uint8, bool, complex64 "
          "supported currently.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::AllocateTensors() {
  if (!consistent_) {
    ReportError("AllocateTensors() called on inconsistent model.");
    return kTfLiteError;
  }

  // Explicit (re)allocation is necessary if nodes have been changed or tensors
  // have been resized. For inputs marked as dynamic, we can't short-circuit the
  // allocation as the client may have done the resize manually.
  if (state_ != kStateUninvokable &&
      !HasDynamicTensorImpl(*context_, inputs())) {
    return kTfLiteOk;
  }

  next_execution_plan_index_to_prepare_ = 0;
  if (memory_planner_) {
    TF_LITE_ENSURE_STATUS(memory_planner_->ResetAllocations());
  }

  TF_LITE_ENSURE_STATUS(PrepareOpsAndTensors());

  state_ = kStateInvokable;

  // Reset the variable tensors to zero after (re)allocating the tensors.
  // Developers shouldn't rely on the side effect of this function to reset
  // variable tesnsors. They should call `ResetVariableTensors` directly
  // instead.
  ResetVariableTensors();

  return kTfLiteOk;
}

// TODO(ycling): Support non-zero default values.
TfLiteStatus Subgraph::ResetVariableTensors() {
  for (auto& tensor : tensors_) {
    if (!tensor.is_variable) {
      continue;
    }

    // Variable tensors have to be `kTfLiteArenaRwPersistent`, and must be
    // allocated after the initial `PrepareOpsAndTensors()` is called.
    TF_LITE_ENSURE_EQ(context_, tensor.allocation_type,
                      kTfLiteArenaRwPersistent);
    TF_LITE_ENSURE(context_, tensor.data.raw != nullptr);

    memset(tensor.data.raw, 0, tensor.bytes);
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::AddNodeWithParameters(
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const char* init_data, size_t init_data_size, void* builtin_data,
    const TfLiteRegistration* registration, int* node_index) {
  if (state_ == kStateInvokableAndImmutable) {
    ReportError("AddNodeWithParameters is disallowed when graph is immutable.");
    return kTfLiteError;
  }
  state_ = kStateUninvokable;

  std::unique_ptr<void, decltype(free)*> builtin_data_deleter(builtin_data,
                                                              free);

  TF_LITE_ENSURE_OK(context_, CheckTensorIndices("node inputs", inputs.data(),
                                                 inputs.size()));
  TF_LITE_ENSURE_OK(
      &context_,
      CheckTensorIndices("node outputs", outputs.data(), outputs.size()));

  int new_node_index = nodes_and_registration_.size();
  if (node_index) *node_index = new_node_index;
  nodes_and_registration_.resize(nodes_and_registration_.size() + 1);
  auto& node_and_reg = nodes_and_registration_.back();
  TfLiteNode& node = node_and_reg.first;
  if (node.inputs) TfLiteIntArrayFree(node.inputs);
  if (node.outputs) TfLiteIntArrayFree(node.outputs);
  if (node.temporaries) TfLiteIntArrayFree(node.temporaries);

  // NOTE, here we are not using move semantics yet, since our internal
  // representation isn't std::vector, but in the future we would like to avoid
  // copies, so we want the interface to take r-value references now.
  node.inputs = ConvertVectorToTfLiteIntArray(inputs);
  node.outputs = ConvertVectorToTfLiteIntArray(outputs);
  node.temporaries = TfLiteIntArrayCreate(0);
  if (init_data) {
    node.user_data = OpInit(*registration, init_data, init_data_size);
  } else {
    node.user_data =
        OpInit(*registration,
               reinterpret_cast<const char*>(builtin_data_deleter.get()), 0);
  }

  node.builtin_data = builtin_data_deleter.release();
  // TODO(ycling): Filling `custom_initial_data` and `custom_initial_data_size`
  // properly for nodes generated by ReplaceNodeSubsetsWithDelegateKernels.

  if (registration->builtin_code == BuiltinOperator_CUSTOM) {
    // When it's a CUSTOM op, the `custom_options` field in the Flatbuffer
    // `Operator` table is passed in.
    node.custom_initial_data = init_data;
    node.custom_initial_data_size = init_data_size;
  } else {
    node.custom_initial_data = nullptr;
    node.custom_initial_data_size = 0;
  }

  node.delegate = nullptr;
  node_and_reg.second = *registration;
  execution_plan_.push_back(new_node_index);
  return kTfLiteOk;
}

TfLiteStatus Subgraph::ResizeInputTensor(int tensor_index,
                                         const std::vector<int>& dims) {
  if (state_ == kStateInvokableAndImmutable) {
    ReportError("ResizeInputTensor is disallowed when graph is immutable.");
    return kTfLiteError;
  }

  // TODO(aselle): All bounds checks can be implemented as one-sided bounds
  // checks by casting to unsigned for efficiency. Profile before doing this.
  TF_LITE_ENSURE(context_,
                 tensor_index < context_->tensors_size && tensor_index >= 0);
  TfLiteTensor* tensor = &context_->tensors[tensor_index];

  // Short-circuit the state change if the dimensions don't change, avoiding
  // unnecessary (re)allocations.
  if (EqualArrayAndTfLiteIntArray(tensor->dims, dims.size(), dims.data())) {
    return kTfLiteOk;
  }

  state_ = kStateUninvokable;
  return ResizeTensorImpl(tensor, ConvertVectorToTfLiteIntArray(dims));
}

TfLiteStatus Subgraph::PrepareOpsStartingAt(
    int first_execution_plan_index, int* last_execution_plan_index_prepared) {
  if (first_execution_plan_index == 0) {
    has_dynamic_tensors_ = false;
  }
  for (int execution_plan_index = first_execution_plan_index;
       execution_plan_index < execution_plan_.size(); execution_plan_index++) {
    int node_index = execution_plan_[execution_plan_index];
    TfLiteNode& node = nodes_and_registration_[node_index].first;
    const TfLiteRegistration& registration =
        nodes_and_registration_[node_index].second;
    EnsureTensorsVectorCapacity();
    if (OpPrepare(registration, &node) == kTfLiteError) {
      return ReportOpError(context_, node, registration, node_index,
                           "failed to prepare");
    }

    *last_execution_plan_index_prepared = execution_plan_index;

    // Discontinue if the node has dynamic outputs. Note that we don't
    // stop for dynamic temporary tensors since they won't affect the
    // sizes of other tensors in the graph.
    if (HasDynamicTensor(*context_, node.outputs)) {
      has_dynamic_tensors_ = true;
      return kTfLiteOk;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::PrepareOpsAndTensors() {
  if (!memory_planner_) {
    memory_planner_.reset(new ArenaPlanner(
        context_, std::unique_ptr<GraphInfo>(new InterpreterInfo(this)),
        /*preserve_inputs=*/true, /*preserve_intermediates*/ false));
    memory_planner_->PlanAllocations();
  }

  int last_exec_plan_index_prepared = 0;

  TF_LITE_ENSURE_STATUS(PrepareOpsStartingAt(
      next_execution_plan_index_to_prepare_, &last_exec_plan_index_prepared));
  TF_LITE_ENSURE_STATUS(memory_planner_->ExecuteAllocations(
      next_execution_plan_index_to_prepare_, last_exec_plan_index_prepared));

  next_execution_plan_index_to_prepare_ = last_exec_plan_index_prepared + 1;
  return kTfLiteOk;
}

TfLiteStatus Subgraph::Invoke() {
  if (!consistent_) {
    ReportError("Invoke called on model that is not consistent.");
    return kTfLiteError;
  }

  TfLiteStatus status = kTfLiteOk;
  if (state_ == kStateUninvokable) {
    ReportError("Invoke called on model that is not ready.");
    return kTfLiteError;
  }

  if (nnapi_delegate_) {
    if (next_execution_plan_index_to_prepare_ == execution_plan_.size()) {
      TF_LITE_ENSURE_OK(context_, nnapi_delegate_->Invoke(this));
      return kTfLiteOk;
    } else {
      // TODO(aselle): In the future, we would like this to be an
      // automatic tflite CPU fallback.
      ReportError(
          "NNAPI was requested, but dependent sized tensors "
          "being used.\n");
      return kTfLiteError;
    }
  }

  // Invocations are always done in node order.
  // Note that calling Invoke repeatedly will cause the original memory plan to
  // be reused, unless either ResizeInputTensor() or AllocateTensors() has been
  // called.
  for (int execution_plan_index = 0;
       execution_plan_index < execution_plan_.size(); execution_plan_index++) {
    if (execution_plan_index == next_execution_plan_index_to_prepare_) {
      TF_LITE_ENSURE_STATUS(PrepareOpsAndTensors());
      TF_LITE_ENSURE(context_, next_execution_plan_index_to_prepare_ >=
                                   execution_plan_index);
    }
    int node_index = execution_plan_[execution_plan_index];
    TfLiteNode& node = nodes_and_registration_[node_index].first;
    const TfLiteRegistration& registration =
        nodes_and_registration_[node_index].second;
    SCOPED_OPERATOR_PROFILE(profiler_, node_index);

    // TODO(ycling): This is an extra loop through inputs to check if the data
    // need to be copied from Delegate buffer to raw memory, which is often not
    // needed. We may want to cache this in prepare to know if this needs to be
    // done for a node or not.
    for (int i = 0; i < node.inputs->size; ++i) {
      int tensor_index = node.inputs->data[i];
      if (tensor_index == kOptionalTensor) {
        continue;
      }
      TfLiteTensor* tensor = &tensors_[tensor_index];
      if (tensor->delegate && tensor->delegate != node.delegate &&
          tensor->data_is_stale) {
        EnsureTensorDataIsReadable(tensor_index);
      }
    }

    EnsureTensorsVectorCapacity();
    tensor_resized_since_op_invoke_ = false;
    if (OpInvoke(registration, &node) == kTfLiteError) {
      status = ReportOpError(context_, node, registration, node_index,
                             "failed to invoke");
    }

    // Force execution prep for downstream ops if the latest op triggered the
    // resize of a dynamic tensor.
    if (tensor_resized_since_op_invoke_ &&
        HasDynamicTensor(*context_, node.outputs)) {
      next_execution_plan_index_to_prepare_ = execution_plan_index + 1;
    }
  }

  return status;
}

TfLiteStatus Subgraph::ResizeTensor(TfLiteContext* context,
                                    TfLiteTensor* tensor,
                                    TfLiteIntArray* new_size) {
  // Note here that context->impl_ is recovering the this pointer for an
  // instance of Interpreter to call into the member function ResizeTensorImpl
  // (this function is static).
  return static_cast<Subgraph*>(context->impl_)
      ->ResizeTensorImpl(tensor, new_size);
}

void Subgraph::ReportErrorImpl(const char* format, va_list args) {
  error_reporter_->Report(format, args);
}

void Subgraph::ReportErrorC(TfLiteContext* context, const char* format, ...) {
  va_list args;
  va_start(args, format);
  auto* f = static_cast<Subgraph*>(context->impl_);
  // Note here that context->impl_ is recovering the this pointer for an
  // instance of Subgraph to call into the member function ReportErrorImpl
  // (this function is static).
  f->ReportErrorImpl(format, args);
  va_end(args);
}

// Entry point for C node plugin API to report an error.
void Subgraph::ReportError(const char* format, ...) {
  va_list args;
  va_start(args, format);
  auto* f = static_cast<Subgraph*>(context_->impl_);
  // Note here that context->impl_ is recovering the this pointer for an
  // instance of Subgraph to call into the member function ReportErrorImpl
  // (this function is static).
  f->ReportErrorImpl(format, args);
  va_end(args);
}

TfLiteStatus Subgraph::AddTensors(int tensors_to_add,
                                  int* first_new_tensor_index) {
  const size_t base_index = tensors_.size();
  if (first_new_tensor_index) *first_new_tensor_index = base_index;
  tensors_.resize(tensors_.size() + tensors_to_add);
  for (size_t i = base_index; i < tensors_.size(); i++) {
    memset(&tensors_[i], 0, sizeof(tensors_[i]));
    tensors_[i].buffer_handle = kTfLiteNullBufferHandle;
  }
  context_->tensors = tensors_.data();
  context_->tensors_size = tensors_.size();
  return kTfLiteOk;
}

TfLiteStatus Subgraph::AddTensors(TfLiteContext* context, int tensors_to_add,
                                  int* first_new_tensor_index) {
  // Note here that context->impl_ is recovering the this pointer for an
  // instance of Interpreter to call into the member function AddTensors
  // (this function is static).
  return static_cast<Subgraph*>(context->impl_)
      ->AddTensors(tensors_to_add, first_new_tensor_index);
}

TfLiteStatus Subgraph::GetNodeAndRegistration(
    int node_index, TfLiteNode** node, TfLiteRegistration** registration) {
  TF_LITE_ENSURE(context_, node_index >= 0);
  auto nodes_size = nodes_and_registration_.size();
  TF_LITE_ENSURE(context_, static_cast<size_t>(node_index) < nodes_size);
  TF_LITE_ENSURE(context_, node != nullptr && registration != nullptr);
  auto& node_and_reg = nodes_and_registration_[node_index];
  *node = &node_and_reg.first;
  *registration = &node_and_reg.second;
  return kTfLiteOk;
}

TfLiteStatus Subgraph::GetNodeAndRegistration(
    struct TfLiteContext* context, int node_index, TfLiteNode** node,
    TfLiteRegistration** registration) {
  return static_cast<Subgraph*>(context->impl_)
      ->GetNodeAndRegistration(node_index, node, registration);
}

TfLiteStatus Subgraph::SetTensorParametersReadOnly(
    int tensor_index, TfLiteType type, const char* name, const size_t rank,
    const int* dims, TfLiteQuantizationParams quantization, const char* buffer,
    size_t bytes, const Allocation* allocation) {
  if (state_ == kStateInvokableAndImmutable) {
    ReportError(
        "SetTensorParametersReadOnly is disallowed when graph is immutable.");
    return kTfLiteError;
  }

  TF_LITE_ENSURE(context_,
                 tensor_index < context_->tensors_size && tensor_index >= 0);
  // For most tensors we know exactly how much memory is necessary so we can
  // ensure the buffer is large enough. However, we need to skip string tensors
  // because their sizes change with the contents of the individual strings.
  if (type != kTfLiteString) {
    size_t required_bytes;
    TF_LITE_ENSURE_OK(context_,
                      BytesRequired(type, dims, rank, &required_bytes));
    TF_LITE_ENSURE_EQ(context_, required_bytes, bytes);
  }

  TfLiteTensor& tensor = context_->tensors[tensor_index];
  if (type == tensor.type &&
      EqualArrayAndTfLiteIntArray(tensor.dims, rank, dims)) {
    // Fast path which does not invalidate the invokable property.
    TfLiteTensorDataFree(&tensor);
    tensor.data.raw = const_cast<char*>(buffer);
    if (!tensor.dims) tensor.dims = ConvertArrayToTfLiteIntArray(rank, dims);
    tensor.params = quantization;
    tensor.allocation_type = kTfLiteMmapRo;
    tensor.allocation = allocation;
  } else {
    state_ = kStateUninvokable;
    TfLiteTensorReset(type, name, ConvertArrayToTfLiteIntArray(rank, dims),
                      quantization, const_cast<char*>(buffer), bytes,
                      kTfLiteMmapRo, allocation, false, &tensor);
  }
  return kTfLiteOk;
}

// Set description of inputs/outputs/data/fptrs for node `node_index`.
// This variant assumes an external buffer has been allocated of size
// bytes. The lifetime of buffer must be ensured to be greater or equal
// to Interpreter.
TfLiteStatus Subgraph::SetTensorParametersReadWrite(
    int tensor_index, TfLiteType type, const char* name, const size_t rank,
    const int* dims, TfLiteQuantizationParams quantization, bool is_variable) {
  if (state_ == kStateInvokableAndImmutable) {
    ReportError(
        "SetTensorParametersReadWrite is disallowed when graph is immutable.");
    return kTfLiteError;
  }
  TF_LITE_ENSURE(context_,
                 tensor_index < context_->tensors_size && tensor_index >= 0);
  size_t required_bytes = 0;
  if (type != kTfLiteString) {
    // These types will be allocated in our arena so we need to record how
    // many bytes we will need based on the dimensions. String tensors are
    // allocated dynamically and we can't know ahead of time how much space
    // they will require.
    TF_LITE_ENSURE_OK(context_,
                      BytesRequired(type, dims, rank, &required_bytes));
  }

  TfLiteAllocationType allocation_type = kTfLiteArenaRw;
  if (type == kTfLiteString) {
    if (is_variable) {
      // We don't have a real use case for string variable tensor.
      ReportError("String variable tensor isn't supported.");
      return kTfLiteError;
    }
    allocation_type = kTfLiteDynamic;
  } else if (is_variable) {
    allocation_type = kTfLiteArenaRwPersistent;
  }

  TfLiteTensorReset(type, name, ConvertArrayToTfLiteIntArray(rank, dims),
                    quantization,
                    /*buffer=*/nullptr, required_bytes, allocation_type,
                    nullptr, is_variable, &context_->tensors[tensor_index]);
  return kTfLiteOk;
}

TfLiteStatus Subgraph::SetExecutionPlan(const std::vector<int>& new_plan) {
  for (int node_index : new_plan) {
    TF_LITE_ENSURE(context_, node_index >= 0 &&
                                 node_index < nodes_and_registration_.size());
  }
  execution_plan_ = new_plan;
  return kTfLiteOk;
}

TfLiteStatus Subgraph::ResizeTensorImpl(TfLiteTensor* tensor,
                                        TfLiteIntArray* new_size) {
  // Note that in theory we could resize kTfLiteArenaRwPersistent tensors too.
  if (tensor->allocation_type == kTfLiteArenaRw ||
      tensor->allocation_type == kTfLiteDynamic ||
      tensor->allocation_type == kTfLiteArenaRwPersistent) {
    tensor_resized_since_op_invoke_ |=
        TfLiteIntArrayEqual(tensor->dims, new_size) == 0;
    if (tensor->type != kTfLiteString) {
      size_t bytesRequired;
      TfLiteStatus status = BytesRequired(tensor->type, new_size->data,
                                          new_size->size, &bytesRequired);
      if (status != kTfLiteOk) {
        TfLiteIntArrayFree(new_size);
        return kTfLiteError;
      }

      // Realloc space for kTfLiteDynamic tensors.
      TfLiteTensorRealloc(bytesRequired, tensor);
      tensor->bytes = bytesRequired;
    }
    if (tensor->dims) TfLiteIntArrayFree(tensor->dims);
    tensor->dims = new_size;

    if (tensor->allocation_type != kTfLiteDynamic) {
      tensor->data.raw = nullptr;
    }
  } else {
    // kTfLiteMmapRo tensors are stored in the flatbuffer and are therefore
    // of fixed size.
    TfLiteIntArrayFree(new_size);
    ReportError("Attempting to resize a fixed-size tensor.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

void Subgraph::UseNNAPI(bool enable) {
  // TODO(aselle): This is a workaround for finding if NNAPI exists.
  // We also need to make sure getLibraryHandle() is renamed to be NNAPI
  // prefixed.
  if (!NNAPIDelegate::IsSupported()) enable = false;
  if (!enable) {
    nnapi_delegate_.reset();
  } else if (!nnapi_delegate_) {
    nnapi_delegate_.reset(new NNAPIDelegate);
  }
}

void Subgraph::SwitchToDelegateContext() {
  context_->GetNodeAndRegistration = GetNodeAndRegistration;
  context_->ReplaceNodeSubsetsWithDelegateKernels =
      ReplaceNodeSubsetsWithDelegateKernels;
  context_->GetExecutionPlan = GetExecutionPlan;
}

void Subgraph::SwitchToKernelContext() {
  context_->GetNodeAndRegistration = [](struct TfLiteContext* context,
                                        int node_index, TfLiteNode** node,
                                        TfLiteRegistration** registration) {
    return ForbiddenContextFunction(context);
  };
  context_->ReplaceNodeSubsetsWithDelegateKernels =
      [](TfLiteContext* context, TfLiteRegistration registration,
         const TfLiteIntArray* nodes_to_replace, TfLiteDelegate* delegate) {
        return ForbiddenContextFunction(context);
      };
  context_->GetExecutionPlan = [](struct TfLiteContext* context,
                                  TfLiteIntArray**) {
    return ForbiddenContextFunction(context);
  };
}

TfLiteStatus Subgraph::ModifyGraphWithDelegate(TfLiteDelegate* delegate) {
  if (state_ == kStateInvokableAndImmutable) {
    ReportError(
        "ModifyGraphWithDelegate is disallowed when graph is immutable.");
    return kTfLiteError;
  }

  if (!(delegate->flags & kTfLiteDelegateFlagsAllowDynamicTensors)) {
    int last_execution_plan_index_prepared;
    TF_LITE_ENSURE_OK(&context_, PrepareOpsStartingAt(
                                     0, &last_execution_plan_index_prepared));
    if (has_dynamic_tensors_) {
      ReportError(
          "Attempting to use a delegate that only supports static-sized "
          "tensors with a graph that has dynamic-sized tensors.");
      return kTfLiteError;
    }
  }

  const bool was_invokable_before_delegate = state_ == kStateInvokable;

  // TODO(aselle): Consider if it is worth storing pointers to delegates.
  // Setup additional context interface.
  SwitchToDelegateContext();

  TfLiteStatus status = delegate->Prepare(context_, delegate);

  // Remove additional context info.
  SwitchToKernelContext();

  TF_LITE_ENSURE_OK(context_, status);

  // If the memory planner has already been created, we need to execute
  // planning again to account for the updated graph topology.
  if (memory_planner_) {
    state_ = kStateUninvokable;
    TF_LITE_ENSURE_OK(context_, memory_planner_->PlanAllocations());
  }

  if (!(delegate->flags & kTfLiteDelegateFlagsAllowDynamicTensors)) {
    // Reset the state to force tensor/op reallocation.
    state_ = kStateUninvokable;
    TF_LITE_ENSURE_OK(context_, AllocateTensors());
    TF_LITE_ENSURE_EQ(context_, state_, kStateInvokable);
    // After using a delegate which doesn't support dynamic tensors, make the
    // entire graph immutable.
    state_ = kStateInvokableAndImmutable;
  } else if (was_invokable_before_delegate) {
    // If the graph was invokable prior to delegate application, flush
    // allocation now to leave it in a consistent state.
    TF_LITE_ENSURE_OK(context_, AllocateTensors());
    TF_LITE_ENSURE_EQ(context_, state_, kStateInvokable);
  }

  return status;
}

}  // namespace tflite
