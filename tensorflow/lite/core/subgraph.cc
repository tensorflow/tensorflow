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

#include <stdarg.h>
#include <stddef.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/experimental/resource/resource_base.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"
#ifdef TFLITE_USE_SIMPLE_MEMORY_PLANNER
#include "tensorflow/lite/simple_planner.h"
#else
#include "tensorflow/lite/arena_planner.h"
#endif

#include <fstream>
#include <iostream>

#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {

namespace {

struct TfLiteQuantizationDeleter {
  void operator()(TfLiteQuantization* q) {
    if (q) TfLiteQuantizationFree(q);
  }
};

using ScopedTfLiteQuantization =
    std::unique_ptr<TfLiteQuantization, TfLiteQuantizationDeleter>;

struct TfLiteSparsityDeleter {
  void operator()(TfLiteSparsity* s) {
    if (s) TfLiteSparsityFree(s);
  }
};

using ScopedTfLiteSparsity =
    std::unique_ptr<TfLiteSparsity, TfLiteSparsityDeleter>;

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
// We're registering this function to several different function to save
// compiled binary size. Please note the restrictions:
// * The type of first parameter have to be `TfLiteContext*`.
// * All parameters must be trivially destructible. (E.g. No C++ class)
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
                          const TensorIntArray& int_array,
                          int* dynamic_tensor_index) {
  for (int i : int_array) {
    if (i == kTfLiteOptionalTensor) continue;
    const TfLiteTensor& tensor = context.tensors[i];
    if (tensor.allocation_type == kTfLiteDynamic) {
      if (dynamic_tensor_index) {
        *dynamic_tensor_index = i;
      }
      return true;
    }
  }
  return false;
}

bool HasDynamicTensor(const TfLiteContext& context,
                      const TfLiteIntArray* int_array,
                      int* dynamic_tensor_index) {
  return HasDynamicTensorImpl(context, TfLiteIntArrayView{int_array},
                              dynamic_tensor_index);
}

// Gets the legacy TfLiteQuantizationParams from the current TfLiteQuantization.
TfLiteQuantizationParams GetLegacyQuantization(
    const TfLiteQuantization& quantization) {
  TfLiteQuantizationParams legacy_quantization;
  legacy_quantization.scale = 0;
  legacy_quantization.zero_point = 0;

  // If the quantization type isn't affine, return the empty
  // legacy_quantization.
  if (quantization.type != kTfLiteAffineQuantization) {
    return legacy_quantization;
  }

  auto* affine_quantization =
      static_cast<TfLiteAffineQuantization*>(quantization.params);
  if (!affine_quantization || !affine_quantization->scale ||
      !affine_quantization->zero_point ||
      affine_quantization->scale->size != 1 ||
      affine_quantization->zero_point->size != 1) {
    return legacy_quantization;
  }

  // We know its per-layer quantization now.
  legacy_quantization.scale = affine_quantization->scale->data[0];
  legacy_quantization.zero_point = affine_quantization->zero_point->data[0];
  return legacy_quantization;
}

static constexpr const char kUnknownCustomOpName[] = "UnknownCustomOp";
const char* GetTFLiteOpName(const TfLiteRegistration& op_reg) {
  if (op_reg.builtin_code == tflite::BuiltinOperator_CUSTOM) {
    const char* const custom_name = op_reg.custom_name;
    return custom_name ? custom_name : kUnknownCustomOpName;
  }
  if (op_reg.builtin_code == tflite::BuiltinOperator_DELEGATE &&
      op_reg.custom_name) {
    return op_reg.custom_name;
  }
  return tflite::EnumNamesBuiltinOperator()[op_reg.builtin_code];
}

// Verifies custom allocation for tensor, if applicable.
TfLiteStatus VerifyCustomAllocationForTensor(
    TfLiteContext* context,
    const std::map<int, TfLiteCustomAllocation>& tensor_idx_to_alloc,
    const int tensor_idx) {
  auto& tensor = context->tensors[tensor_idx];
  if (tensor.allocation_type != kTfLiteCustom) return kTfLiteOk;
  const auto idx_and_alloc = tensor_idx_to_alloc.find(tensor_idx);
  TF_LITE_ENSURE(context, idx_and_alloc != tensor_idx_to_alloc.end());
  if (idx_and_alloc->second.bytes < tensor.bytes) {
    TF_LITE_KERNEL_LOG(context,
                       "Custom allocation is too small for tensor idx: %d",
                       tensor_idx);
    return kTfLiteError;
  }
  return kTfLiteOk;
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

  size_t num_tensors() const override { return subgraph_->tensors_size(); }
  TfLiteTensor* tensor(size_t index) override {
    return subgraph_->tensor(index);
  }
  size_t num_execution_nodes() const override {
    return subgraph_->execution_plan().size();
  }
  size_t num_total_nodes() const override { return subgraph_->nodes_size(); }
  const TfLiteNode& node(size_t index) const override {
    int node_index = subgraph_->execution_plan()[index];
    return subgraph_->nodes_and_registration()[node_index].first;
  }
  size_t node_index(size_t index) const override {
    return subgraph_->execution_plan()[index];
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
                   std::vector<std::unique_ptr<Subgraph>>* subgraphs,
                   resource::ResourceMap* resources,
                   resource::ResourceIDMap* resource_ids,
                   resource::InitializationStatusMap* initialization_status_map)
    : external_contexts_(external_contexts),
      error_reporter_(error_reporter),
      next_execution_plan_index_to_prepare_(0),
      next_execution_plan_index_to_plan_allocation_(0),
      subgraphs_(subgraphs),
      resources_(resources),
      resource_ids_(resource_ids),
      initialization_status_map_(initialization_status_map) {
  context_.impl_ = static_cast<void*>(this);
  context_.ResizeTensor = ResizeTensor;
  context_.ReportError = ReportErrorC;
  context_.AddTensors = AddTensors;
  context_.tensors = nullptr;
  context_.tensors_size = 0;
  context_.allow_fp32_relax_to_fp16 = false;
  context_.recommended_num_threads = -1;
  context_.GetExternalContext = GetExternalContext;
  context_.SetExternalContext = SetExternalContext;
  context_.profiler = nullptr;
  context_.GetTensor = nullptr;
  context_.GetEvalTensor = nullptr;
  context_.GetModelMetadata = GetModelMetadata;

  // Reserve some space for the tensors to avoid excessive resizing.
  tensors_.reserve(kTensorsReservedCapacity);
  nodes_and_registration_.reserve(kTensorsReservedCapacity);
  // Invalid to call these except from TfLiteDelegate
  SwitchToKernelContext();
}

Subgraph::~Subgraph() {
  for (int node_index = 0; node_index < nodes_and_registration_.size();
       ++node_index) {
    CleanupNode(node_index);
  }

  for (size_t i = 0; i < context_.tensors_size; i++) {
    TfLiteTensor* tensor = &context_.tensors[i];
    if (tensor->buffer_handle != kTfLiteNullBufferHandle &&
        tensor->delegate->FreeBufferHandle != nullptr) {
      tensor->delegate->FreeBufferHandle(&context_, tensor->delegate,
                                         &tensor->buffer_handle);
    }
    TfLiteTensorFree(tensor);
  }
}

void Subgraph::CleanupNode(int node_index) {
  TfLiteNode& node = nodes_and_registration_[node_index].first;
  const TfLiteRegistration& registration =
      nodes_and_registration_[node_index].second;
  TfLiteIntArrayFree(node.inputs);
  TfLiteIntArrayFree(node.outputs);
  TfLiteIntArrayFree(node.temporaries);
  TfLiteIntArrayFree(node.intermediates);
  if (node.builtin_data) free(node.builtin_data);
  OpFree(registration, node.user_data);
  node.builtin_data = nullptr;
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
  char* allocation = static_cast<char*>(malloc(allocation_size));

  // Step 3: Fill all data structures.
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

// Assumes that params is not nullptr.
void PopulatePreviewDelegateParams(const NodeSubset& node_subset,
                                   TfLiteDelegateParams* params) {
  // Since these params are used for previewing partitioning, params->delegate
  // is not required.
  params->delegate = nullptr;

  params->nodes_to_replace = TfLiteIntArrayCreate(node_subset.nodes.size());
  CopyVectorToTfLiteIntArray(node_subset.nodes, params->nodes_to_replace);

  params->input_tensors =
      TfLiteIntArrayCreate(node_subset.input_tensors.size());
  CopyVectorToTfLiteIntArray(node_subset.input_tensors, params->input_tensors);

  params->output_tensors =
      TfLiteIntArrayCreate(node_subset.output_tensors.size());
  CopyVectorToTfLiteIntArray(node_subset.output_tensors,
                             params->output_tensors);
}

}  // namespace

TfLiteStatus Subgraph::ReplaceNodeSubsetsWithDelegateKernels(
    TfLiteRegistration registration, const TfLiteIntArray* nodes_to_replace,
    TfLiteDelegate* delegate) {
  // Ignore empty node replacement sets.
  if (!nodes_to_replace->size) {
    return kTfLiteOk;
  }

  // Annotate the registration as DELEGATE op.
  registration.builtin_code = BuiltinOperator_DELEGATE;

  // Analyze the graph to find all independent node_subsets that are either
  // fully not-this-delegate or this-delegate computation.
  InterpreterInfo info(this);
  std::vector<NodeSubset> node_subsets;
  PartitionGraphIntoIndependentNodeSubsets(&info, nodes_to_replace,
                                           &node_subsets);

#ifdef __ANDROID__
  // On Android the log message below is used for diagnosing delegation success
  // also in production builds. Delegation happens sufficiently rarely that the
  // message isn't spammy.
  TFLITE_LOG_PROD(
      tflite::TFLITE_LOG_INFO,
      "Replacing %d node(s) with delegate (%s) node, yielding %zu partitions.",
      nodes_to_replace->size,
      registration.custom_name ? registration.custom_name : "unknown",
      node_subsets.size());
#else   // !__ANDROID__
  // Server-side, delegation may happen so often as to make logging spammy + we
  // don't have a clear need for the diagnostic in production builds.
  TFLITE_LOG(
      tflite::TFLITE_LOG_INFO,
      "Replacing %d node(s) with delegate (%s) node, yielding %zu partitions.",
      nodes_to_replace->size,
      registration.custom_name ? registration.custom_name : "unknown",
      node_subsets.size());
#endif  // __ANDROID__

  execution_plan_.clear();

  for (auto& node_subset : node_subsets) {
    // Subsets claimed by the delegate should have a "macro" op created, the
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
            node_subset.input_tensors, node_subset.output_tensors, {}, nullptr,
            0, params, &registration, &node_index));

        // Initialize the output tensors's delegate-related fields.
        for (int tensor_index : node_subset.output_tensors) {
          TfLiteTensor* tensor = &tensors_[tensor_index];
          TF_LITE_ENSURE(&context_, tensor->delegate == nullptr ||
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
  if (static_cast<int>(type) >= 0 && type < kTfLiteMaxExternalContexts) {
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
  if (static_cast<int>(type) >= 0 && type < kTfLiteMaxExternalContexts) {
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

void Subgraph::FreeDelegatePartitioningData() {
  for (auto& params : partitioning_preview_cache_) {
    TfLiteIntArrayFree(params.nodes_to_replace);
    TfLiteIntArrayFree(params.input_tensors);
    TfLiteIntArrayFree(params.output_tensors);
  }
  partitioning_preview_cache_.clear();
}

TfLiteStatus Subgraph::GetModelMetadata(const char* name, const char** ptr,
                                        size_t* bytes) {
  TF_LITE_ENSURE(&context_, ptr != nullptr);
  TF_LITE_ENSURE(&context_, bytes != nullptr);
  *ptr = nullptr;
  *bytes = 0;
  if (!metadata_) return kTfLiteError;
  const std::string name_str = name;
  auto itr = metadata_->find(name_str);
  if (itr != metadata_->end()) {
    *ptr = itr->second.c_str();
    *bytes = itr->second.size();
    return kTfLiteOk;
  }
  return kTfLiteError;
}

TfLiteStatus Subgraph::GetModelMetadata(const struct TfLiteContext* context,
                                        const char* name, const char** ptr,
                                        size_t* bytes) {
  return static_cast<Subgraph*>(context->impl_)
      ->GetModelMetadata(name, ptr, bytes);
}

TfLiteStatus Subgraph::PreviewDelegatePartitioning(
    const TfLiteIntArray* nodes_to_replace,
    TfLiteDelegateParams** partition_params_array, int* num_partitions) {
  // Ensure partitioning cache is empty.
  FreeDelegatePartitioningData();
  // Defaults.
  if (!partition_params_array || !num_partitions) return kTfLiteError;
  *partition_params_array = nullptr;
  *num_partitions = 0;
  if (!nodes_to_replace->size) {
    return kTfLiteOk;
  }

  // Partition the execution plan into node subsets.
  InterpreterInfo info(this);
  std::vector<NodeSubset> node_subsets;
  PartitionGraphIntoIndependentNodeSubsets(&info, nodes_to_replace,
                                           &node_subsets);

  // Create one TfLiteDelegateParams per node-subset which would be delegated.
  for (auto& node_subset : node_subsets) {
    if (node_subset.type != NodeSubset::kTfPartition) {
      continue;
    }
    partitioning_preview_cache_.emplace_back();
    PopulatePreviewDelegateParams(node_subset,
                                  &partitioning_preview_cache_.back());
    ++*num_partitions;
  }

  *partition_params_array = partitioning_preview_cache_.data();
  return kTfLiteOk;
}

TfLiteStatus Subgraph::PreviewDelegatePartitioning(
    struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
    TfLiteDelegateParams** partition_params_array, int* num_partitions) {
  return static_cast<Subgraph*>(context->impl_)
      ->PreviewDelegatePartitioning(nodes_to_replace, partition_params_array,
                                    num_partitions);
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

TfLiteStatus Subgraph::SetMetadata(
    const std::map<std::string, std::string>* metadata) {
  metadata_ = metadata;
  return kTfLiteOk;
}

void Subgraph::SetCancellationFunction(void* data,
                                       bool (*check_cancelled_func)(void*)) {
  cancellation_data_ = data;
  check_cancelled_func_ = check_cancelled_func;
}

bool Subgraph::IsCancelled() {
  return (check_cancelled_func_ != nullptr) &&
         (*check_cancelled_func_)(cancellation_data_);
}

void Subgraph::ReserveNodes(int count) {
  nodes_and_registration_.reserve(count);
}

TfLiteStatus Subgraph::CheckTensorIndices(const char* label, const int* indices,
                                          int length) {
  // Making sure kTfLiteOptionalTensor is not re-defined to something other than
  // -1.
  static_assert(kTfLiteOptionalTensor == -1,
                "kTfLiteOptionalTensor should be defined -1");

  for (int i = 0; i < length; i++) {
    int index = indices[i];
    // Continue if index == kTfLiteOptionalTensor before additional comparisons
    // below, size_t(-1) is always >= context_tensors_size.
    if (index == kTfLiteOptionalTensor) {
      continue;
    }
    if (index < 0 || static_cast<size_t>(index) >= context_.tensors_size) {
      ReportError(
          "Invalid tensor index %d in %s. The subgraph has %d tensors\n", index,
          label, context_.tensors_size);
      consistent_ = false;
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

// We have two arrays and we need to check that elements from one array don't
// show up in the other. We could sort both arrays and then iterate with two
// pointers from start to finish always increasing the smaller one but since
// these arrays are usually short (<25 elements for inputs, usually <3 for
// outputs), this might be slower than the naive approach (if arrays have size n
// and m, with n >> m ~ O(1), first approach is O(nlogn) whereas the other is
// O(n)). Plus, sorting the input and output arrays might not be something we
// want as it destroys ordering of elements.
//
// If it turns out that this is an issue, we can switch to the other algorithm.
TfLiteStatus Subgraph::CheckInputAndOutputForOverlap(const int* input_indices,
                                                     int num_inputs,
                                                     const int* output_indices,
                                                     int num_outputs) {
  for (int i = 0; i < num_inputs; i++) {
    for (int j = 0; j < num_outputs; j++) {
      if (input_indices[i] == output_indices[j]) {
        ReportError("Tensor %d is both input %d and output %d\n",
                    input_indices[i], i, j);
        consistent_ = false;
        return kTfLiteError;
      }
    }
  }
  return kTfLiteOk;
}

namespace {
// Multiply two sizes and return true if overflow occurred;
// This is based off tensorflow/overflow.h but is simpler as we already
// have unsigned numbers. It is also generalized to work where sizeof(size_t)
// is not 8.
TfLiteStatus MultiplyAndCheckOverflow(size_t a, size_t b, size_t* product) {
  // Multiplying a * b where a and b are size_t cannot result in overflow in a
  // size_t accumulator if both numbers have no non-zero bits in their upper
  // half.
  constexpr size_t size_t_bits = 8 * sizeof(size_t);
  constexpr size_t overflow_upper_half_bit_position = size_t_bits / 2;
  *product = a * b;
  // If neither integers have non-zero bits past 32 bits can't overflow.
  // Otherwise check using slow devision.
  if (TFLITE_EXPECT_FALSE((a | b) >> overflow_upper_half_bit_position != 0)) {
    if (a != 0 && *product / a != b) return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus Subgraph::BytesRequired(TfLiteType type, const int* dims,
                                     size_t dims_size, size_t* bytes) {
  TF_LITE_ENSURE(&context_, bytes != nullptr);
  // When 'dims_size' is 0, we simply assume it's a scalar. Therefore, we start
  // 'count' as 1.
  size_t count = 1;
  for (int k = 0; k < dims_size; k++) {
    size_t old_count = count;
    TF_LITE_ENSURE_MSG(
        &context_,
        MultiplyAndCheckOverflow(old_count, dims[k], &count) == kTfLiteOk,
        "BytesRequired number of elements overflowed.\n");
  }
  size_t type_size = 0;
  TF_LITE_ENSURE_OK(&context_, GetSizeOfType(&context_, type, &type_size));
  TF_LITE_ENSURE_MSG(
      &context_, MultiplyAndCheckOverflow(type_size, count, bytes) == kTfLiteOk,
      "BytesRequired number of bytes overflowed.\n");
  return kTfLiteOk;
}

TfLiteStatus Subgraph::AllocateTensors() {
  TFLITE_SCOPED_TAGGED_DEFAULT_PROFILE(profiler_.get(), "AllocateTensors");
  if (!consistent_) {
    ReportError("AllocateTensors() called on inconsistent model.");
    return kTfLiteError;
  }

  // Restore delegation state if applicable.
  TF_LITE_ENSURE_STATUS(RedoAllDelegates());

  // The runtime doesn't need to adjust any allocations if the state is
  // invokable & no inputs are dynamic (which implies memory plan is unchanged).
  const bool no_reallocations_necessary =
      state_ != kStateUninvokable &&
      !HasDynamicTensorImpl(context_, inputs(), &dynamic_tensor_index_);
  if (no_reallocations_necessary) {
    // If non-persistent memory was released, re-allocate it.
    if (memory_planner_ && !memory_planner_->HasNonPersistentMemory()) {
      memory_planner_->AcquireNonPersistentMemory();
    }
    // Check custom allocations, which may have been modified since last
    // AllocateTensors() call.
    if (!custom_allocations_.empty()) {
      for (const auto& idx_and_alloc : custom_allocations_) {
        const int idx = idx_and_alloc.first;
        TfLiteTensor* tensor_at_index = tensor(idx);
        TF_LITE_ENSURE_EQ(context(), tensor_at_index->allocation_type,
                          kTfLiteCustom);
        TF_LITE_ENSURE_STATUS(VerifyCustomAllocationForTensor(
            context(), custom_allocations_, idx));
      }
    }
    return kTfLiteOk;
  }

  next_execution_plan_index_to_prepare_ = 0;
  next_execution_plan_index_to_plan_allocation_ = 0;
  next_original_execution_plan_index_to_prepare_ = 0;
  if (memory_planner_) {
    TF_LITE_ENSURE_STATUS(memory_planner_->ResetAllocations());
  }

  TF_LITE_ENSURE_STATUS(PrepareOpsAndTensors());

  state_ = kStateInvokable;

  // Reset the variable tensors to zero after (re)allocating the tensors.
  // Developers shouldn't rely on the side effect of this function to reset
  // variable tensors. They should call `ResetVariableTensors` directly
  // instead.
  ResetVariableTensors();

  return kTfLiteOk;
}

// TODO(b/115961645): Support non-zero default values.
TfLiteStatus Subgraph::ResetVariableTensors() {
  for (auto& tensor : tensors_) {
    if (!tensor.is_variable) {
      continue;
    }

    if (tensor.allocation_type == kTfLiteArenaRwPersistent) {
      // If variable tensors allocation type is `kTfLiteArenaRwPersistent`, then
      // they must be allocated after the initial `PrepareOpsAndTensors()` is
      // called.
      TF_LITE_ENSURE(&context_, tensor.data.raw != nullptr);
      tflite::ResetVariableTensor(&tensor);
    } else {
      // If variable tensors allocation type is not `kTfLiteArenaRwPersistent`,
      // then it can only be `kTfLiteCustom` in which case, we do not reset it.
      TF_LITE_ENSURE_EQ(&context_, tensor.allocation_type, kTfLiteCustom);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::AddNodeWithParameters(
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const std::vector<int>& intermediates, const char* init_data,
    size_t init_data_size, void* builtin_data,
    const TfLiteRegistration* registration, int* node_index) {
  std::unique_ptr<void, decltype(free)*> builtin_data_deleter(builtin_data,
                                                              free);
  if (state_ == kStateInvokableAndImmutable) {
    ReportError("AddNodeWithParameters is disallowed when graph is immutable.");
    return kTfLiteError;
  }
  state_ = kStateUninvokable;

  TF_LITE_ENSURE_OK(&context_, CheckTensorIndices("node inputs", inputs.data(),
                                                  inputs.size()));
  TF_LITE_ENSURE_OK(
      &context_,
      CheckTensorIndices("node outputs", outputs.data(), outputs.size()));

  // For builtin ops, inputs and outputs must not overlap. Custom ops must do
  // this check by themselves if they don't support overlapping tensors. This
  // distinction is to allow custom ops to just forward a tensor, reusing it as
  // both input and output.
  if (builtin_data != nullptr) {
    TF_LITE_ENSURE_OK(&context_, CheckInputAndOutputForOverlap(
                                     inputs.data(), inputs.size(),
                                     outputs.data(), outputs.size()));
  }

  int new_node_index = nodes_and_registration_.size();
  if (node_index) *node_index = new_node_index;
  nodes_and_registration_.emplace_back();
  auto& node_and_reg = nodes_and_registration_.back();
  TfLiteNode& node = node_and_reg.first;

  // NOTE, here we are not using move semantics yet, since our internal
  // representation isn't std::vector, but in the future we would like to avoid
  // copies, so we want the interface to take r-value references now.
  node.inputs = ConvertVectorToTfLiteIntArray(inputs);
  node.outputs = ConvertVectorToTfLiteIntArray(outputs);
  node.intermediates = ConvertVectorToTfLiteIntArray(intermediates);
  node.temporaries = TfLiteIntArrayCreate(0);
  if (init_data) {
    node.user_data = OpInit(*registration, init_data, init_data_size);
  } else {
    node.user_data = OpInit(
        *registration, static_cast<const char*>(builtin_data_deleter.get()), 0);
  }

  node.builtin_data = builtin_data_deleter.release();

  if (registration->builtin_code == BuiltinOperator_CUSTOM) {
    // When it's a CUSTOM op, the `custom_options` field in the Flatbuffer
    // `Operator` table is passed in.
    node.custom_initial_data = init_data;
    node.custom_initial_data_size = init_data_size;
  } else {
    node.custom_initial_data = nullptr;
    node.custom_initial_data_size = 0;
  }
  node.might_have_side_effect = OpMightHaveSideEffect(&node, registration);

  node.delegate = nullptr;
  // Copying of registration is required to support unresolved custom ops.
  node_and_reg.second = *registration;
  execution_plan_.push_back(new_node_index);
  return kTfLiteOk;
}

namespace {
// Returns true if any tensor identified by indexes in 'tensor_indexes' is
// of type 'kTfLiteResource'. False otherwise.
bool AnyTensorOfTypeResource(const std::vector<TfLiteTensor>& tensors,
                             const TfLiteIntArray* tensor_indexes) {
  for (int i = 0; i < tensor_indexes->size; ++i) {
    int tensor_index = tensor_indexes->data[i];
    if (tensor_index >= 0 && tensor_index < tensors.size() &&
        tensors[tensor_index].type == kTfLiteResource)
      return true;
  }
  return false;
}

}  // namespace

bool Subgraph::OpMightHaveSideEffect(
    const TfLiteNode* node, const TfLiteRegistration* registration) const {
  // Check if any of the input tensors are of type resource.
  if (AnyTensorOfTypeResource(tensors_, node->inputs)) return true;
  // Check if any of the output tensors are of type resource.
  if (AnyTensorOfTypeResource(tensors_, node->outputs)) return true;
  // Consider control flow ops has side effect, some ops in the control flow
  // subgraph can have side effect.
  if (registration->builtin_code == kTfLiteBuiltinIf ||
      registration->builtin_code == kTfLiteBuiltinWhile ||
      registration->builtin_code == kTfLiteBuiltinCallOnce)
    return true;
  return false;
}

TfLiteStatus Subgraph::ResizeInputTensor(int tensor_index,
                                         const std::vector<int>& dims) {
  const bool delegates_applied = !pre_delegation_execution_plan_.empty();
  const bool graph_is_immutable = state_ == kStateInvokableAndImmutable;
  if (graph_is_immutable && !delegates_applied) {
    ReportError("ResizeInputTensor is disallowed when graph is immutable.");
    return kTfLiteError;
  }

  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);
  TfLiteTensor* tensor = &context_.tensors[tensor_index];

  // Short-circuit the state change if the dimensions don't change, avoiding
  // unnecessary (re)allocations.
  //
  // Note that it's required to check `tensor->data.raw != nullptr`. Otherwise
  // the subgraph won't allocate memory for a dynamic tensor when its size
  // is equal to the original tensor size.
  if (tensor->data.raw != nullptr &&
      EqualArrayAndTfLiteIntArray(tensor->dims, dims.size(), dims.data())) {
    return kTfLiteOk;
  }

  if (graph_is_immutable) {
    // Undo delegation if it resulted in the graph being immutable.
    TF_LITE_ENSURE_STATUS(UndoAllDelegates());
  }
  state_ = kStateUninvokable;
  return ResizeTensorImpl(tensor, ConvertVectorToTfLiteIntArray(dims));
}

TfLiteStatus Subgraph::ResizeInputTensorStrict(int tensor_index,
                                               const std::vector<int>& dims) {
  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);
  TfLiteTensor* tensor = &context_.tensors[tensor_index];

  // Ensure that only unknown dimensions can be resized.
  TF_LITE_ENSURE_EQ(&context_, tensor->dims->size, dims.size());
  for (size_t idx = 0; idx < dims.size(); idx++) {
    // `dims_signature` is not defined when no unknown dimensions are present.
    int dim_signature;
    if (tensor->dims_signature && tensor->dims_signature->size) {
      dim_signature = tensor->dims_signature->data[idx];
    } else {
      dim_signature = tensor->dims->data[idx];
    }

    if (dim_signature != -1 && dim_signature != dims[idx]) {
      ReportError(
          "Attempting to resize dimension %d of tensor %d with value %d to %d. "
          "ResizeInputTensorStrict only allows mutating unknown dimensions "
          "identified by -1.",
          idx, tensor_index, dim_signature, dims[idx]);
      return kTfLiteError;
    }
  }

  return ResizeInputTensor(tensor_index, dims);
}

TfLiteStatus Subgraph::ReleaseNonPersistentMemory() {
  if (memory_planner_) {
    TF_LITE_ENSURE_STATUS(memory_planner_->ReleaseNonPersistentMemory());
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::OpPrepare(const TfLiteRegistration& op_reg,
                                 TfLiteNode* node) {
  if (op_reg.prepare == nullptr) {
    // Check if it's an unresolved custom op.
    if (IsUnresolvedCustomOp(op_reg)) {
      if (IsFlexOp(op_reg.custom_name)) {
        ReportError(
            "Select TensorFlow op(s), included in the given model, is(are) not "
            "supported by this interpreter. Make sure you apply/link the Flex "
            "delegate before inference. For the Android, it can be resolved by "
            "adding \"org.tensorflow:tensorflow-lite-select-tf-ops\" "
            "dependency. See instructions: "
            "https://www.tensorflow.org/lite/guide/ops_select");
      } else {
        ReportError(
            "Encountered unresolved custom op: %s.\nSee instructions: "
            "https://www.tensorflow.org/lite/guide/ops_custom",
            op_reg.custom_name ? op_reg.custom_name : "UnknownOp");
      }
      return kTfLiteError;
    }
    // Resolved ops can have a null Prepare function.
    return kTfLiteOk;
  }
  return op_reg.prepare(&context_, node);
}

TfLiteStatus Subgraph::PrepareOpsStartingAt(
    int first_execution_plan_index, const std::vector<int>& execution_plan,
    int* last_execution_plan_index_prepared) {
  if (first_execution_plan_index == 0) {
    // Forwarding inputs without modification won't be not evaluated in the
    // operators. So, it needs to look up the subgraph's output tensors at the
    // beginning.
    has_dynamic_tensors_ =
        HasDynamicTensorImpl(context_, outputs(), &dynamic_tensor_index_);
  }
  for (int execution_plan_index = first_execution_plan_index;
       execution_plan_index < execution_plan.size(); execution_plan_index++) {
    int node_index = execution_plan[execution_plan_index];
    TfLiteNode& node = nodes_and_registration_[node_index].first;
    const TfLiteRegistration& registration =
        nodes_and_registration_[node_index].second;
    EnsureTensorsVectorCapacity();
    if (OpPrepare(registration, &node) != kTfLiteOk) {
      return ReportOpError(&context_, node, registration, node_index,
                           "failed to prepare");
    }

    *last_execution_plan_index_prepared = execution_plan_index;

    // Discontinue if the node has dynamic outputs. Note that we don't
    // stop for dynamic temporary tensors since they won't affect the
    // sizes of other tensors in the graph.
    if (HasDynamicTensor(context_, node.outputs, &dynamic_tensor_index_)) {
      has_dynamic_tensors_ = true;
      return kTfLiteOk;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::PrepareOpsAndTensors() {
  if (!memory_planner_) {
#ifdef TFLITE_USE_SIMPLE_MEMORY_PLANNER
    memory_planner_.reset(new SimplePlanner(&context_, CreateGraphInfo()));
#else
    memory_planner_.reset(new ArenaPlanner(&context_, CreateGraphInfo(),
                                           preserve_all_tensors_,
                                           kDefaultTensorAlignment));
#endif
    memory_planner_->PlanAllocations();
  }

  // Prepare original execution plan if any applied delegate wants it.
  // If any of the delegates is immutable, this won't be triggered
  // post-delegation (since we undo/redo delegation). For all other cases, other
  // delegates that do shape propagation themselves would still be able to.
  bool prepare_original_plan = false;
  if (!pre_delegation_execution_plan_.empty()) {
    for (int i = 0; i < delegates_applied_.size(); ++i) {
      if ((delegates_applied_[i]->flags &
           kTfLiteDelegateFlagsRequirePropagatedShapes)) {
        prepare_original_plan = true;
        break;
      }
    }
  }
  if (prepare_original_plan) {
    int last_original_exec_plan_index_prepared = 0;
    TF_LITE_ENSURE_STATUS(PrepareOpsStartingAt(
        next_execution_plan_index_to_prepare_, pre_delegation_execution_plan_,
        &last_original_exec_plan_index_prepared));
    next_original_execution_plan_index_to_prepare_ =
        last_original_exec_plan_index_prepared + 1;
  }

  int last_exec_plan_index_prepared = 0;
  TF_LITE_ENSURE_STATUS(
      PrepareOpsStartingAt(next_execution_plan_index_to_prepare_,
                           execution_plan_, &last_exec_plan_index_prepared));
  next_execution_plan_index_to_prepare_ = last_exec_plan_index_prepared + 1;

  // Execute arena allocations.
  TF_LITE_ENSURE_STATUS(memory_planner_->ExecuteAllocations(
      next_execution_plan_index_to_plan_allocation_,
      last_exec_plan_index_prepared));

  if (!custom_allocations_.empty()) {
    // Verify custom allocations for output tensors from the ops that have just
    // been prepared. Other output tensors might be resized later.
    if (!nodes_and_registration_.empty()) {
      for (int node_idx = next_execution_plan_index_to_plan_allocation_;
           node_idx <= last_exec_plan_index_prepared; ++node_idx) {
        TfLiteNode& node = nodes_and_registration_[node_idx].first;
        for (int i = 0; i < node.outputs->size; ++i) {
          const int output_tensor_idx = node.outputs->data[i];
          if (output_tensor_idx == kTfLiteOptionalTensor) continue;
          TF_LITE_ENSURE_STATUS(VerifyCustomAllocationForTensor(
              context(), custom_allocations_, output_tensor_idx));
        }
      }
    }
    // Check input custom allocs only if we just prepared nodes from the idx 0.
    if (next_execution_plan_index_to_plan_allocation_ == 0) {
      for (const int input_tensor_idx : inputs_) {
        if (input_tensor_idx == kTfLiteOptionalTensor) continue;
        TF_LITE_ENSURE_STATUS(VerifyCustomAllocationForTensor(
            context(), custom_allocations_, input_tensor_idx));
      }
    }
  }

  next_execution_plan_index_to_plan_allocation_ =
      last_exec_plan_index_prepared + 1;

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
  } else if (memory_planner_ && !memory_planner_->HasNonPersistentMemory()) {
    ReportError("Non-persistent memory is not available.");
    return kTfLiteError;
  }
  TFLITE_SCOPED_TAGGED_DEFAULT_PROFILE(profiler_.get(), "Invoke");

  // Invocations are always done in node order.
  // Note that calling Invoke repeatedly will cause the original memory plan to
  // be reused, unless either ResizeInputTensor() or AllocateTensors() has been
  // called.
  for (int execution_plan_index = 0;
       execution_plan_index < execution_plan_.size(); execution_plan_index++) {
    if (execution_plan_index == next_execution_plan_index_to_prepare_) {
      TF_LITE_ENSURE_STATUS(PrepareOpsAndTensors());
      TF_LITE_ENSURE(&context_, next_execution_plan_index_to_prepare_ >=
                                    execution_plan_index);
    }
    int node_index = execution_plan_[execution_plan_index];
    TfLiteNode& node = nodes_and_registration_[node_index].first;
    const TfLiteRegistration& registration =
        nodes_and_registration_[node_index].second;

    const char* op_name = nullptr;
    if (profiler_) op_name = GetTFLiteOpName(registration);
    TFLITE_SCOPED_TAGGED_OPERATOR_PROFILE(profiler_.get(), op_name, node_index);

    for (int i = 0; i < node.inputs->size; ++i) {
      int tensor_index = node.inputs->data[i];
      if (tensor_index == kTfLiteOptionalTensor) {
        continue;
      }
      TfLiteTensor* tensor = &tensors_[tensor_index];
      if (tensor->delegate && tensor->delegate != node.delegate &&
          tensor->data_is_stale) {
        TF_LITE_ENSURE_STATUS(EnsureTensorDataIsReadable(tensor_index));
      }
      if (tensor->data.raw == nullptr && tensor->bytes > 0) {
        if (registration.builtin_code == kTfLiteBuiltinReshape && i == 1 &&
            tensor->dims->size != 1) {
          // In general, having a tensor here with no buffer will be an error.
          // However, for the reshape operator, the second input tensor is
          // sometimes only used for the shape, not for the data. Thus, null
          // buffer is ok in this situation.
          // The situation where null buffer is not ok for reshape operator is
          // only when there are 2 inputs given to the node and the one
          // corresponding to the shape (i == 1) is a vector that contains all
          // dimensions. See `GetOutputShape()` function in
          // `tensorflow/lite/kernels/reshape.cc`
          continue;
        } else {
          // In all other cases, we need to return an error as otherwise we will
          // trigger a null pointer dereference (likely).
          ReportError("Input tensor %d lacks data", tensor_index);
          return kTfLiteError;
        }
      }
    }

    if (check_cancelled_func_ != nullptr &&
        check_cancelled_func_(cancellation_data_)) {
      ReportError("Client requested cancel during Invoke()");
      return kTfLiteError;
    }

    EnsureTensorsVectorCapacity();
    tensor_resized_since_op_invoke_ = false;

    // std::string mname = "mobile_bert";
    // std::string mname = "efficientnet_lite";
    // std::string mw = "/weights";
    // std::string mi = "/inputs";
    // std::string mo = "/outputs";

    // std::string mname = "dense_v8";
    // std::string mw = "";
    // std::string mi = "";
    // std::string mo = "";

    // std::string mname = "res50";
    // std::string mw = "/weights";
    // std::string mi = "/inputs";
    // std::string mo = "/outputs";

    // std::string mname = "efficientnet_lite";
    // std::string mw = "/weights";
    // std::string mi = "/inputs";
    // std::string mo = "/outputs";

    // using namespace std;
    // cerr << node_index << "," << op_name << endl;

    // if (registration.builtin_code == 9) {
    //   // using namespace std;
    //   // cout << op_name << ": " << node_index << endl;
    //   const TfLiteTensor* input;
    //   const TfLiteTensor* filter;
    //   TF_LITE_ENSURE_OK(&context_, GetInputSafe(&context_, &node, 0,
    //   &input)); TF_LITE_ENSURE_OK(&context_, GetInputSafe(&context_, &node,
    //   1, &filter));
    //   {
    //     cout << node_index << endl;
    //     if (node_index == 58) {
    //       int k = 0;
    //     }
    //     int rows = filter->dims->data[0];
    //     int cols = filter->dims->data[1];
    //     ofstream myfile;
    //     myfile.open("a_Bert/" + mname + mw + "/wgt_" +
    //                 std::to_string(node_index) + ".csv");
    //     const int8_t* res_pointer = filter->data.int8;
    //     int index = 0;
    //     for (int c = 0; c < rows; c++) {
    //       myfile << endl;
    //       for (int r = 0; r < cols; r++) {
    //         myfile << (int)res_pointer[index] << ",";
    //         index++;
    //       }
    //     }
    //     myfile.close();
    //   }
    //   {
    //     int dof = 0;
    //     if (input->dims->data[0] == 1 && input->dims->size == 3) dof++;
    //     int rows = input->dims->data[dof + 0];
    //     int cols = input->dims->data[dof + 1];
    //     ofstream myfile;
    //     myfile.open("a_Bert/" + mname + mi + "/inp_" +
    //                 std::to_string(node_index) + ".csv");
    //     const int8_t* res_pointer = input->data.int8;
    //     int index = 0;
    //     for (int c = 0; c < rows; c++) {
    //       myfile << endl;
    //       for (int r = 0; r < cols; r++) {
    //         myfile << (int)res_pointer[index] << ",";
    //         index++;
    //       }
    //     }
    //     myfile.close();
    //   }
    // }

    if (OpInvoke(registration, &node) != kTfLiteOk) {
      return ReportOpError(&context_, node, registration, node_index,
                           "failed to invoke");
    }

    // if (registration.builtin_code == 3) {
    //   using namespace std;
    //   TfLiteTensor* output;
    //   TF_LITE_ENSURE_OK(&context_, GetOutputSafe(&context_, &node, 0,
    //   &output));
    //   {
    //     int rows = output->dims->data[3];
    //     int cols = output->dims->data[1] * output->dims->data[2] *
    //                output->dims->data[0];
    //     ofstream myfile;
    //     // myfile.open("aData/" + mname + mo + "/out_" +
    //     //             std::to_string(node_index) + ".csv");
    //     myfile.open("aData/conv/" + std::to_string(node_index) +
    //                 "_del_out_cpu.csv");
    //     int8_t* res_pointer = output->data.int8;
    //     int index = 0;
    //     for (int c = 0; c < cols; c++) {
    //       myfile << endl;
    //       for (int r = 0; r < rows; r++) {
    //         myfile << (int)res_pointer[index] << ",";
    //         index++;
    //       }
    //     }
    //     myfile.close();
    //   }
    // }

    // if (registration.builtin_code == 9 && node_index==2026) {
    //   using namespace std;
    //   TfLiteTensor* output;
    //   TF_LITE_ENSURE_OK(&context_, GetOutputSafe(&context_, &node, 0,
    //   &output));
    //   {
    //      int dof = 0;
    //     if (output->dims->data[0] == 1 && output->dims->size == 3) dof++;
    //     int rows = output->dims->data[dof + 1];
    //     int cols = output->dims->data[dof + 0];
    //     // int rows = output->dims->data[1];
    //     // int cols = output->dims->data[0];
    //     ofstream myfile;
    //     myfile.open("a_Bert/" + mname + mo + "/out_" +
    //                 std::to_string(node_index) + ".csv");
    //     int8_t* res_pointer = output->data.int8;
    //     int index = 0;
    //     for (int c = 0; c < cols; c++) {
    //       myfile << endl;
    //       for (int r = 0; r < rows; r++) {
    //         myfile << (int)res_pointer[index] << ",";
    //         index++;
    //       }
    //     }
    //     myfile.close();
    //   }
    // }

    // Force execution prep for downstream ops if the latest op triggered the
    // resize of a dynamic tensor.
    if (tensor_resized_since_op_invoke_ &&
        HasDynamicTensor(context_, node.outputs, nullptr)) {
      next_execution_plan_index_to_prepare_ = execution_plan_index + 1;

      // This happens when an intermediate dynamic tensor is resized.
      // We don't have to prepare all the ops, but we need to recompute
      // the allocation plan.
      if (next_execution_plan_index_to_plan_allocation_ >
          next_execution_plan_index_to_prepare_) {
        next_execution_plan_index_to_plan_allocation_ =
            next_execution_plan_index_to_prepare_;
        if (memory_planner_) {
          TF_LITE_ENSURE_STATUS(memory_planner_->ResetAllocationsAfter(
              next_execution_plan_index_to_plan_allocation_ - 1));
        }
      }
    }
  }

  return status;
}

TfLiteStatus Subgraph::ResizeTensor(TfLiteContext* context,
                                    TfLiteTensor* tensor,
                                    TfLiteIntArray* new_size) {
  // If the dimensions don't change, avoiding
  // unnecessary (re)allocations.
  //
  // Note that it's required to check `tensor->data.raw != nullptr`. Otherwise
  // the subgraph won't allocate memory for a dynamic tensor when its size
  // is equal to the original tensor size.
  if (tensor->data.raw != nullptr &&
      EqualArrayAndTfLiteIntArray(tensor->dims, new_size->size,
                                  new_size->data)) {
    // A number of clients assume |new_size| remains valid upon success, so
    // swap it in as the new (but logically identical) tensor dims.
    TfLiteIntArrayFree(tensor->dims);
    tensor->dims = new_size;
    return kTfLiteOk;
  }

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
  auto* f = static_cast<Subgraph*>(context_.impl_);
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
  context_.tensors = tensors_.data();
  context_.tensors_size = tensors_.size();
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
  TF_LITE_ENSURE(&context_, node_index >= 0);
  auto nodes_size = nodes_and_registration_.size();
  TF_LITE_ENSURE(&context_, static_cast<size_t>(node_index) < nodes_size);
  TF_LITE_ENSURE(&context_, node != nullptr && registration != nullptr);
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
    const int* dims, TfLiteQuantization quantization, const char* buffer,
    size_t bytes, const Allocation* allocation, TfLiteSparsity* sparsity) {
  // Ensure quantization cleanup on failure.
  ScopedTfLiteQuantization scoped_quantization(&quantization);
  ScopedTfLiteSparsity scoped_sparsity(sparsity);
  if (state_ == kStateInvokableAndImmutable) {
    ReportError(
        "SetTensorParametersReadOnly is disallowed when graph is immutable.");
    return kTfLiteError;
  }

  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);

  // For most tensors we know exactly how much memory is necessary so we can
  // ensure the buffer is large enough. However, we need to skip string tensors
  // and sparse tensors because their sizes change with the contents.
  // TODO(b/145615516): Extend BytesRequired to check sparse tensors.
  if (type != kTfLiteString && type != kTfLiteResource &&
      type != kTfLiteVariant && sparsity == nullptr) {
    size_t required_bytes;
    TF_LITE_ENSURE_OK(&context_,
                      BytesRequired(type, dims, rank, &required_bytes));
    TF_LITE_ENSURE_EQ(&context_, required_bytes, bytes);
  }

  TfLiteTensor& tensor = context_.tensors[tensor_index];
  if (type == tensor.type &&
      EqualArrayAndTfLiteIntArray(tensor.dims, rank, dims)) {
    // Fast path which does not invalidate the invokable property.
    TfLiteTensorDataFree(&tensor);
    TfLiteQuantizationFree(&tensor.quantization);
    tensor.data.raw = const_cast<char*>(buffer);
    if (!tensor.dims) tensor.dims = ConvertArrayToTfLiteIntArray(rank, dims);
    tensor.params = GetLegacyQuantization(quantization);
    tensor.quantization = *scoped_quantization.release();
    tensor.sparsity = scoped_sparsity.release();
    tensor.allocation_type = kTfLiteMmapRo;
    tensor.allocation = allocation;
  } else {
    state_ = kStateUninvokable;
    TfLiteTensorReset(type, name, ConvertArrayToTfLiteIntArray(rank, dims),
                      GetLegacyQuantization(quantization),
                      const_cast<char*>(buffer), bytes, kTfLiteMmapRo,
                      allocation, false, &tensor);
    tensor.quantization = *scoped_quantization.release();
    tensor.sparsity = scoped_sparsity.release();
  }
  return kTfLiteOk;
}

// Set description of inputs/outputs/data/fptrs for node `node_index`.
// This variant assumes an external buffer has been allocated of size
// bytes. The lifetime of buffer must be ensured to be greater or equal
// to Interpreter.
TfLiteStatus Subgraph::SetTensorParametersReadWrite(
    int tensor_index, TfLiteType type, const char* name, const size_t rank,
    const int* dims, TfLiteQuantization quantization, bool is_variable,
    const size_t rank_dims_signature, const int* dims_signature) {
  // Ensure quantization cleanup on failure.
  ScopedTfLiteQuantization scoped_quantization(&quantization);
  if (state_ == kStateInvokableAndImmutable) {
    ReportError(
        "SetTensorParametersReadWrite is disallowed when graph is immutable.");
    return kTfLiteError;
  }
  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);
  size_t required_bytes = 0;
  if (type != kTfLiteString && type != kTfLiteResource &&
      type != kTfLiteVariant) {
    // These types will be allocated in our arena so we need to record how
    // many bytes we will need based on the dimensions. String tensors are
    // allocated dynamically and we can't know ahead of time how much space
    // they will require.
    TF_LITE_ENSURE_OK(&context_,
                      BytesRequired(type, dims, rank, &required_bytes));
  }

  TfLiteAllocationType allocation_type = kTfLiteArenaRw;
  if (type == kTfLiteString || type == kTfLiteResource ||
      type == kTfLiteVariant) {
    if (is_variable) {
      // We don't have a real use case for string variable tensor.
      ReportError("String variable tensor isn't supported.");
      return kTfLiteError;
    }
    allocation_type = kTfLiteDynamic;
  } else if (is_variable) {
    allocation_type = kTfLiteArenaRwPersistent;
  }

  TfLiteTensor& tensor = context_.tensors[tensor_index];
  TfLiteTensorReset(type, name, ConvertArrayToTfLiteIntArray(rank, dims),
                    GetLegacyQuantization(quantization),
                    /*buffer=*/nullptr, required_bytes, allocation_type,
                    nullptr, is_variable, &tensor);
  tensor.quantization = *scoped_quantization.release();
  tensor.dims_signature =
      ConvertArrayToTfLiteIntArray(rank_dims_signature, dims_signature);
  return kTfLiteOk;
}

TfLiteStatus Subgraph::SetExecutionPlan(const std::vector<int>& new_plan) {
  for (int node_index : new_plan) {
    TF_LITE_ENSURE(&context_, node_index >= 0 &&
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
      tensor->allocation_type == kTfLiteArenaRwPersistent ||
      tensor->allocation_type == kTfLitePersistentRo ||
      tensor->allocation_type == kTfLiteCustom) {
    tensor_resized_since_op_invoke_ |=
        TfLiteIntArrayEqual(tensor->dims, new_size) == 0;
    if (tensor->type != kTfLiteString && tensor->type != kTfLiteResource &&
        tensor->type != kTfLiteVariant) {
      size_t bytesRequired;
      TfLiteStatus status = BytesRequired(tensor->type, new_size->data,
                                          new_size->size, &bytesRequired);
      if (status != kTfLiteOk) {
        TfLiteIntArrayFree(new_size);
        return kTfLiteError;
      }

      // Realloc space for heap-allocated tensors.
      TfLiteTensorRealloc(bytesRequired, tensor);
      tensor->bytes = bytesRequired;
    }
    if (tensor->dims) TfLiteIntArrayFree(tensor->dims);
    tensor->dims = new_size;

    // Reset arena-allocated tensors; they will be allocated later.
    if (tensor->allocation_type == kTfLiteArenaRw ||
        tensor->allocation_type == kTfLiteArenaRwPersistent) {
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

void Subgraph::SwitchToDelegateContext() {
  context_.GetNodeAndRegistration = GetNodeAndRegistration;
  context_.ReplaceNodeSubsetsWithDelegateKernels =
      ReplaceNodeSubsetsWithDelegateKernels;
  context_.GetExecutionPlan = GetExecutionPlan;
  context_.PreviewDelegatePartitioning = PreviewDelegatePartitioning;
}

void Subgraph::SwitchToKernelContext() {
  context_.GetNodeAndRegistration = [](struct TfLiteContext* context,
                                       int node_index, TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    return ForbiddenContextFunction(context);
  };
  context_.ReplaceNodeSubsetsWithDelegateKernels =
      [](TfLiteContext* context, TfLiteRegistration registration,
         const TfLiteIntArray* nodes_to_replace, TfLiteDelegate* delegate) {
        return ForbiddenContextFunction(context);
      };
  context_.GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray**) {
    return ForbiddenContextFunction(context);
  };
  context_.PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array,
         int* num_partitions) { return ForbiddenContextFunction(context); };
  // Free any memory that might have been allocated by
  // PreviewDelegatePartitioning.
  FreeDelegatePartitioningData();
}

TfLiteStatus Subgraph::UndoAllDelegates() {
  // Return early if there is nothing to reset to.
  if (pre_delegation_execution_plan_.empty()) return kTfLiteOk;

  // First free all delegate nodes.
  for (int execution_plan_index = 0;
       execution_plan_index < execution_plan_.size(); ++execution_plan_index) {
    int node_index = execution_plan_[execution_plan_index];
    TfLiteNode& node = nodes_and_registration_[node_index].first;
    if (node.delegate == nullptr) {
      continue;
    }
    CleanupNode(node_index);
  }

  // Reset execution plan.
  execution_plan_ = pre_delegation_execution_plan_;
  pre_delegation_execution_plan_.clear();

  // Handling FP16 delegation (if applies).
  //
  // First pass through execution plan to remember mapping of FP16
  // dequantizations in the graph.
  // This is required because delegates that support FP16 could remap supported
  // nodes' inputs to point to their fp16 versions (if delegate supports fp16
  // acceleration). This remapping is performed in FP16GraphPartitionHelper in
  // delegates/utils. We need to undo this remapping to ensure CPU kernels work.
  std::vector<int> fp16_to_fp32(tensors_size(), -1);
  for (int execution_plan_index = 0;
       execution_plan_index < execution_plan_.size(); ++execution_plan_index) {
    int node_index = execution_plan_[execution_plan_index];
    auto& node_and_reg = nodes_and_registration_[node_index];
    const TfLiteNode& node = node_and_reg.first;
    const TfLiteRegistration& reg = node_and_reg.second;
    if (reg.builtin_code == kTfLiteBuiltinDequantize &&
        node.inputs->size == 1 && node.outputs->size == 1) {
      const int input_idx = node.inputs->data[0];
      if (tensors_[input_idx].type == kTfLiteFloat16) {
        fp16_to_fp32[input_idx] = node.outputs->data[0];
      }
    }
  }
  // Second pass through the execution plan to remap applicable nodes' fp16
  // inputs to their original fp32 versions. Note that if a CPU kernel does
  // support fp16, the model will not contain a DEQUANTIZE for its constant
  // input.
  for (int execution_plan_index = 0;
       execution_plan_index < execution_plan_.size(); ++execution_plan_index) {
    int node_index = execution_plan_[execution_plan_index];
    auto& node_and_reg = nodes_and_registration_[node_index];
    const TfLiteNode& node = node_and_reg.first;
    const TfLiteRegistration& reg = node_and_reg.second;
    if (reg.builtin_code == kTfLiteBuiltinDequantize) continue;
    for (int i = 0; i < node.inputs->size; ++i) {
      const int original_input_idx = node.inputs->data[i];
      if (original_input_idx == kTfLiteOptionalTensor) continue;
      if (tensors_[original_input_idx].type == kTfLiteFloat16) {
        node.inputs->data[i] = fp16_to_fp32[original_input_idx];
      }
    }
  }

  // Delegate nodes are appended to nodes_and_registration_. Therefore,
  // cleanup nodes_and_registration_ to only contain nodes from
  // pre_delegation_execution_plan_.
  int max_retained_node_index = 0;
  for (int execution_plan_index = 0;
       execution_plan_index < execution_plan_.size(); ++execution_plan_index) {
    max_retained_node_index = std::max(max_retained_node_index,
                                       execution_plan_[execution_plan_index]);
  }
  nodes_and_registration_.resize(max_retained_node_index + 1);
  // After undoing delegates, the graph is uninvokable, but mutable.
  state_ = kStateUninvokable;

  delegates_undone_ = true;
  return kTfLiteOk;
}

TfLiteStatus Subgraph::RedoAllDelegates() {
  if (!delegates_undone_) return kTfLiteOk;

  delegates_undone_ = false;
  std::vector<TfLiteDelegate*> delegates_to_apply;
  delegates_applied_.swap(delegates_to_apply);
  for (auto* delegate : delegates_to_apply) {
    TF_LITE_ENSURE_STATUS(ModifyGraphWithDelegate(delegate));
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::RemoveAllDelegates() {
  TF_LITE_ENSURE_STATUS(UndoAllDelegates());
  delegates_applied_.clear();
  delegates_undone_ = false;
  TF_LITE_ENSURE_STATUS(EnsureMemoryAllocations());
  return kTfLiteOk;
}

bool Subgraph::HasDelegates() { return !delegates_applied_.empty(); }

bool Subgraph::IsFullyDelegated() const {
  for (const int nid : execution_plan_) {
    const TfLiteNode& node = nodes_and_registration_[nid].first;
    if (node.delegate == nullptr) return false;
  }
  return true;
}

void Subgraph::EnsureTensorsVectorCapacity() {
  const size_t required_capacity = tensors_.size() + kTensorsCapacityHeadroom;
  if (required_capacity > tensors_.capacity()) {
    // Whenever it's required to increase the vector capacity, make it at
    // least twice bigger. The behavior is consistent with the default
    // behavior of GCC STL's `std::vector::resize()`. This avoids frequently
    // allocating and copying the underlying buffer.
    size_t reserved_capacity =
        std::max(required_capacity, tensors_.capacity() * 2);
    tensors_.reserve(reserved_capacity);
    context_.tensors = tensors_.data();
  }
}

TfLiteStatus Subgraph::EnsureMemoryAllocations() {
  if (memory_planner_) {
    state_ = kStateUninvokable;
    TF_LITE_ENSURE_OK(&context_, memory_planner_->PlanAllocations());
  }
  TF_LITE_ENSURE_OK(&context_, AllocateTensors());
  TF_LITE_ENSURE_EQ(&context_, state_, kStateInvokable);
  return kTfLiteOk;
}

TfLiteStatus Subgraph::ModifyGraphWithDelegate(TfLiteDelegate* delegate) {
  TFLITE_SCOPED_TAGGED_DEFAULT_PROFILE(profiler_.get(),
                                       "ModifyGraphWithDelegate");

  if (delegate == nullptr) {
    ReportError("Null delegate.");
    return kTfLiteDelegateError;
  }

  // Resets delegation & leaves graph in consistent state if delegate status is
  // not okay.
  auto reset_delegation_if_not_ok = [this](TfLiteStatus status) {
    if (status != kTfLiteOk) {
      TF_LITE_ENSURE_STATUS(RemoveAllDelegates());
      ReportError(
          "Restored original execution plan after delegate application "
          "failure.");
      return kTfLiteDelegateError;
    }
    return kTfLiteOk;
  };

  // STEP 1: Verify & prepare graph for delegation.
  // ==============================================

  // Restore delegation state if applicable.
  TF_LITE_ENSURE_STATUS(RedoAllDelegates());

  const bool delegate_supports_dynamic_shapes =
      delegate->flags & kTfLiteDelegateFlagsAllowDynamicTensors;
  const auto pre_delegation_state = state_;

  if (state_ == kStateInvokableAndImmutable) {
    // A delegate that doesn't support dynamic shapes was already applied, so
    // we can assume tensor shapes have been propagated & there are no dynamic
    // tensors.
    // Reset the state to force tensor/op reallocation.
    state_ = kStateUninvokable;
  } else if (!delegate_supports_dynamic_shapes) {
    // Check if graph has dynamic tensors by preparing ops.
    int last_execution_plan_index_prepared;
    TF_LITE_ENSURE_STATUS(PrepareOpsStartingAt(
        0, execution_plan_, &last_execution_plan_index_prepared));
    if (has_dynamic_tensors_) {
      TF_LITE_ENSURE_STATUS(EnsureMemoryAllocations());
      ReportError(
          "Attempting to use a delegate that only supports static-sized "
          "tensors with a graph that has dynamic-sized tensors (tensor#%d is a "
          "dynamic-sized tensor).",
          dynamic_tensor_index_);
      return kTfLiteApplicationError;
    }
  }

  if (delegates_applied_.empty()) {
    // This is the first delegate being applied, so remember original execution
    // plan.
    pre_delegation_execution_plan_ = execution_plan_;
  }

  // STEP 2: Delegate replaces applicable nodes with delegate kernels.
  // =================================================================

  // Setup additional context interface.
  SwitchToDelegateContext();
  TfLiteStatus status = delegate->Prepare(&context_, delegate);
  // Remove additional context info.
  SwitchToKernelContext();
  TF_LITE_ENSURE_STATUS(reset_delegation_if_not_ok(status));

  // STEP 3: Leave graph in consistent state based on delegate & previous state.
  // ===========================================================================

  if (!delegate_supports_dynamic_shapes) {
    // CASE 1: Current delegate does not support dynamic shapes.
    // Reset the state to force tensor/op reallocation.
    state_ = kStateUninvokable;
    TF_LITE_ENSURE_STATUS(
        reset_delegation_if_not_ok(EnsureMemoryAllocations()));
    // After using a delegate which doesn't support dynamic tensors, make the
    // entire graph immutable.
    state_ = kStateInvokableAndImmutable;
  } else if (pre_delegation_state == kStateInvokableAndImmutable) {
    // CASE 2: Current delegate supports dynamic shapes, but a previous one
    // does not.
    // Make sure new delegate didn't mark a tensor as dynamic.
    int last_execution_plan_index_prepared;
    TF_LITE_ENSURE_STATUS(reset_delegation_if_not_ok(PrepareOpsStartingAt(
        0, execution_plan_, &last_execution_plan_index_prepared)));
    if (has_dynamic_tensors_) {
      TF_LITE_ENSURE_STATUS(RemoveAllDelegates());
      ReportError(
          "Cannot allow dynamic tensors due to previous delegation, resetting "
          "to original execution plan.");
      return kTfLiteApplicationError;
    }
    // Redo memory allocations & ensure state is set back to original value.
    TF_LITE_ENSURE_STATUS(
        reset_delegation_if_not_ok(EnsureMemoryAllocations()));
    state_ = kStateInvokableAndImmutable;
  } else if (pre_delegation_state == kStateInvokable) {
    // CASE 3: Current delegate supports dynamic shapes, and the graph was
    // previously invokable.
    // Flush allocation now to leave it in a consistent state.
    TF_LITE_ENSURE_STATUS(
        reset_delegation_if_not_ok(EnsureMemoryAllocations()));
  }
  delegates_applied_.push_back(delegate);

  return status;
}

TfLiteStatus Subgraph::SetCustomAllocationForTensor(
    int tensor_index, const TfLiteCustomAllocation& allocation, int64_t flags) {
  TfLiteTensor* tensor = &context_.tensors[tensor_index];
  TF_LITE_ENSURE(context(),
                 (tensor->allocation_type == kTfLiteArenaRw ||
                  tensor->allocation_type == kTfLiteArenaRwPersistent ||
                  tensor->allocation_type == kTfLiteCustom));
  // Don't check allocation.bytes here, we do that after all ops are prepared
  // to allow tensor shape propagation.
  TF_LITE_ENSURE(context(), allocation.data != nullptr);
  if (!(flags & kTfLiteCustomAllocationFlagsSkipAlignCheck)) {
    const intptr_t data_ptr_value = reinterpret_cast<intptr_t>(allocation.data);
    TF_LITE_ENSURE(context(), data_ptr_value % kDefaultTensorAlignment == 0);
  }

  const auto iter_and_success =
      custom_allocations_.insert({tensor_index, allocation});
  if (!iter_and_success.second) {
    iter_and_success.first->second = allocation;
  }

  tensor->allocation_type = kTfLiteCustom;
  tensor->data.data = allocation.data;

  return kTfLiteOk;
}

void Subgraph::SetName(const char* name) {
  if (name) {
    name_ = name;
  } else {
    name_ = "";
  }
}

const std::string& Subgraph::GetName() const { return name_; }

void Subgraph::DumpMemoryPlannerDebugInfo() const {
  if (memory_planner_ == nullptr) return;
  memory_planner_->DumpDebugInfo(execution_plan());
}

TfLiteStatus Subgraph::PreserveAllTensorsExperimental() {
  if (memory_planner_) {
    ReportError(
        "PreserveAllTensorsExperimental called after memory was planned. ");
    return kTfLiteError;
  }
  preserve_all_tensors_ = true;
  return kTfLiteOk;
}

std::unique_ptr<GraphInfo> Subgraph::CreateGraphInfo() {
  return std::unique_ptr<GraphInfo>(new InterpreterInfo(this));
}

}  // namespace tflite
