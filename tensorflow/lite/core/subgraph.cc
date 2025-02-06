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

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common_internal.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/experimental/resource/initialization_status.h"
#include "tensorflow/lite/experimental/resource/resource_base.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/profiling/telemetry/telemetry.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"
#ifdef TFLITE_USE_SIMPLE_MEMORY_PLANNER
#include "tensorflow/lite/simple_planner.h"
#else
#include "tensorflow/lite/arena_planner.h"
#endif
#ifdef TF_LITE_TENSORFLOW_PROFILER
#include "tensorflow/lite/tensorflow_profiler_logger.h"
#endif  // TF_LITE_TENSORFLOW_PROFILER

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
  TF_LITE_KERNEL_LOG(context, "Node number %d (%s) %s.", node_index,
                     registration.custom_name
                         ? registration.custom_name
                         : EnumNameBuiltinOperator(static_cast<BuiltinOperator>(
                               registration.builtin_code)),
                     message);
  return kTfLiteError;
}

// Stub method which returns kTfLiteError when the function is forbidden.
// We're registering this function to several different function to save
// compiled binary size. Please note the restrictions:
// * The type of first parameter have to be `TfLiteContext*`.
// * All parameters must be trivially destructible. (E.g. No C++ class)
TfLiteStatus ForbiddenContextFunction(TfLiteContext* context, ...) {
  TF_LITE_KERNEL_LOG(context,
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

  TfLiteTensor* tensors() override { return subgraph_->tensors(); }

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

  const TfLiteRegistration& registration(size_t index) const override {
    const int node_index = subgraph_->execution_plan()[index];
    return subgraph_->nodes_and_registration()[node_index].second;
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
                   resource::InitializationStatusMap* initialization_status_map,
                   int subgraph_index)
    : external_contexts_(external_contexts),
      registration_externals_(new internal::OperatorsCache),
      error_reporter_(error_reporter),
      next_execution_plan_index_to_prepare_(0),
      next_execution_plan_index_to_plan_allocation_(0),
      subgraphs_(subgraphs),
      subgraph_index_(subgraph_index),
      resources_(resources),
      resource_ids_(resource_ids),
      initialization_status_map_(initialization_status_map),
      options_(nullptr) {
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
    if (tensor->buffer_handle != kTfLiteNullBufferHandle) {
      TfLiteDelegateFreeBufferHandleInternal(&context_, tensor->delegate,
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

// This function template allocates a continuous memory space that contains a
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
//
// Note that the 'delegate' field has to be set by the caller of this function
// template.
//
// This function can also be used with TfLiteOpaqueDelegateParams as a template
// parameter instead of TfLiteDelegateParams, in which case the layout looks
// as follows:
//
// +----------------------------------------------+
// | TfLiteOpaqueDelegateParams                   |
// | struct TfLiteOpaqueDelegate* delegate;       |
// | void* delegate_data;                         |
// | TfLiteIntArray* nodes_to_replace;            |--\
// | TfLiteIntArray* input_tensors;               |--+--\
// | TfLiteIntArray* output_tensors;              |--+--+--\
// +----------------------------------------------+  |  |  |
// | TfLiteIntArray (variable size)               |<-/  |  |
// +----------------------------------------------+     |  |
// | TfLiteIntArray (variable size)               |<----/  |
// +----------------------------------------------+        |
// | TfLiteIntArray (variable size)               |<-------/
// +----------------------------------------------+
//
// Note that the 'delegate' and delegate_data field has to be set by the caller
// of this function template.
template <typename Params>
Params* CreateDelegateParamsImpl(TfLiteDelegate* delegate,
                                 const NodeSubset& node_subset) {
  // Step 1: Calculate the allocation size.
  int allocation_size = sizeof(Params);

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
  Params* params = reinterpret_cast<Params*>(allocation);
  // Callers are expected to fill any fields that sit before the
  // 'nodes_to_replace' field.
  allocation += sizeof(Params);

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

TfLiteDelegateParams* CreateDelegateParams(TfLiteDelegate* delegate,
                                           const NodeSubset& node_subset) {
  TfLiteDelegateParams* params =
      CreateDelegateParamsImpl<TfLiteDelegateParams>(delegate, node_subset);
  params->delegate = delegate;
  return params;
}

TfLiteOpaqueDelegateParams* CreateOpaqueDelegateParams(
    TfLiteDelegate* delegate, const NodeSubset& node_subset) {
  TfLiteOpaqueDelegateParams* params =
      CreateDelegateParamsImpl<TfLiteOpaqueDelegateParams>(delegate,
                                                           node_subset);
  // The following cast is safe only because this code is part of the
  // TF Lite runtime implementation.  Apps using TF Lite should not rely on
  // TfLiteOpaqueDelegate and TfLiteDelegate being equivalent.
  params->delegate = reinterpret_cast<TfLiteOpaqueDelegate*>(delegate);
  params->delegate_data = delegate->opaque_delegate_builder->data;
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

// Returns the 'custom_name' associated with the provided 'registration', or
// "unknown" if the registration does not have a custom name.
//
// Note that 'TfLiteRegistration' has a top-level 'custom_name' field and also
// a nested 'custom_name' field defined inside the optionally set
// 'registration_external' structure.  The top-level field takes precedence over
// the nested field.  'TfLiteRegistration'
// objects can optionally carry a 'TfLiteOperator' pointer in their
// 'registration_external' field.  If that's the case then the
// 'TfLiteRegistration' object is merely a wrapper over a
// 'TfLiteOperator', with all fields except 'registration_external'
// being null, that contains the actual logic that the registration represents.
// See also the comment inside
// 'TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels'.
const char* GetDelegateKernalName(const TfLiteRegistration& registration) {
  if (registration.custom_name) {
    return registration.custom_name;
  }

  if (registration.registration_external &&
      registration.registration_external->custom_name) {
    return registration.registration_external->custom_name;
  }

  return "unknown";
}

}  // namespace

TfLiteStatus Subgraph::PartitionGraph(const TfLiteIntArray* nodes_to_replace,
                                      std::vector<NodeSubset>* node_subsets) {
  const InterpreterInfo info(this);
  // Tensor preservation requires node fusion to be disabled.
  const bool disable_node_fusion = ShouldPreserveAllTensors();
  return tflite::PartitionGraphIntoIndependentNodeSubsets(
      &info, nodes_to_replace, node_subsets,
      /*greedily=*/!DisableDelegateClustering(), control_edges_,
      disable_node_fusion);
}

TfLiteStatus Subgraph::ReplaceNodeSubsetsWithDelegateKernels(
    TfLiteRegistration registration, const TfLiteIntArray* nodes_to_replace,
    TfLiteDelegate* delegate) {
  // Annotate the registration as DELEGATE op.
  registration.builtin_code = BuiltinOperator_DELEGATE;
  if (registration.registration_external) {
    registration.registration_external->builtin_code = BuiltinOperator_DELEGATE;
  }

  // The subgraph is taking ownership of the external registration, in case the
  // user has supplied an opaque delegate.
  if (TfLiteDelegateHasValidOpaqueDelegateBuilder(delegate)) {
    // If the user has supplied an opaque delegate, then they _must_ also use
    // TfLiteOperator.
    if (!registration.registration_external) {
      TFLITE_LOG(
          tflite::TFLITE_LOG_WARNING,
          "For a delegate with the 'opaque_delegate_builder' field set, the "
          "delegate kernel's TfLiteRegistration object must have the "
          "'registration_external' field set.");
      return kTfLiteDelegateError;
    }

    // In this case, the subgraph takes ownership of the external registration.
    OpResolver::OpId op_id{registration.registration_external->builtin_code,
                           registration.registration_external->custom_name,
                           registration.registration_external->version};
    auto [it, inserted] = registration_externals_->emplace(
        op_id,
        std::unique_ptr<TfLiteOperator>(registration.registration_external));
    // If there was already an entry for this op_id in the
    // registration_externals_ cache, the statement above will have
    // no effect on the registration_externals_ cache,
    // but will deallocate registration.registration_externals.
    // To ensure that registration remains valid, we need to use the
    // registration_externals value that was previously in the cache.
    if (!inserted) {
      auto registration_external_from_cache = it->second.get();
      registration.registration_external = registration_external_from_cache;
    }
  }

  // Ignore empty node replacement sets.
  if (!nodes_to_replace->size) {
    return kTfLiteOk;
  }

  // Analyze the graph to find all independent node_subsets that are either
  // fully not-this-delegate or this-delegate computation.
  std::vector<NodeSubset> node_subsets;
  if (PartitionGraph(nodes_to_replace, &node_subsets) == kTfLiteError) {
    return kTfLiteError;
  }

  // On Android the log message below is used for diagnosing delegation success
  // also in production builds. Use VERBOSE here so that the logging is turned
  // off in production builds on other platforms.
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                  "Replacing %d out of %d node(s) with delegate (%s) node, "
                  "yielding %zu partitions "
                  "for subgraph %d.",
                  nodes_to_replace->size, execution_plan_.size(),
                  GetDelegateKernalName(registration), node_subsets.size(),
                  subgraph_index_);

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

        void* delegate_params = nullptr;
        if (TfLiteDelegateHasValidOpaqueDelegateBuilder(delegate)) {
          TfLiteOpaqueDelegateParams* opaque_params =
              CreateOpaqueDelegateParams(delegate, node_subset);
          delegate_params = opaque_params;
        } else {
          TfLiteDelegateParams* params =
              CreateDelegateParams(delegate, node_subset);
          delegate_params = params;
        }
        TF_LITE_ENSURE_STATUS(AddNodeWithParameters(
            node_subset.input_tensors, node_subset.output_tensors, {}, nullptr,
            0, delegate_params, &registration, &node_index));

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

TfLiteStatus Subgraph::AcquireSubgraphContext(
    int subgraph_index, TfLiteContext** acquired_context) {
  TF_LITE_ENSURE(&context_, subgraph_index >= 0);
  TF_LITE_ENSURE(&context_,
                 static_cast<size_t>(subgraph_index) < subgraphs_->size());
  Subgraph* acquired_subgraph = subgraphs_->at(subgraph_index).get();
  acquired_subgraph->SwitchToDelegateContext();
  *acquired_context = acquired_subgraph->context();
  return kTfLiteOk;
}

TfLiteStatus Subgraph::AcquireSubgraphContext(
    struct TfLiteContext* context, int subgraph_index,
    TfLiteContext** acquired_context) {
  return static_cast<Subgraph*>(context->impl_)
      ->AcquireSubgraphContext(subgraph_index, acquired_context);
}

TfLiteStatus Subgraph::ReleaseSubgraphContext(int subgraph_index) {
  TF_LITE_ENSURE(&context_, subgraph_index >= 0);
  TF_LITE_ENSURE(&context_,
                 static_cast<size_t>(subgraph_index) < subgraphs_->size());
  Subgraph* acquired_subgraph = subgraphs_->at(subgraph_index).get();
  acquired_subgraph->SwitchToKernelContext();
  return kTfLiteOk;
}

TfLiteStatus Subgraph::ReleaseSubgraphContext(struct TfLiteContext* context,
                                              int subgraph_index) {
  return static_cast<Subgraph*>(context->impl_)
      ->ReleaseSubgraphContext(subgraph_index);
}

TfLiteStatus Subgraph::MarkSubgraphAsDelegationSkippable(int subgraph_index) {
  TF_LITE_ENSURE(&context_, subgraph_index > 0);
  TF_LITE_ENSURE(&context_,
                 static_cast<size_t>(subgraph_index) < subgraphs_->size());
  subgraphs_->at(subgraph_index)->MarkAsDelegationSkippable();
  return kTfLiteOk;
}

TfLiteStatus Subgraph::GetNodeInitDataMmapInfo(
    const TfLiteNode* node, int* fd,
    int64_t* custom_initial_data_offset_in_file,
    int64_t* custom_initial_data_size) const {
  if (!allocation_) {
    return kTfLiteError;
  }

  if (allocation_->type() != tflite::Allocation::Type::kMMap) {
    return kTfLiteError;
  }

  const MMAPAllocation* mmap_alloc =
      static_cast<const MMAPAllocation*>(allocation_);
  *fd = mmap_alloc->fd();
  if (node->custom_initial_data == nullptr) {
    *custom_initial_data_offset_in_file = -1;
    *custom_initial_data_size = -1;
    // The node does not have custom init data.  Deliberately return 'kTfLiteOk'
    // so that clients can distinguish the following scenarios:
    // - The TF Lite runtime does not have 'MMAPAllocation' available and
    //   therefore can not possibly load the FD/offset/size tuple associcated
    //   with a node's custom init data, return 'kTfLiteError' in this case.
    // - The TF Lite runtime does have 'MMAPAllocation' available, but the
    //   specific 'node' that has been supplied happens to have 'null' set as
    //   its custom init data address.  Return 'kTfLiteOk' but set offset and
    //   size to -1.
    return kTfLiteOk;
  }

  const size_t custom_initial_data_offset =
      reinterpret_cast<const uint8_t*>(node->custom_initial_data) -
      reinterpret_cast<const uint8_t*>(mmap_alloc->mmapped_buffer());

  *custom_initial_data_offset_in_file =
      custom_initial_data_offset + mmap_alloc->mmapped_buffer_offset_in_file();
  *custom_initial_data_size = node->custom_initial_data_size;
  return kTfLiteOk;
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
  std::vector<NodeSubset> node_subsets;
  if (PartitionGraph(nodes_to_replace, &node_subsets) == kTfLiteError) {
    return kTfLiteError;
  }

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
    const std::map<std::string, std::string>* metadata,
    const ControlEdges* control_edges) {
  metadata_ = metadata;
  control_edges_ = control_edges;
  return kTfLiteOk;
}

void Subgraph::SetCancellationFunction(void* data,
                                       bool (*check_cancelled_func)(void*)) {
  cancellation_data_ = data;
  check_cancelled_func_ = check_cancelled_func;
}

TfLiteStatus Subgraph::EnsureTensorDataIsReadable(int tensor_index) {
  TfLiteTensor* t = &tensors_[tensor_index];
  TF_LITE_ENSURE(&context_, t != nullptr);
  TfLiteStatus status = kTfLiteOk;
  if (t->data_is_stale) {
    TF_LITE_ENSURE(&context_, t->delegate != nullptr);
    TF_LITE_ENSURE(&context_, t->buffer_handle != kTfLiteNullBufferHandle);
    status = TfLiteDelegateCopyFromBufferHandleInternal(&context_, t->delegate,
                                                        t->buffer_handle, t);
    t->data_is_stale = false;
  }
  return status;
}

TfLiteStatus Subgraph::EnableCancellation(std::atomic_flag* flag) {
  continue_invocation_ = flag;
  return kTfLiteOk;
}

TfLiteStatus Subgraph::Cancel() {
  if (continue_invocation_) {
    // Sets cancellation flag to false so cancellation check between nodes will
    // cancel the invocation.

    continue_invocation_->clear();
    return kTfLiteOk;
  }
  // Cancellation is not enabled in the interpreter.
  return kTfLiteError;
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

std::vector<int> Subgraph::GetInputTensorsCount() {
  std::vector<int> input_tensors_count(tensors_.size(), 0);
  for (int execution_plan_index = 0;
       execution_plan_index < execution_plan().size(); execution_plan_index++) {
    int node_index = execution_plan()[execution_plan_index];
    TfLiteNode& node = nodes_and_registration_[node_index].first;
    if (node.inputs->size > 0) {
      for (int i = 0; i < node.inputs->size; ++i) {
        const int input_idx = node.inputs->data[i];
        if (input_idx == kTfLiteOptionalTensor) continue;
        input_tensors_count[input_idx]++;
      }
    }
  }
  for (int i = 0; i < outputs_.size(); ++i) {
    input_tensors_count[outputs_[i]]++;
  }
  return input_tensors_count;
}

TfLiteStatus Subgraph::AllocateTensors() {
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

  // Profile "AllocateTensors" only when memory planning is needed.
  TFLITE_SCOPED_TAGGED_DEFAULT_PROFILE(profiler_.get(), "AllocateTensors");

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

  // Initialize the mapping between tensor index and the last execution plan
  // index that uses the tensor.
  InitializeTensorReleaseMap();

  // Temporary tensors allocated during Prepare for nodes which are subsequently
  // delegated are not required and can be freed.
  if (!pre_delegation_execution_plan_.empty()) {
    // NOLINTNEXTLINE - absl::flat_hash_set increases binary size by 106kB.
    std::unordered_set<int> delegated_nodes;
    for (int pre_delegation_node_index : pre_delegation_execution_plan_) {
      delegated_nodes.insert(pre_delegation_node_index);
    }
    for (int pose_delegation_node_index : execution_plan_) {
      delegated_nodes.erase(pose_delegation_node_index);
    }
    for (int node_index : delegated_nodes) {
      TfLiteNode& node = nodes_and_registration_[node_index].first;
      // Free all temporary tensors allocated by delegated nodes.
      for (int i = 0; i < node.temporaries->size; ++i) {
        TfLiteTensor* temporary_tensor = tensor(node.temporaries->data[i]);
        TfLiteTensorDataFree(temporary_tensor);
        temporary_tensor->bytes = 0;
      }
    }
  }
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
                                         const int* const dims_data,
                                         const int rank) {
  if (rank && !dims_data) {
    ReportError("ResizeInputTensor was given a NULL shape.");
    return kTfLiteError;
  }

  const bool delegates_applied = !delegates_applied_.empty();
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
      EqualArrayAndTfLiteIntArray(tensor->dims, rank, dims_data)) {
    return kTfLiteOk;
  }

  if (graph_is_immutable) {
    // Undo delegation if it resulted in the graph being immutable.
    TF_LITE_ENSURE_STATUS(UndoAllDelegates());
  }
  state_ = kStateUninvokable;
  return ResizeTensorImpl(tensor, BuildTfLiteArray(rank, dims_data).release());
}

TfLiteStatus Subgraph::ResizeInputTensor(int tensor_index,
                                         const std::vector<int>& dims) {
  return ResizeInputTensor(tensor_index, dims.data(), dims.size());
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
  state_ = kStateUninvokable;
  if (memory_planner_) {
    TF_LITE_ENSURE_STATUS(memory_planner_->ReleaseNonPersistentMemory());
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::ReleaseMemory() {
  state_ = kStateUninvokable;
  ReleaseNonPersistentMemory();

  // Free dynamic input tensors.
  for (const int input_tensor_idx : inputs_) {
    if (input_tensor_idx == kTfLiteOptionalTensor) continue;
    TfLiteTensor* input_tensor = tensor(input_tensor_idx);
    if (!input_tensor || input_tensor->allocation_type != kTfLiteDynamic)
      continue;
    if (input_tensor->data.raw) {
      TfLiteTensorDataFree(input_tensor);
    }
  }
  // Free dynamic output tensors.
  for (const int output_tensor_idx : outputs_) {
    if (output_tensor_idx == kTfLiteOptionalTensor) continue;
    TfLiteTensor* output_tensor = tensor(output_tensor_idx);
    if (!output_tensor || output_tensor->allocation_type != kTfLiteDynamic)
      continue;
    if (output_tensor->data.raw) {
      TfLiteTensorDataFree(output_tensor);
    }
  }

  return kTfLiteOk;
}

// Give 'op_reg' a chance to initialize itself using the contents of
// 'buffer'. If registration_external is valid, use the 'init' callback from
// that.
void* Subgraph::OpInit(const TfLiteRegistration& op_reg, const char* buffer,
                       size_t length) {
  // Delegates that use the stable delegate API to iterate over the nodes and
  // registrations are presented with ABI stable 'TfLiteOperator'
  // pointers, as opposed to ABI unstable 'TfLiteRegistration' pointers, even
  // for builtin OPs like ADD or SUB.  A knock-on effect of this behavior is
  // that we need to differentiate two scenarios when interacting with a
  // 'TfLiteOperator'.
  // 1. In the 'wrapper' scenario described above we use the 'node_index' field
  //    that points us to the corresponding 'TfLiteRegistration' that holds the
  //    callbacks that need to be invoked.
  // 2. Otherwise the 'TfLiteOperator' is either a stable custom OP,
  //    or a stable delegate kernel, and in both of those cases we need to use
  //    the callbacks stored within the 'TfLiteOperator' itself.
  if (op_reg.registration_external) {
    if (op_reg.registration_external->node_index != -1) {
      TfLiteRegistration* referenced_registration =
          &nodes_and_registration_[op_reg.registration_external->node_index]
               .second;
      if (referenced_registration->init == nullptr) return nullptr;
      return referenced_registration->init(&context_, buffer, length);
    }
    if (op_reg.registration_external->init_with_data) {
      void* user_data = op_reg.registration_external->user_data;
      return op_reg.registration_external->init_with_data(
          user_data, reinterpret_cast<TfLiteOpaqueContext*>(&context_), buffer,
          length);
    }
    if (op_reg.registration_external->init) {
      return op_reg.registration_external->init(
          reinterpret_cast<TfLiteOpaqueContext*>(&context_), buffer, length);
    }
  }
  if (op_reg.init == nullptr) return nullptr;
  return op_reg.init(&context_, buffer, length);
}

TfLiteStatus Subgraph::OpPrepare(const TfLiteRegistration& op_reg,
                                 TfLiteNode* node) {
  // Delegates that use the stable delegate API to iterate over the nodes and
  // registrations are presented with ABI stable 'TfLiteOperator'
  // pointers, as opposed to ABI unstable 'TfLiteRegistration' pointers, even
  // for builtin OPs like ADD or SUB.  A knock-on effect of this behavior is
  // that we need to differentiate two scenarios when interacting with a
  // 'TfLiteOperator'.
  // 1. In the 'wrapper' scenario described above we use the 'node_index' field
  //    that points us to the corresponding 'TfLiteRegistration' that holds the
  //    callbacks that need to be invoked.
  // 2. Otherwise the 'TfLiteOperator' is either a stable custom OP,
  //    or a stable delegate kernel, and in both of those cases we need to use
  //    the callbacks stored within the 'TfLiteOperator' itself.
  if (op_reg.registration_external) {
    if (op_reg.registration_external->node_index != -1) {
      TfLiteRegistration* referenced_registration =
          &nodes_and_registration_[op_reg.registration_external->node_index]
               .second;
      if (referenced_registration->prepare == nullptr) {
        if (IsUnresolvedCustomOp(op_reg)) {
          ReportError(
              "Encountered unresolved custom op: %s.\nSee instructions: "
              "https://www.tensorflow.org/lite/guide/ops_custom ",
              op_reg.custom_name ? op_reg.custom_name : "UnknownOp");
          return kTfLiteUnresolvedOps;
        } else {
          // Resolved ops can have a null Prepare function.
          return kTfLiteOk;
        }
      }
      return referenced_registration->prepare(&context_, node);
    }
    if (op_reg.registration_external->prepare_with_data) {
      // The 'data' field required by the 'prepare' function pointer must be
      // retrieved from the 'registration_external' object itself.
      void* user_data = op_reg.registration_external->user_data;
      return op_reg.registration_external->prepare_with_data(
          user_data, reinterpret_cast<TfLiteOpaqueContext*>(&context_),
          reinterpret_cast<TfLiteOpaqueNode*>(node));
    }
    if (op_reg.registration_external->prepare) {
      return op_reg.registration_external->prepare(
          reinterpret_cast<TfLiteOpaqueContext*>(&context_),
          reinterpret_cast<TfLiteOpaqueNode*>(node));
    }
  }
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
            "https://www.tensorflow.org/lite/guide/ops_custom ",
            op_reg.custom_name ? op_reg.custom_name : "UnknownOp");
      }
      return kTfLiteUnresolvedOps;
    }
    // Resolved ops can have a null Prepare function.
    return kTfLiteOk;
  }
  return op_reg.prepare(&context_, node);
}

// Invoke the operator represented by 'node'.
TfLiteStatus Subgraph::OpInvoke(const TfLiteRegistration& op_reg,
                                TfLiteNode* node) {
  // Delegates that use the stable delegate API to iterate over the nodes and
  // registrations are presented with ABI stable 'TfLiteOperator'
  // pointers, as opposed to ABI unstable 'TfLiteRegistration' pointers, even
  // for builtin OPs like ADD or SUB.  A knock-on effect of this behavior is
  // that we need to differentiate two scenarios when interacting with a
  // 'TfLiteOperator'.
  // 1. In the 'wrapper' scenario described above we use the 'node_index' field
  //    that points us to the corresponding 'TfLiteRegistration' that holds the
  //    callbacks that need to be invoked.
  // 2. Otherwise the 'TfLiteOperator' is either a stable custom OP,
  //    or a stable delegate kernel, and in both of those cases we need to use
  //    the callbacks stored within the 'TfLiteOperator' itself.
  if (op_reg.registration_external) {
    if (op_reg.registration_external->node_index != -1) {
      TfLiteRegistration* referenced_registration =
          &nodes_and_registration_[op_reg.registration_external->node_index]
               .second;
      if (referenced_registration->invoke == nullptr) return kTfLiteError;
      return referenced_registration->invoke(&context_, node);
    }
    if (op_reg.registration_external->invoke_with_data) {
      void* user_data = op_reg.registration_external->user_data;
      return op_reg.registration_external->invoke_with_data(
          user_data, reinterpret_cast<TfLiteOpaqueContext*>(&context_),
          reinterpret_cast<TfLiteOpaqueNode*>(node));
    }
    if (op_reg.registration_external->invoke) {
      return op_reg.registration_external->invoke(
          reinterpret_cast<TfLiteOpaqueContext*>(&context_),
          reinterpret_cast<TfLiteOpaqueNode*>(node));
    }
  }
  if (op_reg.invoke == nullptr) return kTfLiteError;
  return op_reg.invoke(&context_, node);
}

// Let 'op_reg' release any memory it might have allocated via 'OpInit'.
// If registration_external is valid, use the 'free' callback from that.
void Subgraph::OpFree(const TfLiteRegistration& op_reg, void* buffer) {
  // Delegates that use the stable delegate API to iterate over the nodes and
  // registrations are presented with ABI stable 'TfLiteOperator'
  // pointers, as opposed to ABI unstable 'TfLiteRegistration' pointers, even
  // for builtin OPs like ADD or SUB.  A knock-on effect of this behavior is
  // that we need to differentiate two scenarios when interacting with a
  // 'TfLiteOperator'.
  // 1. In the 'wrapper' scenario described above we use the 'node_index' field
  //    that points us to the corresponding 'TfLiteRegistration' that holds the
  //    callbacks that need to be invoked.
  // 2. Otherwise the 'TfLiteOperator' is either a stable custom OP,
  //    or a stable delegate kernel, and in both of those cases we need to use
  //    the callbacks stored within the 'TfLiteOperator' itself.
  if (op_reg.registration_external && buffer) {
    if (op_reg.registration_external->node_index != -1) {
      TfLiteRegistration* referenced_registration =
          &nodes_and_registration_[op_reg.registration_external->node_index]
               .second;
      if (referenced_registration->free == nullptr) return;
      return referenced_registration->free(&context_, buffer);
    }
    if (op_reg.registration_external->free_with_data) {
      void* user_data = op_reg.registration_external->user_data;
      return op_reg.registration_external->free_with_data(
          user_data, reinterpret_cast<TfLiteOpaqueContext*>(&context_), buffer);
    }
    if (op_reg.registration_external->free) {
      return op_reg.registration_external->free(
          reinterpret_cast<TfLiteOpaqueContext*>(&context_), buffer);
    }
  }
  if (op_reg.free == nullptr) return;
  if (buffer) {
    op_reg.free(&context_, buffer);
  }
}

TfLiteStatus Subgraph::MayAllocateOpOutput(TfLiteNode* node) {
  if (ShouldOptimizeMemoryForLargeTensors()) {
    for (int i = 0; i < node->outputs->size; ++i) {
      int tensor_index = node->outputs->data[i];
      if (tensor_index == kTfLiteOptionalTensor) continue;
      TfLiteTensor* tensor = &context_.tensors[tensor_index];
      if (tensor->data.raw == nullptr &&
          tensor->allocation_type == kTfLiteDynamic) {
        TfLiteTensorRealloc(tensor->bytes, tensor);
      }
    }
  }
  return kTfLiteOk;
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
#ifdef TF_LITE_TENSORFLOW_PROFILER
    tflite::OnTfLiteOpPrepare(GetTFLiteOpName(registration), subgraph_index_,
                              node_index);
#endif  // TF_LITE_TENSORFLOW_PROFILER
    const TfLiteStatus op_prepare_status = OpPrepare(registration, &node);
    if (op_prepare_status != kTfLiteOk &&
        op_prepare_status != kTfLiteOutputShapeNotKnown) {
      ReportOpError(&context_, node, registration, node_index,
                    "failed to prepare");
      return op_prepare_status;
    }

    *last_execution_plan_index_prepared = execution_plan_index;

    // Discontinue if the node has dynamic outputs. Note that we don't
    // stop for dynamic temporary tensors since they won't affect the
    // sizes of other tensors in the graph.
    if (HasDynamicTensor(context_, node.outputs, &dynamic_tensor_index_) ||
        op_prepare_status == kTfLiteOutputShapeNotKnown) {
      has_dynamic_tensors_ = true;
      return kTfLiteOk;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::PrepareOpsAndTensors() {
  // Prepare original execution plan if any applied delegate wants it.
  // If any of the delegates is immutable, this won't be triggered
  // post-delegation (since we undo/redo delegation). For all other cases, other
  // delegates that do shape propagation themselves would still be able to.
  bool prepare_original_plan = false;
  if (!pre_delegation_execution_plan_.empty()) {
    for (int i = 0; i < delegates_applied_.size(); ++i) {
      if ((TfLiteDelegateGetFlagsInternal(delegates_applied_[i]) &
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

  if (!memory_planner_) {
#ifdef TFLITE_USE_SIMPLE_MEMORY_PLANNER
    memory_planner_.reset(new SimplePlanner(&context_, CreateGraphInfo()));
#else
    memory_planner_ = std::make_unique<ArenaPlanner>(
        &context_, CreateGraphInfo(), ShouldPreserveAllTensors(),
        kDefaultTensorAlignment, subgraph_index_);
#endif
    memory_planner_->PlanAllocations();
  }

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

TfLiteStatus Subgraph::RemoveUnusedInputs() {
  std::vector<int> input_tensors_count = GetInputTensorsCount();
  // Mark unused inputs as kTfLiteOptionalTensor.
  for (int& tensor_idx : inputs()) {
    if (tensor_idx == kTfLiteOptionalTensor) continue;
    if (input_tensors_count[tensor_idx] == 0) {
      tensor(tensor_idx)->bytes = 0;  // To make it clearer for memory analysis.
      tensor_idx = kTfLiteOptionalTensor;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::Invoke() {
  auto status = InvokeImpl();
  telemetry::TelemetryReportEvent(&context_, "Invoke", status);
  return status;
}
TfLiteStatus Subgraph::InvokeImpl() {
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
#ifdef TF_LITE_TENSORFLOW_PROFILER
  tensorflow::profiler::TraceMe* trace_subgraph =
      tflite::OnTfLiteSubgraphInvoke(name_.c_str(), subgraph_index_);
#endif  // TF_LITE_TENSORFLOW_PROFILER

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
#ifdef TF_LITE_TENSORFLOW_PROFILER
    if (!op_name) {
      op_name = GetTFLiteOpName(registration);
    }
    tensorflow::profiler::TraceMe* trace_op =
        tflite::OnTfLiteOpInvoke(op_name, subgraph_index_, node_index);
#endif  // TF_LITE_TENSORFLOW_PROFILER

    // If per operator profiling flag is set in the delegate, this macro op
    // should not be profiled, thus a nullptr is passed to the ScopedProfile
    bool profile_op =
        !(node.delegate != nullptr &&
          (node.delegate->flags & kTfLiteDelegateFlagsPerOperatorProfiling));
    TFLITE_SCOPED_TAGGED_OPERATOR_PROFILE(
        profile_op ? profiler_.get() : nullptr, op_name, node_index);

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
    // Allocate dynamic tensors which memory is required to be allocated
    // before executing the node.
    MayAllocateOpOutput(&node);

    if (check_cancelled_func_ != nullptr &&
        check_cancelled_func_(cancellation_data_)) {
      ReportError("Client requested cancel during Invoke()");
      return kTfLiteError;
    }

    if (continue_invocation_ && !continue_invocation_->test_and_set()) {
      // `Cancel` is called and cancellation flag is flipped.
      ReportError("Client requested cancel during Invoke()");
      return kTfLiteCancelled;
    }

    EnsureTensorsVectorCapacity();
    tensor_resized_since_op_invoke_ = false;
    if (auto s = OpInvoke(registration, &node); s != kTfLiteOk) {
      auto err = ReportOpError(&context_, node, registration, node_index,
                               "failed to invoke");
      return s == kTfLiteCancelled ? s : err;
    }

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
    // Release dynamic tensor memory if configured by the user.
    MaybeReleaseDynamicTensors(node, node_index);

#ifdef TF_LITE_TENSORFLOW_PROFILER
    tflite::OnTfLiteOpInvokeEnd(trace_op);
#endif  // TF_LITE_TENSORFLOW_PROFILER
  }
#ifdef TF_LITE_TENSORFLOW_PROFILER
  tflite::OnTfLiteSubgraphInvokeEnd(trace_subgraph);
#endif  // TF_LITE_TENSORFLOW_PROFILER
  return status;
}

TfLiteStatus Subgraph::ResizeTensor(TfLiteContext* context,
                                    TfLiteTensor* tensor,
                                    TfLiteIntArray* new_size) {
  // If the dimensions don't change, avoid unnecessary (re)allocations.
  //
  // Note that it's required to check `tensor->data.raw != nullptr`. Otherwise
  // the subgraph won't allocate memory for a dynamic tensor when its size
  // is equal to the original tensor size.
  //
  // We also need to check the bytes count because some direct calls to
  // TfLiteTensorResizeMaybeCopy may lead to inconsistent dims and bytes in a
  // tensor.
  const bool can_reuse_allocation = [tensor, new_size, context] {
    if (tensor->data.raw == nullptr) {
      return false;
    }
    if (!EqualArrayAndTfLiteIntArray(tensor->dims, new_size->size,
                                     new_size->data)) {
      return false;
    }
    // Those data types byte sizes are not handled by `ResizeTensorImpl`.
    if (tensor->type == kTfLiteString || tensor->type == kTfLiteResource ||
        tensor->type == kTfLiteVariant) {
      return true;
    }
    size_t new_bytes = 0;
    tflite::BytesRequired(tensor->type, tensor->dims->data, tensor->dims->size,
                          &new_bytes, context);
    return new_bytes == tensor->bytes;
  }();

  if (can_reuse_allocation) {
    // A number of clients assume |new_size| remains valid upon success, so
    // swap it in as the new (but logically identical) tensor dims.
    if (new_size != tensor->dims) {
      TfLiteIntArrayFree(tensor->dims);
      tensor->dims = new_size;
    }
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
  if (tensors_to_add < 0) return kTfLiteError;
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
    int tensor_index, TfLiteType type, const char* name, const size_t ndims,
    const int* dims, TfLiteQuantization quantization, const char* buffer,
    size_t bytes, const Allocation* allocation, TfLiteSparsity* sparsity,
    const size_t buffer_identifier) {
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
    TF_LITE_ENSURE_OK(
        &context_,
        tflite::BytesRequired(type, dims, ndims, &required_bytes, &context_));
    TF_LITE_ENSURE(&context_, required_bytes <= bytes);
  }

  TfLiteTensor& tensor = context_.tensors[tensor_index];
  if (type == tensor.type &&
      EqualArrayAndTfLiteIntArray(tensor.dims, ndims, dims)) {
    // Fast path which does not invalidate the invokable property.
    TfLiteTensorDataFree(&tensor);
    TfLiteQuantizationFree(&tensor.quantization);
    tensor.data.raw = const_cast<char*>(buffer);
    if (!tensor.dims) tensor.dims = ConvertArrayToTfLiteIntArray(ndims, dims);
    tensor.params = GetLegacyQuantization(quantization);
    tensor.quantization = *scoped_quantization.release();
    tensor.sparsity = scoped_sparsity.release();
    tensor.allocation_type = kTfLiteMmapRo;
    tensor.allocation = allocation;
  } else {
    state_ = kStateUninvokable;
    TfLiteTensorReset(type, name, ConvertArrayToTfLiteIntArray(ndims, dims),
                      GetLegacyQuantization(quantization),
                      const_cast<char*>(buffer), bytes, kTfLiteMmapRo,
                      allocation, false, &tensor);
    tensor.quantization = *scoped_quantization.release();
    tensor.sparsity = scoped_sparsity.release();
  }
  if (buffer_identifier != kTfLiteNoBufferIdentifier) {
    tensor_buffer_identifiers_[tensor_index] = buffer_identifier;
  }
  return kTfLiteOk;
}

// Set description of inputs/outputs/data/fptrs for node `node_index`.
// This variant assumes an external buffer has been allocated of size
// bytes. The lifetime of buffer must be ensured to be greater or equal
// to Interpreter.
TfLiteStatus Subgraph::SetTensorParametersReadWrite(
    int tensor_index, TfLiteType type, const char* name, const size_t ndims,
    const int* dims, TfLiteQuantization quantization, bool is_variable,
    const size_t ndims_signature, const int* dims_signature) {
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
    TF_LITE_ENSURE_OK(
        &context_,
        tflite::BytesRequired(type, dims, ndims, &required_bytes, &context_));
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

  TfLiteTensorReset(type, name, ConvertArrayToTfLiteIntArray(ndims, dims),
                    GetLegacyQuantization(quantization),
                    /*buffer=*/nullptr, required_bytes, allocation_type,
                    nullptr, is_variable, &tensor);
  tensor.quantization = *scoped_quantization.release();
  tensor.dims_signature =
      ConvertArrayToTfLiteIntArray(ndims_signature, dims_signature);
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
      size_t bytes_required;
      TfLiteStatus status =
          tflite::BytesRequired(tensor->type, new_size->data, new_size->size,
                                &bytes_required, &context_);
      if (status != kTfLiteOk) {
        TfLiteIntArrayFree(new_size);
        return kTfLiteError;
      }

      // Realloc space for heap-allocated tensors.
      TfLiteTensorResizeMaybeCopy(bytes_required, tensor, false);
      tensor->bytes = bytes_required;
    }
    if (tensor->dims && tensor->dims != new_size) {
      TfLiteIntArrayFree(tensor->dims);
    }
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

void Subgraph::OptimizeMemoryForLargeTensors(
    int large_tensors_thresholds_in_bytes) {
  for (size_t tensor_index = 0; tensor_index < context_.tensors_size;
       tensor_index++) {
    TfLiteTensor* tensor = &context_.tensors[tensor_index];
    if (tensor->bytes >= large_tensors_thresholds_in_bytes &&
        tensor->allocation_type == kTfLiteArenaRw &&
        // Skip input tensors since they are handled by ResizeInputTensor().
        std::find(inputs_.begin(), inputs_.end(), tensor_index) ==
            inputs_.end()) {
      // Change large tensors' allocation_type and data.raw. This method must be
      // called before AllocateTensors() to avoid handling them by ArenaPlanner.
      tensor->allocation_type = kTfLiteDynamic;
      tensor->data.raw = nullptr;
    }
  }
}

TfLiteStatus Subgraph::SwitchToDelegateContext() {
  TF_LITE_ENSURE(&context_, delegate_context_switch_count_ >= 0);
  if (delegate_context_switch_count_ == 0) {
    context_.GetNodeAndRegistration = GetNodeAndRegistration;
    context_.ReplaceNodeSubsetsWithDelegateKernels =
        ReplaceNodeSubsetsWithDelegateKernels;
    context_.GetExecutionPlan = GetExecutionPlan;
    context_.PreviewDelegatePartitioning = PreviewDelegatePartitioning;
    context_.AcquireSubgraphContext = AcquireSubgraphContext;
    context_.ReleaseSubgraphContext = ReleaseSubgraphContext;
  }
  delegate_context_switch_count_++;
  return kTfLiteOk;
}

TfLiteStatus Subgraph::SwitchToKernelContext() {
  TF_LITE_ENSURE(&context_, delegate_context_switch_count_ >= 1);
  if (delegate_context_switch_count_ == 1) {
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
        [](struct TfLiteContext* context,
           const TfLiteIntArray* nodes_to_replace,
           TfLiteDelegateParams** partition_params_array,
           int* num_partitions) { return ForbiddenContextFunction(context); };
    context_.AcquireSubgraphContext = [](struct TfLiteContext* context,
                                         int subgraph_index,
                                         TfLiteContext** acquired_context) {
      return ForbiddenContextFunction(context);
    };
    context_.ReleaseSubgraphContext = [](struct TfLiteContext* context,
                                         int subgraph_index) {
      return ForbiddenContextFunction(context);
    };
    // Free any memory that might have been allocated by
    // PreviewDelegatePartitioning.
    FreeDelegatePartitioningData();
  }
  delegate_context_switch_count_--;
  return kTfLiteOk;
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

  // Reset all the is_delegation_skippable_ flags in subgraphs.
  for (auto& subgraph : *subgraphs_) {
    subgraph->is_delegation_skippable_ = false;
  }

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
    TF_LITE_ENSURE_STATUS(ModifyGraphWithDelegateImpl(delegate));
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
  auto status = ModifyGraphWithDelegateImpl(delegate);
  telemetry::TelemetryReportEvent(&context_, "ModifyGraphWithDelegate", status);
  return status;
}

TfLiteStatus Subgraph::ModifyGraphWithDelegateImpl(TfLiteDelegate* delegate) {
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
      TfLiteDelegateGetFlagsInternal(delegate) &
      kTfLiteDelegateFlagsAllowDynamicTensors;
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
      TFLITE_LOG_PROD_ONCE(
          tflite::TFLITE_LOG_WARNING,
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
  TfLiteStatus status = TfLiteDelegatePrepareInternal(&context_, delegate);
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

void Subgraph::GetMemoryAllocInfo(SubgraphAllocInfo* alloc_info) const {
  memset(alloc_info, 0, sizeof(SubgraphAllocInfo));
  if (memory_planner_ == nullptr) return;
  memory_planner_->GetAllocInfo(&alloc_info->arena_size,
                                &alloc_info->arena_persist_size);
  for (const auto& tensor : tensors_) {
    if (tensor.allocation_type == kTfLiteDynamic &&
        tensor.data.raw != nullptr) {
      alloc_info->dynamic_size += tensor.bytes;
    }
  }
  if (GetSubgraphIndex() == 0) {
    for (const auto& res : *resources_) {
      alloc_info->resource_size += res.second->GetMemoryUsage();
    }
  }
}

std::unique_ptr<GraphInfo> Subgraph::CreateGraphInfo() {
  return std::unique_ptr<GraphInfo>(new InterpreterInfo(this));
}

void Subgraph::InitializeTensorReleaseMap() {
  for (int i = 0; i < execution_plan_.size(); ++i) {
    int node_index = execution_plan_[i];
    const TfLiteNode& node = nodes_and_registration_[node_index].first;
    for (int input_index = 0; input_index < node.inputs->size; ++input_index) {
      int input_tensor_index = node.inputs->data[input_index];
      TfLiteTensor* input_tensor = tensor(input_tensor_index);
      if (!input_tensor) continue;
      tensor_to_last_op_index_[input_tensor_index] = node_index;
    }
    // Also checks outputs of a node to make sure tensors are released in case
    // when a tensor is not used for input of another node.
    for (int output_index = 0; output_index < node.outputs->size;
         ++output_index) {
      int output_tensor_index = node.outputs->data[output_index];
      TfLiteTensor* output_tensor = tensor(output_tensor_index);
      if (!output_tensor) continue;
      tensor_to_last_op_index_[output_tensor_index] = node_index;
    }
  }
}

void Subgraph::MaybeReleaseDynamicTensors(const TfLiteNode& node,
                                          size_t node_index) {
  if (!ShouldReleaseDynamicTensors()) return;

  // Release input tensors if they're neither graph input tensors nor no
  // longer used by remaining graph execution.
  auto tensorIsInput = [&](int index) {
    for (int idx : inputs_) {
      if (idx == index) return true;
    }
    return false;
  };
  auto tensorIsOutput = [&](int index) {
    for (int idx : outputs_) {
      if (idx == index) return true;
    }
    return false;
  };
  for (int input_index = 0; input_index < node.inputs->size; ++input_index) {
    int input_tensor_index = node.inputs->data[input_index];
    TfLiteTensor* input_tensor = tensor(input_tensor_index);
    if (!input_tensor || input_tensor->allocation_type != kTfLiteDynamic ||
        input_tensor->type == kTfLiteString ||
        input_tensor->type == kTfLiteResource ||
        tensorIsInput(input_tensor_index) || tensorIsOutput(input_tensor_index))
      continue;
    auto it = tensor_to_last_op_index_.find(input_tensor_index);
    if (it != tensor_to_last_op_index_.end() && it->second == node_index) {
      if (input_tensor->data.raw) {
        TfLiteTensorDataFree(input_tensor);
      }
    }
  }

  // Release output tensors if they're neither graph output tensors nor no
  // longer used by remaining graph execution.
  for (int output_index = 0; output_index < node.outputs->size;
       ++output_index) {
    int output_tensor_index = node.outputs->data[output_index];
    TfLiteTensor* output_tensor = tensor(output_tensor_index);
    if (!output_tensor || output_tensor->allocation_type != kTfLiteDynamic ||
        output_tensor->type == kTfLiteString ||
        output_tensor->type == kTfLiteResource ||
        tensorIsInput(output_tensor_index) ||
        tensorIsOutput(output_tensor_index))
      continue;
    auto it = tensor_to_last_op_index_.find(output_tensor_index);
    if (it != tensor_to_last_op_index_.end() && it->second == node_index) {
      if (output_tensor->data.raw) {
        TfLiteTensorDataFree(output_tensor);
      }
    }
  }
}

TfLiteStatus Subgraph::SetBufferHandleImpl(
    TfLiteContext* context, TfLiteTensor* tensor,
    TfLiteBufferHandle buffer_handle, TfLiteDelegate* delegate,
    bool release_existing_buffer_handle) {
  TF_LITE_ENSURE(context, tensor != nullptr);
  TF_LITE_ENSURE(context,
                 tensor->delegate == nullptr || tensor->delegate == delegate);
  tensor->delegate = delegate;
  if (release_existing_buffer_handle &&
      tensor->buffer_handle != kTfLiteNullBufferHandle) {
    TF_LITE_ENSURE_STATUS(TfLiteDelegateFreeBufferHandleInternal(
        context, tensor->delegate, &(tensor->buffer_handle)));
  }
  tensor->buffer_handle = buffer_handle;

  return kTfLiteOk;
}

}  // namespace tflite
