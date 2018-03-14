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

#include "tensorflow/contrib/lite/interpreter.h"
#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include "tensorflow/contrib/lite/arena_planner.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/error_reporter.h"
#include "tensorflow/contrib/lite/graph_info.h"
#include "tensorflow/contrib/lite/kernels/gemm_support.h"
#include "tensorflow/contrib/lite/memory_planner.h"
#include "tensorflow/contrib/lite/nnapi_delegate.h"
#include "tensorflow/contrib/lite/schema/schema_generated.h"
#include "tensorflow/contrib/lite/util.h"

namespace tflite {

namespace {

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

}  // namespace

// A trivial implementation of GraphInfo around the Interpreter.
// NOTE: this interpreter info represents the subset of the
// graph that is executed according to execution plan. Thus,
// the indices are execution plan indices rather than raw node
// indices.
class InterpreterInfo : public GraphInfo {
 public:
  explicit InterpreterInfo(Interpreter* interpreter)
      : interpreter_(interpreter) {}

  size_t num_tensors() const override { return interpreter_->tensors_size(); }
  TfLiteTensor* tensor(size_t index) override {
    return interpreter_->tensor(index);
  }
  size_t num_nodes() const override {
    return interpreter_->execution_plan().size();
  }
  const TfLiteNode& node(size_t index) const override {
    int node_index = interpreter_->execution_plan()[index];
    return interpreter_->node_and_registration(node_index)->first;
  }
  const std::vector<int>& inputs() const override {
    return interpreter_->inputs();
  }
  const std::vector<int>& outputs() const override {
    return interpreter_->outputs();
  }

 public:
  Interpreter* interpreter_;
};

Interpreter::Interpreter(ErrorReporter* error_reporter)
    : error_reporter_(error_reporter ? error_reporter
                                     : DefaultErrorReporter()) {
  context_.impl_ = static_cast<void*>(this);
  context_.ResizeTensor = ResizeTensor;
  context_.ReportError = ReportError;
  context_.AddTensors = AddTensors;
  context_.tensors = nullptr;
  context_.tensors_size = 0;
  context_.eigen_context = nullptr;
  context_.gemm_context = nullptr;
  context_.recommended_num_threads = -1;

  // Invalid to call these these except from TfLiteDelegate
  SetForbiddenContextFunction(&context_.GetNodeAndRegistration);
  SetForbiddenContextFunction(&context_.ReplaceSubgraphsWithDelegateKernels);
  SetForbiddenContextFunction(&context_.GetExecutionPlan);

  // Reserve some space for the tensors to avoid excessive resizing.
  tensors_.reserve(kTensorsReservedCapacity);
  nodes_and_registration_.reserve(kTensorsReservedCapacity);
  next_execution_plan_index_to_prepare_ = 0;
  UseNNAPI(false);
}

Interpreter::~Interpreter() {
  for (auto& nodeAndReg : nodes_and_registration_) {
    TfLiteNode& node = nodeAndReg.first;
    TfLiteIntArrayFree(node.inputs);
    TfLiteIntArrayFree(node.outputs);
    TfLiteIntArrayFree(node.temporaries);
    if (node.builtin_data) free(node.builtin_data);
    OpFree(nodeAndReg.second, node.user_data);
    node.builtin_data = nullptr;
  }

  for (int i = 0; i < context_.tensors_size; i++) {
    TfLiteTensor* tensor = &context_.tensors[i];
    if (tensor->buffer_handle != kTfLiteNullBufferHandle) {
      tensor->delegate->FreeBufferHandle(tensor->delegate,
                                         &tensor->buffer_handle);
    }
    TfLiteTensorFree(tensor);
  }
}

TfLiteStatus Interpreter::ReplaceSubgraphsWithDelegateKernels(
    TfLiteContext* context, TfLiteRegistration registration,
    const TfLiteIntArray* nodes_to_replace, TfLiteDelegate* delegate) {
  return static_cast<Interpreter*>(context->impl_)
      ->ReplaceSubgraphsWithDelegateKernels(registration, nodes_to_replace,
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
                                           const Subgraph& subgraph) {
  // Step 1: Calculate the allocation size.
  int allocation_size = sizeof(TfLiteDelegateParams);

  int nodes_to_replace_size =
      TfLiteIntArrayGetSizeInBytes(subgraph.nodes.size());
  allocation_size += nodes_to_replace_size;

  int input_tensors_size =
      TfLiteIntArrayGetSizeInBytes(subgraph.input_tensors.size());
  allocation_size += input_tensors_size;

  int output_tensors_size =
      TfLiteIntArrayGetSizeInBytes(subgraph.output_tensors.size());
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
  CopyVectorToTfLiteIntArray(subgraph.nodes, params->nodes_to_replace);
  allocation += nodes_to_replace_size;

  params->input_tensors = reinterpret_cast<TfLiteIntArray*>(allocation);
  CopyVectorToTfLiteIntArray(subgraph.input_tensors, params->input_tensors);
  allocation += input_tensors_size;

  params->output_tensors = reinterpret_cast<TfLiteIntArray*>(allocation);
  CopyVectorToTfLiteIntArray(subgraph.output_tensors, params->output_tensors);
  allocation += output_tensors_size;

  return params;
}

}  // namespace

TfLiteStatus Interpreter::ReplaceSubgraphsWithDelegateKernels(
    TfLiteRegistration registration, const TfLiteIntArray* nodes_to_replace,
    TfLiteDelegate* delegate) {
  // Annotate the registration as DELEGATE op.
  registration.builtin_code = BuiltinOperator_DELEGATE;

  // Analyze the graph to find all independent subgraphs that are either
  // fully not-this-delegate or this-delegate computation.
  InterpreterInfo info(this);
  std::vector<Subgraph> subgraphs;
  PartitionGraphIntoIndependentSubgraphs(&info, nodes_to_replace, &subgraphs);

  execution_plan_.clear();
  for (auto& subgraph : subgraphs) {
    // Subgraphs calimed by the delegate should have a "macro" op created, the
    // other subgraphs (kTfNonPartition) just have their nodes added back to
    // the execution plan.
    switch (subgraph.type) {
      case Subgraph::kTfNonPartition:
        for (auto it = subgraph.nodes.begin(); it != subgraph.nodes.end();
             ++it) {
          execution_plan_.push_back(*it);
        }
        break;
      case Subgraph::kTfPartition: {
        int node_index;

        TfLiteDelegateParams* params = CreateDelegateParams(delegate, subgraph);
        AddNodeWithParameters(subgraph.input_tensors, subgraph.output_tensors,
                              nullptr, 0, params, &registration, &node_index);

        // Initialize the output tensors's delegate-related fields.
        for (int tensor_index : subgraph.output_tensors) {
          TfLiteTensor* tensor = &tensors_[tensor_index];
          TF_LITE_ENSURE_EQ(&context_, tensor->delegate, nullptr);
          TF_LITE_ENSURE_EQ(&context_, tensor->buffer_handle,
                            kTfLiteNullBufferHandle);
          // buffer_handle will be filled in delegate's `Prepare`
          // function.
          tensor->delegate = delegate;
        }

        // Associate the node with the delegate.
        TfLiteNode* node = &nodes_and_registration_[node_index].first;
        node->delegate = delegate;
      } break;
      case Subgraph::kTfUnexplored:
        return kTfLiteError;
        break;
    }
  }
  return kTfLiteOk;
}

// Gets an TfLiteIntArray* representing the execution plan. The interpreter owns
// this memory and it is only guaranteed to exist during the invocation of the
// delegate prepare.
TfLiteStatus Interpreter::GetExecutionPlan(TfLiteIntArray** execution_plan) {
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
TfLiteStatus Interpreter::GetExecutionPlan(struct TfLiteContext* context,
                                           TfLiteIntArray** execution_plan) {
  return static_cast<Interpreter*>(context->impl_)
      ->GetExecutionPlan(execution_plan);
}

TfLiteStatus Interpreter::SetInputs(std::vector<int> inputs) {
  TF_LITE_ENSURE_OK(&context_,
                    CheckTensorIndices("inputs", inputs.data(), inputs.size()));
  inputs_ = std::move(inputs);
  return kTfLiteOk;
}

TfLiteStatus Interpreter::SetOutputs(std::vector<int> outputs) {
  TF_LITE_ENSURE_OK(
      &context_, CheckTensorIndices("outputs", outputs.data(), outputs.size()));
  outputs_ = std::move(outputs);
  return kTfLiteOk;
}

TfLiteStatus Interpreter::CheckTensorIndices(const char* label,
                                             const int* indices, int length) {
  // Making sure kOptionalTensor is not re-defined to something other than -1.
  static_assert(kOptionalTensor == -1, "kOptionalTensor should be defined -1");

  for (int i = 0; i < length; i++) {
    int index = indices[i];
    if (index < kOptionalTensor || index >= context_.tensors_size) {
      ReportError(&context_, "Invalid tensor index %d in %s\n", index, label);
      consistent_ = false;
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Interpreter::BytesRequired(TfLiteType type, const int* dims,
                                        int dims_size, size_t* bytes) {
  // TODO(aselle): Check for overflow here using overflow.h in TensorFlow
  // MultiplyWithoutOverflow.
  TF_LITE_ENSURE(&context_, bytes != nullptr);
  size_t count = 1;
  for (int k = 0; k < dims_size; k++) count *= dims[k];
  switch (type) {
    case kTfLiteFloat32:
      *bytes = sizeof(float) * count;
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
    default:
      ReportError(&context_,
                  "Only float32, int32, int64, uint8 supported currently.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Interpreter::AllocateTensors() {
  next_execution_plan_index_to_prepare_ = 0;
  if (memory_planner_) {
    TF_LITE_ENSURE_STATUS(memory_planner_->ResetAllocations());
  }

  if (!consistent_) {
    ReportError(&context_, "AllocateTensors() called on inconsistent model.");
    return kTfLiteError;
  }

  TF_LITE_ENSURE_STATUS(PrepareOpsAndTensors());
  invokable_ = true;
  return kTfLiteOk;
}

TfLiteStatus Interpreter::AddNodeWithParameters(
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const char* init_data, size_t init_data_size, void* builtin_data,
    const TfLiteRegistration* registration, int* node_index) {
  invokable_ = false;

  std::unique_ptr<void, decltype(free)*> builtin_data_deleter(builtin_data,
                                                              free);

  TF_LITE_ENSURE_OK(&context_, CheckTensorIndices("node inputs", inputs.data(),
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
  // properly for nodes generated by ReplaceSubgraphsWithDelegateKernels.

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

TfLiteStatus Interpreter::ResizeInputTensor(int tensor_index,
                                            const std::vector<int>& dims) {
  // TODO(aselle): All bounds checks can be implemented as one-sided bounds
  // checks by casting to unsigned for efficiency. Profile before doing this.

  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);
  invokable_ = false;
  TfLiteIntArray* dims_lite = ConvertVectorToTfLiteIntArray(dims);
  return ResizeTensorImpl(&context_.tensors[tensor_index], dims_lite);
}

// Returns true if at least one tensor in the given list is kTfLiteDynamic.
bool HasDynamicTensor(const TfLiteContext& context,
                      const TfLiteIntArray* tensors) {
  for (int i = 0; i < tensors->size; ++i) {
    const TfLiteTensor& tensor = context.tensors[tensors->data[i]];
    if (tensor.allocation_type == kTfLiteDynamic) {
      return true;
    }
  }
  return false;
}

TfLiteStatus Interpreter::PrepareOpsStartingAt(
    int first_execution_plan_index, int* last_execution_plan_index_prepared) {
  for (int execution_plan_index = first_execution_plan_index;
       execution_plan_index < execution_plan_.size(); execution_plan_index++) {
    int node_index = execution_plan_[execution_plan_index];
    TfLiteNode& node = nodes_and_registration_[node_index].first;
    const TfLiteRegistration& registration =
        nodes_and_registration_[node_index].second;
    EnsureTensorsVectorCapacity();
    if (OpPrepare(registration, &node) == kTfLiteError) {
      return kTfLiteError;
    }

    *last_execution_plan_index_prepared = execution_plan_index;

    // Discontinue if the node has dynamic outputs. Note that we don't
    // stop for dynamic temporary tensors since they won't affect the
    // sizes of other tensors in the graph.
    if (HasDynamicTensor(context_, node.outputs)) {
      break;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Interpreter::PrepareOpsAndTensors() {
  if (!memory_planner_) {
    memory_planner_.reset(new ArenaPlanner(
        &context_, std::unique_ptr<GraphInfo>(new InterpreterInfo(this))));
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

TfLiteStatus Interpreter::Invoke() {
  if (!consistent_) {
    ReportError(&context_, "Invoke called on model that is not consistent.");
    return kTfLiteError;
  }
  if (!invokable_) {
    ReportError(&context_, "Invoke called on model that is not ready.");
    return kTfLiteError;
  }

  TfLiteStatus status = kTfLiteOk;
  if (nnapi_delegate_) {
    if (next_execution_plan_index_to_prepare_ == execution_plan_.size()) {
      TF_LITE_ENSURE_OK(&context_, nnapi_delegate_->Invoke(this));
      return kTfLiteOk;
    } else {
      // TODO(aselle): In the future, we would like this to be an
      // automatic tflite CPU fallback.
      ReportError(&context_,
                  "NNAPI was requested, but dependent sized tensors "
                  "being used.\n");
      return kTfLiteError;
    }
  }

  // Invocations are always done in node order.
  // Note that calling Invoke repeatedly will cause the original memory plan to
  // be reused, unless either ResizeInputTensor() or AllocateTensors() has been
  // called.
  // TODO(b/71913981): we should force recalculation in the presence of dynamic
  // tensors, because they may have new value which in turn may affect shapes
  // and allocations.
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
    if (OpInvoke(registration, &node) == kTfLiteError) {
      status = kTfLiteError;
    }
  }

  return status;
}

TfLiteStatus Interpreter::ResizeTensor(TfLiteContext* context,
                                       TfLiteTensor* tensor,
                                       TfLiteIntArray* new_size) {
  // Note here that context->impl_ is recovering the this pointer for an
  // instance of Interpreter to call into the member function ResizeTensorImpl
  // (this function is static).
  return static_cast<Interpreter*>(context->impl_)
      ->ResizeTensorImpl(tensor, new_size);
}

void Interpreter::ReportErrorImpl(const char* format, va_list args) {
  error_reporter_->Report(format, args);
}

void Interpreter::ReportError(TfLiteContext* context, const char* format, ...) {
  va_list args;
  va_start(args, format);
  auto* f = static_cast<Interpreter*>(context->impl_);
  // Note here that context->impl_ is recovering the this pointer for an
  // instance of Interpreter to call into the member function ReportErrorImpl
  // (this function is static).
  f->ReportErrorImpl(format, args);
  va_end(args);
}

TfLiteStatus Interpreter::AddTensors(int tensors_to_add,
                                     int* first_new_tensor_index) {
  int base_index = tensors_.size();
  if (first_new_tensor_index) *first_new_tensor_index = base_index;
  tensors_.resize(tensors_.size() + tensors_to_add);
  for (int i = base_index; i < tensors_.size(); i++) {
    memset(&tensors_[i], 0, sizeof(tensors_[i]));
    tensors_[i].buffer_handle = kTfLiteNullBufferHandle;
  }
  context_.tensors = tensors_.data();
  context_.tensors_size = tensors_.size();
  return kTfLiteOk;
}

TfLiteStatus Interpreter::AddTensors(TfLiteContext* context, int tensors_to_add,
                                     int* first_new_tensor_index) {
  // Note here that context->impl_ is recovering the this pointer for an
  // instance of Interpreter to call into the member function AddTensors
  // (this function is static).
  return static_cast<Interpreter*>(context->impl_)
      ->AddTensors(tensors_to_add, first_new_tensor_index);
}

TfLiteStatus Interpreter::GetNodeAndRegistration(
    int node_index, TfLiteNode** node, TfLiteRegistration** registration) {
  TF_LITE_ENSURE(&context_, node_index < nodes_size() && node_index >= 0);
  TF_LITE_ENSURE(&context_, node != nullptr && registration != nullptr);
  *node = &nodes_and_registration_[node_index].first;
  *registration = &nodes_and_registration_[node_index].second;
  return kTfLiteOk;
}

TfLiteStatus Interpreter::GetNodeAndRegistration(
    struct TfLiteContext* context, int node_index, TfLiteNode** node,
    TfLiteRegistration** registration) {
  return static_cast<Interpreter*>(context->impl_)
      ->GetNodeAndRegistration(node_index, node, registration);
}

TfLiteStatus Interpreter::SetTensorParametersReadOnly(
    int tensor_index, TfLiteType type, const char* name, const int rank,
    const int* dims, TfLiteQuantizationParams quantization, const char* buffer,
    size_t bytes, const Allocation* allocation) {
  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);
  // For most tensors we know exactly how much memory is necessary so we can
  // ensure the buffer is large enough. However, we need to skip string tensors
  // because their sizes change with the contents of the individual strings.
  if (type != kTfLiteString) {
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
    tensor.data.raw = const_cast<char*>(buffer);
    if (!tensor.dims) tensor.dims = ConvertArrayToTfLiteIntArray(rank, dims);
    tensor.params = quantization;
    tensor.allocation_type = kTfLiteMmapRo;
    tensor.allocation = allocation;
  } else {
    invokable_ = false;
    TfLiteTensorReset(type, name, ConvertArrayToTfLiteIntArray(rank, dims),
                      quantization, const_cast<char*>(buffer), bytes,
                      kTfLiteMmapRo, allocation, &tensor);
  }
  return kTfLiteOk;
}

// Set description of inputs/outputs/data/fptrs for node `node_index`.
// This variant assumes an external buffer has been allocated of size
// bytes. The lifetime of buffer must be ensured to be greater or equal
// to Interpreter.
TfLiteStatus Interpreter::SetTensorParametersReadWrite(
    int tensor_index, TfLiteType type, const char* name, const int rank,
    const int* dims, TfLiteQuantizationParams quantization) {
  invokable_ = false;
  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);
  size_t required_bytes = 0;
  if (type != kTfLiteString) {
    // These types will be allocated in our arena so we need to record how
    // many bytes we will need based on the dimensions. String tensors are
    // allocated dynamically and we can't know ahead of time how much space
    // they will require.
    TF_LITE_ENSURE_OK(&context_,
                      BytesRequired(type, dims, rank, &required_bytes));
  }
  TfLiteTensorReset(type, name, ConvertArrayToTfLiteIntArray(rank, dims),
                    quantization,
                    /*buffer=*/nullptr, required_bytes,
                    type == kTfLiteString ? kTfLiteDynamic : kTfLiteArenaRw,
                    nullptr, &context_.tensors[tensor_index]);
  return kTfLiteOk;
}

TfLiteStatus Interpreter::SetExecutionPlan(const std::vector<int>& new_plan) {
  for (int node_index : new_plan) {
    TF_LITE_ENSURE(&context_, node_index >= 0 && node_index < nodes_size());
  }
  execution_plan_ = new_plan;
  return kTfLiteOk;
}

TfLiteStatus Interpreter::ResizeTensorImpl(TfLiteTensor* tensor,
                                           TfLiteIntArray* new_size) {
  // Note that in theory we could resize kTfLiteArenaRwPersistent tensors too.
  if (tensor->allocation_type == kTfLiteArenaRw ||
      tensor->allocation_type == kTfLiteDynamic) {
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
    ReportError(&context_, "Attempting to resize a fixed-size tensor.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

void Interpreter::UseNNAPI(bool enable) {
  // TODO(aselle): This is a workaround for finding if NNAPI exists.
  // We also need to make sure getLibraryHandle() is renamed to be NNAPI
  // prefixed.
  if (!NNAPIExists()) enable = false;
  if (!enable) {
    nnapi_delegate_.reset();
  } else if (!nnapi_delegate_) {
    nnapi_delegate_.reset(new NNAPIDelegate);
  }
}

void Interpreter::SetNumThreads(int num_threads) {
  context_.recommended_num_threads = num_threads;
}

TfLiteStatus Interpreter::ModifyGraphWithDelegate(TfLiteDelegate* delegate) {
  // TODO(aselle): Consider if it is worth storing pointers to delegates.
  // Setup additional context interface
  context_.GetNodeAndRegistration = GetNodeAndRegistration;
  context_.ReplaceSubgraphsWithDelegateKernels =
      ReplaceSubgraphsWithDelegateKernels;
  context_.GetExecutionPlan = GetExecutionPlan;

  TfLiteStatus status = delegate->Prepare(&context_, delegate);
  // Remove additional context info.
  SetForbiddenContextFunction(&context_.GetNodeAndRegistration);
  SetForbiddenContextFunction(&context_.ReplaceSubgraphsWithDelegateKernels);
  SetForbiddenContextFunction(&context_.GetExecutionPlan);
  return status;
}

TfLiteStatus Interpreter::SetBufferHandle(int tensor_index,
                                          TfLiteBufferHandle buffer_handle,
                                          TfLiteDelegate* delegate) {
  TF_LITE_ENSURE(&context_, tensor_index < tensors_size());
  TfLiteTensor* tensor = &tensors_[tensor_index];

  TF_LITE_ENSURE(&context_,
                 tensor->delegate == nullptr || tensor->delegate == delegate);
  tensor->delegate = delegate;
  if (tensor->buffer_handle != kTfLiteNullBufferHandle) {
    TF_LITE_ENSURE(&context_, tensor->delegate->FreeBufferHandle != nullptr);
    tensor->delegate->FreeBufferHandle(tensor->delegate,
                                       &tensor->buffer_handle);
  }
  tensor->buffer_handle = buffer_handle;

  return kTfLiteOk;
}

TfLiteStatus Interpreter::GetBufferHandle(int tensor_index,
                                          TfLiteBufferHandle* buffer_handle,
                                          TfLiteDelegate** delegate) {
  TF_LITE_ENSURE(&context_, tensor_index < tensors_size());
  TfLiteTensor* tensor = &tensors_[tensor_index];

  *delegate = tensor->delegate;
  *buffer_handle = tensor->buffer_handle;

  return kTfLiteOk;
}

}  // namespace tflite
