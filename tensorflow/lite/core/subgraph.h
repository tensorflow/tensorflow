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
#ifndef TENSORFLOW_LITE_CORE_SUBGRAPH_H_
#define TENSORFLOW_LITE_CORE_SUBGRAPH_H_

#include <stdarg.h>
#include <stddef.h>

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/experimental/resource/initialization_status.h"
#include "tensorflow/lite/experimental/resource/resource_base.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/util.h"

namespace tflite {

class SingleOpModel;  // Class for friend declarations.

namespace delegates {
namespace test_utils {
class TestDelegate;  // Class for friend declarations.
}  // namespace test_utils
}  // namespace delegates

class Subgraph {
 public:
  friend class Interpreter;
  friend class SingleOpModel;

  Subgraph(ErrorReporter* error_reporter,
           TfLiteExternalContext** external_contexts,
           std::vector<std::unique_ptr<Subgraph>>* subgraphs,
           resource::ResourceMap* resources,
           resource::ResourceIDMap* resource_ids,
           resource::InitializationStatusMap* initialization_status_map);

  Subgraph(const Subgraph&) = delete;

  // Subgraphs should be movable but not copyable.
  Subgraph(Subgraph&&) = default;
  Subgraph& operator=(const Subgraph&) = delete;
  virtual ~Subgraph();

  // Provide a list of tensor indexes that are inputs to the model.
  // Each index is bound check and this modifies the consistent_ flag of the
  // interpreter.
  TfLiteStatus SetInputs(std::vector<int> inputs);

  // Provide a list of tensor indexes that are outputs to the model
  // Each index is bound check and this modifies the consistent_ flag of the
  // interpreter.
  TfLiteStatus SetOutputs(std::vector<int> outputs);

  // Provide a list of tensor indexes that are variable tensors.
  // Each index is bound check and this modifies the consistent_ flag of the
  // interpreter.
  TfLiteStatus SetVariables(std::vector<int> variables);

  // Adds a node with the given parameters and returns the index of the new
  // node in `node_index` (optionally). Interpreter will take ownership of
  // `builtin_data` and destroy it with `free`. Ownership of 'init_data'
  // remains with the caller.
  TfLiteStatus AddNodeWithParameters(const std::vector<int>& inputs,
                                     const std::vector<int>& outputs,
                                     const std::vector<int>& intermediates,
                                     const char* init_data,
                                     size_t init_data_size, void* builtin_data,
                                     const TfLiteRegistration* registration,
                                     int* node_index = nullptr);

  // Adds `tensors_to_add` tensors, preserving pre-existing Tensor entries.
  // The value pointed to by `first_new_tensor_index` will be set to the
  // index of the first new tensor if `first_new_tensor_index` is non-null.
  TfLiteStatus AddTensors(int tensors_to_add,
                          int* first_new_tensor_index = nullptr);

  // Set description of inputs/outputs/data/fptrs for node `node_index`.
  // This variant assumes an external buffer has been allocated of size
  // bytes. The lifetime of buffer must be ensured to be greater or equal
  // to Interpreter. `quantization` ownership is passed to the subgraph.
  inline TfLiteStatus SetTensorParametersReadOnly(
      int tensor_index, TfLiteType type, const char* name,
      const std::vector<int>& dims, TfLiteQuantization quantization,
      const char* buffer, size_t bytes, const Allocation* allocation = nullptr,
      TfLiteSparsity* sparsity = nullptr) {
    return SetTensorParametersReadOnly(tensor_index, type, name, dims.size(),
                                       dims.data(), quantization, buffer, bytes,
                                       allocation, sparsity);
  }
  TfLiteStatus SetTensorParametersReadOnly(
      int tensor_index, TfLiteType type, const char* name, const size_t rank,
      const int* dims, TfLiteQuantization quantization, const char* buffer,
      size_t bytes, const Allocation* allocation = nullptr,
      TfLiteSparsity* sparsity = nullptr);

  // Set description of inputs/outputs/data/fptrs for node `node_index`.
  // This variant assumes an external buffer has been allocated of size
  // bytes. The lifetime of buffer must be ensured to be greater or equal
  // to Interpreter. `quantization` ownership is passed to the subgraph.
  inline TfLiteStatus SetTensorParametersReadWrite(
      int tensor_index, TfLiteType type, const char* name,
      const std::vector<int>& dims, TfLiteQuantization quantization,
      bool is_variable = false, const std::vector<int>& dims_signature = {}) {
    if (dims_signature.empty()) {
      return SetTensorParametersReadWrite(tensor_index, type, name, dims.size(),
                                          dims.data(), quantization,
                                          is_variable);
    }
    return SetTensorParametersReadWrite(
        tensor_index, type, name, dims.size(), dims.data(), quantization,
        is_variable, dims_signature.size(), dims_signature.data());
  }
  TfLiteStatus SetTensorParametersReadWrite(
      int tensor_index, TfLiteType type, const char* name, const size_t rank,
      const int* dims, TfLiteQuantization quantization,
      bool is_variable = false, const size_t rank_dims_signature = 0,
      const int* dims_signature = nullptr);

  // Get a mutable tensor data structure.
  TfLiteTensor* tensor(int tensor_index) {
    if (tensor_index < 0 ||
        static_cast<size_t>(tensor_index) >= context_.tensors_size) {
      return nullptr;
    }
    return &context_.tensors[tensor_index];
  }

  // Get an immutable tensor data structure.
  const TfLiteTensor* tensor(int tensor_index) const {
    if (tensor_index < 0 ||
        static_cast<size_t>(tensor_index) >= context_.tensors_size) {
      return nullptr;
    }
    return &context_.tensors[tensor_index];
  }

  // Read only access to list of inputs.
  std::vector<int>& inputs() { return inputs_; }

  // Read only access to list of inputs.
  const std::vector<int>& inputs() const { return inputs_; }

  // Read only access to list of outputs.
  std::vector<int>& outputs() { return outputs_; }

  // Read only access to list of outputs.
  const std::vector<int>& outputs() const { return outputs_; }

  // Read only access to list of variable tensors.
  std::vector<int>& variables() { return variables_; }

  // Read only access to list of variable tensors.
  const std::vector<int>& variables() const { return variables_; }

  // WARNING: Experimental interface, subject to change.
  // TODO(ycling): Move this function to an external context interface.
  resource::ResourceMap& resources() { return *resources_; }

  // WARNING: Experimental interface, subject to change.
  // TODO(b/149099381): Move this function to an external context interface.
  resource::ResourceIDMap& resource_ids() { return *resource_ids_; }

  // WARNING: Experimental interface, subject to change.
  // TODO(b/149099381): Move this function to an external context interface.
  resource::InitializationStatusMap& initialization_status_map() {
    return *initialization_status_map_;
  }

  size_t tensors_size() const { return tensors_.size(); }

  // Return the number of ops in the model.
  size_t nodes_size() const { return nodes_and_registration_.size(); }

  // Return vector of node indices in the order of execution.
  std::vector<int>& execution_plan() { return execution_plan_; }

  // Return read-only vector of node indices in the order of execution.
  const std::vector<int>& execution_plan() const { return execution_plan_; }

  const std::vector<std::pair<TfLiteNode, TfLiteRegistration>>&
  nodes_and_registration() const {
    return nodes_and_registration_;
  }

  // Get a pointer to an operation and registration data structure if in bounds.
  const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration(
      int node_index) const {
    if (node_index < 0 || static_cast<size_t>(node_index) >= nodes_size())
      return nullptr;
    return &nodes_and_registration_[node_index];
  }

  // Change the dimensionality of a given tensor. Note, this is only acceptable
  // for tensor indices that are inputs.
  // Returns status of failure or success.
  TfLiteStatus ResizeInputTensor(int tensor_index,
                                 const std::vector<int>& dims);

  // WARNING: Experimental interface, subject to change
  // Change the dimensionality of a given tensor. This is only acceptable for
  // tensor indices that are inputs or variables. Only unknown dimensions can be
  // resized with this function. Unknown dimensions are indicated as `-1` in the
  // `dims_signature` attribute of a `TfLiteTensor`. Returns status of failure
  // or success.
  TfLiteStatus ResizeInputTensorStrict(int tensor_index,
                                       const std::vector<int>& dims);

  // This releases memory held by non-persistent tensors. It does NOT re-perform
  // memory planning.
  // AllocateTensors needs to be called before next invocation.
  TfLiteStatus ReleaseNonPersistentMemory();

  // Update allocations for all tensors. This will redim dependent tensors using
  // the input tensor dimensionality as given. This is relatively expensive.
  // If you know that your sizes are not changing, you need not call this.
  // Returns status of success or failure.
  TfLiteStatus AllocateTensors();

  // Invoke the subgraph (run the whole graph in dependency order).
  //
  // NOTE: It is possible that the interpreter is not in a ready state
  // to evaluate (i.e. if a ResizeTensor() has been performed without an
  // AllocateTensors().
  // Returns status of success or failure.
  TfLiteStatus Invoke();

  // Entry point for C node plugin API to report an error.
  void ReportError(const char* format, ...);

  // Return the subgraph specific context.
  TfLiteContext* context() { return &context_; }
  const TfLiteContext* context() const { return &context_; }

  // Set the value of an external context.
  void SetExternalContext(TfLiteExternalContextType type,
                          TfLiteExternalContext* ctx);
  // Get the half precision flag.
  // WARNING: This is an experimental API and subject to change.
  bool GetAllowFp16PrecisionForFp32() const {
    return context_.allow_fp32_relax_to_fp16;
  }

  // Sets the cancellation function pointer in order to cancel a request in the
  // middle of a call to Invoke(). The interpreter queries this function during
  // inference, between op invocations; when it returns true, the interpreter
  // will abort execution and return `kTfLiteError`. The `data` parameter
  // contains any data used by the cancellation function, and if non-null,
  // remains owned by the caller.
  // WARNING: This is an experimental API and subject to change.
  void SetCancellationFunction(void* data, bool (*check_cancelled_func)(void*));

  // Ensure the data in `tensor.data` is readable. In case delegate is used,
  // it might require to copy the data from delegate buffer to raw memory.
  // WARNING: This is an experimental API and subject to change.
  TfLiteStatus EnsureTensorDataIsReadable(int tensor_index) {
    TfLiteTensor* t = &tensors_[tensor_index];
    TF_LITE_ENSURE(&context_, t != nullptr);
    if (t->data_is_stale) {
      TF_LITE_ENSURE(&context_, t->delegate != nullptr);
      TF_LITE_ENSURE(&context_, t->buffer_handle != kTfLiteNullBufferHandle);
      TF_LITE_ENSURE(&context_, t->delegate->CopyFromBufferHandle != nullptr);
      TF_LITE_ENSURE_STATUS(t->delegate->CopyFromBufferHandle(
          &context_, t->delegate, t->buffer_handle, t));
      t->data_is_stale = false;
    }
    return kTfLiteOk;
  }

  // The default capacity of `tensors_` vector.
  static constexpr int kTensorsReservedCapacity = 128;
  // The capacity headroom of `tensors_` vector before calling ops'
  // `prepare` and `invoke` function. In these functions, it's guaranteed
  // allocating up to `kTensorsCapacityHeadroom` more tensors won't invalidate
  // pointers to existing tensors.
  static constexpr int kTensorsCapacityHeadroom = 16;

  // Reset all variable tensors to the default value.
  // If a variable tensor doesn't have a buffer, reset it to zero.
  // TODO(b/115961645): Implement - If a variable tensor has a buffer, reset it
  // to the value of the buffer.
  // WARNING: This is an experimental API and subject to change.
  TfLiteStatus ResetVariableTensors();

  void SetProfiler(Profiler* profiler, int associated_subgraph_idx) {
    if (!profiler) {
      profiler_.reset(nullptr);
      context_.profiler = nullptr;
    } else {
      profiler_.reset(
          new SubgraphAwareProfiler(profiler, associated_subgraph_idx));
      context_.profiler = profiler_.get();
    }
  }

  Profiler* GetProfiler() { return profiler_.get(); }

  // Returns a pointer to vector of subgraphs.
  // WARNING: This is an experimental API and subject to change.
  std::vector<std::unique_ptr<Subgraph>>* GetSubgraphs() { return subgraphs_; }

  // True if all tensors in the graph has static size after calling
  // `AllocateTensors` function.
  // Before `AllocateTensors` is called, this will always return true;
  bool HasDynamicTensors() { return has_dynamic_tensors_; }

  // Assigns (or reassigns) a custom memory allocation for the given tensor.
  // `flags` is a bitmask, see TfLiteCustomAllocationFlags.
  // The runtime does NOT take ownership of the underlying memory.
  //
  // NOTE: User needs to call AllocateTensors() after this.
  // Invalid/insufficient buffers will cause an error during AllocateTensors or
  // Invoke (in case of dynamic shapes in the graph).
  //
  // Parameters should satisfy the following conditions:
  // 1. tensor->allocation_type == kTfLiteArenaRw or kTfLiteArenaRwPersistent
  //    In general, this is true for I/O tensors & variable tensors.
  // 2. allocation->data has the appropriate permissions for runtime access
  //    (Read-only for inputs, Read-Write for others), and outlives Interpreter.
  // 3. allocation->bytes >= tensor->bytes.
  //    This condition is checked again if any tensors are resized.
  // 4. allocation->data should be aligned to kDefaultTensorAlignment
  //    defined in lite/util.h. (Currently 64 bytes)
  //    This check is skipped if kTfLiteCustomAllocationFlagsSkipAlignCheck is
  //    set through `flags`.
  // TODO(b/182215910): Expand on this documentation in a g3doc.
  //
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus SetCustomAllocationForTensor(
      int tensor_index, const TfLiteCustomAllocation& allocation,
      int64_t flags = kTfLiteCustomAllocationFlagsNone);

  void SetName(const char* name);
  const std::string& GetName() const;

  // WARNING: This is an experimental API and subject to change.
  // Dumps debugging info by the underlying memory planner.
  // Note: to have minimal binary increase caused by this debug info dump for
  // the TfLite library and allow users to plug-in their own memory planner
  // debugger, we have utilized weak symbols to meet these two requirements. By
  // default, there is no debugging info dumped. However, if the TfLite-provided
  // lite:simple_memory_arena_debug_dump (i.e. containing the strong defintion)
  // is linked to the program, calling this function will output memory usage
  // information about tenosrs and ops.
  void DumpMemoryPlannerDebugInfo() const;

  // WARNING: This is an experimental API and subject to change.
  // Force all intermediate dynamic tensors to be released once they are not
  // used by the model. Please use this configuration with caution, since it
  // might reduce the peak memory usage of the model at the cost of a slower
  // inference speed. This API needs to be called before calling
  // `AllocateTensors`.
  void EnsureDynamicTensorsAreReleased() {
    release_dynamic_tensors_if_unused_ = true;
  }

  // WARNING: This is an experimental API and subject to change.
  // Remove unused inputs of the subgraph. It checks usage of inputs and mark it
  // as kTfLiteOptionalTensor if the input is not used in graph execution.
  // Currently, it's used to remove unused inputs of WHILE cond subgraphs.
  TfLiteStatus RemoveUnusedInputs();

 private:
  friend class InterpreterBuilder;
  friend class TestDelegate;
  // SubgraphAwareProfiler wraps an actual TFLite profiler, such as a
  // BufferedProfiler instance, and takes care of event profiling/tracing in a
  // certain subgraph.
  class SubgraphAwareProfiler : public Profiler {
   public:
    // Constructor should be called with the non-nullptr profiler argument.
    SubgraphAwareProfiler(Profiler* profiler, int64_t subgraph_index)
        : profiler_(profiler), subgraph_index_(subgraph_index) {}
    ~SubgraphAwareProfiler() override {}

    uint32_t BeginEvent(const char* tag, EventType event_type,
                        int64_t event_metadata1,
                        int64_t event_metadata2) override {
      if (!profiler_) return 0;
      return profiler_->BeginEvent(tag, event_type, event_metadata1,
                                   subgraph_index_);
    }

    void EndEvent(uint32_t event_handle) override {
      if (!profiler_) return;
      profiler_->EndEvent(event_handle);
    }

    void EndEvent(uint32_t event_handle, int64_t event_metadata1,
                  int64_t event_metadata2) override {
      if (!profiler_) return;
      profiler_->EndEvent(event_handle, event_metadata1, event_metadata2);
    }

    void AddEvent(const char* tag, EventType event_type, uint64_t start,
                  uint64_t end, int64_t event_metadata1,
                  int64_t event_metadata2) override {
      if (!profiler_) return;
      profiler_->AddEvent(tag, event_type, start, end, event_metadata1,
                          subgraph_index_);
    }

   private:
    // Not own the memory.
    Profiler* const profiler_;
    const int64_t subgraph_index_;
  };

  // Ensure the internal node storage memory allocates at least `count`
  // spots for node. NOTE, this doesn't actually add operators. This is an
  // efficiency optimization that is subject to change.
  // Note: Only used during initialization.
  void ReserveNodes(int count);

  // Overrides execution plan. This bounds checks indices sent in.
  // Note: Only used during initialization.
  TfLiteStatus SetExecutionPlan(const std::vector<int>& new_plan);

  // Prevent 'context_' from accessing functions that are only available to
  // delegated kernels.
  void SwitchToKernelContext();

  // Add delegate-only functions to 'context_'.
  void SwitchToDelegateContext();

  // Give 'op_reg' a chance to initialize itself using the contents of
  // 'buffer'.
  void* OpInit(const TfLiteRegistration& op_reg, const char* buffer,
               size_t length) {
    if (op_reg.init == nullptr) return nullptr;
    return op_reg.init(&context_, buffer, length);
  }

  // Let 'op_reg' release any memory it might have allocated via 'OpInit'.
  void OpFree(const TfLiteRegistration& op_reg, void* buffer) {
    if (op_reg.free == nullptr) return;
    if (buffer) {
      op_reg.free(&context_, buffer);
    }
  }

  // Prepare the given 'node' for execution.
  TfLiteStatus OpPrepare(const TfLiteRegistration& op_reg, TfLiteNode* node);

  // Invoke the operator represented by 'node'.
  TfLiteStatus OpInvoke(const TfLiteRegistration& op_reg, TfLiteNode* node) {
    if (op_reg.invoke == nullptr) return kTfLiteError;
    return op_reg.invoke(&context_, node);
  }

  // Call OpPrepare() for as many ops as possible, allocating memory for their
  // tensors. If an op containing dynamic tensors is found, preparation will be
  // postponed until this function is called again. This allows the interpreter
  // to wait until Invoke() to resolve the sizes of dynamic tensors.
  TfLiteStatus PrepareOpsAndTensors();

  // Call OpPrepare() for all ops starting at 'first_node'. Stop when a
  // dynamic tensors is found or all ops have been prepared. Fill
  // 'last_node_prepared' with the id of the op containing dynamic tensors, or
  // the last in the graph.
  TfLiteStatus PrepareOpsStartingAt(int first_execution_plan_index,
                                    const std::vector<int>& execution_plan,
                                    int* last_execution_plan_index_prepared);

  // Tensors needed by the interpreter. Use `AddTensors` to add more blank
  // tensor entries. Note, `tensors_.data()` needs to be synchronized to the
  // `context_` whenever this std::vector is reallocated. Currently this
  // only happens in `AddTensors()`.
  std::vector<TfLiteTensor> tensors_;

  // Check if an array of tensor indices are valid with respect to the Tensor
  // array.
  // NOTE: this changes consistent_ to be false if indices are out of bounds.
  TfLiteStatus CheckTensorIndices(const char* label, const int* indices,
                                  int length);

  // Check that the input indices and the output indices don't overlap.
  // This is needed because same tensor must not be used both as input and
  // output for an operator.
  // NOTE: this changes consistent_ to be false if indices are out of bounds.
  TfLiteStatus CheckInputAndOutputForOverlap(const int* input_indices,
                                             int num_inputs,
                                             const int* output_indices,
                                             int num_outputs);

  // Compute the number of bytes required to represent a tensor with dimensions
  // specified by the array dims (of length dims_size). Returns the status code
  // and bytes.
  TfLiteStatus BytesRequired(TfLiteType type, const int* dims, size_t dims_size,
                             size_t* bytes);

  // Request an tensor be resized implementation. If the given tensor is of
  // type kTfLiteDynamic it will also be allocated new memory.
  TfLiteStatus ResizeTensorImpl(TfLiteTensor* tensor, TfLiteIntArray* new_size);

  // Report a detailed error string (will be printed to stderr).
  void ReportErrorImpl(const char* format, va_list args);

  // Entry point for C node plugin API to request an tensor be resized.
  static TfLiteStatus ResizeTensor(TfLiteContext* context, TfLiteTensor* tensor,
                                   TfLiteIntArray* new_size);
  // Entry point for C node plugin API to report an error.
  static void ReportErrorC(TfLiteContext* context, const char* format, ...);

  // Entry point for C node plugin API to add new tensors.
  static TfLiteStatus AddTensors(TfLiteContext* context, int tensors_to_add,
                                 int* first_new_tensor_index);

  // WARNING: This is an experimental API and subject to change.
  // Entry point for C API ReplaceNodeSubsetsWithDelegateKernels
  static TfLiteStatus ReplaceNodeSubsetsWithDelegateKernels(
      TfLiteContext* context, TfLiteRegistration registration,
      const TfLiteIntArray* nodes_to_replace, TfLiteDelegate* delegate);

  // Update the execution graph to replace some of the nodes with stub
  // nodes. Specifically any node index that has `nodes[index]==1` will be
  // slated for replacement with a delegate kernel specified by registration.
  // Ownership of 'nodes_to_replace' and 'delegate' remains with the caller.
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus ReplaceNodeSubsetsWithDelegateKernels(
      TfLiteRegistration registration, const TfLiteIntArray* nodes_to_replace,
      TfLiteDelegate* delegate);

  // WARNING: This is an experimental interface that is subject to change.
  // Gets the internal pointer to a TensorFlow lite node by node_index.
  TfLiteStatus GetNodeAndRegistration(int node_index, TfLiteNode** node,
                                      TfLiteRegistration** registration);

  // WARNING: This is an experimental interface that is subject to change.
  // Entry point for C node plugin API to get a node by index.
  static TfLiteStatus GetNodeAndRegistration(struct TfLiteContext*,
                                             int node_index, TfLiteNode** node,
                                             TfLiteRegistration** registration);

  // WARNING: This is an experimental interface that is subject to change.
  // Gets an TfLiteIntArray* representing the execution plan. The interpreter
  // owns this memory and it is only guaranteed to exist during the invocation
  // of the delegate prepare.
  TfLiteStatus GetExecutionPlan(TfLiteIntArray** execution_plan);

  // WARNING: This is an experimental interface that is subject to change.
  // Entry point for C node plugin API to get the execution plan.
  static TfLiteStatus GetExecutionPlan(struct TfLiteContext* context,
                                       TfLiteIntArray** execution_plan);

  // WARNING: This is an experimental interface that is subject to change.
  // Provides a preview of post-delegation partitioning. Each
  // TfLiteDelegateParams in the referenced array corresponds to one instance of
  // the delegate kernel.
  // nodes_to_replace should point to a valid array. partition_params_array &
  // num_partitions should be non-null.
  // Memory allocated by this method is automatically released with another call
  // to PreviewDelegateParitioning, or after TfLiteDelegate::Prepare is done.
  TfLiteStatus PreviewDelegatePartitioning(
      const TfLiteIntArray* nodes_to_replace,
      TfLiteDelegateParams** partition_params_array, int* num_partitions);

  // WARNING: This is an experimental interface that is subject to change.
  // Entry point for C node plugin API to preview delegation partitioning.
  static TfLiteStatus PreviewDelegatePartitioning(
      struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
      TfLiteDelegateParams** partition_params_array, int* num_partitions);

  // Retrieves named metadata from the TFLite model. Returns kTfLiteOk if
  // metadata is successfully obtained.
  // See the Metadata table in TFLite schema.
  TfLiteStatus GetModelMetadata(const char* name, const char** ptr,
                                size_t* bytes);

  // Entry point for C node plugin API to get model metadata based on name.
  static TfLiteStatus GetModelMetadata(const struct TfLiteContext* context,
                                       const char* name, const char** ptr,
                                       size_t* bytes);

  // Used to clear partitioning_preview_cache_, in case
  // PreviewDelegatePartitioning was called.
  void FreeDelegatePartitioningData();

  // Retrieve an existing external context by type.
  TfLiteExternalContext* GetExternalContext(TfLiteExternalContextType type);
  static TfLiteExternalContext* GetExternalContext(
      struct TfLiteContext* context, TfLiteExternalContextType type);

  // Set the value of an external context.
  static void SetExternalContext(struct TfLiteContext* context,
                                 TfLiteExternalContextType type,
                                 TfLiteExternalContext* ctx);

  // WARNING: This is an experimental API and subject to change.
  // Allow a delegate to look at the graph and modify the graph to handle
  // parts of the graph themselves. After this is called, the graph may
  // contain new nodes that replace 1 more nodes.
  // NOTE: If tensors were allocated prior to delegate application, they will
  // be reallocated if the graph was modified (i.e., the caller does *not* need
  // to explicitly call |AllocateTensors()| again). If tensors were unallocated,
  // they will remain unallocated after delegate application.
  // Returns one of the following status codes:
  // 1. kTfLiteOk: Delegation succeeded
  // 2. kTfLiteDelegateError: Delegation failed due to an error *in the
  // delegate*, or the delegate parameter was null. The Subgraph has been
  // restored to its pre-delegation state.
  // NOTE: This reverts all delegates previously applied to the Subgraph.
  // 3. kTfLiteApplicationError : Delegation failed to be applied due to the
  // incompatibility with the TF Lite runtime, e.g., the model graph is already
  // immutable when applying the delegate. However, the Subgraph is still in a
  // invokable state.
  // 4. kTfLiteUnresolvedOps: Delegation failed because the model has an
  // operator that cannot be resolved. This can happen when the op is not
  // registered or built with the TF Lite framework.
  // 5. kTfLiteError: Unexpected/runtime failure.
  TfLiteStatus ModifyGraphWithDelegate(TfLiteDelegate* delegate);

  // This un-applies all delegates that have been applied till now, but retains
  // pointers to them.
  // The old execution plan and nodes are restored.
  TfLiteStatus UndoAllDelegates();

  // This re-applies all delegates that were undone.
  // Does nothing if UndoAllDelegates wasn't previously called.
  TfLiteStatus RedoAllDelegates();

  // This removes all delegates.
  // The old execution plan and nodes are restored. The graph is invokable
  // afterwards.
  TfLiteStatus RemoveAllDelegates();

  // Returns true if the subgraph has delegates applied.
  bool HasDelegates();

  // Returns true if the subgraph has been fully delegated.
  bool IsFullyDelegated() const;

  // Cleanups up data reserved for the given node. Does not remove the {node,
  // registration} pair from nodes_and_registrations_.
  void CleanupNode(int node_index);

  // Ensures that `tensors_` has at least `kTensorsCapacityHeadroom` extra
  // capacity. Calling this function may invalidate existing pointers to
  // tensors. After calling this function, adding `kTensorsCapacityHeadroom`
  // more tensors won't invalidate the pointer to existing tensors.
  void EnsureTensorsVectorCapacity();

  // Ensures the memory required is planned and allocated.
  TfLiteStatus EnsureMemoryAllocations();

  // Returns true if cancellation function returns true.
  bool IsCancelled();

  // Enables preserving intermediates for debugging.
  TfLiteStatus PreserveAllTensorsExperimental();

  // Returns true if 'node' could have side effect (e.g. stateful op).
  // Note that any node that might update other tensors beside op's output
  // are considered to have side effect.
  // So control flow ops like 'If' and 'While' are considered to have
  // side effect because they can have ops that have side effect in the
  // condition and body subgraphs.
  bool OpMightHaveSideEffect(const TfLiteNode* node,
                             const TfLiteRegistration* registration) const;

  // Returns new GraphInfo object based on the current Subgraph.
  std::unique_ptr<GraphInfo> CreateGraphInfo();

  // Store a ptr to the model metadata owned by the Interpreter.
  // Since the lifetime of the Interpreter exceeds the Subgraph, metadata
  // remains valid for the latter's lifetime.
  // Also sets relevant fields on context_ based on known metadata.
  TfLiteStatus SetMetadata(const std::map<std::string, std::string>* metadata);

  // Initializes the mapping between tensor index to the index of the
  // last operation that uses the tensor as input.
  void InitializeTensorReleaseMap();

  // Checks the options for releasing dynamic tensors and release dynamic
  // tensors if configured.
  void MaybeReleaseDynamicInputs(const TfLiteNode& node, size_t node_index);

  // The state of the Interpreter.
  enum State {
    // The interpreter isn't ready to be invoked.
    // `AllocateTensor` need to be called to enter an invokable state.
    kStateUninvokable = 0,
    // The interpreter is ready to be invoked.
    kStateInvokable,
    // The interpreter is ready to be invoked, and graph can't be further
    // modified. The interpreter will enter this state when calling
    // `ModifyGraphWithDelegate` and the delegate doesn't support dynamic
    // tensors.
    kStateInvokableAndImmutable,
  };
  State state_ = kStateUninvokable;

  // A pure C data structure used to communicate with the pure C plugin
  // interface. To avoid copying tensor metadata, this is also the definitive
  // structure to store tensors.
  TfLiteContext context_ = {};

  // A pointer to the external contexts (kTfLiteMaxExternalContexts) array that
  // sits inside the associated TFLite interpreter instance.
  TfLiteExternalContext** external_contexts_;

  // Node inputs/outputs are stored in TfLiteNode and TfLiteRegistration stores
  // function pointers to actual implementation.
  // Nodes should appear in the order in which they are instantiated at runtime.
  // Delegated nodes are appended after all the original ones.
  std::vector<std::pair<TfLiteNode, TfLiteRegistration>>
      nodes_and_registration_;

  // Whether the model is consistent. That is to say if the inputs and outputs
  // of every node and the global inputs and outputs are valid indexes into
  // the tensor array.
  bool consistent_ = true;

  // Array of indices representing the tensors that are inputs to the
  // interpreter.
  std::vector<int> inputs_;

  // Array of indices representing the tensors that are outputs to the
  // interpreter.
  std::vector<int> outputs_;

  // Array of indices representing the tensors that are variable tensors.
  std::vector<int> variables_;

  // The error reporter delegate that tflite will forward queries errors to.
  ErrorReporter* error_reporter_;

  // Index of the next node to prepare.
  // During Invoke(), Interpreter will allocate input tensors first, which are
  // known to be fixed size. Then it will allocate outputs from nodes as many
  // as possible. When there is a node that produces dynamic sized tensor.
  // Interpreter will stop allocating tensors, set the value of next allocate
  // node id, and execute the node to generate the output tensor before continue
  // to allocate successors. This process repeats until all nodes are executed.
  // NOTE: this relies on the order of nodes that is in topological order.
  int next_execution_plan_index_to_prepare_;

  // Only used in cases where a delegate supporting dynamic tensors is applied.
  // This helps prepare the original execution before the post-delegation one,
  // so that tensor shapes propagate.
  int next_original_execution_plan_index_to_prepare_;

  // This is similar to `next_execution_plan_index_to_prepare_`, but it tracks
  // which nodes' allocation is planned with the arena planner.
  //
  // This is a workaround for b/127354079. It shouldn't be necessary if
  // ArenaPlanner can "rewind" to a specific point.
  // TODO(b/127354079): Improve ArenaPlanner and remove this mechanism.
  int next_execution_plan_index_to_plan_allocation_;

  // WARNING: This is an experimental interface that is subject to change.
  // This is a list of node indices (to index into nodes_and_registration).
  // This represents a valid topological sort (dependency ordered) execution
  // plan. In particular, it is valid for this ordering to contain only a
  // subset of the node indices.
  std::vector<int> execution_plan_;

  // This is a copy of the first execution_plan_ before any delegates were
  // applied. It is empty if no delegates were applied to this Subgraph.
  std::vector<int> pre_delegation_execution_plan_;

  // Contains a list of delegates applied by the user so far, in order.
  std::vector<TfLiteDelegate*> delegates_applied_;

  // Set to true if UndoAllDelegates was called, and to false during
  // RedoAllDelegates.
  bool delegates_undone_ = false;

  // In the future, we'd like a TfLiteIntArray compatible representation.
  // TODO(aselle): replace execution_plan_ with this.
  std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> plan_cache_;

  // Used by PreviewDelegateParitioning.
  std::vector<TfLiteDelegateParams> partitioning_preview_cache_;

  std::unique_ptr<MemoryPlanner> memory_planner_;

  // Maps tensor index to custom allocation for all applicable tensors.
  std::map<int, TfLiteCustomAllocation> custom_allocations_;

  // Tracking bit for whether a tensor was resized in the course of an op
  // invocation. This is a useful hint to ensure that dynamic tensor outputs
  // trigger downstream reallocation after op invocation.
  bool tensor_resized_since_op_invoke_ = false;

  // Profiler for this interpreter instance.
  std::unique_ptr<SubgraphAwareProfiler> profiler_;

  // A pointer to vector of subgraphs. The vector is owned by the interpreter.
  std::vector<std::unique_ptr<Subgraph>>* subgraphs_ = nullptr;

  // True if not all tensors in the graph has static size after calling
  // `PrepareOpsStartingAt` function (which is called by the `AllocateTensors`
  // public function).
  // The value is invalid before `PrepareOpStartingAt` is called.
  bool has_dynamic_tensors_ = true;

  // WARNING: This is an experimental interface that is subject to change.
  // This is the index of dynamic tensor which was checked at
  // PrepareOpsStartingAt() when `has_dynamic_tensors_` is set. This information
  // is kept only for user error message.
  int dynamic_tensor_index_ = -1;

  // Reference to cancellation function that can cancel a request in the middle
  // of a call to Invoke(). When this function returns True, a kTfLiteError is
  // thrown by Invoke().
  bool (*check_cancelled_func_)(void*) = nullptr;

  // Reference to data used by the cancellation function in
  // `check_cancelled_func_`.
  void* cancellation_data_ = nullptr;

  // A map of resources. Owned by interpreter and shared by multiple subgraphs.
  resource::ResourceMap* resources_ = nullptr;

  // A map of resources IDs. Owned by interpreter and shared by multiple
  // subgraphs.
  resource::ResourceIDMap* resource_ids_ = nullptr;

  // A map of initialization statuses, that indicate whether the intialization
  // subgraph invocation is done or not.
  resource::InitializationStatusMap* initialization_status_map_;

  // Name of the subgraph (analogous to function name).
  std::string name_;

  // Whether memory planner should be instantiated to retain intermediates for
  // debugging.
  bool preserve_all_tensors_ = false;

  // Model-metadata owned by the Interpreter.
  const std::map<std::string, std::string>* metadata_ = nullptr;

  // Release dynamic tensor's memory once they are not used by the graph.
  bool release_dynamic_tensors_if_unused_ = false;

  // Mapping between tensor index to the last index of the execution plan that
  // uses this tensor.
  std::map<int, int> tensor_to_last_op_index_;
};

}  // namespace tflite
#endif  // TENSORFLOW_LITE_CORE_SUBGRAPH_H_
