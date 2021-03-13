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
/// \file
/// Main abstraction controlling the tflite interpreter.
/// See context.h for the API for defining operations (TfLiteRegistration).
#ifndef TENSORFLOW_LITE_INTERPRETER_H_
#define TENSORFLOW_LITE_INTERPRETER_H_

#include <stddef.h>
#include <stdint.h>

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/common.h"  // IWYU pragma: export
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource/resource_base.h"
#include "tensorflow/lite/external_cpu_backend_context.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/type_to_tflitetype.h"

namespace tflite {

class InterpreterTest;  // Class for friend declarations.

namespace delegates {
class InterpreterUtils;  // Class for friend declarations.

namespace test_utils {
class TestDelegate;  // Class for friend declarations.
}  // namespace test_utils
}  // namespace delegates

/// An interpreter for a graph of nodes that input and output from tensors.
/// Each node of the graph processes a set of input tensors and produces a
/// set of output Tensors. All inputs/output tensors are referenced by index.
///
/// Usage:
///
/// <pre><code>
/// // Create model from file. Note that the model instance must outlive the
/// // interpreter instance.
/// auto model = tflite::FlatBufferModel::BuildFromFile(...);
/// if (model == nullptr) {
///   // Return error.
/// }
/// // Create an Interpreter with an InterpreterBuilder.
/// std::unique_ptr<tflite::Interpreter> interpreter;
/// tflite::ops::builtin::BuiltinOpResolver resolver;
/// if (InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
///   // Return failure.
/// }
/// interpreter->AllocateTensors();
///
/// auto input = interpreter->typed_tensor<float>(0);
/// for (int i = 0; i < input_size; i++) {
///   input[i] = ...;
//  }
/// interpreter.Invoke();
/// </code></pre>
///
/// Note: for nearly all practical use cases, one should not directly construct
/// an Interpreter object, but rather use the InterpreterBuilder.

class Interpreter {
 public:
  // Instantiate an interpreter. All errors associated with reading and
  // processing this model will be forwarded to the error_reporter object.
  //
  // Note, if error_reporter is nullptr, then a default StderrReporter is
  // used. Ownership of 'error_reporter' remains with the caller.
  // WARNING: Use of this constructor outside of an InterpreterBuilder is not
  // recommended.
  explicit Interpreter(ErrorReporter* error_reporter = DefaultErrorReporter());

  ~Interpreter();

  // Interpreters are not copyable as they have non-trivial memory semantics.
  Interpreter(const Interpreter&) = delete;
  Interpreter& operator=(const Interpreter&) = delete;

  // Functions to build interpreter
#ifndef DOXYGEN_SKIP
  /// Provide a list of tensor indexes that are inputs to the model.
  /// Each index is bound check and this modifies the consistent_ flag of the
  /// interpreter.
  TfLiteStatus SetInputs(std::vector<int> inputs);

  /// Provide a list of tensor indexes that are outputs to the model
  /// Each index is bound check and this modifies the consistent_ flag of the
  /// interpreter.
  TfLiteStatus SetOutputs(std::vector<int> outputs);

  /// Provide a list of tensor indexes that are variable tensors.
  /// Each index is bound check and this modifies the consistent_ flag of the
  /// interpreter.
  TfLiteStatus SetVariables(std::vector<int> variables);

  /// Ensure the internal node storage memory allocates at least `count`
  /// spots for node. NOTE, this doesn't actually add operators. This is an
  /// efficiency optimization that is subject to change.
  void ReserveNodes(int count);

  /// Adds a node with the given parameters and returns the index of the new
  /// node in `node_index` (optionally). Interpreter will take ownership of
  /// `builtin_data` and destroy it with `free`. Ownership of 'init_data'
  /// remains with the caller.
  TfLiteStatus AddNodeWithParameters(const std::vector<int>& inputs,
                                     const std::vector<int>& outputs,
                                     const char* init_data,
                                     size_t init_data_size, void* builtin_data,
                                     const TfLiteRegistration* registration,
                                     int* node_index = nullptr);

  /// Adds `tensors_to_add` tensors, preserving pre-existing Tensor entries.
  /// The value pointed to by `first_new_tensor_index` will be set to the
  /// index of the first new tensor if `first_new_tensor_index` is non-null.
  TfLiteStatus AddTensors(int tensors_to_add,
                          int* first_new_tensor_index = nullptr);

  /// Set description of inputs/outputs/data/fptrs for node `node_index`.
  /// This variant assumes an external buffer has been allocated of size
  /// bytes. The lifetime of buffer must be ensured to be greater or equal
  /// to Interpreter.
  TfLiteStatus SetTensorParametersReadOnly(
      int tensor_index, TfLiteType type, const char* name,
      const std::vector<int>& dims, TfLiteQuantization quantization,
      const char* buffer, size_t bytes, const Allocation* allocation = nullptr);

  /// Legacy. Deprecated in favor of above.
  inline TfLiteStatus SetTensorParametersReadOnly(
      int tensor_index, TfLiteType type, const char* name,
      const std::vector<int>& dims, TfLiteQuantizationParams quantization,
      const char* buffer, size_t bytes,
      const Allocation* allocation = nullptr) {
    return SetTensorParametersReadOnly(tensor_index, type, name, dims.size(),
                                       dims.data(), quantization, buffer, bytes,
                                       allocation);
  }

  TfLiteStatus SetTensorParametersReadOnly(
      int tensor_index, TfLiteType type, const char* name, const size_t rank,
      const int* dims, TfLiteQuantizationParams quantization,
      const char* buffer, size_t bytes, const Allocation* allocation = nullptr);

  /// Set description of inputs/outputs/data/fptrs for node `node_index`.
  /// This variant assumes an external buffer has been allocated of size
  /// bytes. The lifetime of buffer must be ensured to be greater or equal
  /// to Interpreter.
  TfLiteStatus SetTensorParametersReadWrite(int tensor_index, TfLiteType type,
                                            const char* name,
                                            const std::vector<int>& dims,
                                            TfLiteQuantization quantization,
                                            bool is_variable = false);

  /// Legacy. Deprecated in favor of above.
  inline TfLiteStatus SetTensorParametersReadWrite(
      int tensor_index, TfLiteType type, const char* name,
      const std::vector<int>& dims, TfLiteQuantizationParams quantization,
      bool is_variable = false,
      const std::vector<int>* dims_signature = nullptr) {
    size_t rank_dims_signature = 0;
    const int* dims_signature_pointer = nullptr;
    if (dims_signature) {
      rank_dims_signature = dims_signature->size();
      dims_signature_pointer = dims_signature->data();
    }
    return SetTensorParametersReadWrite(
        tensor_index, type, name, dims.size(), dims.data(), quantization,
        is_variable, rank_dims_signature, dims_signature_pointer);
  }
  TfLiteStatus SetTensorParametersReadWrite(
      int tensor_index, TfLiteType type, const char* name, const size_t rank,
      const int* dims, TfLiteQuantizationParams quantization,
      bool is_variable = false, const size_t rank_dims_signature = 0,
      const int* dims_signature = nullptr);
#endif  // DOXYGEN_SKIP
  // Functions to access tensor data

  /// Read only access to list of inputs.
  const std::vector<int>& inputs() const { return primary_subgraph().inputs(); }

  /// Return the name of a given input. The given index must be between 0 and
  /// inputs().size().
  const char* GetInputName(int index) const {
    return context_->tensors[inputs()[index]].name;
  }

  /// Read only access to list of outputs.
  const std::vector<int>& outputs() const {
    return primary_subgraph().outputs();
  }

  /// Read only access to list of variable tensors.
  const std::vector<int>& variables() const {
    return primary_subgraph().variables();
  }

  /// Return the name of a given output. The given index must be between 0 and
  /// outputs().size().
  const char* GetOutputName(int index) const {
    return context_->tensors[outputs()[index]].name;
  }

  /// Return the number of tensors in the model.
  size_t tensors_size() const { return context_->tensors_size; }

  /// Return the number of ops in the model.
  size_t nodes_size() const { return primary_subgraph().nodes_size(); }

  /// WARNING: Experimental interface, subject to change
  const std::vector<int>& execution_plan() const {
    return primary_subgraph().execution_plan();
  }

#ifndef DOXYGEN_
  /// WARNING: Experimental interface, subject to change
  /// Overrides execution plan. This bounds checks indices sent in.
  TfLiteStatus SetExecutionPlan(const std::vector<int>& new_plan);
#endif  // DOXYGEN_SKIP

  /// Get a mutable tensor data structure.
  // TODO(aselle): Create a safe ArrayHandle interface to avoid exposing this
  // read/write access to structure
  TfLiteTensor* tensor(int tensor_index) {
    return primary_subgraph().tensor(tensor_index);
  }

  /// Get an immutable tensor data structure.
  const TfLiteTensor* tensor(int tensor_index) const {
    return primary_subgraph().tensor(tensor_index);
  }

  /// Get a pointer to an operation and registration data structure if in
  /// bounds.
  const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration(
      int node_index) const {
    return primary_subgraph().node_and_registration(node_index);
  }

  /// Perform a checked cast to the appropriate tensor type (mutable pointer
  /// version).
  template <class T>
  T* typed_tensor(int tensor_index) {
    if (TfLiteTensor* tensor_ptr = tensor(tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return reinterpret_cast<T*>(tensor_ptr->data.raw);
      }
    }
    return nullptr;
  }

  /// Perform a checked cast to the appropriate tensor type (immutable pointer
  /// version).
  template <class T>
  const T* typed_tensor(int tensor_index) const {
    if (const TfLiteTensor* tensor_ptr = tensor(tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return reinterpret_cast<const T*>(tensor_ptr->data.raw);
      }
    }
    return nullptr;
  }

  /// WARNING: Experimental interface, subject to change
  /// Returns list of all names of different method signatures defined
  /// in the model.
  /// Note, pointers returned have lifetime same as the Interpreter object.
  std::vector<const std::string*> signature_def_names() const {
    std::vector<const std::string*> method_names;
    method_names.reserve(signature_defs_.size());
    for (const auto& sig_def : signature_defs_) {
      method_names.emplace_back(&sig_def.method_name);
    }
    return method_names;
  }

  /// WARNING: Experimental interface, subject to change
  /// Returns the mapping of inputs to tensor index in the signature
  /// specified through 'method_name'.
  /// If invalid name passed, an empty list will be returned.
  const std::map<std::string, uint32_t>& signature_inputs(
      const char* method_name) const {
    for (const auto& sig_def : signature_defs_) {
      if (sig_def.method_name == method_name) return sig_def.inputs;
    }
    static const std::map<std::string, uint32_t>* default_empty_list =
        new std::map<std::string, uint32_t>();
    return *default_empty_list;
  }

  /// WARNING: Experimental interface, subject to change
  /// Returns the mapping of outputs to tensor index in the signature
  /// specified through 'method_name'.
  /// If invalid name passed, an empty list will be returned.
  const std::map<std::string, uint32_t>& signature_outputs(
      const char* method_name) const {
    for (const auto& sig_def : signature_defs_) {
      if (sig_def.method_name == method_name) return sig_def.outputs;
    }
    static const std::map<std::string, uint32_t>* default_empty_list =
        new std::map<std::string, uint32_t>();
    return *default_empty_list;
  }

  /// WARNING: Experimental interface, subject to change
  /// Returns the input tensor identified by 'signature_input_name' in the
  /// signature identified by 'signature_method_name'.
  /// Returns nullptr if not found.
  TfLiteTensor* input_tensor_by_signature_name(
      const char* signature_input_name, const char* signature_method_name) {
    const int tensor_index = GetTensorIndexFromSignatureDefName(
        signature_input_name, signature_method_name, /*is_input=*/true);
    return tensor_index == -1 ? nullptr : tensor(tensor_index);
  }

  /// WARNING: Experimental interface, subject to change
  /// Returns the output tensor identified by 'signature_output_name' in the
  /// signature identified by 'signature_method_name'.
  /// Returns nullptr if not found.
  const TfLiteTensor* output_tensor_by_signature_name(
      const char* signature_output_name,
      const char* signature_method_name) const {
    const int tensor_index = GetTensorIndexFromSignatureDefName(
        signature_output_name, signature_method_name, /*is_input=*/false);
    return tensor_index == -1 ? nullptr : tensor(tensor_index);
  }

  /// Return a mutable pointer to the given input tensor. The given index must
  /// be between 0 and inputs().size().
  TfLiteTensor* input_tensor(size_t index) { return tensor(inputs()[index]); }

  /// Return an immutable pointerto the given input tensor. The given index must
  /// be between 0 and inputs().size().
  const TfLiteTensor* input_tensor(size_t index) const {
    return tensor(inputs()[index]);
  }

  /// Return a mutable pointer into the data of a given input tensor. The given
  /// index must be between 0 and inputs().size().
  template <class T>
  T* typed_input_tensor(int index) {
    return typed_tensor<T>(inputs()[index]);
  }

  /// Return an immutable pointer into the data of a given input tensor. The
  /// given index must be between 0 and inputs().size().
  template <class T>
  const T* typed_input_tensor(int index) const {
    return typed_tensor<T>(inputs()[index]);
  }

  /// Return a mutable pointer to the given output tensor. The given index must
  /// be between 0 and outputs().size().
  TfLiteTensor* output_tensor(size_t index) { return tensor(outputs()[index]); }

  /// Return an immutable pointer to the given output tensor. The given index
  /// must be between 0 and outputs().size().
  const TfLiteTensor* output_tensor(size_t index) const {
    return tensor(outputs()[index]);
  }

  /// Return a mutable pointer into the data of a given output tensor. The given
  /// index must be between 0 and outputs().size().
  template <class T>
  T* typed_output_tensor(int index) {
    return typed_tensor<T>(outputs()[index]);
  }

  /// Return an immutable pointer into the data of a given output tensor. The
  /// given index must be between 0 and outputs().size().
  template <class T>
  const T* typed_output_tensor(int index) const {
    return typed_tensor<T>(outputs()[index]);
  }

  /// Change the dimensionality of a given tensor. Note, this is only acceptable
  /// for tensor indices that are inputs or variables.
  /// Returns status of failure or success. Note that this doesn't actually
  /// resize any existing buffers. A call to AllocateTensors() is required to
  /// change the tensor input buffer.
  TfLiteStatus ResizeInputTensor(int tensor_index,
                                 const std::vector<int>& dims);

  // WARNING: Experimental interface, subject to change
  // Change the dimensionality of a given tensor. This is only acceptable for
  // tensor indices that are inputs or variables. Only unknown dimensions can be
  // resized with this function. Unknown dimensions are indicated as `-1` in the
  // `dims_signature` attribute of a `TfLiteTensor`. Returns status of failure
  // or success.  Note that this doesn't actually resize any existing buffers.
  /// A call to AllocateTensors() is required to change the tensor input buffer.
  TfLiteStatus ResizeInputTensorStrict(int tensor_index,
                                       const std::vector<int>& dims);

  // This releases memory held by non-persistent tensors. It does NOT re-perform
  // memory planning.
  // AllocateTensors needs to be called before next invocation.
  /// WARNING: Experimental interface, subject to change
  TfLiteStatus ReleaseNonPersistentMemory();

  // Update allocations for all tensors. This will redim dependent tensors
  // using the input tensor dimensionality as given. This is relatively
  // expensive. This *must be* called after the interpreter has been created
  // and before running inference (and accessing tensor buffers), and *must be*
  // called again if (and only if) an input tensor is resized. Returns status of
  // success or failure.
  TfLiteStatus AllocateTensors();

  /// Invoke the interpreter (run the whole graph in dependency order).
  ///
  /// NOTE: It is possible that the interpreter is not in a ready state
  /// to evaluate (i.e. if a ResizeTensor() has been performed without an
  /// AllocateTensors().
  /// Returns status of success or failure.
  TfLiteStatus Invoke();

  /// Set the number of threads available to the interpreter.
  ///
  /// NOTE: num_threads should be >= -1.
  /// User may pass -1 to let the TFLite interpreter set the no of threads
  /// available to itself.
  TfLiteStatus SetNumThreads(int num_threads);

  /// Allow float16 precision for FP32 calculation when possible.
  /// Default: not allow.
  ///
  /// WARNING: This API is deprecated: prefer controlling this via delegate
  /// options, e.g. `tflite::StatefulNnApiDelegate::Options::allow_fp16' or
  /// `TfLiteGpuDelegateOptionsV2::is_precision_loss_allowed`.
  /// This method will be removed in a future release.
  void SetAllowFp16PrecisionForFp32(bool allow);

  /// Get the half precision flag.
  /// WARNING: This is an experimental API and subject to change.
  bool GetAllowFp16PrecisionForFp32() const {
    return context_->allow_fp32_relax_to_fp16;
  }

  /// Sets the cancellation function pointer in order to cancel a request in the
  /// middle of a call to Invoke(). The interpreter queries this function during
  /// inference, between op invocations; when it returns true, the interpreter
  /// will abort execution and return `kTfLiteError`. The `data` parameter
  /// contains any data used by the cancellation function, and if non-null,
  /// remains owned by the caller.
  /// WARNING: This is an experimental API and subject to change.
  void SetCancellationFunction(void* data, bool (*check_cancelled_func)(void*));

  /// Allow a delegate to look at the graph and modify the graph to handle
  /// parts of the graph themselves. After this is called, the graph may
  /// contain new nodes that replace 1 more nodes.
  /// 'delegate' must outlive the interpreter.
  /// Returns one of the following four status codes:
  /// 1. kTfLiteOk: Success.
  /// 2. kTfLiteDelegateError: Delegation failed due to an error in the
  /// delegate, or the delegate parameter was null. The Interpreter has been
  /// restored to its pre-delegation state.
  /// NOTE: This undoes all delegates previously applied to the Interpreter.
  /// 3. kTfLiteApplicationError : Delegation failed to be applied due to the
  /// incompatibility with the TfLite runtime, e.g., the model graph is already
  /// immutable when applying the delegate. However, the interpreter could still
  /// be invoked.
  /// 4. kTfLiteError: Unexpected/runtime failure.
  /// WARNING: This is an experimental API and subject to change.
  TfLiteStatus ModifyGraphWithDelegate(TfLiteDelegate* delegate);

  // Owning handle to a TfLiteDelegate instance.
  using TfLiteDelegatePtr =
      std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

  /// Same as ModifyGraphWithDelegate except this interpreter takes
  /// ownership of the provided delegate.
  /// WARNING: This is an experimental API and subject to change.
  template <typename Delegate, typename Deleter>
  inline TfLiteStatus ModifyGraphWithDelegate(
      std::unique_ptr<Delegate, Deleter> delegate) {
    Deleter deleter = std::move(delegate.get_deleter());

    // Note that we retain ownership of the delegate even if graph modification
    // fails, as delegate use will be in an indeterminate state at that point.
    owned_delegates_.emplace_back(
        delegate.release(), [deleter](TfLiteDelegate* delegate_to_delete) {
          deleter(
              static_cast<typename std::unique_ptr<Delegate, Deleter>::pointer>(
                  delegate_to_delete));
        });
    return ModifyGraphWithDelegate(owned_delegates_.back().get());
  }

  /// This overload is *never* OK. TfLiteDelegate is a C structure, so it has no
  /// virtual destructor. The default deleter of the unique_ptr does not know
  /// how to delete C++ objects deriving from TfLiteDelegate.
  TfLiteStatus ModifyGraphWithDelegate(
      std::unique_ptr<TfLiteDelegate> delegate) = delete;

  /// Ensure the data in `tensor.data` is readable. In case delegate is used,
  /// it might require to copy the data from delegate buffer to raw memory.
  /// WARNING: This is an experimental API and subject to change.
  TfLiteStatus EnsureTensorDataIsReadable(int tensor_index) {
    return primary_subgraph().EnsureTensorDataIsReadable(tensor_index);
  }

  /// Set the delegate buffer handle to a tensor. It can be called in the
  /// following cases:
  /// 1. Set the buffer handle to a tensor that's not being written by a
  ///    delegate. For example, feeding an OpenGL texture as the input of the
  ///    inference graph.
  /// 2. Set the buffer handle to a tensor that uses the same delegate.
  ///    For example, set an OpenGL texture as the output of inference, while
  ///    the node which produces output is an OpenGL delegate node.
  /// WARNING: This is an experimental API and subject to change.
  TfLiteStatus SetBufferHandle(int tensor_index,
                               TfLiteBufferHandle buffer_handle,
                               TfLiteDelegate* delegate);

  /// Get the delegate buffer handle, and the delegate which can process the
  /// buffer handle.
  /// WARNING: This is an experimental API and subject to change.
  TfLiteStatus GetBufferHandle(int tensor_index,
                               TfLiteBufferHandle* buffer_handle,
                               TfLiteDelegate** delegate);

  /// Sets the profiler to tracing execution. The caller retains ownership
  /// of the profiler and must ensure its validity.
  /// WARNING: This is an experimental API and subject to change.
  void SetProfiler(Profiler* profiler);

  /// Same as SetProfiler except this interpreter takes ownership
  /// of the provided profiler.
  /// WARNING: This is an experimental API and subject to change.
  void SetProfiler(std::unique_ptr<Profiler> profiler);

  /// Gets the profiler used for op tracing.
  /// WARNING: This is an experimental API and subject to change.
  Profiler* GetProfiler();

  // The default capacity of `tensors_` vector.
  static constexpr int kTensorsReservedCapacity = 128;
  /// The capacity headroom of `tensors_` vector before calling ops'
  /// `prepare` and `invoke` function. In these functions, it's guaranteed
  /// allocating up to `kTensorsCapacityHeadroom` more tensors won't invalidate
  /// pointers to existing tensors.
  static constexpr int kTensorsCapacityHeadroom = 16;

  /// Set if buffer handle output is allowed.
  //
  /// When using hardware delegation, Interpreter will make the data of output
  /// tensors available in `tensor->data` by default. If the application can
  /// consume the buffer handle directly (e.g. reading output from OpenGL
  /// texture), it can set this flag to false, so Interpreter won't copy the
  /// data from buffer handle to CPU memory. WARNING: This is an experimental
  /// API and subject to change.
  void SetAllowBufferHandleOutput(bool allow_buffer_handle_output) {
    allow_buffer_handle_output_ = allow_buffer_handle_output;
  }

  /// Reset all variable tensors to the default value.
  /// If a variable tensor doesn't have a buffer, reset it to zero.
  /// TODO(b/115961645): Implement - If a variable tensor has a buffer, reset it
  /// to the value of the buffer.
  /// WARNING: This is an experimental API and subject to change.
  TfLiteStatus ResetVariableTensors();

  /// Retrieve an operator's description of its work, for profiling purposes.
  const char* OpProfilingString(const TfLiteRegistration& op_reg,
                                const TfLiteNode* node) const {
    if (op_reg.profiling_string == nullptr) return nullptr;
    return op_reg.profiling_string(context_, node);
  }

  // Set the value of an external context. TFLite interpreter doesn't take the
  // memory ownership of this external context 'ctx', and the context should
  // outlive the TFLite interpreter.
  void SetExternalContext(TfLiteExternalContextType type,
                          TfLiteExternalContext* ctx);

  // Assigns (or reassigns) a custom memory allocation for the given tensor.
  // `flags` is a bitmask, see TfLiteCustomAllocationFlags.
  // The runtime does NOT take ownership of the underlying memory.
  //
  // NOTE: User needs to call AllocateTensors() after this. In case of input
  // resizing, buffers will be checked for required data size during
  // AllocateTensors().
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
  //
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus SetCustomAllocationForTensor(
      int tensor_index, const TfLiteCustomAllocation& allocation,
      int64_t flags = kTfLiteCustomAllocationFlagsNone);

#ifndef DOXYGEN_SKIP
  /// Adds `subgraphs_to_add` subgraphs, preserving pre-existing Subgraph
  /// entries. The value pointed to by `first_new_subgraph_index` will be set to
  /// the index of the first new subgraph if `first_new_subgraph_index` is
  /// non-null.
  /// WARNING: This is an experimental API and subject to change.
  void AddSubgraphs(int subgraphs_to_add,
                    int* first_new_subgraph_index = nullptr);

  /// Return the number of subgraphs in the model.
  /// WARNING: This is an experimental API and subject to change.
  size_t subgraphs_size() const { return subgraphs_.size(); }

  /// Get a pointer to a subgraph if in bounds.
  /// WARNING: This is an experimental API and subject to change.
  Subgraph* subgraph(int subgraph_index) {
    if (subgraph_index < 0 ||
        static_cast<size_t>(subgraph_index) >= subgraphs_size())
      return nullptr;
    return &*subgraphs_[subgraph_index];
  }

  /// WARNING: Experimental interface, subject to change
  Subgraph& primary_subgraph() {
    return *subgraphs_.front();  /// Safe as subgraphs_ always has 1 entry.
  }

  /// WARNING: Experimental interface, subject to change
  const Subgraph& primary_subgraph() const {
    return *subgraphs_.front();  // Safe as subgraphs_ always has 1 entry.
  }

  /// WARNING: Experimental interface, subject to change
  // Get the error reporter associated with this interpreter.
  ErrorReporter* error_reporter() const { return error_reporter_; }

#endif  // DOXYGEN_SKIP

 private:
  // Structure representing SignatureDef inputs/outputs.
  struct SignatureDef {
    // Maps name in signature def as key to index of the tensor in the model.
    std::map<std::string, uint32_t> inputs;
    // Maps name in signature def as key to index of the tensor in the model.
    std::map<std::string, uint32_t> outputs;
    // The method name for this signature.
    std::string method_name;
    // The key of this SignatureDef in the SavedModel signature def map.
    std::string signature_def_key;
  };
  friend class InterpreterBuilder;
  friend class tflite::InterpreterTest;
  friend class tflite::delegates::InterpreterUtils;
  friend class tflite::delegates::test_utils::TestDelegate;

  /// Set the value of an external context.
  static void SetExternalContext(struct TfLiteContext* context,
                                 TfLiteExternalContextType type,
                                 TfLiteExternalContext* ctx);

  // Helper method that return the tensor index that corresponds to
  // a name in a SignatureDef. Defined by 'signature_method_name', and
  // 'signature_tensor_name'.
  // If 'is_input' is true then the tensor is checked in input tensors,
  // otherwise it will be checked in output tensors.
  // Returns -1 if the tensor is not found.
  int GetTensorIndexFromSignatureDefName(const char* signature_tensor_name,
                                         const char* signature_method_name,
                                         bool is_input) const {
    // Iterate directly and don't use other methods to avoid extra allocation.
    for (const auto& signature : signature_defs_) {
      if (signature.method_name != signature_method_name) continue;
      auto& signature_list = (is_input ? signature.inputs : signature.outputs);
      auto tensor_iter = signature_list.find(signature_tensor_name);
      if (tensor_iter == signature_list.end()) return -1;
      return tensor_iter->second;
    }
    return -1;
  }

  // Sets the profiler to all subgraphs.
  void SetSubgraphProfiler();

  // Remove delegates (for fallback behaviour). The interpreter is invokable
  // afterwards.
  TfLiteStatus RemoveAllDelegates();

  // Returns true if delegates have been applied.
  bool HasDelegates();

  // Returns true if cancellation function returns true.
  bool IsCancelled();

  // Sets the list of signature defs in the model.
  void SetSignatureDef(std::vector<SignatureDef> signature_defs) {
    signature_defs_ = std::move(signature_defs);
  }

  // A pure C data structure used to communicate with the pure C plugin
  // interface. To avoid copying tensor metadata, this is also the definitive
  // structure to store tensors.
  // This is the primary subgraph context.
  TfLiteContext* context_ = nullptr;

  // The error reporter delegate that tflite will forward queries errors to.
  ErrorReporter* error_reporter_ = nullptr;

  // List of delegates that have been installed and are owned by this
  // interpreter instance. Useful if client delegate ownership is burdensome.
  // WARNING: This is an experimental API and subject to change.
  // TODO(b/116667551): Use TfLiteExternalContext for storing state.
  std::vector<
      std::unique_ptr<TfLiteDelegate, std::function<void(TfLiteDelegate*)>>>
      owned_delegates_;

  // Profiler that has been installed and is owned by this interpreter instance.
  // Useful if client profiler ownership is burdensome.
  std::unique_ptr<Profiler> owned_profiler_;

  // Points to the installed Profiler instance.
  Profiler* installed_profiler_ = nullptr;

  bool allow_buffer_handle_output_ = false;

  // List of active external contexts.
  TfLiteExternalContext* external_contexts_[kTfLiteMaxExternalContexts];

  // The default external cpu backend context. After an TFLite interpreter is
  // initialized, 'external_contexts_[kTfLiteCpuBackendContext]' is set to point
  // to this object. However, if this element value is overwritten via calling
  // 'SetExternalContext(kTfLiteCpuBackendContext, ...)', we will reset this to
  // nullptr if necessary.
  std::unique_ptr<ExternalCpuBackendContext> own_external_cpu_backend_context_;

  // Subgraphs
  std::vector<std::unique_ptr<Subgraph>> subgraphs_;

  // A map of resources. Owned by interpreter and shared by multiple subgraphs.
  resource::ResourceMap resources_;

  // Indicating delegates that the TFLite interpreter will apply by default.
  // An empty one means there's no delegate to be applied by default or
  // delegates have been applied and doesn't need to be applied again.
  std::vector<TfLiteDelegatePtr> lazy_delegate_providers_;

  // List of signature def mapping inputs/output to tensor ids.
  // We just keep track of tensor index.
  std::vector<SignatureDef> signature_defs_;
};

}  // namespace tflite
#endif  // TENSORFLOW_LITE_INTERPRETER_H_
