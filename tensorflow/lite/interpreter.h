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
// Main abstraction controlling the tflite interpreter.
// See context.h for the API for defining operations (TfLiteRegistration).
#ifndef TENSORFLOW_LITE_INTERPRETER_H_
#define TENSORFLOW_LITE_INTERPRETER_H_

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/c/c_api_internal.h"  // IWYU pragma: export
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/experimental/resource_variable/resource_variable.h"
#include "tensorflow/lite/external_cpu_backend_context.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/type_to_tflitetype.h"

namespace tflite {

/// An interpreter for a graph of nodes that input and output from tensors.
/// Each node of the graph processes a set of input tensors and produces a
/// set of output Tensors. All inputs/output tensors are referenced by index.
///
/// Usage:
///
/// <pre><code>
/// // Create basic model
/// Interpreter foo(2, 1);
/// foo.SetTensorParametersReadWrite(0, ...);
/// foo.SetTensorParametersReadOnly(1, ...);
/// foo.SetNodeParameters(0, ...)
/// // Resize input array to 1 length.
/// foo.ResizeInputTensor(0, 1);
/// foo.AllocateTensors();
/// // Install array data
/// foo.typed_tensor<float>(0)[0] = 3;
/// foo.Invoke();
/// foo.typed_tensor<float>(0)[0] = 4;
/// foo.Invoke();
/// // Resize input array and set data.
/// foo.ResizeInputTensor(0, 2);
/// foo.AllocateTensors();
/// foo.typed_tensor<float>(0)[0] = 4;
/// foo.typed_tensor<float>(0)[1] = 8;
/// foo.Invoke();
/// </code></pre>
///

class Interpreter {
 public:
  /// Instantiate an interpreter. All errors associated with reading and
  /// processing this model will be forwarded to the error_reporter object.
  //
  /// Note, if error_reporter is nullptr, then a default StderrReporter is
  /// used. Ownership of 'error_reporter' remains with the caller.
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
      bool is_variable = false) {
    return SetTensorParametersReadWrite(tensor_index, type, name, dims.size(),
                                        dims.data(), quantization, is_variable);
  }
  TfLiteStatus SetTensorParametersReadWrite(
      int tensor_index, TfLiteType type, const char* name, const size_t rank,
      const int* dims, TfLiteQuantizationParams quantization,
      bool is_variable = false);
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
  /// Returns status of failure or success.
  /// TODO(aselle): Consider implementing ArraySlice equivalent to make this
  ///   more adept at accepting data without an extra copy. Use absl::ArraySlice
  ///   if our partners determine that dependency is acceptable.
  TfLiteStatus ResizeInputTensor(int tensor_index,
                                 const std::vector<int>& dims);

  // This releases memory held by non-persistent tensors. It does NOT re-perform
  // memory planning.
  // AllocateTensors needs to be called before next invocation.
  /// WARNING: Experimental interface, subject to change
  TfLiteStatus ReleaseNonPersistentMemory();

  /// Update allocations for all tensors. This will redim dependent tensors
  /// using the input tensor dimensionality as given. This is relatively
  /// expensive. If you know that your sizes are not changing, you need not call
  /// this. Returns status of success or failure.
  TfLiteStatus AllocateTensors();

  /// Invoke the interpreter (run the whole graph in dependency order).
  ///
  /// NOTE: It is possible that the interpreter is not in a ready state
  /// to evaluate (i.e. if a ResizeTensor() has been performed without an
  /// AllocateTensors().
  /// Returns status of success or failure.
  TfLiteStatus Invoke();

  /// Enable or disable the NN API (true to enable)
  void UseNNAPI(bool enable);

  /// Set the number of threads available to the interpreter.
  void SetNumThreads(int num_threads);

  /// Allow float16 precision for FP32 calculation when possible.
  /// default: not allow.
  /// WARNING: This is an experimental API and subject to change.
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
  /// WARNING: This is an experimental API and subject to change.
  TfLiteStatus ModifyGraphWithDelegate(TfLiteDelegate* delegate);

  // Owning handle to a TfLiteDelegate instance.
  using TfLiteDelegatePtr =
      std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

  /// Same as ModifyGraphWithDelegate except this interpreter takes
  /// ownership of the provided delegate. Be sure to construct the unique_ptr
  /// with a suitable destruction function.
  /// WARNING: This is an experimental API and subject to change.
  TfLiteStatus ModifyGraphWithDelegate(TfLiteDelegatePtr delegate);

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
#endif  // DOXYGEN_SKIP

 private:
  friend class InterpreterBuilder;
  friend class InterpreterTest;

  /// Set the value of an external context.
  static void SetExternalContext(struct TfLiteContext* context,
                                 TfLiteExternalContextType type,
                                 TfLiteExternalContext* ctx);

  // A pure C data structure used to communicate with the pure C plugin
  // interface. To avoid copying tensor metadata, this is also the definitive
  // structure to store tensors.
  // This is the primary subgraph context.
  TfLiteContext* context_;

  // The error reporter delegate that tflite will forward queries errors to.
  ErrorReporter* error_reporter_;

  // List of delegates that have been installed and are owned by this
  // interpreter instance. Useful if client delegate ownership is burdensome.
  // WARNING: This is an experimental API and subject to change.
  // TODO(b/116667551): Use TfLiteExternalContext for storing state.
  std::vector<TfLiteDelegatePtr> owned_delegates_;

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

  // A map of resource variables. Owned by interpreter and shared by multiple
  // subgraphs.
  ResourceVariableMap resource_variables_;
};

}  // namespace tflite
#endif  // TENSORFLOW_LITE_INTERPRETER_H_
