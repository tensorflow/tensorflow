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
#ifndef TENSORFLOW_CONTRIB_LITE_INTERPRETER_H_
#define TENSORFLOW_CONTRIB_LITE_INTERPRETER_H_

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tensorflow/contrib/lite/allocation.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/error_reporter.h"
#include "tensorflow/contrib/lite/memory_planner.h"
#include "tensorflow/contrib/lite/profiling/profiler.h"

namespace tflite {

// Map statically from a c++ type to a TfLiteType (used below for safe casts).
template <class T>
constexpr TfLiteType typeToTfLiteType() {
  return kTfLiteNoType;
}
template <>
constexpr TfLiteType typeToTfLiteType<int>() {
  return kTfLiteInt32;
}
template <>
constexpr TfLiteType typeToTfLiteType<int64_t>() {
  return kTfLiteInt64;
}
template <>
constexpr TfLiteType typeToTfLiteType<float>() {
  return kTfLiteFloat32;
}
template <>
constexpr TfLiteType typeToTfLiteType<unsigned char>() {
  return kTfLiteUInt8;
}
template <>
constexpr TfLiteType typeToTfLiteType<bool>() {
  return kTfLiteBool;
}

// Forward declare since NNAPIDelegate uses Interpreter.
class NNAPIDelegate;

// An interpreter for a graph of nodes that input and output from tensors.
// Each node of the graph processes a set of input tensors and produces a
// set of output Tensors. All inputs/output tensors are referenced by index.
//
// Usage:
//
// -- Create basic model
// Interpreter foo(2, 1);
// foo.SetTensorParametersReadWrite(0, ...);
// foo.SetTensorParametersReadOnly(1, ...);
// foo.SetNodeParameters(0, ...)
//
// -- Resize input array to 1 length.
// foo.ResizeInputTensor(0, 1);
// foo.AllocateTensors();
// -- Install array data
// foo.typed_tensor<float>(0)[0] = 3;
// foo.Invoke();
// foo.typed_tensor<float>(0)[0] = 4;
// foo.Invoke();
// -- Resize input array and set data.
// foo.ResizeInputTensor(0, 2);
// foo.AllocateTensors();
// foo.typed_tensor<float>(0)[0] = 4;
// foo.typed_tensor<float>(0)[1] = 8;
// foo.Invoke();
//

struct TfLiteIntArrayDeleter {
  void operator()(TfLiteIntArray* a) {
    if (a) TfLiteIntArrayFree(a);
  }
};

class Interpreter {
 public:
  // Instantiate an interpreter. All errors associated with reading and
  // processing this model will be forwarded to the error_reporter object.
  //
  // Note, if error_reporter is nullptr, then a default StderrReporter is
  // used.
  explicit Interpreter(ErrorReporter* error_reporter = DefaultErrorReporter());

  ~Interpreter();

  Interpreter(const Interpreter&) = delete;
  Interpreter& operator=(const Interpreter&) = delete;

  // Functions to build interpreter

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
  // to Interpreter.
  inline TfLiteStatus SetTensorParametersReadOnly(
      int tensor_index, TfLiteType type, const char* name,
      const std::vector<int>& dims, TfLiteQuantizationParams quantization,
      const char* buffer, size_t bytes,
      const Allocation* allocation = nullptr) {
    return SetTensorParametersReadOnly(tensor_index, type, name, dims.size(),
                                       dims.data(), quantization, buffer, bytes,
                                       allocation);
  };

  TfLiteStatus SetTensorParametersReadOnly(
      int tensor_index, TfLiteType type, const char* name, const size_t rank,
      const int* dims, TfLiteQuantizationParams quantization,
      const char* buffer, size_t bytes, const Allocation* allocation = nullptr);

  // Set description of inputs/outputs/data/fptrs for node `node_index`.
  // This variant assumes an external buffer has been allocated of size
  // bytes. The lifetime of buffer must be ensured to be greater or equal
  // to Interpreter.
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

  // Functions to access tensor data

  // Read only access to list of inputs.
  const std::vector<int>& inputs() const { return inputs_; }

  // Return the name of a given input. The given index must be between 0 and
  // inputs().size().
  const char* GetInputName(int index) const {
    return context_.tensors[inputs_[index]].name;
  }

  // Read only access to list of outputs.
  const std::vector<int>& outputs() const { return outputs_; }

  // Read only access to list of variable tensors.
  const std::vector<int>& variables() const { return variables_; }

  // Return the name of a given output. The given index must be between 0 and
  // outputs().size().
  const char* GetOutputName(int index) const {
    return context_.tensors[outputs_[index]].name;
  }

  // Return the number of tensors in the model.
  size_t tensors_size() const { return context_.tensors_size; }

  // Return the number of ops in the model.
  size_t nodes_size() const { return nodes_and_registration_.size(); }

  // WARNING: Experimental interface, subject to change
  const std::vector<int>& execution_plan() const { return execution_plan_; }

  // WARNING: Experimental interface, subject to change
  // Overrides execution plan. This bounds checks indices sent in.
  TfLiteStatus SetExecutionPlan(const std::vector<int>& new_plan);

  // Get a mutable tensor data structure.
  // TODO(aselle): Create a safe ArrayHandle interface to avoid exposing this
  // read/write access to structure
  TfLiteTensor* tensor(int tensor_index) {
    if (tensor_index >= context_.tensors_size || tensor_index < 0)
      return nullptr;
    return &context_.tensors[tensor_index];
  }

  // Get an immutable tensor data structure.
  const TfLiteTensor* tensor(int tensor_index) const {
    if (tensor_index >= context_.tensors_size || tensor_index < 0)
      return nullptr;
    return &context_.tensors[tensor_index];
  }

  // Get a pointer to an operation and registration data structure if in bounds.
  const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration(
      int node_index) const {
    if (node_index >= nodes_and_registration_.size() || node_index < 0)
      return nullptr;
    return &nodes_and_registration_[node_index];
  }

  // Perform a checked cast to the appropriate tensor type (mutable pointer
  // version).
  template <class T>
  T* typed_tensor(int tensor_index) {
    if (TfLiteTensor* tensor_ptr = tensor(tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return reinterpret_cast<T*>(tensor_ptr->data.raw);
      }
    }
    return nullptr;
  }

  // Perform a checked cast to the appropriate tensor type (immutable pointer
  // version).
  template <class T>
  const T* typed_tensor(int tensor_index) const {
    if (const TfLiteTensor* tensor_ptr = tensor(tensor_index)) {
      if (tensor_ptr->type == typeToTfLiteType<T>()) {
        return reinterpret_cast<const T*>(tensor_ptr->data.raw);
      }
    }
    return nullptr;
  }

  // Return a mutable pointer into the data of a given input tensor. The given
  // index must be between 0 and inputs().size().
  template <class T>
  T* typed_input_tensor(int index) {
    return typed_tensor<T>(inputs_[index]);
  }

  // Return an immutable pointer into the data of a given input tensor. The
  // given index must be between 0 and inputs().size().
  template <class T>
  const T* typed_input_tensor(int index) const {
    return typed_tensor<T>(inputs_[index]);
  }

  // Return a mutable pointer into the data of a given output tensor. The given
  // index must be between 0 and outputs().size().
  template <class T>
  T* typed_output_tensor(int index) {
    return typed_tensor<T>(outputs_[index]);
  }

  // Return an immutable pointer into the data of a given output tensor. The
  // given index must be between 0 and outputs().size().
  template <class T>
  const T* typed_output_tensor(int index) const {
    return typed_tensor<T>(outputs_[index]);
  }

  // Change the dimensionality of a given tensor. Note, this is only acceptable
  // for tensor indices that are inputs.
  // Returns status of failure or success.
  // TODO(aselle): Consider implementing ArraySlice equivalent to make this
  //   more adept at accepting data without an extra copy. Use absl::ArraySlice
  //   if our partners determine that dependency is acceptable.
  TfLiteStatus ResizeInputTensor(int tensor_index,
                                 const std::vector<int>& dims);

  // Update allocations for all tensors. This will redim dependent tensors using
  // the input tensor dimensionality as given. This is relatively expensive.
  // If you know that your sizes are not changing, you need not call this.

  // Returns status of success or failure.
  TfLiteStatus AllocateTensors();

  // Invoke the interpreter (run the whole graph in dependency order).
  //
  // NOTE: It is possible that the interpreter is not in a ready state
  // to evaluate (i.e. if a ResizeTensor() has been performed without an
  // AllocateTensors().
  // Returns status of success or failure.
  TfLiteStatus Invoke();

  // Enable or disable the NN API (true to enable)
  void UseNNAPI(bool enable);

  // Set the number of threads available to the interpreter.
  void SetNumThreads(int num_threads);

  // Allow a delegate to look at the graph and modify the graph to handle
  // parts of the graph themselves. After this is called, the graph may
  // contain new nodes that replace 1 more nodes.
  // WARNING: This is an experimental API and subject to change.
  TfLiteStatus ModifyGraphWithDelegate(TfLiteDelegate* delegate,
                                       bool allow_dynamic_tensors = false);

  // Ensure the data in `tensor.data` is readable. In case delegate is used,
  // it might require to copy the data from delegate buffer to raw memory.
  // WARNING: This is an experimental API and subject to change.
  TfLiteStatus EnsureTensorDataIsReadable(int tensor_index) {
    TF_LITE_ENSURE(&context_, tensor_index < tensors_size());
    TfLiteTensor* tensor = &tensors_[tensor_index];
    if (tensor->data_is_stale) {
      TF_LITE_ENSURE(&context_, tensor->delegate != nullptr);
      TF_LITE_ENSURE(&context_,
                     tensor->buffer_handle != kTfLiteNullBufferHandle);
      // This can be null if the delegate doesn't use its own buffer.
      TF_LITE_ENSURE(&context_,
                     tensor->delegate->CopyFromBufferHandle != nullptr);
      tensor->delegate->CopyFromBufferHandle(tensor->delegate,
                                             tensor->buffer_handle,
                                             tensor->data.raw, tensor->bytes);
      tensor->data_is_stale = false;
    }
    return kTfLiteOk;
  }

  // Set the delegate buffer handle to a tensor. It can be called in the
  // following cases:
  // 1. Set the buffer handle to a tensor that's not being written by a
  //    delegate. For example, feeding an OpenGL texture as the input of the
  //    inference graph.
  // 2. Set the buffer handle to a tensor that uses the same delegate.
  //    For example, set an OpenGL texture as the output of inference, while
  //    the node which produces output is an OpenGL delegate node.
  // WARNING: This is an experimental API and subject to change.
  TfLiteStatus SetBufferHandle(int tensor_index,
                               TfLiteBufferHandle buffer_handle,
                               TfLiteDelegate* delegate);

  // Get the delegate buffer handle, and the delegate which can process the
  // buffer handle.
  // WARNING: This is an experimental API and subject to change.
  TfLiteStatus GetBufferHandle(int tensor_index,
                               TfLiteBufferHandle* buffer_handle,
                               TfLiteDelegate** delegate);

  void SetProfiler(profiling::Profiler* profiler) { profiler_ = profiler; }

  profiling::Profiler* GetProfiler() { return profiler_; }

  // The default capacity of `tensors_` vector.
  static constexpr int kTensorsReservedCapacity = 128;
  // The capacity headroom of `tensors_` vector before calling ops'
  // `prepare` and `invoke` function. In these functions, it's guaranteed
  // allocating up to `kTensorsCapacityHeadroom` more tensors won't invalidate
  // pointers to existing tensors.
  static constexpr int kTensorsCapacityHeadroom = 16;

  // Set if buffer handle output is allowed.
  //
  // When using hardware delegation, Interpreter will make the data of output
  // tensors available in `tensor->data` by default. If the application can
  // consume the buffer handle directly (e.g. reading output from OpenGL
  // texture), it can set this flag to false, so Interpreter won't copy the data
  // from buffer handle to CPU memory.
  // WARNING: This is an experimental API and subject to change.
  void SetAllowBufferHandleOutput(bool allow_buffer_handle_output) {
    allow_buffer_handle_output_ = allow_buffer_handle_output;
  }

  // Reset all variable tensors to zero.
  // WARNING: This is an experimental API and subject to change.
  TfLiteStatus ResetVariableTensorsToZero();

 private:
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
  TfLiteStatus OpPrepare(const TfLiteRegistration& op_reg, TfLiteNode* node) {
    if (op_reg.prepare == nullptr) return kTfLiteOk;
    return op_reg.prepare(&context_, node);
  }

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

  // Compute the number of bytes required to represent a tensor with dimensions
  // specified by the array dims (of length dims_size). Returns the status code
  // and bytes.
  TfLiteStatus BytesRequired(TfLiteType type, const int* dims, size_t dims_size,
                             size_t* bytes);

  // Request an tensor be resized implementation. If the given tensor is of
  // type kTfLiteDynamic it will also be allocated new memory.
  TfLiteStatus ResizeTensorImpl(TfLiteTensor* tensor, TfLiteIntArray* new_size);

  // Report a detailed error string (will be printed to stderr).
  // TODO(aselle): allow user of class to provide alternative destinations.
  void ReportErrorImpl(const char* format, va_list args);

  // Entry point for C node plugin API to request an tensor be resized.
  static TfLiteStatus ResizeTensor(TfLiteContext* context, TfLiteTensor* tensor,
                                   TfLiteIntArray* new_size);
  // Entry point for C node plugin API to report an error.
  static void ReportError(TfLiteContext* context, const char* format, ...);

  // Entry point for C node plugin API to add new tensors.
  static TfLiteStatus AddTensors(TfLiteContext* context, int tensors_to_add,
                                 int* first_new_tensor_index);

  // WARNING: This is an experimental API and subject to change.
  // Entry point for C API ReplaceSubgraphsWithDelegateKernels
  static TfLiteStatus ReplaceSubgraphsWithDelegateKernels(
      TfLiteContext* context, TfLiteRegistration registration,
      const TfLiteIntArray* nodes_to_replace, TfLiteDelegate* delegate);

  // Update the execution graph to replace some of the nodes with stub
  // nodes. Specifically any node index that has `nodes[index]==1` will be
  // slated for replacement with a delegate kernel specified by registration.
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus ReplaceSubgraphsWithDelegateKernels(
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
  // Gets an TfLiteIntArray* representing the execution plan. The caller owns
  // this memory and must free it with TfLiteIntArrayFree().
  TfLiteStatus GetExecutionPlan(TfLiteIntArray** execution_plan);

  // WARNING: This is an experimental interface that is subject to change.
  // Entry point for C node plugin API to get the execution plan
  static TfLiteStatus GetExecutionPlan(struct TfLiteContext* context,
                                       TfLiteIntArray** execution_plan);

  // Ensures that `tensors_` has at least `kTensorsCapacityHeadroom` extra
  // capacity. Calling this function may invalidate existing pointers to
  // tensors. After calling this function, adding `kTensorsCapacityHeadroom`
  // more tensors won't invalidate the pointer to existing tensors.
  void EnsureTensorsVectorCapacity() {
    const size_t required_capacity = tensors_size() + kTensorsCapacityHeadroom;
    if (required_capacity > tensors_.capacity()) {
      tensors_.reserve(required_capacity);
      context_.tensors = tensors_.data();
    }
  }

  // The state of the Interpreter.
  enum State {
    // The interpreter isn't ready to be invoked.
    // `AllocateTensor` need to be called to enter an invokable state.
    kStateUninvokable = 0,
    // The interpreter is ready to be invoked.
    kStateInvokable,
    // The interpreter is ready to be invoked, and graph can't be further
    // modified. The interpreter will enter this state when calling
    // `ModifyGraphWithDelegate` with `allow_dynamic_tensors=false`.
    kStateInvokableAndImmutable,
  };
  State state_ = kStateUninvokable;

  // A pure C data structure used to communicate with the pure C plugin
  // interface. To avoid copying tensor metadata, this is also the definitive
  // structure to store tensors.
  TfLiteContext context_;

  // Node inputs/outputs are stored in TfLiteNode and TfLiteRegistration stores
  // function pointers to actual implementation.
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

  // WARNING: This is an experimental interface that is subject to change.
  // This is a list of node indices (to index into nodes_and_registration).
  // This represents a valid topological sort (dependency ordered) execution
  // plan. In particular, it is valid for this ordering to contain only a
  // subset of the node indices.
  std::vector<int> execution_plan_;

  // In the future, we'd like a TfLiteIntArray compatible representation.
  // TODO(aselle): replace execution_plan_ with this.
  std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> plan_cache_;

  // Whether to delegate to NN API
  std::unique_ptr<NNAPIDelegate> nnapi_delegate_;

  std::unique_ptr<MemoryPlanner> memory_planner_;

  bool allow_buffer_handle_output_ = false;

  // Profiler for this interpreter instance.
  profiling::Profiler* profiler_;
};

}  // namespace tflite
#endif  // TENSORFLOW_CONTRIB_LITE_INTERPRETER_H_
