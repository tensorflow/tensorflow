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
#include "tensorflow/lite/experimental/resource/initialization_status.h"
#include "tensorflow/lite/experimental/resource/resource_base.h"
#include "tensorflow/lite/external_cpu_backend_context.h"
#include "tensorflow/lite/internal/signature_def.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "tensorflow/lite/signature_runner.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/type_to_tflitetype.h"

namespace tflite {

class InterpreterTest;  // Class for friend declarations.

namespace delegates {
class InterpreterUtils;  // Class for friend declarations.

namespace test_utils {
class TestDelegation;  // Class for friend declarations.
}  // namespace test_utils
}  // namespace delegates

namespace interpreter_wrapper {
class InterpreterWrapper;  // Class for friend declarations.
}  // namespace interpreter_wrapper

/// Options class for `Interpreter`.
/// WARNING: This is an experimental API and subject to change.
class InterpreterOptions {
 public:
  InterpreterOptions()
      : experimental_preserve_all_tensors_(false),
        experimental_ensure_dynamic_tensors_are_released_(false),
        experimental_dynamic_allocation_for_large_tensors_(0) {}

  /// Preserving all intermediates tensors for debugging.
  /// WARNING: This is an experimental API and subject to change.
  void SetPreserveAllTensors(bool value = true) {
    experimental_preserve_all_tensors_ = value;
  }

  /// Returns if the `experimental_preserve_all_tensors_` feature is enabled.
  /// WARNING: This is an experimental API and subject to change.
  bool GetPreserveAllTensors() { return experimental_preserve_all_tensors_; }

  /// Force all intermediate dynamic tensors to be released once they are not
  /// used by the model. Please use this configuration with caution, since it
  /// might reduce the peak memory usage of the model at the cost of a slower
  /// inference speed.
  /// WARNING: This is an experimental API and subject to change.
  void SetEnsureDynamicTensorsAreReleased(bool value = true) {
    experimental_ensure_dynamic_tensors_are_released_ = value;
  }

  /// Returns if the `experimental_ensure_dynamic_tensors_are_released_` feature
  /// is enabled.
  /// WARNING: This is an experimental API and subject to change.
  bool GetEnsureDynamicTensorsAreReleased() {
    return experimental_ensure_dynamic_tensors_are_released_;
  }

  /// Use dynamic tensor allocation method for large tensors instead of static
  /// memory planner. It improves peak memory usage but there could be some
  /// latency impact. The value is used to determine large tensors.
  /// WARNING: This is an experimental API and subject to change.
  void SetDynamicAllocationForLargeTensors(int value) {
    if (value > 0) {
      experimental_dynamic_allocation_for_large_tensors_ = value;
    }
  }

  /// Returns the size threshold for dynamic tensor allocation method.
  /// It returns zero if the feature is not enabled.
  /// WARNING: This is an experimental API and subject to change.
  int GetDynamicAllocationForLargeTensors() {
    return experimental_dynamic_allocation_for_large_tensors_;
  }

 private:
  bool experimental_preserve_all_tensors_;
  bool experimental_ensure_dynamic_tensors_are_released_;
  int experimental_dynamic_allocation_for_large_tensors_;
};

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
/// if (interpreter->AllocateTensors() != kTfLiteOk) {
///   // Return failure.
/// }
///
/// auto input = interpreter->typed_tensor<float>(0);
/// for (int i = 0; i < input_size; i++) {
///   input[i] = ...;
//  }
/// interpreter->Invoke();
/// </code></pre>
///
/// Note: For nearly all practical use cases, one should not directly construct
/// an Interpreter object, but rather use the InterpreterBuilder.
///
/// WARNING: This class is *not* thread-safe. The client is responsible for
/// ensuring serialized interaction to avoid data races and undefined behavior.
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

  /// Returns a pointer to an operation and registration data structure if in
  /// bounds from the primary subgraph(subgraph_[0]).
  const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration(
      int node_index) const {
    return primary_subgraph().node_and_registration(node_index);
  }

  /// Returns a pointer to an operation and registration data structure if in
  /// bounds.
  const std::pair<TfLiteNode, TfLiteRegistration>* node_and_registration(
      int subgraph_index, int node_index) const {
    return subgraph(subgraph_index)->node_and_registration(node_index);
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
  /// Returns list of all keys of different method signatures defined in the
  /// model.
  /// Note, pointers returned have lifetime same as the Interpreter object.
  std::vector<const std::string*> signature_keys() const {
    std::vector<const std::string*> signature_keys;
    signature_keys.reserve(signature_defs_.size());
    for (const auto& sig_def : signature_defs_) {
      signature_keys.emplace_back(&sig_def.signature_key);
    }
    return signature_keys;
  }

  /// WARNING: Experimental interface, subject to change
  /// Returns a pointer to the SignatureRunner instance to run the part of the
  /// graph identified by a SignatureDef. The nullptr is returned if the given
  /// signature key is not valid.
  /// If you need to specify delegates, you have to do that before calling this
  /// function. This function will additionally apply default delegates. Thus,
  /// applying delegates after that might lead to undesirable behaviors.
  /// Note, the pointed instance has lifetime same as the Interpreter object
  /// and the SignatureRunner class is *not* thread-safe.
  SignatureRunner* GetSignatureRunner(const char* signature_key);

  /// WARNING: Experimental interface, subject to change
  /// Return the subgraph index that corresponds to a SignatureDef, defined by
  /// 'signature_key'.
  /// If invalid name passed, -1 will be returned.
  int GetSubgraphIndexFromSignature(const char* signature_key) const {
    for (const auto& signature : signature_defs_) {
      if (signature.signature_key == signature_key) {
        return signature.subgraph_index;
      }
    }
    return -1;
  }

  /// WARNING: Experimental interface, subject to change
  /// Returns the mapping of inputs to tensor index in the signature
  /// specified through 'signature_key'.
  /// If invalid name passed, an empty list will be returned.
  const std::map<std::string, uint32_t>& signature_inputs(
      const char* signature_key) const {
    for (const auto& sig_def : signature_defs_) {
      if (sig_def.signature_key == signature_key) return sig_def.inputs;
    }
    static const std::map<std::string, uint32_t>* default_empty_list =
        new std::map<std::string, uint32_t>();
    return *default_empty_list;
  }

  /// WARNING: Experimental interface, subject to change
  /// Returns the mapping of outputs to tensor index in the signature
  /// specified through 'signature_key'.
  /// If invalid name passed, an empty list will be returned.
  const std::map<std::string, uint32_t>& signature_outputs(
      const char* signature_key) const {
    for (const auto& sig_def : signature_defs_) {
      if (sig_def.signature_key == signature_key) return sig_def.outputs;
    }
    static const std::map<std::string, uint32_t>* default_empty_list =
        new std::map<std::string, uint32_t>();
    return *default_empty_list;
  }

  /// WARNING: Experimental interface, subject to change
  /// Returns the input tensor identified by 'signature_input_name' in the
  /// signature identified by 'signature_key'.
  /// Returns nullptr if not found.
  TfLiteTensor* input_tensor_by_signature(const char* signature_input_name,
                                          const char* signature_key) {
    const int subgraph_index = GetSubgraphIndexFromSignature(signature_key);
    if (subgraph_index == -1) return nullptr;
    const int tensor_index = GetTensorIndexFromSignature(
        signature_input_name, signature_key, /*is_input=*/true);
    if (tensor_index == -1) return nullptr;
    return subgraph(subgraph_index)->tensor(tensor_index);
  }

  /// WARNING: Experimental interface, subject to change
  /// Returns the output tensor identified by 'signature_output_name' in the
  /// signature identified by 'signature_key'.
  /// Returns nullptr if not found.
  const TfLiteTensor* output_tensor_by_signature(
      const char* signature_output_name, const char* signature_key) const {
    const int subgraph_index = GetSubgraphIndexFromSignature(signature_key);
    if (subgraph_index == -1) return nullptr;
    const int tensor_index = GetTensorIndexFromSignature(
        signature_output_name, signature_key, /*is_input=*/false);
    if (tensor_index == -1) return nullptr;
    return subgraph(subgraph_index)->tensor(tensor_index);
  }

  /// Return a mutable pointer to the given input tensor. The given index must
  /// be between 0 and inputs().size().
  TfLiteTensor* input_tensor(size_t index) { return tensor(inputs()[index]); }

  /// Return an immutable pointer to the given input tensor. The given index
  /// must be between 0 and inputs().size().
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

  /// Change the dimensionality of a given tensor. This is only acceptable for
  /// tensor indices that are inputs or variables. Only unknown dimensions can
  /// be resized with this function. Unknown dimensions are indicated as `-1` in
  /// the `dims_signature` attribute of a `TfLiteTensor`. Returns status of
  /// failure or success.  Note that this doesn't actually resize any existing
  /// buffers. A call to AllocateTensors() is required to change the tensor
  /// input buffer.
  TfLiteStatus ResizeInputTensorStrict(int tensor_index,
                                       const std::vector<int>& dims);

  /// This releases memory held by non-persistent tensors. It does NOT
  /// re-perform memory planning. AllocateTensors needs to be called before next
  /// invocation. WARNING: Experimental interface, subject to change
  TfLiteStatus ReleaseNonPersistentMemory();


  /// Update allocations for all tensors. This will redim dependent tensors
  /// using the input tensor dimensionality as given. This is relatively
  /// expensive. This *must be* called after the interpreter has been created
  /// and before running inference (and accessing tensor buffers), and *must be*
  /// called again if (and only if) an input tensor is resized. Returns status
  /// of success or failure.  Will fail if any of the ops in the model (other
  /// than those which were rewritten by delegates, if any) are not supported by
  /// the Interpreter's OpResolver.
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
  /// NOTE: `num_threads` should be >= -1. Setting `num_threads` to 0 has the
  /// effect to disable multithreading, which is equivalent to setting
  /// `num_threads` to 1. If set to the value -1, the number of threads used
  /// will be implementation-defined and platform-dependent.
  ///
  /// As TfLite interpreter could internally apply a TfLite delegate by default
  /// (i.e. XNNPACK), the number of threads that are available to the default
  /// delegate *should be* set via InterpreterBuilder APIs as follows:
  ///
  ///     std::unique_ptr<tflite::Interpreter> interpreter;
  ///     tflite::InterpreterBuilder builder(tflite model, op resolver);
  ///     builder.SetNumThreads(...)
  ///     ASSERT_EQ(builder(&interpreter), kTfLiteOk);
  ///
  /// WARNING: This API is deprecated: prefer using
  /// `InterpreterBuilder::SetNumThreads`, as documented above.
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
  /// Returns one of the following status codes:
  /// 1. kTfLiteOk: Success.
  /// 2. kTfLiteDelegateError: Delegation failed due to an error in the
  /// delegate, or the delegate parameter was null. The Interpreter has been
  /// restored to its pre-delegation state.
  /// NOTE: This undoes all delegates previously applied to the Interpreter.
  /// 3. kTfLiteApplicationError : Delegation failed to be applied due to the
  /// incompatibility with the TfLite runtime, e.g., the model graph is already
  /// immutable when applying the delegate. However, the interpreter could still
  /// be invoked.
  /// 4. kTfLiteUnresolvedOps: Delegation failed because the model has an
  /// operator that cannot be resolved. This can happen when the op is not
  /// registered or built with the TF Lite framework.
  /// 5. kTfLiteError: Unexpected/runtime failure.
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
  ///
  /// When using hardware delegation, Interpreter will make the data of output
  /// tensors available in `tensor->data` by default. If the application can
  /// consume the buffer handle directly (e.g. reading output from OpenGL
  /// texture), it can set this flag to false, so Interpreter won't copy the
  /// data from buffer handle to CPU memory.
  /// WARNING: This is an experimental API and subject to change.
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

  /// Assigns (or reassigns) a custom memory allocation for the given tensor.
  /// `flags` is a bitmask, see TfLiteCustomAllocationFlags.
  /// The runtime does NOT take ownership of the underlying memory.
  ///
  /// NOTE: User needs to call AllocateTensors() after this.
  /// Invalid/insufficient buffers will cause an error during AllocateTensors or
  /// Invoke (in case of dynamic shapes in the graph).
  ///
  /// Parameters should satisfy the following conditions:
  /// 1. tensor->allocation_type == kTfLiteArenaRw or kTfLiteArenaRwPersistent
  ///    In general, this is true for I/O tensors & variable tensors.
  /// 2. allocation->data has the appropriate permissions for runtime access
  ///    (Read-only for inputs, Read-Write for others), and outlives
  ///    Interpreter.
  /// 3. allocation->bytes >= tensor->bytes.
  ///    This condition is checked again if any tensors are resized.
  /// 4. allocation->data should be aligned to kDefaultTensorAlignment
  ///    defined in lite/util.h. (Currently 64 bytes)
  ///    This check is skipped if kTfLiteCustomAllocationFlagsSkipAlignCheck is
  ///    set through `flags`.
  ///
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus SetCustomAllocationForTensor(
      int tensor_index, const TfLiteCustomAllocation& allocation,
      int64_t flags = kTfLiteCustomAllocationFlagsNone);

  /// Apply InterpreterOptions which tunes behavior of the interpreter.
  /// WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus ApplyOptions(InterpreterOptions* options);

#ifndef DOXYGEN_SKIP
  /// Return the number of subgraphs in the model.
  /// WARNING: This is an experimental API and subject to change.
  size_t subgraphs_size() const { return subgraphs_.size(); }

  /// Get a pointer to a subgraph if in bounds.
  /// WARNING: This is an experimental API and subject to change.
  const Subgraph* subgraph(int subgraph_index) const {
    if (subgraph_index < 0 ||
        static_cast<size_t>(subgraph_index) >= subgraphs_size()) {
      return nullptr;
    }
    return subgraphs_[subgraph_index].get();
  }

  /// WARNING: This is an experimental API and subject to change.
  Subgraph* subgraph(int subgraph_index) {
    return const_cast<Subgraph*>(
        static_cast<const Interpreter*>(this)->subgraph(subgraph_index));
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
  friend class InterpreterBuilder;
  friend class tflite::InterpreterTest;
  friend class tflite::delegates::InterpreterUtils;
  friend class tflite::delegates::test_utils::TestDelegation;
  friend class tflite::interpreter_wrapper::InterpreterWrapper;

  /// Set the value of an external context.
  static void SetExternalContext(struct TfLiteContext* context,
                                 TfLiteExternalContextType type,
                                 TfLiteExternalContext* ctx);

  // Helper method that return the tensor index that corresponds to
  // a name in a SignatureDef. Defined by 'signature_key', and
  // 'signature_tensor_name'.
  // If 'is_input' is true then the tensor is checked in input tensors,
  // otherwise it will be checked in output tensors.
  // Returns -1 if the tensor is not found.
  int GetTensorIndexFromSignature(const char* signature_tensor_name,
                                  const char* signature_key,
                                  bool is_input) const {
    // Iterate directly and don't use other methods to avoid extra allocation.
    for (const auto& signature : signature_defs_) {
      if (signature.signature_key != signature_key) continue;
      auto& signature_list = (is_input ? signature.inputs : signature.outputs);
      auto tensor_iter = signature_list.find(signature_tensor_name);
      if (tensor_iter == signature_list.end()) return -1;
      return tensor_iter->second;
    }
    return -1;
  }

  // Applies TFLite default delegates.
  TfLiteStatus ApplyLazyDelegateProviders();

  // Private non-experimental implementation of ModifyGraphWithDelegate.
  // Unlike ModifyGraphWithDelegate, ModifyGraphWithDelegateImpl is defined in
  // interpreter.cc rather than in interpreter_experimental.cc, so it can be
  // used to implement other non-experimental methods.
  TfLiteStatus ModifyGraphWithDelegateImpl(TfLiteDelegate* delegate);

  // Same as ModifyGraphWithDelegateImpl except that it takes ownership of the
  // delegate.
  template <typename Delegate, typename Deleter>
  inline TfLiteStatus ModifyGraphWithDelegateImpl(
      std::unique_ptr<Delegate, Deleter>&& delegate) {
    Deleter deleter = std::move(delegate.get_deleter());

    // Note that we retain ownership of the delegate even if graph modification
    // fails, as delegate use will be in an indeterminate state at that point.
    owned_delegates_.emplace_back(
        delegate.release(), [deleter](TfLiteDelegate* delegate_to_delete) {
          deleter(
              static_cast<typename std::unique_ptr<Delegate, Deleter>::pointer>(
                  delegate_to_delete));
        });
    return ModifyGraphWithDelegateImpl(owned_delegates_.back().get());
  }

  // Overrides execution plan. ImplThis bounds checks indices sent in.
  // Note: Only used during initialization.
  TfLiteStatus SetExecutionPlan(const std::vector<int>& new_plan);

  // Sets the profiler to all subgraphs.
  void SetSubgraphProfiler();

  // Remove delegates (for fallback behaviour). The interpreter is invokable
  // afterwards.
  TfLiteStatus RemoveAllDelegates();

  // Returns true if delegates have been applied.
  bool HasDelegates();

  // Returns true if the model has been fully delegated.
  bool IsFullyDelegated() const;

  // Returns true if cancellation function returns true.
  bool IsCancelled();

  // Sets the list of signature defs in the model.
  void SetSignatureDef(std::vector<internal::SignatureDef> signature_defs) {
    signature_defs_ = std::move(signature_defs);
  }

  // Sets model metadata as a mapping of name (key) and buffer (value) strings.
  // Used by InterpreterBuilder, should be called after setting up subgraphs.
  TfLiteStatus SetMetadata(const std::map<std::string, std::string>& metadata);

  /// Adds `subgraphs_to_add` subgraphs, preserving pre-existing Subgraph
  /// entries. The value pointed to by `first_new_subgraph_index` will be set to
  /// the index of the first new subgraph if `first_new_subgraph_index` is
  /// non-null.
  void AddSubgraphs(int subgraphs_to_add,
                    int* first_new_subgraph_index = nullptr);

  /// Implementation of SetProfiler.
  /// Unlike SetProfiler, this is defined in interpreter.cc rather than in
  /// intepreter_experimental.cc, so it can be used by interpreter_builder.cc.
  void SetProfilerImpl(std::unique_ptr<Profiler> profiler);

  TfLiteStatus ApplyOptionsImpl(InterpreterOptions* options);

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

  // A map of resource Ids. Owned by interpreter and shared by multiple
  // subgraphs.
  resource::ResourceIDMap resource_ids_;

  // A map of intialization statuses, that indicate whether the intialization
  // subgraph invocation is done or not. Owned by interpreter and shared by
  // multiple subgraphs.
  resource::InitializationStatusMap initialization_status_map_;

  // Indicating delegates that the TFLite interpreter will apply by default.
  // An empty one means there's no delegate to be applied by default or
  // delegates have been applied and doesn't need to be applied again.
  using TfLiteDelegateCreator =
      std::function<TfLiteDelegatePtr(int /*num_threads*/)>;
  using TfLiteDelegateCreators = std::vector<TfLiteDelegateCreator>;
  TfLiteDelegateCreators lazy_delegate_providers_;

  // List of SignatureDefs obtained from the model.
  std::vector<internal::SignatureDef> signature_defs_;

  // Map of signature key to its corresponding SignatureRunner object.
  // A SignatureRunner is basically a wrapper of the Subgraph corresponding to
  // its SignatureDef.
  std::map<std::string, SignatureRunner> signature_runner_map_;

  // Model metadata stored as mapping of name (key) to buffer (value).
  // Data is mapped from the Metadata in TFLite flatbuffer model.
  std::map<std::string, std::string> metadata_;
};

}  // namespace tflite
#endif  // TENSORFLOW_LITE_INTERPRETER_H_
