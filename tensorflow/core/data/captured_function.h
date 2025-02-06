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
#ifndef TENSORFLOW_CORE_DATA_CAPTURED_FUNCTION_H_
#define TENSORFLOW_CORE_DATA_CAPTURED_FUNCTION_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class Device;
class OpKernelContext;
class ResourceMgr;

namespace data {

class CapturedFunction;
class InstantiatedCapturedFunction;

// Creates an iterator for a dataset which is created by applying the given
// function to the given input element.
absl::Status MakeIteratorFromInputElement(
    IteratorContext* ctx, const DatasetBaseIterator* parent,
    const std::vector<Tensor>& input_element, int64_t thread_index,
    const InstantiatedCapturedFunction& inst_captured_func,
    absl::string_view prefix, std::unique_ptr<IteratorBase>* out_iterator);

// Creates an iterator for a dataset which is created by applying the given
// function to the given input element. Pass non-null `node` to record
// processing time for modeling Iterator's GetNext() resource usage.
absl::Status MakeIteratorFromInputElement(
    IteratorContext* ctx, const DatasetBaseIterator* parent,
    const std::vector<Tensor>& input_element, int64_t thread_index,
    const InstantiatedCapturedFunction& inst_captured_func,
    absl::string_view prefix, std::unique_ptr<IteratorBase>* out_iterator,
    const std::shared_ptr<model::Node>& node);

struct ShortCircuitInfo {
  std::vector<int> indices;
  std::vector<bool> can_move;
};

// Metadata shared across all captures of the same function.
class FunctionMetadata {
 public:
  struct Params {
    bool use_inter_op_parallelism = true;
    bool use_default_device = true;
  };

  // Creates a new instance of the `FunctionMetadata` class, fetching function
  // from a context argument.
  static absl::Status Create(tensorflow::OpKernelConstruction* ctx,
                             const string& func_name, Params params,
                             std::shared_ptr<FunctionMetadata>* out_metadata);

  // Creates a new instance of the `FunctionMetadata` class, using the provided
  // function.
  static absl::Status Create(tensorflow::OpKernelConstruction* ctx,
                             NameAttrList&& func, Params params,
                             std::shared_ptr<FunctionMetadata>* out_metadata);

  // Returns the named list of function arguments.
  const NameAttrList& func() const { return func_; }

  // Returns a borrowed pointer to the function library that contains the
  // transitive closure of definitions used by the function.
  const FunctionLibraryDefinition* lib_def() const { return lib_def_.get(); }

  // Returns short-circuit information.
  const ShortCircuitInfo& short_circuit_info() const {
    return short_circuit_info_;
  }

  // Indicates whether a default device should be used for executing function
  // ops.
  bool use_default_device() const { return use_default_device_; }

  // Indicates whether to use inter-op parallelism for execution of the
  // function.
  bool use_inter_op_parallelism() const { return use_inter_op_parallelism_; }

  // Indicates whether the function should a multi-device function backend.
  bool use_multi_device_function() const { return use_multi_device_function_; }

 private:
  FunctionMetadata(NameAttrList&& func, Params params)
      : func_(std::move(func)),
        use_default_device_(params.use_default_device),
        use_inter_op_parallelism_(params.use_inter_op_parallelism) {}

  NameAttrList func_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_ = nullptr;
  ShortCircuitInfo short_circuit_info_;
  bool use_default_device_ = true;
  bool use_inter_op_parallelism_ = true;
  bool use_multi_device_function_ = true;
};

// Constructs and stores the parameters for the CapturedFunction Instantiate
// function.
struct InstantiateCapturedFunctionParams {
  explicit InstantiateCapturedFunctionParams(IteratorContext* ctx) {
    flr = ctx->flr();
    function_handle_cache = ctx->function_handle_cache();
    runner = ctx->runner();
  }

  explicit InstantiateCapturedFunctionParams(OpKernelContext* ctx) {
    flr = ctx->function_library();
    function_handle_cache = nullptr;
    runner = ctx->runner();
  }

  FunctionLibraryRuntime* flr;
  FunctionHandleCache* function_handle_cache;
  std::function<void(std::function<void()>)>* runner;
};

// A `CapturedFunction` encapsulates a TensorFlow function, plus any "captured"
// arguments that it closed over in the user program.
class CapturedFunction {
 public:
  // Creates a new instance using a list of named attributes, fetching captured
  // inputs from a context argument.
  static absl::Status Create(OpKernelContext* ctx,
                             std::shared_ptr<const FunctionMetadata> metadata,
                             const string& argument_name,
                             std::unique_ptr<CapturedFunction>* out_function);

  // Creates a new instance using a list of named attributes, using provided
  // captured inputs.
  static absl::Status Create(OpKernelContext* ctx,
                             std::shared_ptr<const FunctionMetadata> metadata,
                             std::vector<Tensor>&& captured_inputs,
                             std::unique_ptr<CapturedFunction>* out_function);

  // Adds the definition of this captured function into the given graph,
  // returning its captured inputs and types through the respective output
  // arguments.
  absl::Status AddToGraph(SerializationContext* ctx,
                          DatasetBase::DatasetGraphDefBuilder* b,
                          std::vector<Node*>* other_arguments,
                          DataTypeVector* other_arguments_types) const;

  // Instantiates this function for use in the given context, providing an
  // InstantiatedCapturedFunction that can be used to execute functions.
  absl::Status Instantiate(IteratorContext* ctx,
                           std::unique_ptr<InstantiatedCapturedFunction>*
                               instantiated_captured_function);

  absl::Status Instantiate(InstantiateCapturedFunctionParams params,
                           std::unique_ptr<InstantiatedCapturedFunction>*
                               instantiated_captured_function);

  // Determines whether the captured function is stateful.
  absl::Status CheckExternalState() const;

  // Returns the additional captured inputs that will be passed to the function.
  const std::vector<Tensor>& captured_inputs() const {
    return captured_inputs_;
  }

  // Returns the named list of function arguments.
  const NameAttrList& func() const { return metadata_->func(); }

  // Returns the transitive set of function definition required to instantiate
  // this function.
  const FunctionLibraryDefinition* lib_def() const {
    return metadata_->lib_def();
  }

  // If every function output corresponds to one of its inputs, the method
  // returns the mapping from output indices to input indices. Otherwise, it
  // returns an empty list.
  const ShortCircuitInfo& short_circuit_info() const {
    return metadata_->short_circuit_info();
  }

  // Indicates whether the function should use inter op parallelism.
  bool use_inter_op_parallelism() const {
    return metadata_->use_inter_op_parallelism();
  }

 private:
  CapturedFunction(std::shared_ptr<const FunctionMetadata> metadata,
                   std::vector<Tensor> captured_inputs);

  absl::Status IsMultiDevice(FunctionLibraryRuntime* flr,
                             bool* is_multi_device) const;

  const std::shared_ptr<const FunctionMetadata> metadata_;
  const std::vector<Tensor> captured_inputs_;

  CapturedFunction(const CapturedFunction&) = delete;
  void operator=(const CapturedFunction&) = delete;
};

// `InstantiatedCapturedFunction` encapsulates all the runtime support needed
// to execute a tensorflow function.
//
// While `CapturedFunction` encapsulates constant attributes of the function,
// such as its name and captured arguments, `InstantiatedCapturedFunction`
// encapsulates runtime aspects, such as `FunctionLibraryRuntime` and function
// handle.
//
// The `Iterator` related classes use `InstantiatedCapturedFunction` to execute
// functions outside of the normal `OpKernel::Compute()` context.
class InstantiatedCapturedFunction {
 public:
  // Runs the instantiated captured function. This method takes ownership of
  // the tensors in `args`, in order to be able to deallocate them as early as
  // possible. Use `RunWithBorrowedArgs()` if the caller needs to retain
  // ownership of the `args`.
  absl::Status Run(IteratorContext* ctx, std::vector<Tensor>&& args,
                   std::vector<Tensor>* rets) const;

  // Runs the instantiated captured function. This method takes ownership of
  // the tensors in `args`, in order to be able to deallocate them as early as
  // possible. Use `RunWithBorrowedArgs()` if the caller needs to retain
  // ownership of the `args`. Pass non-null `node` to record processing time
  // for modeling Iterator's GetNext() resource usage. When non-null node is
  // provided, the pre-requisite is that the calling thread has previously
  // called `DatasetBaseIterator::RecordStart().
  absl::Status Run(IteratorContext* ctx, std::vector<Tensor>&& args,
                   std::vector<Tensor>* rets,
                   const std::shared_ptr<model::Node>& node) const;

  // Synchronously runs the captured function on the given `args`, and stores
  // the results in `*rets`. Prefer to use `Run()` or `RunAsync()` when
  // possible.
  absl::Status RunWithBorrowedArgs(IteratorContext* ctx,
                                   const std::vector<Tensor>& args,
                                   std::vector<Tensor>* rets) const;

  // Synchronously runs the captured function on the given `args`, and stores
  // the results in `*rets`. Prefer to use `Run()` or `RunAsync()` when
  // possible. Pass non-null `node` to record processing time for modeling
  // Iterator's GetNext() resource usage. When non-null node is provided, the
  // pre-requisite is that the calling thread has previously called
  // `DatasetBaseIterator::RecordStart().
  absl::Status RunWithBorrowedArgs(
      IteratorContext* ctx, const std::vector<Tensor>& args,
      std::vector<Tensor>* rets,
      const std::shared_ptr<model::Node>& node) const;

  // Synchronously runs the captured function on the given `args`, and stores
  // the results in `*rets`. Prefer to use `Run()` or `RunAsync()` when
  // possible. This can be useful for calling a captured function in cases where
  // an `IteratorContext*` is not available (such as a destructor).
  //
  // TODO(b/144278100): Avoid running functions without IteratorContext.
  absl::Status RunInstantiated(const std::vector<Tensor>& args,
                               std::vector<Tensor>* rets);

  // Asynchronously runs the captured function on the given `args`, stores the
  // results in `*rets`, and calls the given `done` callback when the function
  // returns. This method takes ownership of the tensors in `args`, in order to
  // be able to deallocate them as early as possible. Pass non-null `node` to
  // record processing time for modeling Iterator's GetNext() resource usage.
  // When non-null node is provided, the pre-requisite is that the calling
  // thread has previously called `DatasetBaseIterator::RecordStart().
  void RunAsync(IteratorContext* ctx, std::vector<Tensor>&& args,
                std::vector<Tensor>* rets,
                FunctionLibraryRuntime::DoneCallback done,
                const std::shared_ptr<model::Node>& node) const {
    RunAsync(*(ctx->runner()), ctx->cancellation_manager(),
             ctx->collective_executor(), std::move(args), rets, done, node);
  }

  // A version of `RunAsync` that does not take an `IteratorContext` but a
  // runner, a cancellation manager, and a collective executor.
  void RunAsync(std::function<void(std::function<void()>)> runner,
                CancellationManager* parent_cancellation_manager,
                CollectiveExecutor* collective_executor,
                std::vector<Tensor>&& args, std::vector<Tensor>* rets,
                FunctionLibraryRuntime::DoneCallback done,
                const std::shared_ptr<model::Node>& node) const;

  std::string func_name() const { return captured_func_->func().name(); }

 private:
  friend class CapturedFunction;

  InstantiatedCapturedFunction(
      FunctionLibraryRuntime* lib, FunctionLibraryRuntime::Handle f_handle,
      DataTypeVector ret_types,
      std::function<void(std::function<void()>)> runner,
      CapturedFunction* captured_func, bool is_multi_device);

  // Determines whether a rendezvous object should be created when running the
  // instantiated function.
  bool ShouldCreateRendezvous() const;

  FunctionLibraryRuntime* const lib_;  // Not owned.
  const FunctionLibraryRuntime::Handle f_handle_;
  const DataTypeVector ret_types_;
  // Note: We capture the runner at function instantiation time to be able to
  // run the function without `IteratorContext` via `RunInstantiated`.
  std::function<void(std::function<void()>)> captured_runner_;
  CapturedFunction* const captured_func_;  // Not owned.
  const bool is_multi_device_;

  InstantiatedCapturedFunction(const InstantiatedCapturedFunction&) = delete;
  void operator=(const InstantiatedCapturedFunction&) = delete;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_CAPTURED_FUNCTION_H_
