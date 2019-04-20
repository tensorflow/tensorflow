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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_CAPTURED_FUNCTION_H_
#define TENSORFLOW_CORE_KERNELS_DATA_CAPTURED_FUNCTION_H_

#include <memory>
#include <vector>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
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

Status MakeIteratorFromInputElement(
    IteratorContext* ctx, const std::vector<Tensor>& input_element,
    int64 thread_index, const InstantiatedCapturedFunction& inst_captured_func,
    StringPiece prefix, std::unique_ptr<IteratorBase>* out_iterator);

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
  ~InstantiatedCapturedFunction();

  // Runs the instantiated captured function. This method takes ownership of
  // the tensors in `args`, in order to be able to deallocate them as early as
  // possible. Use `RunWithBorrowedArgs()` if the caller needs to retain
  // ownership of the `args`.
  Status Run(IteratorContext* ctx, std::vector<Tensor>&& args,
             std::vector<Tensor>* rets) const;

  // Synchronously runs the captured function on the given `args`, and stores
  // the results in `*rets`. Prefer to use `Run()` or `RunAsync()` when
  // possible.
  Status RunWithBorrowedArgs(IteratorContext* ctx,
                             const std::vector<Tensor>& args,
                             std::vector<Tensor>* rets) const;

  // Synchronously runs the captured function on the given `args`, and stores
  // the results in `*rets`. Prefer to use `Run()` or `RunAsync()` when
  // possible. This can be useful for calling a captured
  // function in cases where an `IteratorContext*` is not available
  // (such as a destructor).
  Status RunInstantiated(const std::vector<Tensor>& args,
                         std::vector<Tensor>* rets);

  // Asynchronously runs the captured function on the given `args`, stores
  // the results in `*rets`, and calls the given `done` callback when the
  // function returns. This method takes ownership of the tensors in `args`,
  // in order to be able to deallocate them as early as possible.
  void RunAsync(IteratorContext* ctx, std::vector<Tensor>&& args,
                std::vector<Tensor>* rets,
                FunctionLibraryRuntime::DoneCallback done,
                const string& prefix) const;

  // Returns a step ID for use when running an `InstantiatedCapturedFunction`.
  static int64 generate_step_id() {
    // Choose a step ID that is guaranteed not to clash with any
    // Session-generated step ID. DirectSession only generates
    // non-negative step IDs (contiguous, starting from 0), and
    // MasterSession generates 56-bit random step IDs whose MSB is
    // always 0, so a negative random step ID should suffice.
    return -std::abs(static_cast<int64>(random::New64()));
  }

 private:
  InstantiatedCapturedFunction(
      FunctionLibraryRuntime* lib, FunctionLibraryRuntime::Handle f_handle,
      DataTypeVector ret_types,
      std::function<void(std::function<void()>)> runner,
      CapturedFunction* captured_func);

  friend class CapturedFunction;

  FunctionLibraryRuntime* const lib_;
  const FunctionLibraryRuntime::Handle f_handle_;
  const DataTypeVector ret_types_;
  std::function<void(std::function<void()>)> captured_runner_;
  CapturedFunction* const captured_func_;

  TF_DISALLOW_COPY_AND_ASSIGN(InstantiatedCapturedFunction);
};

// A `CapturedFunction` encapsulates a TensorFlow function, plus any "captured"
// arguments that it closed over in the user program.
class CapturedFunction {
 public:
  struct Params {
    bool use_inter_op_parallelism = true;
    bool is_multi_device_function = false;
    std::shared_ptr<FunctionLibraryDefinition> lib_def = nullptr;
  };

  // Creates a new instance using a list of named attributes, fetching captured
  // inputs from a context argument.
  static Status Create(const NameAttrList& func, OpKernelContext* ctx,
                       const string& argument_name, Params params,
                       std::unique_ptr<CapturedFunction>* out_function);

  // Creates a new instance using a list of named attributes, using provided
  // captured inputs.
  static Status Create(const NameAttrList& func, OpKernelContext* ctx,
                       std::vector<Tensor>&& captured_inputs, Params params,
                       std::unique_ptr<CapturedFunction>* out_function);

  // Adds the definition of this captured function into the given graph,
  // returning its captured inputs and types through the respective output
  // arguments.
  Status AddToGraph(SerializationContext* ctx,
                    DatasetBase::DatasetGraphDefBuilder* b,
                    std::vector<Node*>* other_arguments,
                    DataTypeVector* other_arguments_types) const;

  // Instantiates this function for use in the given context, providing an
  // InstantiatedCapturedFunction that can be used to execute functions.
  Status Instantiate(IteratorContext* ctx,
                     std::unique_ptr<InstantiatedCapturedFunction>*
                         instantiated_captured_function);

  // Returns the additional captured inputs that will be passed to the function.
  const std::vector<Tensor>& captured_inputs() const {
    return captured_inputs_;
  }

  // Returns the named list of function arguments.
  const NameAttrList& func() const { return func_; }

  // Indicates whether the function is multi-device.
  bool is_multi_device_function() const { return is_multi_device_function_; }

  // Returns the transitive set of function definition required to instantiate
  // this function.
  const FunctionLibraryDefinition* lib_def() const { return lib_def_.get(); }

  // Indicates whether the function should use inter op parallelism.
  bool use_inter_op_parallelism() const { return use_inter_op_parallelism_; }

 private:
  CapturedFunction(const NameAttrList& func,
                   std::vector<Tensor> captured_inputs, Params params);

  const NameAttrList func_;
  const std::vector<Tensor> captured_inputs_;
  const bool use_inter_op_parallelism_;
  const bool is_multi_device_function_;
  std::shared_ptr<const FunctionLibraryDefinition> lib_def_;

  TF_DISALLOW_COPY_AND_ASSIGN(CapturedFunction);
};
}  // namespace data

// TODO(b/114112161): Remove these aliases when all users have moved over to the
// `tensorflow::data` namespace.
using data::CapturedFunction;

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_CAPTURED_FUNCTION_H_
