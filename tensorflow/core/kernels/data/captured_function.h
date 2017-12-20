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
#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DATA_CAPTURED_FUNCTION_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DATA_CAPTURED_FUNCTION_H_

#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class Device;
class OpKernelContext;
class ResourceMgr;

// A `CapturedFunction` encapsulates a TensorFlow function and all of
// the runtime support required to execute it.
//
// The `Dataset`-related classes use `CapturedFunction` to execute
// TensorFlow functions outside a the normal `OpKernel::Compute()`
// context.
//
// NOTE(mrry): Here we are taking a conservative approach to dealing with
// ownership of the various framework and runtime objects that are needed
// to execute functions. We copy the function library *definition* (i.e.
// a set of FunctionDefs) out of this kernel's context's function library
// *runtime*, then we use that together with a specially-created
// ThreadPoolDevice to build a new FunctionLibraryRuntime for the Dataset.
//
// We need to do this (or refactor the ownership of framework components
// in each of the session implementations) to make it possible to close
// down a ParallelMapDataset::Iterator when its session is closed.
//
// TODO(mrry): Clean this up. Investigate whether it would be possible to
// reuse the session's FunctionLibraryRuntime(s) or Device(s).
class CapturedFunction {
 public:
  // NOTE(mrry): The `captured_inputs` are passed by value. For
  // efficiency, you are recommended to move this argument into the call.
  static Status Create(OpKernelContext* ctx, const NameAttrList& func,
                       int graph_def_version,
                       std::vector<Tensor> captured_inputs,
                       std::unique_ptr<CapturedFunction>* out_function);

  // Synchronously runs the captured function on the given `args`, and stores
  // the results in `*rets`. This method takes ownership of the tensors in
  // `args`, in order to be able to deallocate them as early as possible.
  // Use `RunWithBorrowedArgs()` if the caller needs to retain ownership of
  // the `args`.
  Status Run(FunctionLibraryRuntime::Options f_opts, std::vector<Tensor>&& args,
             std::vector<Tensor>* rets);

  // Synchronously runs the captured function on the given `args`, and stores
  // the results in `*rets`. Prefer to use `Run()` or `RunAsync()` when
  // possible.
  Status RunWithBorrowedArgs(FunctionLibraryRuntime::Options f_opts,
                             const std::vector<Tensor>& args,
                             std::vector<Tensor>* rets);

  // Asynchronously runs the captured function on the given `args`, stores
  // the results in `*rets`, and calls the given `done` callback when the
  // function returns. This method takes ownership of the tensors in `args`,
  // in order to be able to deallocate them as early as possible.
  void RunAsync(FunctionLibraryRuntime::Options f_opts,
                std::vector<Tensor>&& args, std::vector<Tensor>* rets,
                FunctionLibraryRuntime::DoneCallback done);

  // Returns a borrowed pointer to the `ResourceManager` used when this
  // function is run.
  ResourceMgr* resource_manager() const { return device_->resource_manager(); }

  // Returns that additional captured inputs that will be passed to the function
  // when `Run*()` is called.
  const std::vector<Tensor>& captured_inputs() { return captured_inputs_; }

  // Returns a step ID for use when running a `CapturedFunction`.
  static int64 generate_step_id() {
    // Choose a step ID that is guaranteed not to clash with any
    // Session-generated step ID. DirectSession only generates
    // non-negative step IDs (contiguous, starting from 0), and
    // MasterSession generates 56-bit random step IDs whose MSB is
    // always 0, so a negative random step ID should suffice.
    return -std::abs(static_cast<int64>(random::New64()));
  }

 private:
  CapturedFunction(Device* device, std::unique_ptr<DeviceMgr> device_mgr,
                   std::unique_ptr<FunctionLibraryDefinition> flib_def,
                   std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
                   FunctionLibraryRuntime* lib,
                   FunctionLibraryRuntime::Handle f_handle,
                   std::vector<Tensor> captured_inputs,
                   DataTypeSlice ret_types);

  Device* const device_;  // owned by device_mgr_.
  const std::unique_ptr<DeviceMgr> device_mgr_;
  const std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  const std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  FunctionLibraryRuntime* const lib_;  // owned by pflr_.
  const FunctionLibraryRuntime::Handle f_handle_;
  const std::vector<Tensor> captured_inputs_;
  DataTypeSlice ret_types_;  // owned by pflr_.

  TF_DISALLOW_COPY_AND_ASSIGN(CapturedFunction);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DATA_CAPTURED_FUNCTION_H_
