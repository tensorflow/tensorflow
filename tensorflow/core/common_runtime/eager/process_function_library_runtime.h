/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_PROCESS_FUNCTION_LIBRARY_RUNTIME_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_PROCESS_FUNCTION_LIBRARY_RUNTIME_H_

#include <memory>
#include <unordered_map>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"
#endif  // IS_MOBILE_PLATFORM

namespace tensorflow {
namespace eager {

// A ProcessFunctionLibraryRuntime which supports running functions with inputs
// on remote devices.
// TODO(b/134094971): Support outputting tensors on remote devices.
class EagerProcessFunctionLibraryRuntime
    : public ProcessFunctionLibraryRuntime {
 public:
  EagerProcessFunctionLibraryRuntime(
      const DeviceMgr* device_mgr, Env* env, const ConfigProto* config,
      int graph_def_version, const FunctionLibraryDefinition* lib_def,
      const OptimizerOptions& optimizer_options,
      thread::ThreadPool* thread_pool = nullptr,
      DistributedFunctionLibraryRuntime* parent = nullptr,
      const CustomKernelCreator* custom_kernel_creator = nullptr)
      : ProcessFunctionLibraryRuntime(
            device_mgr, env, config, graph_def_version, lib_def,
            optimizer_options, thread_pool, parent, custom_kernel_creator) {}

#if !defined(IS_MOBILE_PLATFORM)
  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::Handle handle,
           const FunctionArgsInterface& args, std::vector<Tensor>* rets,
           FunctionLibraryRuntime::DoneCallback done) const override;

 private:
  void RunRemoteDevice(
      const FunctionLibraryRuntime::Options& opts,
      FunctionLibraryRuntime::Handle local_handle, const InternalArgsView& args,
      std::vector<Tensor>* rets,
      FunctionLibraryRuntime::DoneCallback done) const override;
#endif  // IS_MOBILE_PLATFORM
};

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_PROCESS_FUNCTION_LIBRARY_RUNTIME_H_
