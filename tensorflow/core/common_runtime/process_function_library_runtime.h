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
#ifndef THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_FUNCTION_LIBRARY_RUNTIME_H_
#define THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_FUNCTION_LIBRARY_RUNTIME_H_

#include <unordered_map>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// A class that stores all the FunctionLibraryRuntime objects, one per device.
class ProcessFunctionLibraryRuntime {
 public:
  // Creates FunctionLibraryRuntime objects for each device in the provided
  // DeviceMgr. Caller needs to make sure that device_mgr and lib_def outlive
  // this object.
  ProcessFunctionLibraryRuntime(const DeviceMgr* device_mgr, Env* env,
                                int graph_def_version,
                                const FunctionLibraryDefinition* lib_def,
                                const OptimizerOptions& optimizer_options);

  ProcessFunctionLibraryRuntime(const DeviceMgr* device_mgr, Env* env,
                                int graph_def_version,
                                const FunctionLibraryDefinition* lib_def,
                                const OptimizerOptions& optimizer_options,
                                CustomKernelCreator custom_kernel_creator);

  // Given a list of attrs on a function, extracts the "_target" attribute which
  // indicates which device to run the function on. If it can't find the _target
  // attribute, returns "". Canonicalizes the device name.
  static string ObtainFunctionTarget(const AttrSlice& attrs);

  static const char kDefaultFLRDevice[];
  // Returns the FunctionLibraryRuntime for the corresponding device_name.
  FunctionLibraryRuntime* GetFLR(const string& device_name);

  // For a given canonicalized key signature of the function instantiated
  // on device `device_name` and a `local_handle`, creates a handle and returns
  // that value. Use core/common_runtime/framework/function.h::Canonicalize
  // to canonicalize the function signature.
  FunctionLibraryRuntime::Handle AddHandle(
      const string& function_key, const string& device_name,
      FunctionLibraryRuntime::LocalHandle local_handle);

  // Returns a handle if found for the given key, else returns kInvalidHandle.
  FunctionLibraryRuntime::Handle GetHandle(const string& function_key) const;

  // For the given handle instantiated on device `device_name` returns the local
  // index of instantiation of that function. If the function was not
  // instantiated on `device_name` returns kInvalidLocalHandle.
  FunctionLibraryRuntime::LocalHandle GetHandleOnDevice(
      const string& device_name, FunctionLibraryRuntime::Handle handle);

  // Returns true if function with handle `handle` was instantiated on device
  // `device_name`.
  bool IsInstantiatedOnDevice(const string& device_name,
                              FunctionLibraryRuntime::Handle handle);

  // Instantiates the function. See framework/function.h for more details.
  // Allows for function_name to be instantiated on different devices
  // as specified in attrs.
  Status Instantiate(const string& function_name, AttrSlice attrs,
                     FunctionLibraryRuntime::Handle* handle);

  // Runs the function with given `handle`. Function could have been
  // instantiated on any device. More details in framework/function.h
  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::Handle handle, gtl::ArraySlice<Tensor> args,
           std::vector<Tensor>* rets,
           FunctionLibraryRuntime::DoneCallback done);

 private:
  mutable mutex mu_;

  // Holds all the function invocations here.
  std::unordered_map<string, FunctionLibraryRuntime::Handle> table_
      GUARDED_BY(mu_);
  std::vector<std::pair<string, FunctionLibraryRuntime::LocalHandle>>
      function_data_ GUARDED_BY(mu_);
  std::unordered_map<string, std::unique_ptr<FunctionLibraryRuntime>> flr_map_;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_FUNCTION_LIBRARY_RUNTIME_H_
