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
  // DeviceMgr. Caller needs to make sure that device_mgr, lib_def and parent
  // (if provided) outlive this object.
  ProcessFunctionLibraryRuntime(const DeviceMgr* device_mgr, Env* env,
                                int graph_def_version,
                                const FunctionLibraryDefinition* lib_def,
                                const OptimizerOptions& optimizer_options,
                                DistributedFunctionLibraryRuntime* parent);

  ProcessFunctionLibraryRuntime(const DeviceMgr* device_mgr, Env* env,
                                int graph_def_version,
                                const FunctionLibraryDefinition* lib_def,
                                const OptimizerOptions& optimizer_options,
                                CustomKernelCreator custom_kernel_creator,
                                DistributedFunctionLibraryRuntime* parent);

  ProcessFunctionLibraryRuntime(const DeviceMgr* device_mgr, Env* env,
                                int graph_def_version,
                                const FunctionLibraryDefinition* lib_def,
                                const OptimizerOptions& optimizer_options);

  ProcessFunctionLibraryRuntime(const DeviceMgr* device_mgr, Env* env,
                                int graph_def_version,
                                const FunctionLibraryDefinition* lib_def,
                                const OptimizerOptions& optimizer_options,
                                CustomKernelCreator custom_kernel_creator);

  // Sends `tensors_to_send` from `source_device` to `target_device` using
  // `rendezvous`. `key_prefix` is used as a prefix for the keys sent to the
  // Rendezvous. `device_context` should be the DeviceContext of the device
  // doing the sending. `alloc_attrs` should either be empty or be the size of
  // `tensors_to_send` and indicates how the input tensors are allocated. Method
  // takes references on each of the `tensors_to_send`. Method doesn't block.
  static Status SendTensors(const string& source_device,
                            const string& target_device,
                            const string& key_prefix, int64 src_incarnation,
                            gtl::ArraySlice<Tensor> tensors_to_send,
                            DeviceContext* device_context,
                            const std::vector<AllocatorAttributes>& alloc_attrs,
                            Rendezvous* rendezvous);

  typedef std::function<void(const Status&)> StatusCallback;

  // Receives `received_tensors` from `target_device` (originally sent from
  // `source_device`) using `rendezvous`. Uses `key_prefix` to construct the
  // keys to be retrieved. `device_context` should be for the device receiving
  // the tensors. `alloc_attrs` indicates how to allocate the received
  // tensors and should either be empty or `num_tensors` in size. Method doesn't
  // block and calls `done` when `num_tensors` are fetched.
  static void ReceiveTensorsAsync(
      const string& source_device, const string& target_device,
      const string& key_prefix, int64 src_incarnation, int64 num_tensors,
      DeviceContext* device_context,
      const std::vector<AllocatorAttributes>& alloc_attrs,
      Rendezvous* rendezvous, std::vector<Tensor>* received_tensors,
      const StatusCallback& done);

  static const char kDefaultFLRDevice[];
  // Returns the FunctionLibraryRuntime for the corresponding device_name.
  FunctionLibraryRuntime* GetFLR(const string& device_name);

  // Returns the device incarnation for the given device_name.
  Status GetDeviceIncarnation(const string& device_name, int64* incarnation);

  // For a given canonicalized key signature of the function instantiated
  // on device `device_name` and a `local_handle`, creates a handle and returns
  // that value. Uses core/common_runtime/framework/function.h::Canonicalize
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
                     const FunctionLibraryRuntime::InstantiateOptions& options,
                     FunctionLibraryRuntime::Handle* handle);

  // Delegates to the local FLR that owns state corresponding to `handle` and
  // tells it to release it. If the `handle` isnt' needed at all, the local FLR
  // might call RemoveHandle on this to get rid of the state owned by the Proc
  // FLR.
  Status ReleaseHandle(FunctionLibraryRuntime::Handle handle);

  // Runs the function with given `handle`. Function could have been
  // instantiated on any device. More details in framework/function.h
  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::Handle handle, gtl::ArraySlice<Tensor> args,
           std::vector<Tensor>* rets,
           FunctionLibraryRuntime::DoneCallback done);

 private:
  // For a given device_name, returns a DeviceContext for copying
  // tensors to/from the device.
  Status GetDeviceContext(const string& device_name,
                          DeviceContext** device_context);

  // Looks up the information for the given `handle` and returns the name
  // of the device where the function is registered.
  string GetDeviceName(FunctionLibraryRuntime::Handle handle);

  // Removes handle from the state owned by this object.
  Status RemoveHandle(FunctionLibraryRuntime::Handle handle);

  friend class FunctionLibraryRuntimeImpl;

  mutable mutex mu_;

  struct FunctionData {
    const string target_device;
    const FunctionLibraryRuntime::LocalHandle local_handle;

    FunctionData(const string& target_device,
                 FunctionLibraryRuntime::LocalHandle local_handle)
        : target_device(target_device), local_handle(local_handle) {}
    FunctionData() : FunctionData("", -1) {}
  };

  const DeviceMgr* const device_mgr_;
  const FunctionLibraryDefinition* lib_def_;
  // Holds all the function invocations here.
  std::unordered_map<string, FunctionLibraryRuntime::Handle> table_
      GUARDED_BY(mu_);
  std::unordered_map<FunctionLibraryRuntime::Handle, FunctionData>
      function_data_ GUARDED_BY(mu_);
  std::unordered_map<Device*, std::unique_ptr<FunctionLibraryRuntime>> flr_map_;
  int next_handle_ GUARDED_BY(mu_);
  DistributedFunctionLibraryRuntime* const parent_;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_FUNCTION_LIBRARY_RUNTIME_H_
