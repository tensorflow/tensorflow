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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_FUNCTION_LIBRARY_RUNTIME_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_FUNCTION_LIBRARY_RUNTIME_H_

#include <unordered_map>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// A class that stores all the FunctionLibraryRuntime objects, one per device.
class ProcessFunctionLibraryRuntime {
 public:
  // Creates FunctionLibraryRuntime objects for each device in the provided
  // DeviceMgr. Caller needs to make sure that device_mgr, lib_def and parent
  // (if provided) outlive this object.
  ProcessFunctionLibraryRuntime(
      const DeviceMgr* device_mgr, Env* env, int graph_def_version,
      const FunctionLibraryDefinition* lib_def,
      const OptimizerOptions& optimizer_options,
      thread::ThreadPool* thread_pool = nullptr,
      DistributedFunctionLibraryRuntime* parent = nullptr,
      const CustomKernelCreator* custom_kernel_creator = nullptr);

  ~ProcessFunctionLibraryRuntime() {
    // Deleting the FunctionLibraryRuntime map will delete the function handles
    // registered in it, which may call ReleaseHandle in this class again to
    // release their sub-function. These circular calls may casue segfault
    // since the flr_map_ may has already been deleted. Explicitly releasing
    // flr_map_ here and checking flr_map_ in ReleaseHandle to avoid this.
    flr_map_.reset();
  }

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
      StatusCallback done);

  static const char kDefaultFLRDevice[];
  // Returns the FunctionLibraryRuntime for the corresponding device_name.
  FunctionLibraryRuntime* GetFLR(const string& device_name) const;

  // Returns the return types for the function identified by handle `h`.
  Status GetRetTypes(FunctionLibraryRuntime::Handle h,
                     DataTypeVector* ret_types);

  // Returns the device incarnation for the given device_name.
  Status GetDeviceIncarnation(const string& device_name,
                              int64* incarnation) const;

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
  // instantiated on `device_name` or the function is multi-device,
  // returns kInvalidLocalHandle.
  FunctionLibraryRuntime::LocalHandle GetHandleOnDevice(
      const string& device_name, FunctionLibraryRuntime::Handle handle) const;

  // Fills `output_devices` with the devices on which the results will
  // be produced. If some output is produced on CPU, the corresponding Device*
  // is set to nullptr. If some output is DT_RESOURCE, the corresponding Device*
  // is set to the device backing the resource.
  // REQUIRES: `handle` identifies a multi-device function.
  Status GetOutputDevices(FunctionLibraryRuntime::Handle handle,
                          std::vector<Device*>* output_devices) const;

  // Returns true if function with handle `handle` was instantiated on device
  // `device_name`. Returns false for multi-device functions.
  bool IsInstantiatedOnDevice(const string& device_name,
                              FunctionLibraryRuntime::Handle handle) const;

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
  // For multi-device functions, calls ReleaseHandle on local FLRs for each
  // component function that is part of this multi-device function.
  // Each local FLR might call RemoveHandle on this.
  Status ReleaseHandle(FunctionLibraryRuntime::Handle handle);

  // Runs the function with given `handle`. Function could have been
  // instantiated on any device. More details in framework/function.h
  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::Handle handle, gtl::ArraySlice<Tensor> args,
           std::vector<Tensor>* rets,
           FunctionLibraryRuntime::DoneCallback done) const;
  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::Handle handle, CallFrameInterface* frame,
           FunctionLibraryRuntime::DoneCallback done) const;

  const DeviceMgr* device_mgr() { return device_mgr_; }

  const DeviceSet* device_set() { return &device_set_; }

  const FunctionLibraryDefinition* GetFunctionLibraryDefinition() const {
    return lib_def_;
  }

 private:
  friend class FunctionLibraryRuntimeImpl;

  using DeviceAndFHandle = std::pair<string, FunctionLibraryRuntime::Handle>;
  using ArgAndRetIndices = std::pair<std::vector<int>, std::vector<int>>;
  using ArgAndRetAllocAttrs = std::pair<std::vector<AllocatorAttributes>,
                                        std::vector<AllocatorAttributes>>;

  FunctionLibraryRuntime::Handle AddHandleLocked(
      const string& function_key, const string& device_name,
      FunctionLibraryRuntime::LocalHandle local_handle)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Structure to keep track of how a component function (a single-device
  // piece of a multi-device function) fits into the multi-device function.
  struct ComponentFunctionData {
    // The handle for the instantiated component function.
    FunctionLibraryRuntime::Handle handle_;
    // arg_indices_.size() is the number of arguments to the component function.
    // The i-th argument of the component function comes from the
    // `arg_indices_[i]`-th argument of the multi-device function.
    std::vector<int> arg_indices_;
    // ret_indices_.size() is the number of return values of the component
    // function.  The i-th return value of the component function goes to the
    // `ret_indices_[i]`-th return value of the multi-device function.
    std::vector<int> ret_indices_;
    // arg_alloc_attrs_[i] are the allocator attributes of the i-th argument to
    // the component function.
    std::vector<AllocatorAttributes> arg_alloc_attrs_;
    // ret_alloc_attrs_[i] are the allocator attributes of the i-th return value
    // of the component function.
    std::vector<AllocatorAttributes> ret_alloc_attrs_;
  };

  // Data structure holding information for a single instantiated multi-device
  // function.
  // The fields are filled in during instantiation. Once the object is
  // added to mdevice_data_, all fields are constant.
  struct MultiDeviceFunctionData {
    MultiDeviceFunctionData(const string& function_name,
                            const string& function_key, int num_outputs,
                            FunctionLibraryDefinition&& lib_def,
                            DataTypeVector ret_types)
        : function_name_(function_name),
          function_key_(function_key),
          instantiation_counter_(1),
          lib_def_(std::move(lib_def)),
          num_outputs_(num_outputs),
          ret_types_(std::move(ret_types)) {}

    const string function_name_;
    const string function_key_;
    uint64 instantiation_counter_;
    // A library that contains definitions of component functions and their
    // transitive dependencies.
    FunctionLibraryDefinition lib_def_;
    // Stored here to resize the output tensor vector when function is run.
    const int num_outputs_;
    DataTypeVector ret_types_;

    // Maps the device name to the information about the component function
    // be run on this device.
    std::unordered_map<string, ComponentFunctionData> glue_;
  };

  // For a given device_name, returns a DeviceContext for copying
  // tensors to/from the device.
  Status GetDeviceContext(const string& device_name,
                          DeviceContext** device_context) const;

  // Looks up the information for the given `handle` and returns the name
  // of the device where the function is registered.
  string GetDeviceName(FunctionLibraryRuntime::Handle handle) const;

  // Removes handle from the state owned by this object.
  Status RemoveHandle(FunctionLibraryRuntime::Handle handle);

  // Clones ProcessFunctionLibraryRuntime and FunctionLibraryDefinition
  // (transferring ownership of both to the caller). Note that the
  // ProcessFunctionLibraryRuntime borrows a pointer to the
  // FunctionLibraryDefinition and so the FunctionLibraryDefinition should
  // outlive the ProcessFunctionLibraryRuntime.
  //
  // The `skip_flib_def` argument controls whether the method should clone the
  // FunctionLibraryDefinition (default behavior) or return an empty function
  // library. The latter is used by tf.data, which manages
  // FunctionLibraryDefinitions for its functions independently (and passes
  // these into the FunctionLibraryRuntime through an overlay), to avoid linear
  // runtime w.r.t. to number of functions in the current function library.
  Status Clone(Env* env, int graph_def_version,
               const OptimizerOptions& optimizer_options,
               const CustomKernelCreator* custom_kernel_creator,
               std::unique_ptr<FunctionLibraryDefinition>* out_lib_def,
               std::unique_ptr<ProcessFunctionLibraryRuntime>* out_pflr,
               bool skip_flib_def = false) const;

  Status ReleaseMultiDeviceHandle(FunctionLibraryRuntime::Handle handle);

  // If handle represents a multi-device function, returns the multi-device
  // data associated with handle. Else, nullptr.
  MultiDeviceFunctionData* IsMultiDevice(
      FunctionLibraryRuntime::Handle handle) const;

  Status InstantiateMultiDevice(
      const string& function_name, AttrSlice attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      FunctionLibraryRuntime::Handle* handle);

  FunctionLibraryRuntime::Handle AddMultiDeviceHandle(
      const std::unique_ptr<MultiDeviceFunctionData> data,
      const string& function_key);

  // TODO(iga): Reword
  // Pins each arg that emits a `DT_RESOURCE` tensor to the device on which the
  // corresponding resource lives. This ensures that the Placer assigns ops that
  // access these resources to the appropriate devices.
  Status PinArgsAndRets(const std::vector<string>& input_devices,
                        const std::vector<string>& output_devices,
                        const DeviceSet& device_set,
                        const std::vector<Node*>& arg_nodes,
                        const std::vector<Node*>& ret_nodes) const;

  struct CleanUpItem {
    string device;
    uint64 step_id;
    FunctionLibraryRuntime::Handle local_handle;
  };

  void RunInternal(const FunctionLibraryRuntime::Options& opts,
                   FunctionLibraryRuntime::Handle handle,
                   gtl::ArraySlice<Tensor> args, std::vector<Tensor>* rets,
                   std::vector<std::unique_ptr<CleanUpItem>>* cleanup_items,
                   FunctionLibraryRuntime::DoneCallback done) const;

  void RunMultiDevice(const FunctionLibraryRuntime::Options& opts,
                      FunctionLibraryRuntime::Handle handle,
                      gtl::ArraySlice<Tensor> args, std::vector<Tensor>* rets,
                      std::vector<std::unique_ptr<CleanUpItem>>* cleanup_items,
                      FunctionLibraryRuntime::DoneCallback done) const;
  void CleanUp(std::vector<std::unique_ptr<CleanUpItem>>* items,
               FunctionLibraryRuntime::DoneCallback done) const;

  // Data structure holding information for a single instantiated remote
  // (to be executed on `target_device`) function.
  class FunctionData {
   public:
    FunctionData(const string& target_device,
                 FunctionLibraryRuntime::LocalHandle local_handle,
                 const string& function_key)
        : target_device_(target_device),
          local_handle_(local_handle),
          function_key_(function_key) {}

    string target_device() { return target_device_; }
    const string& function_key() { return function_key_; }

    FunctionLibraryRuntime::LocalHandle local_handle() {
      mutex_lock l(mu_);
      return local_handle_;
    }

    // Initializes the FunctionData object by potentially making an Initialize
    // call to the DistributedFunctionLibraryRuntime.
    Status DistributedInit(
        DistributedFunctionLibraryRuntime* parent, const string& function_name,
        const FunctionLibraryDefinition& lib_def, AttrSlice attrs,
        const FunctionLibraryRuntime::InstantiateOptions& options);

   private:
    mutex mu_;

    const string target_device_;
    FunctionLibraryRuntime::LocalHandle local_handle_ GUARDED_BY(mu_);
    const string function_key_;
    bool init_started_ GUARDED_BY(mu_) = false;
    Status init_result_ GUARDED_BY(mu_);
    Notification init_done_;
  };

  mutable mutex mu_;

  Env* const env_;
  const DeviceMgr* const device_mgr_;
  DeviceSet device_set_;
  const FunctionLibraryDefinition* lib_def_;
  thread::ThreadPool* default_thread_pool_;

  // Holds all the function instantiations. Maps function_keys to handles.
  std::unordered_map<string, FunctionLibraryRuntime::Handle> table_
      GUARDED_BY(mu_);

  // Function data for instantitated remote functions.
  std::unordered_map<FunctionLibraryRuntime::Handle,
                     std::unique_ptr<FunctionData>>
      function_data_ GUARDED_BY(mu_);

  // Function data for instantiated multi-device functions.
  std::unordered_map<FunctionLibraryRuntime::Handle,
                     std::unique_ptr<MultiDeviceFunctionData>>
      mdevice_data_ GUARDED_BY(mu_);

  std::unique_ptr<
      std::unordered_map<Device*, std::unique_ptr<FunctionLibraryRuntime>>>
      flr_map_;
  int next_handle_ GUARDED_BY(mu_);
  DistributedFunctionLibraryRuntime* const parent_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_FUNCTION_LIBRARY_RUNTIME_H_
