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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/stats_publisher_interface.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tsl/platform/thread_annotations.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"
#endif  // !IS_MOBILE_PLATFORM

namespace tensorflow {

class FunctionArgsInterface {
 public:
  virtual ~FunctionArgsInterface() {}

  virtual bool HasRemoteOrPackedInputs() const = 0;

  virtual absl::Status GetLocalArg(const FunctionArgIndex& index,
                                   Tensor* val) const = 0;

  virtual std::vector<Tensor> GetLocalTensors() const = 0;

#if !defined(IS_MOBILE_PLATFORM)
  virtual absl::Status GetRemoteArg(const FunctionArgIndex& index,
                                    eager::RemoteTensorHandle* val) const {
    return errors::Unimplemented(
        "Serializing a remote argument is not implemented.");
  }
#endif  // IS_MOBILE_PLATFORM
};

// A class that stores all the FunctionLibraryRuntime objects, one per device.
class ProcessFunctionLibraryRuntime {
 public:
  // Creates FunctionLibraryRuntime objects for each device in the provided
  // DeviceMgr. Caller needs to make sure that device_mgr, lib_def and parent
  // (if provided) outlive this object.
  ProcessFunctionLibraryRuntime(
      const DeviceMgr* device_mgr, Env* env, const ConfigProto* config,
      int graph_def_version, const FunctionLibraryDefinition* lib_def,
      const OptimizerOptions& optimizer_options,
      thread::ThreadPool* thread_pool = nullptr,
      DistributedFunctionLibraryRuntime* parent = nullptr,
      const SessionMetadata* session_metadata = nullptr,
      Rendezvous::Factory rendezvous_factory = Rendezvous::Factory(),
      StatsPublisherFactory stats_publisher_factory = CreateNoOpStatsPublisher);

  ~ProcessFunctionLibraryRuntime() {
    // Deleting the FunctionLibraryRuntime map will delete the function handles
    // registered in it, which may call ReleaseHandle in this class again to
    // release their sub-function. These circular calls may cause segfault
    // since the flr_map_ may have already been deleted. Explicitly releasing
    // flr_map_ here and checking flr_map_ in ReleaseHandle to avoid this.
    flr_map_.reset();
  }

  // Sends `tensors_to_send` from `source_device` to `target_device` using
  // `rendezvous`. `key_prefix` is used as a prefix for the keys sent to the
  // Rendezvous. `device_context` should be the DeviceContext of the device
  // doing the sending. `alloc_attrs` should either be empty or be the size of
  // `tensors_to_send` and indicates how the input tensors are allocated. Method
  // takes references on each of the `tensors_to_send`. Method doesn't block.
  static absl::Status SendTensors(
      const string& source_device, const string& target_device,
      const string& key_prefix, int64_t src_incarnation,
      absl::Span<const Tensor> tensors_to_send, DeviceContext* device_context,
      const std::vector<AllocatorAttributes>& alloc_attrs,
      RendezvousInterface* rendezvous);

  // Receives `received_tensors` from `target_device` (originally sent from
  // `source_device`) using `rendezvous`. Uses `key_prefix` to construct the
  // keys to be retrieved. `device_context` should be for the device receiving
  // the tensors. `alloc_attrs` indicates how to allocate the received
  // tensors and should either be empty or `num_tensors` in size. Method doesn't
  // block and calls `done` when `num_tensors` are fetched.
  static void ReceiveTensorsAsync(
      const string& source_device, const string& target_device,
      const string& key_prefix, int64_t src_incarnation, int64_t num_tensors,
      DeviceContext* device_context,
      const std::vector<AllocatorAttributes>& alloc_attrs,
      RendezvousInterface* rendezvous, std::vector<Tensor>* received_tensors,
      StatusCallback done);

  static const char kDefaultFLRDevice[];
  // Returns the FunctionLibraryRuntime for the corresponding device_name.
  FunctionLibraryRuntime* GetFLR(const string& device_name) const;

  // Returns the return types for the function identified by handle `h`.
  absl::Status GetRetTypes(FunctionLibraryRuntime::Handle h,
                           DataTypeVector* ret_types);

  // Returns the device incarnation for the given device_name.
  absl::Status GetDeviceIncarnation(const string& device_name,
                                    int64_t* incarnation) const;

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
  //
  // If `include_multi_device` is true and `handle` is a multi-device function
  // with a single component that is placed on `device_name`, then this method
  // will return the local handle for that component.
  FunctionLibraryRuntime::LocalHandle GetHandleOnDevice(
      const string& device_name, FunctionLibraryRuntime::Handle handle,
      bool include_multi_device = false) const;

  // Fills `output_devices` with the devices on which the results will
  // be produced. If some output is produced on CPU, the corresponding Device*
  // is set to nullptr. If some output is DT_RESOURCE, the corresponding Device*
  // is set to the device backing the resource.
  // REQUIRES: `handle` identifies a multi-device function.
  absl::Status GetOutputDevices(FunctionLibraryRuntime::Handle handle,
                                std::vector<Device*>* output_devices) const;

  // Instantiates the function. See framework/function.h for more details.
  // Allows for function_name to be instantiated on different devices
  // as specified in attrs.
  absl::Status Instantiate(
      const string& function_name, AttrSlice attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      FunctionLibraryRuntime::Handle* handle);

  // Finalizes the function library runtime by calling
  // FunctionLibraryRuntime::Finalize on all local FLRs. The Instantiate method
  // should not be called after Finalize is called.
  absl::Status Finalize();

  // Returns whether the function represented by the given handle needs to
  // execute cross process.
  absl::Status IsCrossProcess(FunctionLibraryRuntime::Handle handle,
                              bool* is_cross_process) const;

  // Delegates to the local FLR that owns state corresponding to `handle` and
  // tells it to release it. If the `handle` isn't needed at all, the local FLR
  // might call RemoveHandle on this to get rid of the state owned by the Proc
  // FLR.
  // For multi-device functions, calls ReleaseHandle on local FLRs for each
  // component function that is part of this multi-device function.
  // Each local FLR might call RemoveHandle on this.
  absl::Status ReleaseHandle(FunctionLibraryRuntime::Handle handle);

  // Runs the function with given `handle`. Function could have been
  // instantiated on any device. More details in framework/function.h
  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::Handle handle, absl::Span<const Tensor> args,
           std::vector<Tensor>* rets,
           FunctionLibraryRuntime::DoneCallback done) const;
  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::Handle handle, CallFrameInterface* frame,
           FunctionLibraryRuntime::DoneCallback done) const;

  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::Handle handle,
           const FunctionArgsInterface& args, std::vector<FunctionRet>* rets,
           FunctionLibraryRuntime::DoneCallback done) const;

  absl::Status RunSync(const FunctionLibraryRuntime::Options& opts,
                       FunctionLibraryRuntime::Handle handle,
                       absl::Span<const Tensor> args,
                       std::vector<Tensor>* rets) const;
  absl::Status RunSync(const FunctionLibraryRuntime::Options& opts,
                       FunctionLibraryRuntime::Handle handle,
                       CallFrameInterface* frame) const;

  const DeviceMgr* device_mgr() { return device_mgr_; }

  const std::shared_ptr<DeviceSet> device_set() const {
    tf_shared_lock l(mu_);
    return device_set_;
  }

  // Initialize the set of local and remote devices and corresponding flr for op
  // device selection.
  void InitializeDeviceAndFlr();

  const ConfigProto* config() const { return config_ ? &(*config_) : nullptr; }

  const FunctionLibraryDefinition* GetFunctionLibraryDefinition() const {
    return lib_def_;
  }

  // Add a CompositeDevice to `device_set_`
  void AddCompositeDevice(CompositeDevice* d) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    device_set_->AddDevice(d);
    composite_devices_.push_back(d);
  }

 protected:
  friend class FunctionLibraryRuntimeImpl;

  struct InternalArgs {
    std::vector<FunctionArg> args;
#if !defined(IS_MOBILE_PLATFORM)
    // Holds the RemoteTensorHandles referred by args.
    std::vector<std::unique_ptr<eager::RemoteTensorHandle>> remote_args;
#endif  // IS_MOBILE_PLATFORM
  };

  // Structure detailing the asynchronous assumptions of a component function,
  // such as whether it can support synchronous execution and any information
  // needed to execute in proper order to resolve inter-subgraph dependencies.
  class AsyncAttributes {
   public:
    enum Summary { kSafeForSync = 0, kSendOnly, kRecvOnly, kAsyncRequired };

    AsyncAttributes()
        : allow_control_flow_sync_execution_(false), summary_(kSafeForSync) {}
    explicit AsyncAttributes(const Graph* graph,
                             bool allow_control_flow_sync_execution)
        : allow_control_flow_sync_execution_(allow_control_flow_sync_execution),
          summary_(Summarize(graph)) {}
    Summary summary() const { return summary_; }
    bool allow_control_flow_sync_execution() const {
      return allow_control_flow_sync_execution_;
    }

   private:
    // This data member should be initialized before the summary_.
    bool allow_control_flow_sync_execution_;
    Summary summary_;
    Summary Summarize(const Graph* graph);
  };

  // Structure to keep track of how a component function (a single-device
  // piece of a multi-device function) fits into the multi-device function.
  struct ComponentFunctionData {
    // The handle for the instantiated component function.
    FunctionLibraryRuntime::Handle handle;
    // The name for the component function.
    string name;
    // arg_indices.size() is the number of arguments to the component function.
    // The i-th argument of the component function comes from the
    // `arg_indices[i]`-th argument of the multi-device function.
    std::vector<FunctionArgIndex> arg_indices;
    // ret_indices.size() is the number of return values of the component
    // function.  The i-th return value of the component function goes to the
    // `ret_indices[i]`-th return value of the multi-device function.
    std::vector<int> ret_indices;
    // arg_alloc_attrs[i] are the allocator attributes of the i-th argument to
    // the component function.
    std::vector<AllocatorAttributes> arg_alloc_attrs;
    // ret_alloc_attrs[i] are the allocator attributes of the i-th return value
    // of the component function.
    std::vector<AllocatorAttributes> ret_alloc_attrs;

    AsyncAttributes async_attributes;
  };

  // Data structure holding information for a single instantiated multi-device
  // function.
  // The fields are filled in during instantiation. Once the object is
  // added to mdevice_data_, all fields are constant.
  struct MultiDeviceFunctionData {
    MultiDeviceFunctionData(const string& function_name,
                            const string& function_key, int num_outputs,
                            DataTypeVector ret_types)
        : function_name_(function_name),
          function_key_(function_key),
          instantiation_counter_(1),
          num_outputs_(num_outputs),
          ret_types_(std::move(ret_types)),
          is_cross_process_(false),
          has_remote_outputs(false) {}

    const string function_name_;
    const string function_key_;
    uint64 instantiation_counter_;
    // Stored here to resize the output tensor vector when function is run.
    const int num_outputs_;
    DataTypeVector ret_types_;

    // Indicates whether this function needs to execute cross process.
    bool is_cross_process_;
    // Indicates whether this function has remote outputs.
    bool has_remote_outputs;

    //  Indicates if running this function synchronously is both allowed + safe.
    bool enable_sync_execution;

    // Maps the device name to the information about the component function
    // be run on this device.
    std::unordered_map<string, ComponentFunctionData> glue_;
  };

  struct CleanUpItem {
    string device;
    uint64 step_id;
    FunctionLibraryRuntime::Handle local_handle;
  };

  // If `handle` represents a multi-device function, returns the multi-device
  // data associated with `handle`. Else, nullptr.
  MultiDeviceFunctionData* IsMultiDevice(
      FunctionLibraryRuntime::Handle handle) const;

  DistributedFunctionLibraryRuntime* const parent_;

 private:
  FunctionLibraryRuntime::Handle AddHandleLocked(
      const string& function_key, const string& device_name,
      FunctionLibraryRuntime::LocalHandle local_handle)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // For a given device_name, returns a DeviceContext for copying
  // tensors to/from the device.
  absl::Status GetDeviceContext(const string& device_name,
                                DeviceContext** device_context) const;

  // Looks up the information for the given `handle` and returns the name
  // of the device where the function is registered.
  string GetDeviceName(FunctionLibraryRuntime::Handle handle) const;

  // Removes handle from the state owned by this object.
  absl::Status RemoveHandle(FunctionLibraryRuntime::Handle handle);

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
  absl::Status Clone(Env* env, int graph_def_version,
                     const OptimizerOptions& optimizer_options,
                     std::unique_ptr<FunctionLibraryDefinition>* out_lib_def,
                     std::unique_ptr<ProcessFunctionLibraryRuntime>* out_pflr,
                     bool skip_flib_def = false) const;

  absl::Status ReleaseMultiDeviceHandle(FunctionLibraryRuntime::Handle handle);

  absl::Status InstantiateMultiDevice(
      const string& function_name, AttrSlice attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      FunctionLibraryRuntime::Handle* handle);

  void InstantiateRemote(
      const string& function_name, AttrSlice attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      FunctionLibraryRuntime::Handle* handle,
      FunctionLibraryRuntime::DoneCallback done);

  FunctionLibraryRuntime::Handle AddMultiDeviceHandle(
      const std::unique_ptr<MultiDeviceFunctionData> data,
      const string& function_key);

  bool HasMultiDeviceHandle(FunctionLibraryRuntime::Handle handle) const;

  void RunInternal(const FunctionLibraryRuntime::Options& opts,
                   FunctionLibraryRuntime::Handle handle,
                   absl::Span<const FunctionArg> args,
                   std::vector<FunctionRet>* rets,
                   std::vector<std::unique_ptr<CleanUpItem>>* cleanup_items,
                   FunctionLibraryRuntime::DoneCallback done) const;

  absl::Status CreateRendezvous(
      FunctionLibraryRuntime::Options& opts,
      tsl::core::RefCountPtr<Rendezvous>* created_rendezvous) const;

  FunctionLibraryRuntime::DoneCallback ApplyCleanUpToDoneCallback(
      std::vector<std::unique_ptr<CleanUpItem>>* items,
      FunctionLibraryRuntime::DoneCallback done,
      const FunctionLibraryRuntime::Options& opts,
      tsl::core::RefCountPtr<Rendezvous> rendezvous) const;

  void CleanUp(std::vector<std::unique_ptr<CleanUpItem>>* items,
               FunctionLibraryRuntime::DoneCallback done) const;

  static absl::Status GetComponentArgs(absl::Span<const Tensor> args,
                                       const ComponentFunctionData& comp_data,
                                       InternalArgs* comp_args);

#if !defined(IS_MOBILE_PLATFORM)
  static absl::Status GetComponentArgs(const FunctionArgsInterface& args,
                                       const ComponentFunctionData& comp_data,
                                       InternalArgs* comp_args);
#endif  // IS_MOBILE_PLATFORM

  std::vector<string> GetOrderedSubgraphs(
      const MultiDeviceFunctionData* data) const;

  absl::Status PrepareRunMultiDevice(
      const FunctionLibraryRuntime::Options& opts,
      FunctionLibraryRuntime::Handle handle,
      const MultiDeviceFunctionData** data) const;

  absl::Status RunMultiDeviceSync(
      const FunctionLibraryRuntime::Options& opts,
      FunctionLibraryRuntime::Handle handle, std::vector<FunctionRet>* rets,
      std::function<absl::Status(const ComponentFunctionData& comp_data,
                                 InternalArgs* args)>
          get_component_args) const;

  void RunMultiDeviceAsync(
      const FunctionLibraryRuntime::Options& opts,
      FunctionLibraryRuntime::Handle handle, std::vector<FunctionRet>* rets,
      std::vector<std::unique_ptr<CleanUpItem>>* cleanup_items,
      FunctionLibraryRuntime::DoneCallback done,
      std::function<absl::Status(const ComponentFunctionData& comp_data,
                                 InternalArgs* args)>
          get_component_args) const;

  void PublishSubgraphs(
      const std::string& function_name,
      std::vector<core::RefCountPtr<FunctionRecord>>&& function_records);

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

    const string& target_device() { return target_device_; }
    const string& function_key() { return function_key_; }

    FunctionLibraryRuntime::LocalHandle local_handle() {
      mutex_lock l(mu_);
      return local_handle_;
    }

    // Initializes the FunctionData object by potentially making an Initialize
    // call to the DistributedFunctionLibraryRuntime.
    void DistributedInit(
        DistributedFunctionLibraryRuntime* parent, const string& function_name,
        const FunctionLibraryDefinition& lib_def, AttrSlice attrs,
        const FunctionLibraryRuntime::InstantiateOptions& options,
        FunctionLibraryRuntime::DoneCallback done);

    bool is_cross_process() {
      mutex_lock l(mu_);
      return is_cross_process_;
    }

   private:
    mutex mu_;

    const string target_device_;
    FunctionLibraryRuntime::LocalHandle local_handle_ TF_GUARDED_BY(mu_);
    const string function_key_;
    bool is_cross_process_ TF_GUARDED_BY(mu_) = false;
    bool init_started_ TF_GUARDED_BY(mu_) = false;
    absl::Status init_result_ TF_GUARDED_BY(mu_);
    Notification init_done_;
  };

  mutable mutex mu_;

  Env* const env_;
  const std::optional<const ConfigProto> config_;
  const DeviceMgr* const device_mgr_;
  const FunctionLibraryDefinition* lib_def_;
  thread::ThreadPool* default_thread_pool_;

  // Cluster update can reinitialize the device_set_ due to remote device
  // changes. At the same time, InstantiateMultiDevice can use the cached
  // devices to instantiate multi-worker functions. Function instantiation would
  // fail if it spans the changed remote devices.
  std::shared_ptr<DeviceSet> device_set_ TF_GUARDED_BY(mu_);

  // Composite devices owned by a EagerContext.
  std::vector<CompositeDevice*> composite_devices_ TF_GUARDED_BY(mu_);

  // Holds all the function instantiations. Maps function_keys to handles.
  std::unordered_map<string, FunctionLibraryRuntime::Handle> table_
      TF_GUARDED_BY(mu_);

  // Function data for instantiated remote functions.
  std::unordered_map<FunctionLibraryRuntime::Handle,
                     std::unique_ptr<FunctionData>>
      function_data_ TF_GUARDED_BY(mu_);

  // Function data for instantiated multi-device functions.
  std::unordered_map<FunctionLibraryRuntime::Handle,
                     std::unique_ptr<MultiDeviceFunctionData>>
      mdevice_data_ TF_GUARDED_BY(mu_);

  std::unique_ptr<
      std::unordered_map<Device*, core::RefCountPtr<FunctionLibraryRuntime>>>
      flr_map_;
  int next_handle_ TF_GUARDED_BY(mu_);
  const SessionMetadata* const session_metadata_;
  const Rendezvous::Factory rendezvous_factory_;

  const OptimizerOptions optimizer_options_;
  const int graph_def_version_;

  StatsPublisherFactory stats_publisher_factory_;
  // Holds all stats publishers, one for publishing subgraphs of each
  // instantiated function.
  std::vector<std::unique_ptr<StatsPublisherInterface>> stats_publishers_
      TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_FUNCTION_LIBRARY_RUNTIME_H_
