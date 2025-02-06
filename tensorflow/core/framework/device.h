/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// A Device is a something that can perform computations as part of a
// model.  Devices can be local (runs computation on this machine), or
// remote (contacts a device local to another machine using an RPC to
// do the work).  Devices are registered in a DeviceSet, which is also
// responsible for the Device <-> id mapping.
//
// Device names
// * Every Device should have a unique name with the format:
//     /job:___/replica:___/task:___/(gpu|cpu):___
//   An example name would be "/job:train/replica:0/task:3/device:GPU:2".
// * Task numbers are within the specified replica, so there are as
//   many "task zeros" as replicas.

#ifndef TENSORFLOW_CORE_FRAMEWORK_DEVICE_H_
#define TENSORFLOW_CORE_FRAMEWORK_DEVICE_H_

#include <memory>
#include <string>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class Device : public DeviceBase {
 public:
  // Callback type that takes a Status and returns void.
  typedef std::function<void(const absl::Status&)> DoneCallback;

  Device(Env* env, const DeviceAttributes& device_attributes);
  ~Device() override;

  // A compare function that orders devices by their parsed name.
  static bool LessByParsedName(const Device& a, const Device& b) {
    return a.parsed_name() < b.parsed_name();
  }

  // Full name of this device (see top comment).
  const std::string& name() const override { return device_attributes_.name(); }

  // Parsed name of this device
  const DeviceNameUtils::ParsedName& parsed_name() const override {
    return parsed_name_;
  }

  // Describes what kind of device this is.  This is intended to be
  // human-readable and not computer-parsed, except that two devices
  // with the same device_type() are expected to perform similarly
  // (both from a computation and communication perspective).
  const std::string& device_type() const override {
    return device_attributes_.device_type();
  }

  // Returns an aggregation of device attributes.
  const DeviceAttributes& attributes() const override {
    return device_attributes_;
  }

  // Performs the actual compute function.
  //
  // Subclasses may override this function if they wish to perform
  // some initialization before each compute.
  virtual void Compute(OpKernel* op_kernel, OpKernelContext* context) {
    op_kernel->Compute(context);
  }

  // Asynchronous kernel's compute.
  virtual void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                            AsyncOpKernel::DoneCallback done) {
    op_kernel->ComputeAsync(context, std::move(done));
  }

  // Blocks until all operations queued on the device at the time of
  // the call have completed.  Returns any error pending on the device
  // at completion.
  virtual absl::Status Sync() = 0;

  // Calls the given callback when all operations queued on the device at the
  // time of the call have completed. The callback is passed any error pending
  // on the device at completion.
  // TODO(b/112409994): Consolidate these two APIs, removing the synchronous
  // version.
  virtual void Sync(const DoneCallback& done);

  // On session completion, the executor may call Device::Sync() depending on
  // flag settings. Override this to return false for devices that don't allow
  // such calls. Instead, these devices must use other mechanisms (such as
  // num_deferred_ops) to ensure the device has finished processing necessary
  // work at session completion. In addition, for these devices, RefreshStatus
  // must be called at session completion to retrieve execution result status.
  //
  // Devices that override this function must also implement RefreshStatus.
  virtual bool AllowsSyncOnCompletion() const { return true; }

  // This is used in conjunction with AllowsSyncOnCompletion to allow the
  // executor to get execution result status at session completion.
  //
  // For supported devices, this call returns the underlying device stream's
  // current status in a non-blocking way, without using blocking calls such as
  // Stream::BlockHostUntilDone or Device::Sync. When applicable, the device
  // status is also updated with the retrieved stream status.
  virtual absl::Status RefreshStatus() {
    return errors::Unimplemented(
        "RefreshStatus is not supported on this device.");
  }

  // Optionally modify the device's GraphDef before execution.
  //
  // This method should be considered experimental and is supplied to enable
  // prototyping of TensorFlow device implementations that need to modify
  // the GraphDef before execution.
  //
  // 'graph' supplies the partition of the graph assigned to this
  // device.
  virtual absl::Status MaybeRewriteGraph(std::unique_ptr<Graph>* /*graph*/) {
    return absl::OkStatus();
  }

  // Sets `out_context` a new DeviceContext* for executing a graph, or nullptr
  // if the device does not support contexts. Returns an error status if any
  // error occurred while trying to create a context, otherwise OK.
  //
  // The caller takes ownership of one reference on the output DeviceContext*,
  // and should call Unref().
  virtual absl::Status TryGetDeviceContext(DeviceContext** out_context) {
    *out_context = nullptr;
    return absl::OkStatus();
  }

  // Returns the op segment of this device.  The caller can reuse op
  // kernels registered for the same session running on this device.
  OpSegment* op_segment() { return &op_seg_; }

  // Returns the resource manager associated w/ this device.
  virtual ResourceMgr* resource_manager() { return rmgr_; }

  // Summarizes the status of this Device, for debugging.
  std::string DebugString() const { return device_attributes_.DebugString(); }

  // Assembles the parameter components into a complete DeviceAttributes value.
  static DeviceAttributes BuildDeviceAttributes(
      const std::string& name, DeviceType device, Bytes memory_limit,
      const DeviceLocality& locality, const std::string& physical_device_desc);

  static DeviceAttributes BuildDeviceAttributes(
      const std::string& name, DeviceType device, Bytes memory_limit,
      const DeviceLocality& locality) {
    // Pass in an empty string as physical device name.
    return BuildDeviceAttributes(name, device, memory_limit, locality, "");
  }

  // Updates `attributes()`, indicating the XLA global ID associated with this
  // device. This ID is unique across clients in a multi-client setup. For TPUs
  // this does not happen until the TPU system has been initialized.
  void set_xla_global_id(int64_t id) override {
    device_attributes_.set_xla_global_id(id);
  }

  // Clears the resource manager associated with this device.
  void ClearResourceMgr() { rmgr_->Clear(); }

  virtual bool IsLocal() const { return true; }

  // Informs if this Device can be used as a caller in RemoteCall operation.
  virtual bool IsRemoteCallAllowed() const;

  // Whether to merge the host_to_device copy stream with the compute stream.
  // Only useful for GPU devices.
  virtual bool merge_host_to_device_stream() const { return false; }

  // Whether to merge the device_to_host copy stream with the compute stream.
  // Only useful for GPU devices.
  virtual bool merge_device_to_host_stream() const { return false; }

  // Whether to merge the device_to_device copy streams with the compute stream.
  // Only useful for GPU devices.
  virtual bool merge_device_to_device_stream() const { return false; }

 protected:
  void DeleteResourceMgr() {
    delete rmgr_;
    rmgr_ = nullptr;
  }

 private:
  DeviceAttributes device_attributes_;
  DeviceNameUtils::ParsedName parsed_name_;

  // op_seg_ maps session handle and op name to OpKernel objects.
  OpSegment op_seg_;

  // Resources associated w/ this device. E.g., shared variables, etc.
  ResourceMgr* rmgr_ = nullptr;

  Device(const Device&) = delete;
  void operator=(const Device&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_DEVICE_H_
