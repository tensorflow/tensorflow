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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_H_

#include <memory>
#include <string>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_attributes.pb_text.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class DeviceMgr;

class Device : public DeviceBase {
 public:
  // Callback type that takes a Status and returns void.
  typedef std::function<void(const Status&)> DoneCallback;

  Device(Env* env, const DeviceAttributes& device_attributes);
  ~Device() override;

  // Full name of this device (see top comment).
  const string& name() const override { return device_attributes_.name(); }

  // Parsed name of this device
  const DeviceNameUtils::ParsedName& parsed_name() const {
    return parsed_name_;
  }

  // Describes what kind of device this is.  This is intended to be
  // human-readable and not computer-parsed, except that two devices
  // with the same device_type() are expected to perform similarly
  // (both from a computation and communication perspective).
  const string& device_type() const { return device_attributes_.device_type(); }

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

  // Takes ownership of the references in tensors. If necessary, a
  // device may override this method to keep a reference to the
  // accessed tensors until the async computation has completed.
  virtual void ConsumeListOfAccessedTensors(
      DeviceContext* context, const TensorReferenceVector& tensors) {
    for (const auto& ref : tensors) {
      ref.Unref();
    }
  }

  // If true, and tracing is enabled, the `tracing::ScopedAnnotation()` tracing
  // mechanism will be used instead of `tracing::ScopedActivity()`. Some devices
  // may override this method to use annotations, which enable child activities
  // (such as GPU kernel launches) to be related to the OpKernel invocation.
  virtual bool TraceUsingAnnotations() const { return false; }

  // Blocks until all operations queued on the device at the time of
  // the call have completed.  Returns any error pending on the device
  // at completion.
  virtual Status Sync() = 0;

  // Calls the given callback when all operations queued on the device at the
  // time of the call have completed. The callback is passed any error pending
  // on the device at completion.
  // TODO(b/112409994): Consolidate these two APIs, removing the synchronous
  // version.
  virtual void Sync(const DoneCallback& done);

  // Override this to return true for devices that require a Sync() call before
  // session completion.
  virtual bool RequiresSyncOnCompletion() const { return false; }

  // Optionally modify the device's GraphDef before execution.
  //
  // This method should be considered experimental and is supplied to enable
  // prototyping of TensorFlow device implementations that need to modify
  // the GraphDef before execution.
  //
  // 'graph' supplies the partition of the graph assigned to this
  // device.
  virtual Status MaybeRewriteGraph(std::unique_ptr<Graph>* /*graph*/) {
    return Status::OK();
  }

  // Fill in the context map for the graph. Default behavior is to do
  // nothing.
  //
  // The caller takes ownership over the DeviceContext objects given
  // by the device.
  virtual Status FillContextMap(const Graph* graph,
                                DeviceContextMap* device_context_map) {
    return Status::OK();
  }

  // Returns the op segment of this device.  The caller can reuse op
  // kernels registered for the same session running on this device.
  OpSegment* op_segment() { return &op_seg_; }

  // Returns the resource manager associated w/ this device.
  virtual ResourceMgr* resource_manager() { return rmgr_; }

  // Returns the device manager that owns this device, or nullptr if this Device
  // is not owned by a device manager.
  DeviceMgr* device_mgr() const { return device_mgr_; }

  // Summarizes the status of this Device, for debugging.
  string DebugString() const { return ProtoDebugString(device_attributes_); }

  // Assembles the parameter components into a complete DeviceAttributes value.
  static DeviceAttributes BuildDeviceAttributes(
      const string& name, DeviceType device, Bytes memory_limit,
      const DeviceLocality& locality, const string& physical_device_desc);

  static DeviceAttributes BuildDeviceAttributes(
      const string& name, DeviceType device, Bytes memory_limit,
      const DeviceLocality& locality) {
    // Pass in an empty string as physical device name.
    return BuildDeviceAttributes(name, device, memory_limit, locality, "");
  }

  // Clears the resource manager associated with this device.
  void ClearResourceMgr() { rmgr_->Clear(); }

 protected:
  void DeleteResourceMgr() {
    delete rmgr_;
    rmgr_ = nullptr;
  }

 private:
  friend class DeviceMgr;

  // Pointer to the device manager that owns this device. Not owned.
  DeviceMgr* device_mgr_ = nullptr;

  const DeviceAttributes device_attributes_;
  DeviceNameUtils::ParsedName parsed_name_;

  // op_seg_ maps session handle and op name to OpKernel objects.
  OpSegment op_seg_;

  // Resources associated w/ this device. E.g., shared variables, etc.
  ResourceMgr* rmgr_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(Device);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_H_
