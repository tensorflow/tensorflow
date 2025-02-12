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

// The XlaDevice executes a TensorFlow graph using the XLA linear algebra
// runtime.
//
// Operators assigned to an XlaDevice are compiled into XLA computations.
// Tensors on an XlaDevice are thin wrappers around XLA ScopedShapedBuffers.
//
// XlaDevice is instantiated separately for each XLA backend (e.g., CPU or GPU),
// under different names (e.g., XLA_CPU or XLA_GPU).

#ifndef TENSORFLOW_COMPILER_JIT_XLA_DEVICE_H_
#define TENSORFLOW_COMPILER_JIT_XLA_DEVICE_H_
#include <set>

#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/local_client.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/tfrt/common/async_value_tensor.h"

namespace tensorflow {

class XlaDevice : public LocalDevice {
 public:
  // Given a tensor, sets `xla::Shape*` the shape of tensor's representation
  // on device, fully padded. On error, the contents of `xla::Shape*`
  // are undefined.
  typedef std::function<absl::Status(const Tensor&, xla::Shape*)> PaddedShapeFn;

  // Wrapper class to store metadata about the XlaDevice, where it can be
  // retrieved e.g., when lazily creating the XlaCompilationCache device.
  class Metadata {
   public:
    Metadata(int device_ordinal, se::Platform* platform,
             const DeviceType& device_type,
             std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
                 shape_determination_fns,
             PaddedShapeFn padded_shape_fn, bool use_multiple_streams);

    // The index of the device on this host.
    int device_ordinal() const;

    se::Platform* platform() const;
    xla::LocalClient* client() const;
    const DeviceType& jit_device_type() const;
    const XlaShapeLayoutHelpers::ShapeDeterminationFns&
    default_shape_determination_fns() const {
      return shape_determination_fns_.at(0);
    }
    const PaddedShapeFn& padded_shape_fn() const { return padded_shape_fn_; }

    bool UseMultipleStreams() const { return use_multiple_streams_; }

   private:
    const int device_ordinal_;
    const DeviceType device_type_;
    se::Platform* platform_;  // Not owned.
    std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
        shape_determination_fns_;
    PaddedShapeFn padded_shape_fn_;
    const bool use_multiple_streams_;

    Metadata(const Metadata&) = delete;
    void operator=(const Metadata&) = delete;
  };

  // Sets `*metadata` to the XlaDevice Metadata in the XLA device used by `ctx`.
  static absl::Status GetMetadata(OpKernelContext* ctx,
                                  const Metadata** metadata);

  // Sets `*metadata` to the XlaDevice Metadata in the XLA device used by `ctx`.
  static absl::Status GetMetadata(OpKernelConstruction* ctx,
                                  const Metadata** metadata);

  // Sets `*metadata` to the XlaDevice Metadata in the XLA device used by
  // `device`.
  static absl::Status GetMetadataFromDevice(
      DeviceBase* device, const XlaDevice::Metadata** metadata);

  struct Options {
    // The StreamExecutor platform. Not owned. Must be non-null.
    se::Platform* platform = nullptr;

    // The device name's prefix (e.g., "/task:7")
    string device_name_prefix;

    // The name of the XLA device (e.g., "XLA_CPU")
    string device_name;

    // The number of the device.
    int device_ordinal = -1;

    // The name of the compilation device (e.g., "XLA_CPU_JIT");
    string compilation_device_name;

    // If 'use_multiple_streams' is true, we create separate streams for
    // compute, host-to-device, and device-to-host communication.
    bool use_multiple_streams = false;

    // If true, the XLA devices with the same device ordinal will share the same
    // compute stream. Otherwise each XLA device will having their own compute
    // streams.
    bool use_global_compute_stream = false;

    // A vector of ShapeDeterminationFn (i.e., a bundle of LayoutSelectionFn,
    // ShapeRepresentationFn). Each bundle describes how the on-host shapes of
    // a) argument and return value, for entry computations b) variables, for
    // all computations, should be represented in XLA. Parameters/return values
    // will be shaped according to the function pair, and reshaped back to/from
    // their declared shapes for computations. Must be non-empty.
    std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
        shape_determination_fns;

    // If padded_shape_fn is empty, a default implementation that returns
    // the logical on-device shape without padding is used.
    PaddedShapeFn padded_shape_fn;

    // Set of devices to use. This controls which of the devices on the given
    // platform will have resources allocated. For GPUs this will be
    // filled from visible_gpu_devices list from session configuration.
    std::optional<std::set<int>> allowed_devices;
  };

  // Creates a new XLA Device.
  XlaDevice(const SessionOptions& session_options, const Options& options);
  ~XlaDevice() override;

  Allocator* GetAllocator(AllocatorAttributes attr) override
      TF_LOCKS_EXCLUDED(mu_);
  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
  void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;
  absl::Status Sync() override;

  absl::Status TryGetDeviceContext(DeviceContext** out_context) override
      TF_LOCKS_EXCLUDED(mu_);

  absl::Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                   const AllocatorAttributes alloc_attrs,
                                   Tensor* tensor) override
      TF_LOCKS_EXCLUDED(mu_);

  absl::Status MakeTensorFromProto(DeviceContext* device_context,
                                   const TensorProto& tensor_proto,
                                   const AllocatorAttributes alloc_attrs,
                                   Tensor* tensor);

  const Metadata& metadata() { return xla_metadata_; }

  // Ensures the DeviceContext associated with this XlaDevice is created and
  // valid (i.e. all streams are ok). If any state is not valid, a new
  // DeviceContext will be created.
  //
  // TODO(b/111859745): The Eager context needs to call this method to recover
  // from failures.
  absl::Status EnsureDeviceContextOk() TF_LOCKS_EXCLUDED(mu_);

  // Two convenient methods to get the underlying device context.
  // Get the default device context, created by the first
  // shape_representation_fn.
  absl::StatusOr<DeviceContext*> GetDeviceContextDefault();
  // Get the device context given the index.
  absl::StatusOr<DeviceContext*> GetDeviceContextWithIndex(int index);

  // Instructs this XlaDevice to set a AcceleratorDeviceInfo, which holds extra
  // information for GPU and TPU devices.
  absl::Status UseAcceleratorDeviceInfo() TF_LOCKS_EXCLUDED(mu_);

  // Instructs this XlaDevice to return 'sync_on_completion' for
  // AllowsSyncOnCompletion().
  void SetAllowsSyncOnCompletion(bool sync_on_completion)
      TF_LOCKS_EXCLUDED(mu_);
  bool AllowsSyncOnCompletion() const override TF_LOCKS_EXCLUDED(mu_);

  // Installs an error handling callback when RefreshStatus sees !status.ok().
  void SetHandleDeviceErrorCallback(std::function<absl::Status()> callback);

  absl::Status RefreshStatus() override TF_LOCKS_EXCLUDED(mu_);

 private:
  absl::StatusOr<xla::LocalClient*> GetOrCreateClient() const;
  Allocator* GetAllocatorLocked(AllocatorAttributes attr)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  absl::Status EnsureStreamOkLocked(xla::Backend* backend, const string& name,
                                    std::shared_ptr<se::Stream>* stream,
                                    bool* stream_was_changed)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Return a vector of device context, ordered by the sequence in the given
  // shape_representation_fns.
  absl::StatusOr<std::vector<DeviceContext*>> GetDeviceContextLocked()
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Handles error when RefreshStatus sees !status.ok().
  absl::Status HandleDeviceError();

  mutable mutex mu_;
  // The metadata of this XlaDevice.
  const Metadata xla_metadata_;
  // Which hardware device in the client's platform this XlaDevice controls.
  const int device_ordinal_;
  // The name/type of this XlaDevice. eg. "XLA_GPU".
  const DeviceType device_name_;
  // The name of the device that is used to compile Ops for this XlaDevice.
  const DeviceType jit_device_name_;
  // The platform for this device.
  se::Platform* const platform_;  // Not owned.
  // Intra-op threads to spawn (from SessionOptions).
  const int intra_op_parallelism_threads_;
  // Memory allocator associated with this device.
  Allocator* xla_allocator_ TF_GUARDED_BY(mu_) = nullptr;  // Not owned.
  std::unique_ptr<AsyncValueAllocator> pjrt_allocator_ TF_GUARDED_BY(mu_);

  // Stream associated with this device. Operations enqueued on this
  // stream are executed on the device. Operations include data
  // copying back and forth between CPU and the device, and
  // computations enqueued by XLA.
  std::shared_ptr<se::Stream> stream_ TF_GUARDED_BY(mu_);
  // If false, only stream_ is valid and all computation and transfers use
  // stream_. If true, computation is performed by stream_ and transfers are
  // performed by host_to_device/device_to_device stream or borrowing a stream
  // for each device to host transfer.
  const bool use_multiple_streams_;
  // If use_multiple_streams_, host to device transfers are performed using this
  // stream.
  std::shared_ptr<se::Stream> host_to_device_stream_ TF_GUARDED_BY(mu_);
  // If use_multiple_streams_, transfers between different devices are performed
  // using these streams.
  std::vector<std::shared_ptr<se::Stream>> device_to_device_streams_
      TF_GUARDED_BY(mu_);

  // See comments in options.
  std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
      shape_determination_fns_;

  // A list of the device context accessed by all users of the XlaDevice, set by
  // calls to EnsureDeviceContextOk. The number of device conetexts is based on
  // the number of shape representation functions in XlaDevice::Options. If
  // accelerator_device_info_ is non-null, this pointer is also filled in to
  // that struct. DeviceContext is a ref-counted object.
  std::vector<DeviceContext*> device_contexts_ TF_GUARDED_BY(mu_);

  // Holds extra information for GPU and TPU devices, e.g. the device context.
  bool use_accelerator_device_info_ TF_GUARDED_BY(mu_) = false;
  std::unique_ptr<DeviceBase::AcceleratorDeviceInfo> accelerator_device_info_
      TF_GUARDED_BY(mu_);

  // Thread pool used for running closures
  std::unique_ptr<thread::ThreadPool> thread_pool_;

  // True if the device allows XlaDevice::Sync to be called on completion
  // regardless of status.
  bool sync_on_completion_ TF_GUARDED_BY(mu_) = true;

  // A callback that will be invoked when RefreshStatus sees a status error.
  std::function<absl::Status()> device_error_callback_ TF_GUARDED_BY(mu_);

  // Set of devices to use. This controls which of the devices on the given
  // platform will have resources allocated. For GPUs this will be
  // filled from visible_gpu_devices list from session configuration.
  std::optional<std::set<int>> allowed_devices_;

  const bool use_global_compute_stream_;

  // A static vector with device_ordinal as its index, describing the global
  // compute streams used in each XLA device. It is only used if
  // `use_global_compute_stream` in `XlaDevice::Options` is set to true.
  static mutex global_mu_;
  static std::vector<std::shared_ptr<se::Stream>>* global_compute_streams_
      TF_GUARDED_BY(global_mu_);
};

// Builds OpKernel registrations on 'device' for the JIT operators
// registered on 'jit_device'. Returns ownership of a XlaDeviceOpRegistrations
// object that encapsulates the kernel registrations.
struct XlaDeviceOpRegistrations {
  std::vector<std::unique_ptr<kernel_factory::OpKernelRegistrar>>
      op_kernel_registrars;
};

XlaDeviceOpRegistrations* RegisterXlaDeviceKernels(
    const char* device, const char* jit_device,
    OpKernel* (*factory)(OpKernelConstruction*),
    absl::string_view kernel_class_name);

XlaDeviceOpRegistrations* RegisterXlaDeviceKernels(const char* device,
                                                   const char* jit_device);

absl::Status DefaultPaddedShapeFn(const Tensor& tensor, xla::Shape* shape);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_DEVICE_H_
