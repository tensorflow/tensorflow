/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_H_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "absl/types/variant.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle_data.h"
#include "tensorflow/core/common_runtime/function.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle_data.h"
#endif  // IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/lib/core/stringpiece.h"

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// Associates a Tensor and a Device, used in the eager runtime. Internal version
// of the TFE_TensorHandle struct and the python EagerTensor class
// (unrelated to python TensorHandle).
class TensorHandle : public core::RefCounted {
  // Custom devices do many of the same things as physical Devices, but have a
  // much more restricted interface. We pass around ambiguous pointers since
  // TensorHandles may be placed either on custom or physical devices.
  using VariantDevice = absl::variant<Device*, CustomDevice*>;

  // TensorHandle for dtype != DT_RESOURCE
  TensorHandle(tensorflow::Tensor&& t, Device* d, Device* op_device,
               Device* resource_device, EagerContext* ctx);
  // TensorHandle for dtype == DT_RESOURCE
  TensorHandle(tensorflow::Tensor&& t, Device* d, Device* op_device,
               EagerContext* ctx);
  TensorHandle(tensorflow::Tensor&& t, CustomDevice* d, EagerContext* ctx);
  TensorHandle(Device* d, Device* op_device, Device* resource_device,
               DataType dtype, EagerContext* ctx);

#if !defined(IS_MOBILE_PLATFORM)
  TensorHandle(int64 op_id, int32 output_num, const string& remote_task,
               DataType dtype, Device* device, EagerContext* ctx);
  TensorHandle(int64 op_id, int32 output_num, DataType dtype, Device* device,
               EagerContext* ctx);
#endif  // IS_MOBILE_PLATFORM

 public:
  // TensorHandle with no assigned device
  static Status CreateLocalHandle(const tensorflow::Tensor& t,
                                  TensorHandle** h);
  static Status CreateLocalHandle(tensorflow::Tensor&& t, Device* d,
                                  Device* op_device, EagerContext* ctx,
                                  TensorHandle** h);
  static Status CreateLocalHandle(tensorflow::Tensor&& t, Device* d,
                                  Device* op_device, Device* resource_device,
                                  EagerContext* ctx, TensorHandle** h);
  static Status CreateLocalHandle(tensorflow::Tensor&& t, CustomDevice* d,
                                  EagerContext* ctx, TensorHandle** h);
  static Status CreateEmptyLocalHandle(Device* d, Device* op_device,
                                       Device* resource_device, DataType dtype,
                                       EagerContext* ctx, TensorHandle** h);
#if !defined(IS_MOBILE_PLATFORM)
  static Status CreateUnshapedRemoteHandle(int64 op_id, int32 output_num,
                                           const string& remote_task,
                                           DataType dtype, Device* d,
                                           EagerContext* ctx, TensorHandle** h);
  static Status CreateLazyRemoteHandle(int64 op_id, int32 output_num,
                                       DataType dtype, Device* d,
                                       EagerContext* ctx, TensorHandle** h);
#endif  // IS_MOBILE_PLATFORM

  ~TensorHandle() override { DVLOG(3) << "Deleting TensorHandle " << this; }

  // Return the Tensor from the default device.
  Status Tensor(const tensorflow::Tensor** t) const;
  // Return the Tensor from the specified device which could be either the
  // default device or a local mirror. The device pointer should be nullptr if
  // requesting the HostCPU.
  Status TensorFromDevice(const Device* d, const tensorflow::Tensor** t) const;

  // Return the TensorValue from the specified device which could be either the
  // default device or a local mirror. The device pointer should be nullptr if
  // requesting the HostCPU.
  Status TensorValue(const Device* d, tensorflow::TensorValue* t);

  VariantDevice device() const { return device_; }
  Device* op_device() const { return op_device_; }
  Device* resource_device() const { return resource_device_; }

  VariantDevice DeviceOrHostCPU(const EagerContext& ctx) const;

  Status Shape(tensorflow::TensorShape* shape);
  Status NumDims(int* num_dims) const;
  Status Dim(int dim_index, int64* dim) const;
  Status NumElements(int64* num_elements) const;

  Status Unprotect(const Device* d);

  // Checks if a mirror tensor exists for the specified device. Mirrors are only
  // maintained for local devices, like CPUs & GPUs. Note a mirror may be empty,
  // as it is still to be set by an async operation.
  bool HasLocalMirror(const Device* d) const;
  // Add an empty mirror placeholder for the specified device. The expectation
  // is this will be populated by a call to SetTensor.
  Status AddEmptyLocalMirror(const Device* d);
  // Add a local mirror. This will fail if an empty local mirror was previously
  // added. For that case, SetTensor should be used instead.
  Status AddLocalMirror(tensorflow::Tensor&& tensor, const Device* d);

#if !defined(IS_MOBILE_PLATFORM)
  bool HasRemoteMirror(const Device* d, uint64 context_view_id) const;
  bool HasResourceShapeMirror(const Device* d, uint64 context_view_id) const;

  Status AddUnshapedRemoteMirror(const Device* d, int64 op_id, int output_num,
                                 const string& remote_task, EagerContext* ctx);
  Status AddResourceShapeMirror(const Device* d, int64 op_id, int output_num,
                                EagerContext* ctx);

  // Return the op_id and output num if the handle refers to a remote tensor.
  Status RemoteAddress(const Device* d, int64* op_id, int32* output_num) const;

  // Called on an async remote tensor once it's shape has been determined. This
  // transitions the tensor handle from a non-ready to a ready state by
  // replacing the backing data abstraction to allow for the shape to be
  // queried.
  // This method or Poison must be called exactly once for remote tensors that
  // were created without a known shape.
  Status SetRemoteShape(const TensorShape& shape, const Device* d,
                        uint64 context_view_id);

  // Poisons either this handle or a remote mirror with error `status`.
  // Poisoning means that the handle will become ready and methods trying
  // to access the remote shape will return this error `status`.
  // Exactly one of SetRemoteShape or PoisonRemote methods must be called on a
  // unshaped handle on a remote device.
  void PoisonRemote(Status status, const Device* d, uint64 context_view_id);
#endif

  // Sets the `tensor` for this async non-ready handle making it ready.
  // This method or Poison must be called exactly once for non-ready async
  // handles to make them ready.
  Status SetTensor(tensorflow::Tensor&& tensor, const Device* d);

  // Poisons either this handle or a local mirror with error `status`.
  // Poisoning means that the handle will become ready and methods trying
  // to access the actual tensor or shape will return this error `status`.
  // Exactly one of SetTensor or Poison methods must be called on a non-ready
  // tensor for a specific device.
  void Poison(Status status, const Device* d);

  Status CopyToDevice(const EagerContext& ctx, tensorflow::Device* d,
                      tensorflow::Tensor* output);

  Status InferenceShape(
      shape_inference::InferenceContext* const inference_context,
      shape_inference::ShapeHandle* shape_handle);
  void SetInferenceShape(
      shape_inference::InferenceContext* const inference_context,
      const shape_inference::ShapeHandle& shape_handle);
  Status CopyInferenceShape(TensorHandle* other);

  // Warning: can return nullptr for CPU tensors.
  EagerContext* Context() { return ctx_; }

  // dtype for the handle. It must be the same as t.dtype() once the handle is
  // ready.
  const DataType dtype;

  bool OnHostCPU() const {
    return (
        device_.index() == 0 &&
        (absl::get<Device*>(device_) == nullptr ||
         (ctx_ != nullptr && ctx_->HostCPU() == absl::get<Device*>(device_))));
  }

  bool IsRemote() const;
  void EnableImplicitMirroring() { implicit_mirroring_ = true; }
  bool ImplicitMirroring() const { return implicit_mirroring_; }

  string DebugString() const;

  struct ResourceHandleInfo {
    std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes;
    std::vector<string> allowed_devices;
  };

  void SetResourceHandleInfo(ResourceHandleInfo&& resource_handle_info);

  // If this TensorHandle is 1) a local tensor, and 2) a resource handle,
  // return data types, shapes and allowed devices of the underlying resource.
  Status GetResourceHandleInfo(ResourceHandleInfo* result);
  Status GetResourceHandleDtypesAndShapes(
      std::vector<DtypeAndPartialTensorShape>* result);
  Status GetResourceAllowedDevices(std::vector<string>* result);

 private:
  // The TensorHandleData can either represent a local or remote tensor handle.
  // Further, it can be in a non-ready state. It would become ready with a call
  // to either SetTensor or SetRemoteShape which replaces the underlying data
  // with a ready version of the tensor handle data.
  bool IsReady() const;

  Status GetResourceHandleInfoImpl(std::function<void()> set_resource_info);

  VariantDevice const device_;

  // Device in which the op producing this tensor was executed. Equals to
  // device_ for constant tensors.
  // Can be nullptr if the op producing this tensor was a function executed
  // with function library runtime.
  tensorflow::Device* const op_device_;

  // If the tensor dtype is DT_RESOURCE, resource_device_ holds the device
  // backing the resource. Else resource_device_ is nullptr.
  tensorflow::Device* const resource_device_;

  mutable mutex mu_;

  // Map of local mirrors. This can include both ready and non-ready mirrors.
  std::unordered_map<const tensorflow::Device*, LocalTensorHandleData>
      local_mirrors_ TF_GUARDED_BY(mu_);
#if !defined(IS_MOBILE_PLATFORM)
  // TODO(yujingzhang): Remove resource_shape_mirrors_ once scalable per-replica
  // variable is ready, since we could get the shape locally without remote copy
  // then.
  std::unordered_map<string, RemoteTensorHandleData> resource_shape_mirrors_
      TF_GUARDED_BY(mu_);
  // TODO(gjn): Is std::map the most optimal choice here? Perhaps this should be
  // a fixed size map.
  std::unordered_map<string, RemoteTensorHandleData> remote_mirrors_
      TF_GUARDED_BY(mu_);
#endif

  // `ctx` is only guaranteed to be set if the handle is not "ready". This is
  // typically true when the handle was produced during async execution.
  // `ctx` object is not owned and should outlive this handle.
  //
  // TODO(b/150614042): Reference count EagerContext to ensure that 'device_' of
  // a TensorHandle does not outlive the EagerContext from which it came?
  EagerContext* const ctx_;

  // Does not need synchronization because it can be accessed only after
  // WaitReady() has returned. At that point, is_poisoned_ is immutable.
  Status is_poisoned_;
  bool implicit_mirroring_;

  // If this TensorHandle 1) is a local tensor, and 2) is a resource handle or
  // refers to a remote resource handle, we store data types, shapes and allowed
  // devices for the underlying resource.
  ResourceHandleInfo resource_handle_info_;

  // Does not need synchronization because it can be accessed only after
  // WaitReady() has returned. At that point, data_ is immutable.
#if !defined(IS_MOBILE_PLATFORM)
  absl::variant<LocalTensorHandleData, RemoteTensorHandleData> data_;
#else
  absl::variant<LocalTensorHandleData> data_;
#endif

  PartialTensorShape inference_shape_;
};

// Checks whether a VariantDevice contains a custom device.
bool VariantDeviceIsCustom(absl::variant<Device*, CustomDevice*> device);

// Wraps device->name() or CustomDevice->name().
string VariantDeviceName(absl::variant<Device*, CustomDevice*> device);

// Wraps device->DebugString() or CustomDevice->name().
string VariantDeviceDebugString(absl::variant<Device*, CustomDevice*> device);

// Indicates either HostCPU or an unset physical device. We never set a null
// CustomDevice*.
const absl::variant<Device*, CustomDevice*> kVariantDeviceNull =
    static_cast<Device*>(nullptr);

// Returns the device backing the resource. Else, returns nullptr.
Device* GetResourceDevice(const ResourceHandle& handle, EagerContext* ctx);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_TENSOR_HANDLE_H_
