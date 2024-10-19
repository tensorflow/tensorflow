/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/jit/pjrt_base_device.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_context.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/tfrt/common/async_value_tensor.h"

namespace tensorflow {

class NextPluggableDeviceAllocator;

class NextPluggableDevice : public PjRtBaseDevice {
 public:
  struct Options {
    // The device name's prefix (e.g., "/task:7")
    string device_name_prefix;

    // The name of the  device (e.g., "GPU")
    string device_name;

    // The name of the compilation device (e.g., "XLA_TPU_JIT");
    string compilation_device_name;

    // The TfDeviceId.
    int device_ordinal = -1;

    // A vector of ShapeDeterminationFn (i.e., a bundle of LayoutSelectionFn,
    // ShapeRepresentationFn). Each bundle describes how the on-host shapes of
    // a) argument and return value, for entry computations b) variables, for
    // all computations, should be represented in XLA. Parameters/return values
    // will be shaped according to the function pair, and reshaped back to/from
    // their declared shapes for computations. Must be non-empty.
    std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
        shape_determination_fns;
  };

  NextPluggableDevice(const SessionOptions& session_options,
                      const Options& options);

  ~NextPluggableDevice() override;

  Allocator* GetAllocator(AllocatorAttributes attr) override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;

  void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;

  absl::Status Sync() override;

  void Sync(const DoneCallback& done) override;

  absl::Status TryGetDeviceContext(DeviceContext** out_context) override;

  absl::Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                   AllocatorAttributes alloc_attrs,
                                   Tensor* tensor) override;

  int GetDeviceOrdinal() const { return device_ordinal_; }

 private:
  int device_ordinal_;
  // Need to use RefCountPtr since DeviceContext is a ref counted object.
  core::RefCountPtr<DeviceContext> device_context_;
  std::unique_ptr<NextPluggableDeviceAllocator> tfnpd_allocator_;
  std::unique_ptr<AsyncValueAllocator> pjrt_allocator_;
  Allocator* allocator_ = nullptr;  // Not owned.
  std::unique_ptr<DeviceBase::AcceleratorDeviceInfo> accelerator_device_info_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_NEXT_PLUGGABLE_DEVICE_H_
