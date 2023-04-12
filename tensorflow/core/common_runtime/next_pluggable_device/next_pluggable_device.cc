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

#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device.h"

#include <memory>
#include <utility>

#include "absl/flags/flag.h"
#include "tensorflow/compiler/jit/pjrt_device_context.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_allocator.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

// TODO(b/263832956): remove this flag when next_pluggable_device is open
// sourced.
ABSL_FLAG(bool, next_pluggable_device_use_pjrt, true,
          "Use PjRtClient for data transfer and compile on demand op in next "
          "pluggable device.");

namespace tensorflow {

// TODO(chuanhao): implement an API to query device memory, and make
// memory_limit a parameter instead of hard coding.
static DeviceAttributes BuildNextPluggableDeviceAttributes(
    const string& name_prefix, const string& device_name, int device_ordinal) {
  return Device::BuildDeviceAttributes(
      absl::StrCat(name_prefix, "/device:", device_name, ":", device_ordinal),
      DeviceType(device_name), Bytes(16ULL << 30), DeviceLocality(),
      absl::StrCat("device: ", device_name, " device"));
}

NextPluggableDevice::NextPluggableDevice(const SessionOptions& session_options,
                                         const Options& options)
    : PjRtBaseDevice(
          session_options,
          PjRtBaseDevice::Options(options.device_name_prefix,
                                  options.device_name, options.device_ordinal,
                                  options.compilation_device_name,
                                  options.shape_determination_fns)),
      device_ordinal_(options.device_ordinal) {
  allocator_ = std::make_unique<NextPluggableDeviceAllocator>(device_ordinal_);
  if (absl::GetFlag(FLAGS_next_pluggable_device_use_pjrt)) {
    // TODO(b/262472386) Support shape_determination_fns through
    // TFNPD_XlaShapeToDeviceShapeRepresentation.
    if (!options.shape_determination_fns.empty()) {
      device_context_ = core::RefCountPtr<DeviceContext>(
          new PjRtDeviceContext(options.shape_determination_fns[0]));
    } else {
      XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns{
          UseNoPreferenceLayoutFn(), IdentityShapeRepresentationFn()};
      device_context_ = core::RefCountPtr<DeviceContext>(
          new PjRtDeviceContext(shape_determination_fns));
    }
  } else {
    device_context_ = core::RefCountPtr<DeviceContext>(
        new NextPluggableDeviceContext(device_ordinal_));
  }

  // Must set accelerator_device_info, otherwise TF will treat this device as
  // CPU device.
  auto accelerator_device_info =
      std::make_unique<DeviceBase::AcceleratorDeviceInfo>();
  accelerator_device_info->default_context = device_context_.get();
  set_tensorflow_accelerator_device_info(accelerator_device_info.get());
  accelerator_device_info_ = std::move(accelerator_device_info);
}

NextPluggableDevice::~NextPluggableDevice() = default;

Allocator* NextPluggableDevice::GetAllocator(AllocatorAttributes attr) {
  if (attr.on_host()) {
    return cpu_allocator();
  }
  return allocator_.get();
}

void NextPluggableDevice::Compute(OpKernel* op_kernel,
                                  OpKernelContext* context) {
  VLOG(1) << "NextPluggableDevice::Compute " << op_kernel->name() << ":"
          << op_kernel->type_string();
  op_kernel->Compute(context);
}

void NextPluggableDevice::ComputeAsync(AsyncOpKernel* op_kernel,
                                       OpKernelContext* context,
                                       AsyncOpKernel::DoneCallback done) {
  VLOG(1) << "NextPluggableDevice::ComputeAsync " << op_kernel->name() << ":"
          << op_kernel->type_string();
  op_kernel->ComputeAsync(context, done);
}

// TODO(chuanhao): implement NextPluggableDevice::Sync().
Status NextPluggableDevice::Sync() { return OkStatus(); }

// TODO(chuanhao): implement NextPluggableDevice::Sync().
void NextPluggableDevice::Sync(const DoneCallback& done) {}

Status NextPluggableDevice::TryGetDeviceContext(DeviceContext** out_context) {
  *out_context = device_context_.get();
  (*out_context)->Ref();
  return OkStatus();
}

Status NextPluggableDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }

  Status status;
  if (alloc_attrs.on_host()) {
    *tensor = parsed;
  } else {
    Allocator* allocator = GetAllocator(alloc_attrs);
    Tensor copy(allocator, parsed.dtype(), parsed.shape());
    TF_RETURN_IF_ERROR(
        device_context_->CopyCPUTensorToDeviceSync(&parsed, this, &copy));
    *tensor = copy;
  }
  VLOG(2) << "Allocated tensor at " << DMAHelper::base(tensor);
  return status;
}

}  // namespace tensorflow
