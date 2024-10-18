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
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/reffed_status_callback.h"

ABSL_FLAG(bool, next_pluggable_device_use_pjrt_allocator, true,
          "Use PjRtAllocator in next pluggable device.");

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
  if (absl::GetFlag(FLAGS_next_pluggable_device_use_pjrt_allocator)) {
    pjrt_allocator_ = std::make_unique<AsyncValueAllocator>();
    allocator_ = pjrt_allocator_.get();
  } else {
    tfnpd_allocator_ =
        std::make_unique<NextPluggableDeviceAllocator>(device_ordinal_);
    allocator_ = tfnpd_allocator_.get();
  }

  if (!options.shape_determination_fns.empty()) {
    device_context_ = core::RefCountPtr<DeviceContext>(
        new PjRtDeviceContext(options.shape_determination_fns[0]));
  } else {
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns{
        UseNoPreferenceLayoutFn(), IdentityShapeRepresentationFn()};
    device_context_ = core::RefCountPtr<DeviceContext>(
        new PjRtDeviceContext(shape_determination_fns));
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
  return allocator_;
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
absl::Status NextPluggableDevice::Sync() { return absl::OkStatus(); }

// TODO(chuanhao): implement NextPluggableDevice::Sync().
void NextPluggableDevice::Sync(const DoneCallback& done) { done(Sync()); }

absl::Status NextPluggableDevice::TryGetDeviceContext(
    DeviceContext** out_context) {
  *out_context = device_context_.get();
  (*out_context)->Ref();
  return absl::OkStatus();
}

absl::Status NextPluggableDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        absl::StrCat("Cannot parse tensor from proto: ",
                                     tensor_proto.DebugString()));
  }

  absl::Status status;
  if (alloc_attrs.on_host()) {
    *tensor = parsed;
    VLOG(2) << "Allocated tensor at " << DMAHelper::base(tensor);
    return status;
  }

  if (parsed.dtype() != DT_VARIANT) {
    Allocator* allocator = GetAllocator(alloc_attrs);
    Tensor copy(allocator, parsed.dtype(), parsed.shape());
    TF_RETURN_IF_ERROR(
        device_context_->CopyCPUTensorToDeviceSync(&parsed, this, &copy));
    *tensor = copy;
    VLOG(2) << "Allocated tensor at " << DMAHelper::base(tensor);
    return status;
  }
  const Variant* from = parsed.flat<Variant>().data();
  Tensor copy(cpu_allocator(), DT_VARIANT, parsed.shape());
  Variant* copy_variant = copy.flat<Variant>().data();

  std::list<Notification> notifications;
  auto copier = [this, &alloc_attrs, &notifications, &status](
                    const Tensor& from, Tensor* to) {
    // Copier isn't run in a multithreaded environment, so we don't
    // have to worry about the notifications list being modified in parallel.
    notifications.emplace_back();
    Notification& n = *notifications.rbegin();

    StatusCallback done = [&n, &status](const absl::Status& s) {
      if (status.ok()) {
        status.Update(s);
      }
      n.Notify();
    };
    if (!DMAHelper::CanUseDMA(&from)) {
      absl::Status err =
          absl::Status(absl::StatusCode::kInternal,
                       absl::StrCat("NextPluggableDevice copy from non-DMA ",
                                    DataTypeString(from.dtype()), " tensor"));
      done(err);
      return err;
    }

    auto* copy_dst =
        new Tensor(GetAllocator(alloc_attrs), from.dtype(), from.shape());

    // If the tensor is not initialized, we likely ran out of memory.
    if (!copy_dst->IsInitialized()) {
      delete copy_dst;
      absl::Status err =
          absl::Status(absl::StatusCode::kResourceExhausted,
                       absl::StrCat("OOM when allocating tensor of shape ",
                                    from.shape().DebugString(), " and type ",
                                    DataTypeString(from.dtype())));
      done(err);
      return err;
    }

    auto wrapped_done = [to, copy_dst,
                         done = std::move(done)](const absl::Status& s) {
      if (s.ok()) {
        *to = std::move(*copy_dst);
      }
      delete copy_dst;
      done(s);
    };

    device_context_->CopyCPUTensorToDevice(&from, this, copy_dst,
                                           std::move(wrapped_done),
                                           true /*sync_dst_compute*/);
    return absl::OkStatus();
  };

  absl::Status s;
  for (int64_t ix = 0; ix < parsed.NumElements(); ++ix) {
    s = VariantDeviceCopy(VariantDeviceCopyDirection::HOST_TO_DEVICE, from[ix],
                          &copy_variant[ix], copier);
    if (!s.ok()) {
      break;
    }
  }
  for (auto& n : notifications) {
    n.WaitForNotification();
  }
  if (!s.ok()) {
    return s;
  }
  *tensor = std::move(copy);
  return status;
}

}  // namespace tensorflow
