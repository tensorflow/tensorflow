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

#include "tensorflow/core/common_runtime/copy_tensor.h"

#include <atomic>
#include <utility>
#include <vector>
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/util/reffed_status_callback.h"

namespace tensorflow {
namespace {

struct RegistrationInfo {
  RegistrationInfo(DeviceType s, DeviceType r, CopyTensor::CopyFunction cf)
      : sender_device_type(std::move(s)),
        receiver_device_type(std::move(r)),
        copy_function(cf) {}
  DeviceType sender_device_type;
  DeviceType receiver_device_type;
  CopyTensor::CopyFunction copy_function;
};

// We use a vector instead of a map since we expect there to be very
// few registrations.
std::vector<RegistrationInfo>* MutableRegistry() {
  static std::vector<RegistrationInfo>* registry =
      new std::vector<RegistrationInfo>;
  return registry;
}

void CopyHostToDevice(const Tensor* input, Allocator* cpu_allocator,
                      Allocator* out_allocator, StringPiece edge_name,
                      Device* dst, Tensor* output,
                      DeviceContext* recv_dev_context, StatusCallback done) {
  if (input->dtype() == DT_VARIANT) {
    Tensor copy(cpu_allocator, DT_VARIANT, input->shape());
    auto* status_cb = new ReffedStatusCallback(std::move(done));
    core::ScopedUnref status_cb_unref(status_cb);

    auto wrapped_done = [status_cb](const Status& s) {
      status_cb->UpdateStatus(s);
      status_cb->Unref();
    };
    auto copier = std::bind(
        [dst, recv_dev_context, out_allocator, status_cb](
            StatusCallback wrapped_done_,
            // Begin unbound arguments
            const Tensor& from, Tensor* to) {
          if (!DMAHelper::CanUseDMA(&from)) {
            Status err = errors::InvalidArgument(
                "During Variant Host->Device Copy: "
                "non-DMA-copy attempted of tensor type: ",
                DataTypeString(from.dtype()));
            status_cb->UpdateStatus(err);
            return err;
          }
          if (status_cb->ok()) {
            status_cb->Ref();
            *to = Tensor(out_allocator, from.dtype(), from.shape());
            recv_dev_context->CopyCPUTensorToDevice(&from, dst, to,
                                                    wrapped_done_);
            return Status::OK();
          } else {
            return status_cb->status();
          }
        },
        std::move(wrapped_done), std::placeholders::_1, std::placeholders::_2);

    const Variant* v = input->flat<Variant>().data();
    Variant* v_out = copy.flat<Variant>().data();
    Status s_copy_init;
    for (int64 i = 0; i < input->NumElements(); ++i) {
      s_copy_init = VariantDeviceCopy(
          VariantDeviceCopyDirection::HOST_TO_DEVICE, v[i], &v_out[i], copier);
      if (!s_copy_init.ok()) {
        status_cb->UpdateStatus(s_copy_init);
        break;
      }
    }
    if (s_copy_init.ok()) {
      *output = std::move(copy);
    }
  } else {
    recv_dev_context->CopyCPUTensorToDevice(input, dst, output,
                                            std::move(done));
  }
}

void CopyDeviceToHost(const Tensor* input, Allocator* cpu_allocator,
                      Allocator* out_allocator, StringPiece edge_name,
                      Device* src, Tensor* output,
                      DeviceContext* send_dev_context, StatusCallback done) {
  if (input->dtype() == DT_VARIANT) {
    Tensor copy(cpu_allocator, DT_VARIANT, input->shape());
    auto* status_cb = new ReffedStatusCallback(std::move(done));
    core::ScopedUnref status_cb_unref(status_cb);

    auto wrapped_done = [status_cb](const Status& s) {
      status_cb->UpdateStatus(s);
      status_cb->Unref();
    };
    auto copier = std::bind(
        [edge_name, src, send_dev_context, out_allocator, status_cb](
            StatusCallback wrapped_done_,
            // Begin unbound arguments
            const Tensor& from, Tensor* to) {
          if (!DMAHelper::CanUseDMA(&from)) {
            Status err = errors::InvalidArgument(
                "During Variant Device->Host Copy: "
                "non-DMA-copy attempted of tensor type: ",
                DataTypeString(from.dtype()));
            status_cb->UpdateStatus(err);
            return err;
          }
          if (status_cb->ok()) {
            status_cb->Ref();
            *to = Tensor(out_allocator, from.dtype(), from.shape());
            send_dev_context->CopyDeviceTensorToCPU(&from, edge_name, src, to,
                                                    wrapped_done_);
            return Status::OK();
          } else {
            return status_cb->status();
          }
        },
        std::move(wrapped_done), std::placeholders::_1, std::placeholders::_2);

    const Variant* v = input->flat<Variant>().data();
    Variant* v_out = copy.flat<Variant>().data();
    Status s_copy_init;
    for (int64 i = 0; i < input->NumElements(); ++i) {
      s_copy_init = VariantDeviceCopy(
          VariantDeviceCopyDirection::DEVICE_TO_HOST, v[i], &v_out[i], copier);
      if (!s_copy_init.ok()) {
        status_cb->UpdateStatus(s_copy_init);
        break;
      }
    }
    if (s_copy_init.ok()) {
      *output = std::move(copy);
    }
  } else {
    send_dev_context->CopyDeviceTensorToCPU(input, edge_name, src, output,
                                            std::move(done));
  }
}

void CopyDeviceToDevice(CopyTensor::CopyFunction copy_function,
                        Allocator* cpu_allocator, Allocator* out_allocator,
                        DeviceContext* send_dev_context,
                        DeviceContext* recv_dev_context, Device* src,
                        Device* dst, const AllocatorAttributes src_alloc_attr,
                        const AllocatorAttributes dst_alloc_attr,
                        const Tensor* input, Tensor* output,
                        StatusCallback done) {
  if (input->dtype() == DT_VARIANT) {
    Tensor copy(cpu_allocator, DT_VARIANT, input->shape());
    auto* status_cb = new ReffedStatusCallback(std::move(done));
    core::ScopedUnref status_cb_unref(status_cb);

    auto wrapped_done = [status_cb](const Status& s) {
      status_cb->UpdateStatus(s);
      status_cb->Unref();
    };
    auto copier = std::bind(
        [copy_function, src, dst, src_alloc_attr, dst_alloc_attr,
         recv_dev_context, send_dev_context, out_allocator,
         status_cb](StatusCallback wrapped_done_,
                    // Begin unbound arguments
                    const Tensor& from, Tensor* to) {
          if (!DMAHelper::CanUseDMA(&from)) {
            Status err = errors::InvalidArgument(
                "During Variant Device->Device Copy: "
                "non-DMA-copy attempted of tensor type: ",
                DataTypeString(from.dtype()));
            status_cb->UpdateStatus(err);
            return err;
          }
          if (status_cb->ok()) {
            status_cb->Ref();
            *to = Tensor(out_allocator, from.dtype(), from.shape());
            copy_function(send_dev_context, recv_dev_context, src, dst,
                          src_alloc_attr, dst_alloc_attr, &from, to,
                          std::move(wrapped_done_));
            return Status::OK();
          } else {
            return status_cb->status();
          }
        },
        std::move(wrapped_done), std::placeholders::_1, std::placeholders::_2);

    const Variant* v = input->flat<Variant>().data();
    Variant* v_out = copy.flat<Variant>().data();
    Status s_copy_init;
    for (int64 i = 0; i < input->NumElements(); ++i) {
      s_copy_init =
          VariantDeviceCopy(VariantDeviceCopyDirection::DEVICE_TO_DEVICE, v[i],
                            &v_out[i], copier);
      if (!s_copy_init.ok()) {
        status_cb->UpdateStatus(s_copy_init);
        break;
      }
    }
    if (s_copy_init.ok()) {
      *output = std::move(copy);
    }
  } else {
    copy_function(send_dev_context, recv_dev_context, src, dst, src_alloc_attr,
                  dst_alloc_attr, input, output, std::move(done));
  }
}

}  // namespace

// static
void CopyTensor::ViaDMA(StringPiece edge_name, DeviceContext* send_dev_context,
                        DeviceContext* recv_dev_context, Device* src,
                        Device* dst, const AllocatorAttributes src_alloc_attr,
                        const AllocatorAttributes dst_alloc_attr,
                        const Tensor* input, Tensor* output,
                        StatusCallback done) {
  tracing::ScopedAnnotation annotation(edge_name);
  VLOG(1) << "Copy " << edge_name;

  const DeviceType src_device_type(
      src_alloc_attr.on_host() ? DEVICE_CPU : src->attributes().device_type());
  const DeviceType dst_device_type(
      dst_alloc_attr.on_host() ? DEVICE_CPU : dst->attributes().device_type());
  const bool non_cpu_src = src_device_type != DeviceType(DEVICE_CPU);
  const bool non_cpu_dst = dst_device_type != DeviceType(DEVICE_CPU);

  // TODO(phawkins): choose an allocator optimal for both the src and dst
  // devices, not just the src device.
  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_gpu_compatible(true);
  host_alloc_attrs.set_on_host(true);
  Allocator* cpu_allocator = src->GetAllocator(host_alloc_attrs);
  Allocator* out_allocator = dst->GetAllocator(dst_alloc_attr);

  // E.g., gpu -> gpu
  if (non_cpu_src && non_cpu_dst) {
    // Device to device copy.  Look through registry for an appropriate
    // CopyFunction.
    std::vector<RegistrationInfo>* registry = MutableRegistry();
    for (const RegistrationInfo& ri : *registry) {
      if (ri.sender_device_type == src_device_type &&
          ri.receiver_device_type == dst_device_type) {
        CopyDeviceToDevice(ri.copy_function, cpu_allocator, out_allocator,
                           send_dev_context, recv_dev_context, src, dst,
                           src_alloc_attr, dst_alloc_attr, input, output,
                           std::move(done));
        return;
      }
    }

    // Fall back to copying via the host.
    VLOG(1) << "No function registered to copy from devices of type "
            << src_device_type.type() << " to devices of type "
            << dst_device_type.type()
            << ". Falling back to copying via the host.";

    Tensor* cpu_tensor =
        new Tensor(cpu_allocator, input->dtype(), input->shape());
    std::function<void(const Status&)> delete_and_done = std::bind(
        [cpu_tensor](StatusCallback done_,
                     // Begin unbound arguments.
                     const Status& status) {
          delete cpu_tensor;
          done_(status);
        },
        std::move(done), std::placeholders::_1);
    std::function<void(const Status&)> then_copy_to_other_device = std::bind(
        [delete_and_done, recv_dev_context, cpu_tensor, cpu_allocator,
         out_allocator, edge_name, dst, output](StatusCallback delete_and_done_,
                                                // Begin unbound arguments.
                                                Status status) {
          if (!status.ok()) {
            delete_and_done_(status);
            return;
          }
          CopyHostToDevice(cpu_tensor, cpu_allocator, out_allocator, edge_name,
                           dst, output, recv_dev_context,
                           std::move(delete_and_done_));
        },
        std::move(delete_and_done), std::placeholders::_1);
    CopyDeviceToHost(input, cpu_allocator, out_allocator, edge_name, src,
                     cpu_tensor, send_dev_context,
                     std::move(then_copy_to_other_device));
    return;
  }

  // E.g., gpu -> cpu
  if (non_cpu_src && !non_cpu_dst) {
    // Device to host copy.
    CopyDeviceToHost(input, cpu_allocator, out_allocator, edge_name, src,
                     output, send_dev_context, std::move(done));
    return;
  }

  // E.g., cpu -> gpu
  if (!non_cpu_src && non_cpu_dst) {
    // Host to Device copy.
    CopyHostToDevice(input, cpu_allocator, out_allocator, edge_name, dst,
                     output, recv_dev_context, std::move(done));
    return;
  }

  // cpu -> cpu
  CHECK(!non_cpu_src && !non_cpu_dst);
  *output = *input;
  done(Status::OK());
}

// static
Status CopyTensor::Register(DeviceType sender_device_type,
                            DeviceType receiver_device_type,
                            CopyFunction copy_function) {
  std::vector<RegistrationInfo>* registry = MutableRegistry();
  registry->emplace_back(sender_device_type, receiver_device_type,
                         copy_function);
  return Status::OK();
}

}  // namespace tensorflow
