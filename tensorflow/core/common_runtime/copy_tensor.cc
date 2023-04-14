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
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/util/reffed_status_callback.h"

namespace tensorflow {
namespace {

struct RegistrationInfo {
  RegistrationInfo(DeviceType s, DeviceType r, CopyTensor::CopyFunction cf,
                   bool is_pluggable_device)
      : sender_device_type(std::move(s)),
        receiver_device_type(std::move(r)),
        copy_function(cf),
        is_pluggable_device(is_pluggable_device) {}
  DeviceType sender_device_type;
  DeviceType receiver_device_type;
  CopyTensor::CopyFunction copy_function;
  bool is_pluggable_device;
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
                      DeviceContext* recv_dev_context, StatusCallback done,
                      bool sync_dst_compute, bool sync_dst_recv) {
  if (input->dtype() == DT_VARIANT) {
    Tensor copy(cpu_allocator, DT_VARIANT, input->shape());
    auto* status_cb = new ReffedStatusCallback(std::move(done));
    core::ScopedUnref status_cb_unref(status_cb);

    auto wrapped_done = [status_cb](const Status& s) {
      status_cb->UpdateStatus(s);
      status_cb->Unref();
    };
    auto copier = [dst, recv_dev_context, out_allocator, status_cb,
                   cpu_allocator, edge_name, sync_dst_compute, sync_dst_recv,
                   wrapped_done = std::move(wrapped_done)](const Tensor& from,
                                                           Tensor* to) {
      if (from.dtype() == DT_VARIANT) {
        status_cb->Ref();
        CopyHostToDevice(&from, cpu_allocator, out_allocator, edge_name, dst,
                         to, recv_dev_context, wrapped_done, sync_dst_compute,
                         sync_dst_recv);
        return OkStatus();
      } else {
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
          recv_dev_context->CopyCPUTensorToDevice(
              &from, dst, to, wrapped_done, sync_dst_compute, sync_dst_recv);
          return OkStatus();
        } else {
          return status_cb->status();
        }
      }
    };

    const Variant* v = input->flat<Variant>().data();
    Variant* v_out = copy.flat<Variant>().data();
    Status s_copy_init;
    for (int64_t i = 0; i < input->NumElements(); ++i) {
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
  } else if (input->dtype() == DT_RESOURCE) {
    *output = *input;
    done(OkStatus());
  } else {
    recv_dev_context->CopyCPUTensorToDevice(input, dst, output, std::move(done),
                                            sync_dst_compute, sync_dst_recv);
  }
}

void CopyDeviceToDevice(CopyTensor::CopyFunction copy_function,
                        Allocator* cpu_allocator, Allocator* out_allocator,
                        DeviceContext* send_dev_context,
                        DeviceContext* recv_dev_context, Device* src,
                        Device* dst, const AllocatorAttributes src_alloc_attr,
                        const AllocatorAttributes dst_alloc_attr,
                        const Tensor* input, Tensor* output,
                        int dev_to_dev_stream_index, StatusCallback done) {
  if (input->dtype() == DT_VARIANT) {
    Tensor copy(cpu_allocator, DT_VARIANT, input->shape());
    auto* status_cb = new ReffedStatusCallback(std::move(done));
    core::ScopedUnref status_cb_unref(status_cb);

    auto wrapped_done = [status_cb](const Status& s) {
      status_cb->UpdateStatus(s);
      status_cb->Unref();
    };
    auto copier = [copy_function, cpu_allocator, src, dst, src_alloc_attr,
                   dst_alloc_attr, recv_dev_context, send_dev_context,
                   out_allocator, status_cb, dev_to_dev_stream_index,
                   wrapped_done = std::move(wrapped_done)](
                      // Begin unbound arguments
                      const Tensor& from, Tensor* to) {
      if (from.dtype() == DT_VARIANT) {
        status_cb->Ref();
        CopyDeviceToDevice(copy_function, cpu_allocator, out_allocator,
                           send_dev_context, recv_dev_context, src, dst,
                           src_alloc_attr, dst_alloc_attr, &from, to,
                           dev_to_dev_stream_index, wrapped_done);
        return OkStatus();
      } else {
        if (!DMAHelper::CanUseDMA(&from)) {
          Status err = errors::InvalidArgument(
              "During Variant Device->Device Copy: ", src->name(), " to ",
              dst->name(), " non-DMA-copy attempted of tensor type: ",
              DataTypeString(from.dtype()));
          status_cb->UpdateStatus(err);
          return err;
        }
        if (status_cb->ok()) {
          status_cb->Ref();
          *to = Tensor(out_allocator, from.dtype(), from.shape());
          copy_function(send_dev_context, recv_dev_context, src, dst,
                        src_alloc_attr, dst_alloc_attr, &from, to,
                        dev_to_dev_stream_index, wrapped_done);
          return OkStatus();
        } else {
          return status_cb->status();
        }
      }
    };

    const Variant* v = input->flat<Variant>().data();
    Variant* v_out = copy.flat<Variant>().data();
    Status s_copy_init;
    for (int64_t i = 0; i < input->NumElements(); ++i) {
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
  } else if (input->dtype() == DT_RESOURCE) {
    *output = *input;
    done(OkStatus());
  } else {
    copy_function(send_dev_context, recv_dev_context, src, dst, src_alloc_attr,
                  dst_alloc_attr, input, output, dev_to_dev_stream_index,
                  std::move(done));
  }
}

}  // namespace

// static
void CopyTensor::ViaDMA(StringPiece edge_name, DeviceContext* send_dev_context,
                        DeviceContext* recv_dev_context, Device* src,
                        Device* dst, const AllocatorAttributes src_alloc_attr,
                        const AllocatorAttributes dst_alloc_attr,
                        const Tensor* input, Tensor* output,
                        int dev_to_dev_stream_index, StatusCallback done,
                        bool sync_dst_compute, bool sync_dst_recv) {
  profiler::ScopedAnnotation annotation(
      [&] { return absl::StrCat("#edge_name=", edge_name, "#"); });
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
    // TODO(penpornk): Revisit the lookup mechanism after PR #43611 (device
    // alias) is resolved.
    const bool src_device_is_pluggable =
        DeviceFactory::IsPluggableDevice(src_device_type.type_string());
    for (const RegistrationInfo& ri : *registry) {
      if (ri.sender_device_type == src_device_type &&
          ri.receiver_device_type == dst_device_type) {
        if (src_device_is_pluggable && !ri.is_pluggable_device) continue;
        CopyDeviceToDevice(ri.copy_function, cpu_allocator, out_allocator,
                           send_dev_context, recv_dev_context, src, dst,
                           src_alloc_attr, dst_alloc_attr, input, output,
                           dev_to_dev_stream_index, std::move(done));
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
    auto delete_and_done = [cpu_tensor,
                            done = std::move(done)](const Status& status) {
      delete cpu_tensor;
      done(status);
    };
    auto then_copy_to_other_device =
        [delete_and_done = std::move(delete_and_done), recv_dev_context,
         cpu_tensor, cpu_allocator, out_allocator, edge_name, dst, output,
         sync_dst_compute, sync_dst_recv](Status status) {
          if (!status.ok()) {
            delete_and_done(status);
            return;
          }
          CopyHostToDevice(cpu_tensor, cpu_allocator, out_allocator, edge_name,
                           dst, output, recv_dev_context,
                           std::move(delete_and_done), sync_dst_compute,
                           sync_dst_recv);
        };
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
                     output, recv_dev_context, std::move(done),
                     sync_dst_compute, sync_dst_recv);
    return;
  }

  // cpu -> cpu
  CHECK(!non_cpu_src && !non_cpu_dst);
  *output = *input;
  done(OkStatus());
}

// static
Status CopyTensor::Register(DeviceType sender_device_type,
                            DeviceType receiver_device_type,
                            CopyFunction copy_function,
                            bool is_pluggable_device) {
  std::vector<RegistrationInfo>* registry = MutableRegistry();
  registry->emplace_back(sender_device_type, receiver_device_type,
                         copy_function, is_pluggable_device);
  return OkStatus();
}

namespace {

// The following registrations enable a DT_VARIANT tensor element that contains
// a wrapped `tensorflow::Tensor` to be copied between devices.
static Status WrappedTensorDeviceCopy(
    const Tensor& from, Tensor* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
  if (DMAHelper::CanUseDMA(&from)) {
    TF_RETURN_IF_ERROR(copy(from, to));
  } else {
    *to = from;
  }

  return OkStatus();
}

#define REGISTER_WRAPPED_TENSOR_COPY(DIRECTION)         \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      Tensor, DIRECTION, WrappedTensorDeviceCopy)

REGISTER_WRAPPED_TENSOR_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_WRAPPED_TENSOR_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_WRAPPED_TENSOR_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

}  // namespace

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
    auto copier = [edge_name, src, send_dev_context, out_allocator, status_cb,
                   cpu_allocator, wrapped_done = std::move(wrapped_done)](
                      const Tensor& from, Tensor* to) {
      if (from.dtype() == DT_VARIANT) {
        status_cb->Ref();
        CopyDeviceToHost(&from, cpu_allocator, out_allocator, edge_name, src,
                         to, send_dev_context, wrapped_done);
        return OkStatus();
      } else {
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
                                                  wrapped_done);
          return OkStatus();
        } else {
          return status_cb->status();
        }
      }
    };

    const Variant* v = input->flat<Variant>().data();
    Variant* v_out = copy.flat<Variant>().data();
    Status s_copy_init;
    for (int64_t i = 0; i < input->NumElements(); ++i) {
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
  } else if (input->dtype() == DT_RESOURCE) {
    *output = *input;
    done(OkStatus());
  } else {
    send_dev_context->CopyDeviceTensorToCPU(input, edge_name, src, output,
                                            std::move(done));
  }
}

}  // namespace tensorflow
