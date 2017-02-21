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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"

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

}  // namespace

// static
void CopyTensor::ViaDMA(StringPiece edge_name, DeviceContext* send_dev_context,
                        DeviceContext* recv_dev_context, Device* src,
                        Device* dst, const AllocatorAttributes src_alloc_attr,
                        const AllocatorAttributes dst_alloc_attr,
                        const Tensor* input, Tensor* output,
                        StatusCallback done) {
  port::Tracing::ScopedAnnotation annotation(edge_name);
  VLOG(1) << "Copy " << edge_name;

  const DeviceType src_device_type(
      src_alloc_attr.on_host() ? DEVICE_CPU : src->attributes().device_type());
  const DeviceType dst_device_type(
      dst_alloc_attr.on_host() ? DEVICE_CPU : dst->attributes().device_type());
  const bool non_cpu_src = src_device_type != DeviceType(DEVICE_CPU);
  const bool non_cpu_dst = dst_device_type != DeviceType(DEVICE_CPU);

  // E.g., gpu -> gpu
  if (non_cpu_src && non_cpu_dst) {
    // Device to device copy.  Look through registry for an appropriate
    // CopyFunction.
    std::vector<RegistrationInfo>* registry = MutableRegistry();
    for (const RegistrationInfo& ri : *registry) {
      if (ri.sender_device_type == src_device_type &&
          ri.receiver_device_type == dst_device_type) {
        ri.copy_function(send_dev_context, recv_dev_context, src, dst,
                         src_alloc_attr, dst_alloc_attr, input, output, done);
        return;
      }
    }

    // Fall back to copying via the host.
    VLOG(1) << "No function registered to copy from devices of type "
            << src_device_type.type() << " to devices of type "
            << dst_device_type.type()
            << ". Falling back to copying via the host.";

    // TODO(phawkins): choose an allocator optimal for both the src and dst
    // devices, not just the src device.
    AllocatorAttributes host_alloc_attrs;
    host_alloc_attrs.set_gpu_compatible(true);
    host_alloc_attrs.set_on_host(true);
    Allocator* cpu_allocator = src->GetAllocator(host_alloc_attrs);
    Tensor* cpu_tensor =
        new Tensor(cpu_allocator, input->dtype(), input->shape());
    auto delete_and_done = [cpu_tensor, done](const Status& status) {
      delete cpu_tensor;
      done(status);
    };
    send_dev_context->CopyDeviceTensorToCPU(
        input, edge_name, src, cpu_tensor,
        [recv_dev_context, cpu_tensor, dst, output,
         delete_and_done](const Status& status) {
          if (!status.ok()) {
            delete_and_done(status);
            return;
          }
          recv_dev_context->CopyCPUTensorToDevice(cpu_tensor, dst, output,
                                                  delete_and_done);
        });
    return;
  }

  // E.g., gpu -> cpu
  if (non_cpu_src && !non_cpu_dst) {
    // Device to host copy.
    send_dev_context->CopyDeviceTensorToCPU(input, edge_name, src, output,
                                            done);
    return;
  }

  // E.g., cpu -> gpu
  if (!non_cpu_src && non_cpu_dst) {
    // Host to Device copy.
    recv_dev_context->CopyCPUTensorToDevice(input, dst, output, done);
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
