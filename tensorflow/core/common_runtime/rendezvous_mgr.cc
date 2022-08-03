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

#include "tensorflow/core/common_runtime/rendezvous_mgr.h"

#include <unordered_set>

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"

namespace tensorflow {

namespace {
void SameWorkerRecvDone(const DeviceMgr* device_mgr,
                        const Rendezvous::ParsedKey& parsed,
                        const Rendezvous::Args& send_args,
                        const Rendezvous::Args& recv_args, const Tensor& in,
                        Tensor* out, StatusCallback done) {
  // Do a quick copy (sharing the underlying buffer) if both tensors
  // are on host memory.
  const bool src_host =
      (send_args.alloc_attrs.on_host() || parsed.src.type == "CPU");
  const bool dst_host =
      (recv_args.alloc_attrs.on_host() || parsed.dst.type == "CPU");
  if (src_host && dst_host) {
    if (VLOG_IS_ON(3)) {
      bool src_override =
          send_args.alloc_attrs.on_host() && !(parsed.src.type == "CPU");
      bool dst_override =
          recv_args.alloc_attrs.on_host() && !(parsed.dst.type == "CPU");
      if (src_override || dst_override) {
        VLOG(3) << "Shortcut to keep tensor on host (src_override "
                << src_override << " and dst_override " << dst_override
                << ") tensor dtype:" << DataTypeString(in.dtype()) << " "
                << parsed.FullKey();
      }
    }
    *out = in;
    done(OkStatus());
    return;
  }

  // This copy must involve a non-CPU device. Hence, "in" must support DMA
  // (e.g., string tensors do not work on GPU).  Variant copy DMA
  // checks happen inside CopyTensor::ViaDMA.
  if (!DataTypeCanUseMemcpy(in.dtype()) && in.dtype() != DT_VARIANT &&
      in.dtype() != DT_RESOURCE) {
    done(errors::InvalidArgument(
        "Non-DMA-safe ", DataTypeString(in.dtype()),
        " tensor may not be copied from/to a device. Key: ", parsed.FullKey()));
    return;
  }

  Device* src_device;
  Status s = device_mgr->LookupDevice(parsed.src_device, &src_device);
  if (!s.ok()) {
    done(s);
    return;
  }
  Device* dst_device;
  s = device_mgr->LookupDevice(parsed.dst_device, &dst_device);
  if (!s.ok()) {
    done(s);
    return;
  }

  profiler::ScopedMemoryDebugAnnotation op_annotation(
      "SameWorkerRecvDone", 0, "dynamic", in.dtype(),
      [&in]() { return in.shape().DebugString(); });
  AllocatorAttributes attr = recv_args.alloc_attrs;
  attr.set_gpu_compatible(send_args.alloc_attrs.gpu_compatible() ||
                          recv_args.alloc_attrs.gpu_compatible());
  Allocator* out_allocator = dst_device->GetAllocator(attr);
  bool sync_dst_compute = true;
  if (in.dtype() != DT_VARIANT) {
    // Variants are handled by CopyTensor::ViaDMA.
    AllocationAttributes aa;
    uint64 safe_alloc_frontier = dst_device->SafeAllocFrontier(0);
    std::function<uint64()> freed_by_func = [dst_device,
                                             &safe_alloc_frontier]() {
      safe_alloc_frontier = dst_device->SafeAllocFrontier(safe_alloc_frontier);
      return safe_alloc_frontier;
    };
    if ((parsed.dst.type == "GPU" ||
         DeviceFactory::IsPluggableDevice(parsed.dst.type)) &&
        safe_alloc_frontier > 0) {
      // There's a timestamped allocator at work, so use it instead
      // of sync_dst_compute.
      aa.freed_by_func = &freed_by_func;
      sync_dst_compute = false;
    }
    Tensor copy(out_allocator, in.dtype(), in.shape(), aa);
    *out = copy;
    if (in.shape().num_elements() > 0 && out->data() == nullptr) {
      done(tensorflow::errors::ResourceExhausted(
          "SameWorkerRecvDone unable to allocate output tensor. Key: ",
          parsed.FullKey()));
      return;
    }
  }

  CopyTensor::ViaDMA(
      parsed.edge_name, send_args.device_context, recv_args.device_context,
      src_device, dst_device, send_args.alloc_attrs, recv_args.alloc_attrs, &in,
      out, 0 /*dev_to_dev_stream_index*/, std::move(done), sync_dst_compute);
}

void IntraProcessRecvAsyncImpl(const DeviceMgr* device_mgr,
                               LocalRendezvous* local,
                               const RendezvousInterface::ParsedKey& parsed,
                               const Rendezvous::Args& recv_args,
                               RendezvousInterface::DoneCallback done) {
  VLOG(1) << "IntraProcessRendezvous Recv " << local << " " << parsed.FullKey();

  profiler::ScopedMemoryDebugAnnotation op_annotation("RecvAsync");
  // Recv the tensor from local_.
  local->RecvAsync(
      parsed, recv_args,
      [device_mgr, parsed, done = std::move(done)](
          const Status& status, const Rendezvous::Args& send_args,
          const Rendezvous::Args& recv_args, const Tensor& in,
          bool is_dead) mutable {
        // If "in" is an uninitialized tensor, do copy-construction to
        // preserve the uninitialized state, along with data type and shape
        // info, which is useful for debugger purposes.
        Tensor* out = in.IsInitialized() ? new Tensor : new Tensor(in);

        auto final_callback = [send_args, recv_args, out, is_dead,
                               done = std::move(done)](const Status& s) {
          done(s, send_args, recv_args, *out, is_dead);
          delete out;
        };

        if (status.ok() && in.IsInitialized()) {
          SameWorkerRecvDone(device_mgr, parsed, send_args, recv_args, in, out,
                             std::move(final_callback));
        } else {
          final_callback(status);
        }
      });
}

}  // namespace

RefCountedIntraProcessRendezvous::RefCountedIntraProcessRendezvous(
    const DeviceMgr* device_mgr)
    : device_mgr_(device_mgr), local_(this) {}

RefCountedIntraProcessRendezvous::~RefCountedIntraProcessRendezvous() {}

Status RefCountedIntraProcessRendezvous::Send(const ParsedKey& key,
                                              const Rendezvous::Args& args,
                                              const Tensor& val,
                                              const bool is_dead) {
  VLOG(1) << "IntraProcessRendezvous Send " << this << " " << key.FullKey();
  return local_.Send(key, args, val, is_dead);
}

void RefCountedIntraProcessRendezvous::RecvAsync(const ParsedKey& key,
                                                 const Rendezvous::Args& args,
                                                 DoneCallback done) {
  VLOG(1) << "IntraProcessRendezvous Recv " << this << " " << key.FullKey();
  IntraProcessRecvAsyncImpl(device_mgr_, &local_, key, args, std::move(done));
}

void RefCountedIntraProcessRendezvous::StartAbort(const Status& s) {
  local_.StartAbort(s);
}

Status RefCountedIntraProcessRendezvous::GetLocalRendezvousStatus() {
  return local_.status();
}

PrivateIntraProcessRendezvous::PrivateIntraProcessRendezvous(
    const DeviceMgr* device_mgr)
    : device_mgr_(device_mgr), local_(nullptr) {}

PrivateIntraProcessRendezvous::~PrivateIntraProcessRendezvous() {}

Status PrivateIntraProcessRendezvous::Send(const ParsedKey& key,
                                           const Rendezvous::Args& args,
                                           const Tensor& val,
                                           const bool is_dead) {
  DVLOG(1) << "IntraProcessRendezvous Send " << this << " " << key.FullKey();
  return local_.Send(key, args, val, is_dead);
}

void PrivateIntraProcessRendezvous::RecvAsync(const ParsedKey& key,
                                              const Rendezvous::Args& args,
                                              DoneCallback done) {
  DVLOG(1) << "StackAllocatedIntraProcessRendezvous Recv " << this << " "
           << key.FullKey();
  IntraProcessRecvAsyncImpl(device_mgr_, &local_, key, args, std::move(done));
}

void PrivateIntraProcessRendezvous::StartAbort(const Status& s) {
  local_.StartAbort(s);
}

}  // end namespace tensorflow
