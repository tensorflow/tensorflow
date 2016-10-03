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
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

IntraProcessRendezvous::IntraProcessRendezvous(const DeviceMgr* device_mgr)
    : device_mgr_(device_mgr), local_(NewLocalRendezvous()) {}

IntraProcessRendezvous::~IntraProcessRendezvous() { local_->Unref(); }

Status IntraProcessRendezvous::Send(const ParsedKey& parsed,
                                    const Rendezvous::Args& args,
                                    const Tensor& val, const bool is_dead) {
  VLOG(1) << "IntraProcessRendezvous Send " << this << " " << parsed.FullKey();
  {
    mutex_lock l(mu_);
    if (!status_.ok()) return status_;
  }

  // Buffers "val" and "device_context" in local_.
  return local_->Send(parsed, args, val, is_dead);
}

Status IntraProcessRendezvous::ParseKey(const string& key, bool is_src,
                                        Rendezvous::ParsedKey* parsed) {
  {
    mutex_lock l(mu_);
    if (!status_.ok()) return status_;
  }
  TF_RETURN_IF_ERROR(Rendezvous::ParseKey(key, parsed));
  return Status::OK();
}

void IntraProcessRendezvous::SameWorkerRecvDone(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& send_args,
    const Rendezvous::Args& recv_args, const Tensor& in, Tensor* out,
    StatusCallback done) {
  // Do a quick copy (sharing the underlying buffer) if both tensors
  // are on host memory.
  const bool src_host =
      (send_args.alloc_attrs.on_host() || parsed.src.type == "CPU");
  const bool dst_host =
      (recv_args.alloc_attrs.on_host() || parsed.dst.type == "CPU");
  if (src_host && dst_host) {
    *out = in;
    done(Status::OK());
    return;
  }

  // This copy must involve a non-CPU device. Hence, "in" must support DMA
  // (e.g., string tensors do not work on GPU).
  if (!DataTypeCanUseMemcpy(in.dtype())) {
    done(errors::InvalidArgument("Non-DMA-safe ", DataTypeString(in.dtype()),
                                 " tensor may not be copied from/to a GPU."));
    return;
  }

  Device* src_device;
  Status s = device_mgr_->LookupDevice(parsed.src_device, &src_device);
  if (!s.ok()) {
    done(s);
    return;
  }
  Device* dst_device;
  s = device_mgr_->LookupDevice(parsed.dst_device, &dst_device);
  if (!s.ok()) {
    done(s);
    return;
  }

  AllocatorAttributes attr = recv_args.alloc_attrs;
  attr.set_gpu_compatible(send_args.alloc_attrs.gpu_compatible() ||
                          recv_args.alloc_attrs.gpu_compatible());
  Allocator* out_allocator = dst_device->GetAllocator(attr);
  Tensor copy(out_allocator, in.dtype(), in.shape());
  *out = copy;

  CopyTensor::ViaDMA(parsed.edge_name, send_args.device_context,
                     recv_args.device_context, src_device, dst_device,
                     send_args.alloc_attrs, recv_args.alloc_attrs, &in, out,
                     done);
}

void IntraProcessRendezvous::RecvAsync(const ParsedKey& parsed,
                                       const Rendezvous::Args& recv_args,
                                       DoneCallback done) {
  VLOG(1) << "IntraProcessRendezvous Recv " << this << " " << parsed.FullKey();

  // Recv the tensor from local_.
  local_->RecvAsync(parsed, recv_args, [this, parsed, done](
                                           const Status& status,
                                           const Rendezvous::Args& send_args,
                                           const Rendezvous::Args& recv_args,
                                           const Tensor& in, bool is_dead) {
    Status s = status;

    // If "in" is an uninitialized tensor, do copy-construction to preserve
    // the uninitialized state, along with data type and shape info, which
    // is useful for debugger purposes.
    Tensor* out = in.IsInitialized() ? new Tensor : new Tensor(in);

    StatusCallback final_callback = [done, send_args, recv_args, out,
                                     is_dead](const Status& s) {
      done(s, send_args, recv_args, *out, is_dead);
      delete out;
    };

    if (s.ok() && in.IsInitialized()) {
      SameWorkerRecvDone(parsed, send_args, recv_args, in, out, final_callback);
    } else {
      final_callback(s);
    }
  });
}

void IntraProcessRendezvous::StartAbort(const Status& s) {
  CHECK(!s.ok());
  local_->StartAbort(s);
}

}  // end namespace tensorflow
