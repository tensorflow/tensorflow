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
#include "tensorflow/core/common_runtime/collective_rma_local.h"

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace tensorflow {

void CollectiveRemoteAccessLocal::StartAbort(const Status& s) {
  buf_rendezvous_.StartAbort(s);
}

void CollectiveRemoteAccessLocal::RecvFromPeer(
    const string& peer_device, const string& peer_task, bool peer_is_local,
    const string& key, Device* to_device, DeviceContext* to_device_ctx,
    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
    const DeviceLocality& client_locality, int dev_to_dev_stream_index,
    CancellationManager* cancellation_manager, const StatusCallback& done) {
  VLOG(1) << "RecvFromPeer " << this << " from " << peer_device << " key "
          << key;
  if (!peer_is_local) {
    done(
        errors::Internal("CollectiveRemoteAccessLocal::RecvFromPeer "
                         "called with peer_is_local=false"));
    return;
  }

  Device* from_device;
  Status status = dev_mgr_->LookupDevice(peer_device, &from_device);
  if (!status.ok()) {
    done(status);
    return;
  }

  auto consumer_callback = [to_tensor, to_device_ctx, to_device, to_alloc_attr,
                            dev_to_dev_stream_index,
                            done](const Status& status,
                                  BufRendezvous::Hook* hook) {
    Status s = status;
    if (s.ok()) {
      if (hook == nullptr) {
        s = errors::Internal("Invalid null hook in ConsumeBuf callback");
      }
    } else {
      if (hook != nullptr) {
        LOG(ERROR) << "Got hook " << hook << " with status " << s
                   << " from ConsumeBuf";
      }
    }

    if (s.ok()) {
      int64_t recv_bytes = to_tensor->TotalBytes();
      CHECK_EQ(recv_bytes, hook->prod_value->TotalBytes());
      MemCpyAsync(hook->prod_ctx,    // src DeviceContext
                  to_device_ctx,     // dst DeviceContext
                  hook->prod_dev,    // src Device
                  to_device,         // dst Device
                  hook->prod_attr,   // src AllocatorAttributes
                  to_alloc_attr,     // dst AllocatorAttributes
                  hook->prod_value,  // src Tensor*
                  to_tensor,         // dst Tensor*
                  dev_to_dev_stream_index,
                  [hook, done](const Status& memcpy_status) {
                    // This callback may be executing in the GPUEventMgr
                    // pool in which case it must be very short duration
                    // and non-blocking (except e.g. for queue insertion).
                    // It would be safer, though expensive, to transfer
                    // to another thread here.
                    done(memcpy_status);
                    BufRendezvous::DoneWithHook(hook);
                  });
    } else {
      done(s);
      if (hook != nullptr) {
        BufRendezvous::DoneWithHook(hook);
      }
    }
  };
  buf_rendezvous_.ConsumeBuf(key, from_device->name(),
                             from_device->attributes().incarnation(),
                             consumer_callback, cancellation_manager);
}

void CollectiveRemoteAccessLocal::PostToPeer(
    const string& peer_device, const string& peer_task, const string& key,
    Device* from_device, DeviceContext* from_device_ctx,
    const AllocatorAttributes& from_alloc_attr, const Tensor* from_tensor,
    const DeviceLocality& client_locality,
    CancellationManager* cancellation_manager, const StatusCallback& done) {
  VLOG(1) << "PostToPeer " << this << " key " << key
          << " step_id_=" << step_id_;
  buf_rendezvous_.ProvideBuf(key, from_device, from_device_ctx, from_tensor,
                             from_alloc_attr, done, cancellation_manager);
}

void CollectiveRemoteAccessLocal::CheckPeerHealth(const string& peer_task,
                                                  int64_t timeout_in_ms,
                                                  const StatusCallback& done) {
  // Assume local devices are always healthy.
  done(errors::Internal(
      "CheckPeerHealth is not supposed to be called for local collectives"));
}

/*static*/
void CollectiveRemoteAccessLocal::MemCpyAsync(
    DeviceContext* src_dev_ctx, DeviceContext* dst_dev_ctx, Device* src_dev,
    Device* dst_dev, const AllocatorAttributes& src_attr,
    const AllocatorAttributes& dst_attr, const Tensor* src, Tensor* dst,
    int dev_to_dev_stream_index, const StatusCallback& done) {
  // We want a real copy to happen, i.e. the bytes inside of src should be
  // transferred to the buffer backing dst.  If src and dst are on different
  // devices then CopyTensor::ViaDMA will do just that.  But if they're both
  // the same CPU, then it will actually just reset dst to point to src.
  // Since this routine is used for copying between devices and within a
  // device, we need to detect and bypass the wrong-semantics case.
  const DeviceType src_device_type(
      src_attr.on_host() ? DEVICE_CPU : src_dev->attributes().device_type());
  const DeviceType dst_device_type(
      dst_attr.on_host() ? DEVICE_CPU : dst_dev->attributes().device_type());
  const bool non_cpu_src = src_device_type != DeviceType(DEVICE_CPU);
  const bool non_cpu_dst = dst_device_type != DeviceType(DEVICE_CPU);
  // For GPU devices when only one compute stream is used (the default)
  // the OpKernelContext does not supply a DeviceContext.  It's assumed
  // that all nodes use the default context.
  if (src_dev_ctx == nullptr && src_device_type == DEVICE_GPU) {
    const DeviceBase::GpuDeviceInfo* dev_info =
        src_dev->tensorflow_gpu_device_info();
    CHECK(dev_info);
    src_dev_ctx = dev_info->default_context;
  }
  if (dst_dev_ctx == nullptr && dst_device_type == DEVICE_GPU) {
    const DeviceBase::GpuDeviceInfo* dev_info =
        src_dev->tensorflow_gpu_device_info();
    CHECK(dev_info);
    dst_dev_ctx = dev_info->default_context;
  }
  if (non_cpu_src) CHECK(src_dev_ctx);
  if (non_cpu_dst) CHECK(dst_dev_ctx);
  if (non_cpu_src || non_cpu_dst) {
    CopyTensor::ViaDMA("",  // edge name (non-existent)
                       src_dev_ctx, dst_dev_ctx, src_dev, dst_dev, src_attr,
                       dst_attr, src, dst, dev_to_dev_stream_index, done);
  } else {
    int64_t bytes = src->TotalBytes();
    DCHECK_EQ(dst->TotalBytes(), bytes);
    memcpy(DMAHelper::base(dst), DMAHelper::base(src), bytes);
    done(Status::OK());
  }
}

}  // namespace tensorflow
