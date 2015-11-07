#include "tensorflow/core/common_runtime/rendezvous_mgr.h"

#include <unordered_set>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#if (!defined(PLATFORM_POSIX_ANDROID) && !defined(PLATFORM_GOOGLE_ANDROID)) && \
    (defined(PLATFORM_GOOGLE) || GOOGLE_CUDA)
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#endif
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

namespace {

void CopyTensorBetweenDevices(const string& id, DeviceContext* send_dev_context,
                              DeviceContext* recv_dev_context, Device* src,
                              Device* dst,
                              const AllocatorAttributes src_alloc_attr,
                              const AllocatorAttributes dst_alloc_attr,
                              const Tensor* input, Tensor* output,
                              std::function<void(const Status&)> done) {
  if (src->attributes().device_type() != dst->attributes().device_type()) {
    done(errors::Unimplemented(
        "Copy between device types not yet implemented: src=", src->name(),
        " dst=", dst->name()));
  } else if (src->attributes().device_type() != "CPU") {
    done(errors::Unimplemented(
        "Copy between non-CPU devices not yet implemented"));
  }
  *output = *input;
  done(Status::OK());
}

#if (!defined(PLATFORM_POSIX_ANDROID) && !defined(PLATFORM_GOOGLE_ANDROID)) && \
    (defined(PLATFORM_GOOGLE) || GOOGLE_CUDA)
constexpr auto CopyTensorBetweenDevicesFunc = &GPUUtil::CopyViaDMA;
#else
constexpr auto CopyTensorBetweenDevicesFunc = &CopyTensorBetweenDevices;
#endif

}  // end namespace

IntraProcessRendezvous::IntraProcessRendezvous(const DeviceMgr* device_mgr)
    : device_mgr_(device_mgr), local_(NewLocalRendezvous()) {}

IntraProcessRendezvous::~IntraProcessRendezvous() { local_->Unref(); }

Status IntraProcessRendezvous::Send(const string& key,
                                    const Rendezvous::Args& args,
                                    const Tensor& val, const bool is_dead) {
  VLOG(1) << "IntraProcessRendezvous Send " << this << " " << key;
  {
    mutex_lock l(mu_);
    if (!status_.ok()) return status_;
  }
  Rendezvous::ParsedKey parsed;
  TF_RETURN_IF_ERROR(Rendezvous::ParseKey(key, &parsed));

  // Buffers "val" and "device_context" in local_.
  return local_->Send(key, args, val, is_dead);
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

  CopyTensorBetweenDevicesFunc(parsed.edge_name, send_args.device_context,
                               recv_args.device_context, src_device, dst_device,
                               send_args.alloc_attrs, recv_args.alloc_attrs,
                               &in, out, done);
}

void IntraProcessRendezvous::RecvAsync(const string& key,
                                       const Rendezvous::Args& recv_args,
                                       DoneCallback done) {
  VLOG(1) << "IntraProcessRendezvous Recv " << this << " " << key;

  Rendezvous::ParsedKey parsed;
  Status s = ParseKey(key, false /*!is_src*/, &parsed);
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), false);
    return;
  }

  // Recv the tensor from local_.
  local_->RecvAsync(key, recv_args, [this, parsed, done](
                                        const Status& status,
                                        const Rendezvous::Args& send_args,
                                        const Rendezvous::Args& recv_args,
                                        const Tensor& in, bool is_dead) {
    Status s = status;
    Tensor* out = new Tensor;
    StatusCallback final_callback = [done, send_args, recv_args, out,
                                     is_dead](const Status& s) {
      done(s, send_args, recv_args, *out, is_dead);
      delete out;
    };

    if (s.ok()) {
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
