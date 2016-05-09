/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"

#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

BaseRendezvousMgr::BaseRendezvousMgr(const WorkerEnv* env) : worker_env_(env) {}

BaseRendezvousMgr::~BaseRendezvousMgr() {
  for (auto& p : table_) {
    BaseRemoteRendezvous* rendez = p.second;
    rendez->StartAbort(errors::Aborted("Shutdown"));
    rendez->Unref();
  }
}

Rendezvous* BaseRendezvousMgr::Find(int64 step_id) {
  return FindOrCreate(step_id);
}

BaseRemoteRendezvous* BaseRendezvousMgr::FindOrCreate(int64 step_id) {
  mutex_lock l(mu_);
  Table::iterator iter = table_.find(step_id);
  if (iter == table_.end()) {
    auto rr = Create(step_id, worker_env_);
    iter = table_.insert({step_id, rr}).first;
  }
  iter->second->Ref();
  return iter->second;
}

void BaseRendezvousMgr::RecvLocalAsync(int64 step_id, const string& key,
                                       Rendezvous::DoneCallback done) {
  BaseRemoteRendezvous* rendez = FindOrCreate(step_id);
  rendez->RecvLocalAsync(
      key, [rendez, done](const Status& s, const Rendezvous::Args& send_args,
                          const Rendezvous::Args& recv_args, const Tensor& v,
                          bool dead) {
        rendez->Unref();
        done(s, send_args, recv_args, v, dead);
      });
}

Status BaseRendezvousMgr::RecvLocal(int64 step_id, const string& key,
                                    Tensor* val, bool* is_dead) {
  Status ret;
  Notification n;
  RecvLocalAsync(step_id, key,
                 [val, is_dead, &ret, &n](const Status& s,
                                          const Rendezvous::Args& send_args,
                                          const Rendezvous::Args& recv_args,
                                          const Tensor& v, const bool dead) {
                   ret = s;
                   *val = v;
                   *is_dead = dead;
                   n.Notify();
                 });
  n.WaitForNotification();
  return ret;
}

void BaseRendezvousMgr::Cleanup(int64 step_id) {
  Rendezvous* rendez = nullptr;
  {
    mutex_lock l(mu_);
    Table::iterator iter = table_.find(step_id);
    if (iter != table_.end()) {
      rendez = iter->second;
      table_.erase(iter);
    }
  }
  if (!rendez) return;
  rendez->StartAbort(errors::Aborted("Cleanup ", step_id));
  rendez->Unref();
}

void BaseRendezvousMgr::CleanupAll() {
  std::vector<Rendezvous*> rendezs;
  {
    mutex_lock l(mu_);
    for (const auto& entry : table_) {
      rendezs.push_back(entry.second);
    }
    table_.clear();
  }
  for (auto rendez : rendezs) {
    rendez->StartAbort(errors::Aborted("Shutdown"));
    rendez->Unref();
  }
}

BaseRemoteRendezvous::BaseRemoteRendezvous(const WorkerEnv* env, int64 step_id,
                                           bool tolerate_dup_recv)
    : env_(env),
      step_id_(step_id),
      tolerate_dup_recv_(tolerate_dup_recv),
      local_(NewLocalRendezvous(tolerate_dup_recv)) {}

BaseRemoteRendezvous::~BaseRemoteRendezvous() {
  CHECK(active_.empty());
  local_->Unref();
}

// Returns true if "device_name" is a valid full name of local device
// of the "worker".  This helper is purely based on the worker name
// and device name and does no lookups in the worker->device_mgr.
static bool IsLocalDevice(const WorkerEnv& worker,
                          const StringPiece device_name) {
  return device_name.starts_with(worker.worker_name);
}

Status BaseRemoteRendezvous::Send(const string& key,
                                  const Rendezvous::Args& args,
                                  const Tensor& val, const bool is_dead) {
  VLOG(1) << "BaseRemoteRendezvous Send " << this << " " << key;
  {
    mutex_lock l(mu_);
    if (!status_.ok()) return status_;
  }
  Rendezvous::ParsedKey parsed;
  TF_RETURN_IF_ERROR(Rendezvous::ParseKey(key, &parsed));
  if (!IsLocalDevice(*env_, parsed.src_device)) {
    return errors::InvalidArgument("Invalid rendezvous key (src): ", key, " @ ",
                                   env_->worker_name);
  }
  // Buffers "val" and "device_context" in local_.
  return local_->Send(key, args, val, is_dead);
}

Status BaseRemoteRendezvous::ParseKey(const string& key, bool is_src,
                                      Rendezvous::ParsedKey* parsed) {
  {
    mutex_lock l(mu_);
    if (!status_.ok()) return status_;
  }
  TF_RETURN_IF_ERROR(Rendezvous::ParseKey(key, parsed));
  if (is_src && !IsLocalDevice(*env_, parsed->src_device)) {
    return errors::InvalidArgument("Invalid rendezvous key (src): ", key, " @ ",
                                   env_->worker_name);
  }
  if (!is_src && !IsLocalDevice(*env_, parsed->dst_device)) {
    return errors::InvalidArgument("Invalid rendezvous key (dst): ", key, " @ ",
                                   env_->worker_name);
  }
  return Status::OK();
}

void BaseRemoteRendezvous::SameWorkerRecvDone(
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

  // This copy must involve a GPU. Hence, "in" must support DMA
  // (e.g., string tensors do not work on GPU).
  if (!DMAHelper::CanUseDMA(&in)) {
    done(errors::InvalidArgument("Non-DMA-safe ", DataTypeString(in.dtype()),
                                 " tensor may not be copied from/to a GPU."));
    return;
  }

  Device* src_device;
  Status s = env_->device_mgr->LookupDevice(parsed.src_device, &src_device);
  if (!s.ok()) {
    done(s);
    return;
  }
  Device* dst_device;
  s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_device);
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

  // The following function takes care of cpu->gpu, gpu->cpu, gpu->gpu copies,
  // etc.
  CopyTensor::ViaDMA(parsed.edge_name, send_args.device_context,
                     recv_args.device_context, src_device, dst_device,
                     send_args.alloc_attrs, recv_args.alloc_attrs, &in, out,
                     done);
}

bool BaseRemoteRendezvous::IsSameWorker(DeviceNameUtils::ParsedName src,
                                        DeviceNameUtils::ParsedName dst) {
  return DeviceNameUtils::IsSameAddressSpace(src, dst);
}

void BaseRemoteRendezvous::RecvAsync(const string& key,
                                     const Rendezvous::Args& recv_args,
                                     DoneCallback done) {
  VLOG(1) << "RemoteRendezvous Recv " << this << " " << key;

  Rendezvous::ParsedKey parsed;
  Status s = ParseKey(key, false /*!is_src*/, &parsed);
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), false);
    return;
  }

  // Are src and dst in the same worker?
  if (IsSameWorker(parsed.src, parsed.dst)) {
    // Recv the tensor from local_.
    local_->RecvAsync(
        key, recv_args, [this, parsed, done](const Status& status,
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
            SameWorkerRecvDone(parsed, send_args, recv_args, in, out,
                               final_callback);
          } else {
            final_callback(s);
          }
        });
    return;
  } else {
    RecvFromRemoteAsync(key, parsed, recv_args, done);
  }
}

void BaseRemoteRendezvous::RecvLocalAsync(const string& key,
                                          DoneCallback done) {
  Rendezvous::ParsedKey parsed;
  Status s = ParseKey(key, true /* is_src */, &parsed);
  if (!s.ok()) {
    done(s, Args(), Args(), Tensor(), false);
    return;
  }
  local_->RecvAsync(key, Args(), done);
}

void BaseRemoteRendezvous::StartAbort(const Status& s) {
  CHECK(!s.ok());
  local_->StartAbort(s);
  {
    // Aborts all active RecvTensor calls.
    mutex_lock l(mu_);
    if (status_.ok()) {
      status_ = s;
      for (BaseRecvTensorCall* call : active_) {
        call->StartAbort(s);
      }
      active_.clear();
    }
  }
}

void BaseRemoteRendezvous::RegisterCall(BaseRecvTensorCall* call) {
  mutex_lock l(mu_);
  if (!status_.ok()) {
    call->StartAbort(status_);
  } else {
    CHECK(active_.insert(call).second);
  }
}

void BaseRemoteRendezvous::DeregisterCall(BaseRecvTensorCall* call) {
  mutex_lock l(mu_);
  active_.erase(call);
}

}  // end namespace tensorflow
