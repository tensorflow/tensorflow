/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"

namespace tensorflow {

static void StartAbortRendevous(Rendezvous* rendez, const Status& s) {
  rendez->StartAbort(s);
  rendez->Unref();
}

BaseRendezvousMgr::BaseRendezvousMgr(const WorkerEnv* worker_env)
    : worker_env_(worker_env) {}

BaseRendezvousMgr::~BaseRendezvousMgr() {
  for (auto& p : table_) {
    auto rendez = p.second;
    StartAbortRendevous(rendez, errors::Aborted("Shutdown"));
  }
}

RemoteRendezvous* BaseRendezvousMgr::Find(int64_t step_id) {
  return FindOrCreate(step_id);
}

BaseRemoteRendezvous* BaseRendezvousMgr::FindOrCreate(int64_t step_id) {
  mutex_lock l(mu_);
  auto iter = table_.find(step_id);
  if (iter == table_.end()) {
    auto rr = Create(step_id, worker_env_);
    iter = table_.insert({step_id, rr}).first;
  }
  iter->second->Ref();
  return iter->second;
}

void BaseRendezvousMgr::RecvLocalAsync(int64_t step_id,
                                       const Rendezvous::ParsedKey& parsed,
                                       Rendezvous::DoneCallback done) {
  auto rendez = FindOrCreate(step_id);
  auto done_cb = [rendez, done = std::move(done)](
                     const Status& s, const Rendezvous::Args& send_args,
                     const Rendezvous::Args& recv_args, const Tensor& v,
                     bool dead) {
    rendez->Unref();
    done(s, send_args, recv_args, v, dead);
  };
  rendez->RecvLocalAsync(parsed, std::move(done_cb));
}

Status BaseRendezvousMgr::RecvLocal(int64_t step_id,
                                    const Rendezvous::ParsedKey& parsed,
                                    Tensor* val, bool* is_dead) {
  Status ret;
  Notification n;
  RecvLocalAsync(step_id, parsed,
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

void BaseRendezvousMgr::Cleanup(int64_t step_id) {
  Rendezvous* rendez = nullptr;
  {
    mutex_lock l(mu_);
    auto iter = table_.find(step_id);
    if (iter != table_.end()) {
      rendez = iter->second;
      table_.erase(iter);
    }
  }
  if (rendez) {
    StartAbortRendevous(rendez, errors::Aborted("Cleanup ", step_id));
  }
}

void BaseRendezvousMgr::CleanupAll() {
  mutex_lock l(mu_);
  for (auto iter = table_.begin(); iter != table_.end(); iter++) {
    iter->second->Unref();
  }
}

BaseRemoteRendezvous::BaseRemoteRendezvous(const WorkerEnv* env,
                                           int64_t step_id)
    : env_(env),
      step_id_(step_id),
      num_shards_(env_->experimental_num_shards),
      local_(NewLocalRendezvous(num_shards_)),
      session_(nullptr) {
  DCHECK_GT(env_->experimental_num_shards, 0);
}

BaseRemoteRendezvous::~BaseRemoteRendezvous() {
  calls_.clear();
  local_->Unref();
}

// Returns true if "device_name" is a valid full name of local device
// of the "worker".  This helper is purely based on the worker name
// and device name and does no lookups in the worker->device_mgr.
static bool IsLocalDevice(const StringPiece worker_name,
                          const StringPiece device_name) {
  return absl::StartsWith(device_name, worker_name);
}

Status BaseRemoteRendezvous::Initialize(WorkerSession* session) {
  CHECK_NE(session, nullptr) << "session must not be null!";
  std::vector<DeferredCall> deferred_calls;
  {
    mutex_lock l(mu_);
    if (session_ != nullptr) {
      if (session_->worker_name() == session->worker_name()) {
        VLOG(1) << "Skipping rendezvous re-initialization.";
        return OkStatus();
      }
      Status s = errors::Internal(
          "Double init! Worker names would have changed from: ",
          session_->worker_name(), " -> ", session->worker_name());
      LOG(WARNING) << s;
      return s;
    }
    session_ = session;
    std::swap(deferred_calls, deferred_calls_);
  }
  for (auto& call : deferred_calls) {
    RecvLocalAsyncInternal(call.parsed, std::move(call.done));
  }
  return OkStatus();
}

WorkerSession* BaseRemoteRendezvous::session() {
  tf_shared_lock l(mu_);
  return session_;
}

bool BaseRemoteRendezvous::is_initialized() {
  tf_shared_lock l(mu_);
  return is_initialized_locked();
}

Status BaseRemoteRendezvous::Send(const Rendezvous::ParsedKey& parsed,
                                  const Rendezvous::Args& args,
                                  const Tensor& val, const bool is_dead) {
  VLOG(1) << "BaseRemoteRendezvous Send " << this << " " << parsed.FullKey();
  WorkerSession* sess = nullptr;
  {
    tf_shared_lock l(mu_);
    if (!status_.ok()) return status_;
    DCHECK(is_initialized_locked());
    sess = session_;
  }

  if (!IsLocalDevice(sess->worker_name(), parsed.src_device)) {
    return errors::InvalidArgument(
        "Invalid rendezvous key (src): ", parsed.FullKey(), " @ ",
        sess->worker_name());
  }

  // Buffers "val" and "device_context" in local_.
  return local_->Send(parsed, args, val, is_dead);
}

Status BaseRemoteRendezvous::ValidateDevices(const ParsedKey& parsed,
                                             bool is_src) {
  // Cache session pointer to avoid repeatedly taking & releasing the lock
  // (e.g. calling session())
  WorkerSession* sess = nullptr;
  {
    tf_shared_lock l(mu_);
    if (!status_.ok()) return status_;
    if (!is_initialized_locked()) {
      return errors::Internal("ValidateDevices called before initialization.");
    }
    sess = session_;
  }
  if (is_src && !IsLocalDevice(sess->worker_name(), parsed.src_device)) {
    return errors::InvalidArgument(
        "Invalid rendezvous key (src): ", parsed.FullKey(), " @ ",
        sess->worker_name());
  }
  if (!is_src && !IsLocalDevice(sess->worker_name(), parsed.dst_device)) {
    return errors::InvalidArgument(
        "Invalid rendezvous key (dst): ", parsed.FullKey(), " @ ",
        sess->worker_name());
  }
  return OkStatus();
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
    done(OkStatus());
    return;
  }

  // This copy must involve a GPU. Hence, "in" must support DMA
  // (e.g., string tensors do not work on GPU).  Variant copy DMA
  // checks happen inside CopyTensor::ViaDMA.
  if (!DMAHelper::CanUseDMA(&in) && in.dtype() != DT_VARIANT &&
      in.dtype() != DT_RESOURCE) {
    done(errors::InvalidArgument(
        "Non-DMA-safe ", DataTypeString(in.dtype()),
        " tensor may not be copied from/to a device. Key: ", parsed.FullKey()));
    return;
  }

  WorkerSession* sess = session();
  Device* src_device;
  Status s = sess->device_mgr()->LookupDevice(parsed.src_device, &src_device);
  if (!s.ok()) {
    done(s);
    return;
  }
  Device* dst_device;
  s = sess->device_mgr()->LookupDevice(parsed.dst_device, &dst_device);
  if (!s.ok()) {
    done(s);
    return;
  }

  profiler::ScopedMemoryDebugAnnotation op_annotation(
      "SameWorkerRecvDone", step_id_, "dynamic", in.dtype(),
      [&in]() { return in.shape().DebugString(); });
  AllocatorAttributes attr = recv_args.alloc_attrs;
  attr.set_gpu_compatible(send_args.alloc_attrs.gpu_compatible() ||
                          recv_args.alloc_attrs.gpu_compatible());
  Allocator* out_allocator = dst_device->GetAllocator(attr);
  AllocationAttributes allocation_attr;
  uint64 safe_alloc_frontier = dst_device->SafeAllocFrontier(0);
  bool sync_dst_compute = (safe_alloc_frontier == 0);
  std::function<uint64()> freed_by_func = [dst_device, &safe_alloc_frontier]() {
    safe_alloc_frontier = dst_device->SafeAllocFrontier(safe_alloc_frontier);
    return safe_alloc_frontier;
  };
  if (!sync_dst_compute) {
    allocation_attr.freed_by_func = &freed_by_func;
  }
  if (in.dtype() != DT_VARIANT) {
    // Variants are handled by CopyTensor::ViaDMA.
    Tensor copy(out_allocator, in.dtype(), in.shape(), allocation_attr);
    *out = copy;
  }

  // The following function takes care of cpu->gpu, gpu->cpu, gpu->gpu copies,
  // etc.
  CopyTensor::ViaDMA(
      parsed.edge_name, send_args.device_context, recv_args.device_context,
      src_device, dst_device, send_args.alloc_attrs, recv_args.alloc_attrs, &in,
      out, 0 /*dev_to_dev_stream_index*/, std::move(done), sync_dst_compute);
}

bool BaseRemoteRendezvous::IsSameWorker(DeviceNameUtils::ParsedName src,
                                        DeviceNameUtils::ParsedName dst) {
  return DeviceNameUtils::IsSameAddressSpace(src, dst);
}

void BaseRemoteRendezvous::RecvAsync(const ParsedKey& parsed,
                                     const Rendezvous::Args& recv_args,
                                     DoneCallback done) {
  VLOG(1) << "RemoteRendezvous Recv " << this << " " << parsed.FullKey();
  Status s = ValidateDevices(parsed, false /*!is_src*/);
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), false);
    return;
  }

  // ValidateDevices() returns an error status if the rendezvous is not
  // initialized.
  DCHECK(is_initialized()) << "RecvAsync called when uninitialized (key: "
                           << parsed.FullKey() << ").";

  profiler::ScopedMemoryDebugAnnotation op_annotation("RecvAsync", step_id_);
  // Are src and dst in the same worker?
  if (IsSameWorker(parsed.src, parsed.dst)) {
    // Recv the tensor from local_.
    local_->RecvAsync(
        parsed, recv_args,
        [this, parsed, done](
            const Status& status, const Rendezvous::Args& send_args,
            const Rendezvous::Args& recv_args, const Tensor& in, bool is_dead) {
          VLOG(2) << "RemoteRendezvous Finished Recv " << this << " "
                  << parsed.FullKey();
          Tensor* out = new Tensor;
          StatusCallback final_callback = [done, send_args, recv_args, out,
                                           is_dead](const Status& s) {
            done(s, send_args, recv_args, *out, is_dead);
            delete out;
          };

          if (status.ok()) {
            SameWorkerRecvDone(parsed, send_args, recv_args, in, out,
                               std::move(final_callback));
          } else {
            final_callback(status);
          }
        });
    return;
  } else {
    RecvFromRemoteAsync(parsed, recv_args, std::move(done));
  }
}

void BaseRemoteRendezvous::RecvLocalAsync(const ParsedKey& parsed,
                                          DoneCallback done) {
  // Test whether the rendezvous is initialized using a shared lock, to avoid
  // the need for exclusive access in the common case.
  if (TF_PREDICT_FALSE(!is_initialized())) {
    mutex_lock l(mu_);
    if (!is_initialized_locked()) {
      // RecvLocalAsync can be called (due to an incoming RecvTensor RPC from a
      // remote worker) before the RunStep (or PartialRunStep) RPC from the
      // master arrives. RecvLocalAsync thus buffers the arguments until after
      // the RemoteRendezvous is Initialize()'d, when it completes the
      // rendezvous logic. At some point after Initialize() is called, a Tensor
      // is produced locally that will then be sent in response to the incoming
      // RPC.
      DeferredCall call(parsed, std::move(done));
      deferred_calls_.push_back(call);
      return;
    }
  }
  RecvLocalAsyncInternal(parsed, std::move(done));
}

void BaseRemoteRendezvous::RecvLocalAsyncInternal(const ParsedKey& parsed,
                                                  DoneCallback done) {
  Status s = ValidateDevices(parsed, true /* is_src */);
  if (!s.ok()) {
    done(s, Args(), Args(), Tensor(), false);
    return;
  }
  local_->RecvAsync(parsed, Args(), std::move(done));
}

void BaseRemoteRendezvous::StartAbort(const Status& s) {
  CHECK(!s.ok());
  // If the status passed in is a cancelled or aborted error, mark it as
  // "derived" for the rendezvous. Derived status messages are ignored when
  // aggregating errors across devices: this allows us to prefer our original
  // status message over any cancellation related errors.
  Status derived_status = s;
  if (errors::IsCancelled(s) || errors::IsAborted(s)) {
    derived_status = StatusGroup::MakeDerived(s);
  }

  local_->StartAbort(derived_status);

  bool status_ok = false;
  {
    mutex_lock l(mu_);
    status_ok = status_.ok();
    if (status_ok) {
      status_ = derived_status;
    }
  }

  if (!status_ok) {
    return;
  }

  // Aborts all active RecvTensor calls.
  absl::flat_hash_set<BaseRecvTensorCall*> calls;
  {
    mutex_lock l(calls_mu_);
    for (auto& it : calls_) {
      for (auto& bucket : it.second->buckets) {
        mutex_lock l(bucket.mu);
        calls.merge(bucket.calls);
      }
    }
    calls_.clear();
  }

  for (auto& call : calls) {
    call->StartAbort(derived_status);
  }
}

void BaseRemoteRendezvous::RegisterCall(BaseRecvTensorCall* call,
                                        const Rendezvous::Args& args) {
  CancellationManager* cm = args.cancellation_manager;
  bool already_cancelled = false;
  {
    tf_shared_lock l(mu_);
    if (!status_.ok()) {
      call->StartAbort(status_);
      return;
    }
  }

  CancellationToken token = CancellationManager::kInvalidToken;
  absl::flat_hash_map<CancellationManager*,
                      std::unique_ptr<PendingCalls>>::iterator it;
  bool buckets_found = false;
  {
    tf_shared_lock l(calls_mu_);
    it = calls_.find(cm);
    if (cm != nullptr) {
      already_cancelled = cm->IsCancelled();
    }
    buckets_found = it != calls_.end();
    if (buckets_found && !already_cancelled) {
      // Fast path for Cancellation manager that has been managed by this class.
      it->second->num_calls.fetch_add(1);
      auto& bucket =
          it->second->buckets[absl::Hash<void*>{}(call) % num_shards_];
      mutex_lock l(bucket.mu);
      bool emplaced = bucket.calls.emplace(call).second;
      CHECK(emplaced);  // Crash OK.
      return;
    }
  }
  if (!buckets_found) {
    mutex_lock l(calls_mu_);
    it = calls_.find(cm);
    if (it == calls_.end()) {
      if (cm != nullptr) {
        token = cm->get_cancellation_token();
      }
      it = calls_
               .emplace(cm,
                        std::make_unique<PendingCalls>(token, 1, num_shards_))
               .first;

      if (cm != nullptr) {
        already_cancelled = !cm->RegisterCallback(token, [this, cm]() {
          // Abort all the RecvTensor calls associated with thie cancellation
          // manager.
          absl::flat_hash_map<CancellationManager*,
                              std::unique_ptr<PendingCalls>>::iterator it;
          absl::flat_hash_set<BaseRecvTensorCall*> calls;
          {
            mutex_lock l(calls_mu_);
            it = calls_.find(cm);
            if (it == calls_.end()) {
              return;
            }
            for (auto& bucket : it->second->buckets) {
              {
                mutex_lock l(bucket.mu);
                calls.merge(bucket.calls);
              }
            }
            calls_.erase(cm);
          }
          for (auto& call : calls) {
            call->StartAbort(
                errors::Cancelled("RecvFromRemoteAsync is cancelled."));
          }
        });

        if (already_cancelled) {
          calls_.erase(cm);
        } else {
          auto& bucket =
              it->second->buckets[absl::Hash<void*>{}(call) % num_shards_];
          mutex_lock l(bucket.mu);
          bool emplaced = bucket.calls.emplace(call).second;
          CHECK(emplaced);  // Crash OK.
          return;
        }
      }
    }
  }

  if (already_cancelled) {
    call->StartAbort(errors::Cancelled("RecvFromRemoteAsync is cancelled."));
  }
}

void BaseRemoteRendezvous::DeregisterCall(BaseRecvTensorCall* call,
                                          const Rendezvous::Args& args) {
  auto cm = args.cancellation_manager;
  bool is_last_call = false;
  {
    tf_shared_lock l(calls_mu_);
    auto it = calls_.find(cm);
    if (it == calls_.end()) {
      return;
    }
    auto& bucket = it->second->buckets[absl::Hash<void*>{}(call) % num_shards_];
    {
      mutex_lock l(bucket.mu);
      bucket.calls.erase(call);
    }
    is_last_call = it->second->num_calls.fetch_sub(1) == 1 && cm != nullptr;
  }
  if (is_last_call) {
    mutex_lock l(calls_mu_);
    auto it = calls_.find(cm);
    if (it->second->num_calls == 0) {
      cm->TryDeregisterCallback(it->second->token);
      calls_.erase(it);
    }
  }
}

BaseRemoteRendezvous::DeferredCall::DeferredCall(const ParsedKey& parsed,
                                                 DoneCallback done)
    : parsed(parsed), done(std::move(done)) {}

}  // end namespace tensorflow
