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

#include <functional>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"

namespace tensorflow {

namespace {

size_t HashCall(BaseRecvTensorCall* call) {
  // Salt hash with "42" to avoid using the same hash function for the shard key
  // and the hashtable contained within the the shard itself.
  return absl::HashOf(call, 42);
}

}  // namespace

BaseRendezvousMgr::BaseRendezvousMgr(const WorkerEnv* worker_env)
    : cache_(new RendezvousCache<BaseRemoteRendezvous>()),
      worker_env_(worker_env) {}

BaseRendezvousMgr::~BaseRendezvousMgr() = default;

tsl::core::RefCountPtr<RemoteRendezvous> BaseRendezvousMgr::Find(
    int64_t step_id) {
  return FindOrCreate(step_id);
}

tsl::core::RefCountPtr<BaseRemoteRendezvous> BaseRendezvousMgr::FindOrCreate(
    int64_t step_id) {
  return cache_->FindOrCreate(
      step_id, [this, step_id]() { return Create(step_id, worker_env_); });
}

void BaseRendezvousMgr::RecvLocalAsync(int64_t step_id,
                                       const Rendezvous::ParsedKey& parsed,
                                       Rendezvous::DoneCallback done) {
  FindOrCreate(step_id)->RecvLocalAsync(parsed, std::move(done));
}

absl::Status BaseRendezvousMgr::RecvLocal(int64_t step_id,
                                          const Rendezvous::ParsedKey& parsed,
                                          Tensor* val, bool* is_dead) {
  absl::Status ret;
  absl::Notification n;
  RecvLocalAsync(step_id, parsed,
                 [val, is_dead, &ret, &n](const absl::Status& s,
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

BaseRemoteRendezvous::BaseRemoteRendezvous(const WorkerEnv* env,
                                           int64_t step_id)
    : env_(env),
      step_id_(step_id),
      num_shards_(env_->experimental_num_shards),
      local_(this, num_shards_),
      session_(nullptr) {
  DCHECK_GT(env_->experimental_num_shards, 0);
}

BaseRemoteRendezvous::~BaseRemoteRendezvous() {
  VLOG(5) << "BaseRemoteRendezvous::~BaseRemoteRendezvous() " << this;
  {
    mutex_lock l(calls_mu_);
    calls_.clear();
  }
}

// Returns true if "device_name" is a valid full name of local device
// of the "worker". This helper is purely based on the worker name
// and device name and does no lookups in the worker->device_mgr.
static bool IsLocalDevice(const absl::string_view worker_name,
                          const absl::string_view device_name) {
  return absl::StartsWith(device_name, worker_name);
}

// Returns true if the parsed device name is empty. An empty src device
// is used to represent a Recv from the local host device when
// the host device name is not known at the time when the graph node is
// emitted.
static bool IsImplicitLocalDevice(
    const DeviceNameUtils::ParsedName parsed_device_name) {
  return !DeviceNameUtils::HasSomeDetails(parsed_device_name);
}

absl::Status BaseRemoteRendezvous::Initialize(WorkerSession* session) {
  CHECK_NE(session, nullptr) << "session must not be null!";
  std::vector<DeferredCall> deferred_calls;
  {
    mutex_lock l(mu_);
    if (session_ != nullptr) {
      if (session_->worker_name() == session->worker_name()) {
        VLOG(1) << "Skipping rendezvous re-initialization.";
        return absl::OkStatus();
      }
      absl::Status s = errors::Internal(
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
  return absl::OkStatus();
}

WorkerSession* BaseRemoteRendezvous::session() {
  tf_shared_lock l(mu_);
  return session_;
}

bool BaseRemoteRendezvous::is_initialized() {
  tf_shared_lock l(mu_);
  return is_initialized_locked();
}

absl::Status BaseRemoteRendezvous::Send(const Rendezvous::ParsedKey& parsed,
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

  if (!IsImplicitLocalDevice(parsed.src) &&
      !IsLocalDevice(sess->worker_name(), parsed.src_device)) {
    return errors::InvalidArgument(
        "Invalid rendezvous key (src): ", parsed.FullKey(), " @ ",
        sess->worker_name());
  }

  // Buffers "val" and "device_context" in local_.
  return local_.Send(parsed, args, val, is_dead);
}

absl::Status BaseRemoteRendezvous::ValidateDevices(const ParsedKey& parsed,
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
  if (is_src && !IsImplicitLocalDevice(parsed.src) &&
      !IsLocalDevice(sess->worker_name(), parsed.src_device)) {
    return errors::InvalidArgument(
        "Invalid rendezvous key (src): ", parsed.FullKey(), " @ ",
        sess->worker_name());
  }
  if (!is_src && !IsLocalDevice(sess->worker_name(), parsed.dst_device)) {
    return errors::InvalidArgument(
        "Invalid rendezvous key (dst): ", parsed.FullKey(), " @ ",
        sess->worker_name());
  }
  return absl::OkStatus();
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
    done(absl::OkStatus());
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
  absl::Status s =
      sess->device_mgr()->LookupDevice(parsed.src_device, &src_device);
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

  tsl::profiler::ScopedMemoryDebugAnnotation op_annotation(
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
  absl::Status s = ValidateDevices(parsed, false /*!is_src*/);
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), false);
    return;
  }

  // ValidateDevices() returns an error status if the rendezvous is not
  // initialized.
  DCHECK(is_initialized()) << "RecvAsync called when uninitialized (key: "
                           << parsed.FullKey() << ").";

  tsl::profiler::ScopedMemoryDebugAnnotation op_annotation("RecvAsync",
                                                           step_id_);
  // Are src and dst in the same worker?
  // At this point parsed.dst must be a local device asserted by the previous
  // call to ValidateDevices.
  if (IsImplicitLocalDevice(parsed.src) ||
      IsSameWorker(parsed.src, parsed.dst)) {
    // Recv the tensor from local_.
    local_.RecvAsync(
        parsed, recv_args,
        [this, parsed, done](
            const absl::Status& status, const Rendezvous::Args& send_args,
            const Rendezvous::Args& recv_args, const Tensor& in, bool is_dead) {
          VLOG(2) << "RemoteRendezvous Finished Local Recv " << this << " "
                  << parsed.FullKey();
          Tensor* out = new Tensor;
          StatusCallback final_callback = [done, send_args, recv_args, out,
                                           is_dead](const absl::Status& s) {
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
    // Keep current rendezvous alive while the recv is inflight.
    this->Ref();
    RecvFromRemoteAsync(parsed, recv_args,
                        [this, parsed, done](const absl::Status& status,
                                             const Rendezvous::Args& send_args,
                                             const Rendezvous::Args& recv_args,
                                             const Tensor& in, bool is_dead) {
                          VLOG(2) << "RemoteRendezvous Finished Remote Recv "
                                  << this << " " << parsed.FullKey();
                          done(status, send_args, recv_args, in, is_dead);
                          this->Unref();
                        });
  }
}

void BaseRemoteRendezvous::RecvLocalAsync(const ParsedKey& parsed,
                                          DoneCallback done) {
  VLOG(2) << "RemoteRendezvous RecvLocal " << this << " " << parsed.FullKey();
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
      deferred_calls_.emplace_back(parsed, std::move(done), GetNewRef(this));
      return;
    }
  }
  RecvLocalAsyncInternal(parsed, std::move(done));
}

void BaseRemoteRendezvous::RecvLocalAsyncInternal(const ParsedKey& parsed,
                                                  DoneCallback done) {
  absl::Status s = ValidateDevices(parsed, true /* is_src */);
  if (!s.ok()) {
    done(s, Args(), Args(), Tensor(), false);
    return;
  }
  local_.RecvAsync(parsed, Args(), std::move(done));
}

void BaseRemoteRendezvous::StartAbort(const absl::Status& s) {
  CHECK(!s.ok());
  // If the status passed in is a cancelled or aborted error, mark it as
  // "derived" for the rendezvous. Derived status messages are ignored when
  // aggregating errors across devices: this allows us to prefer our original
  // status message over any cancellation related errors.
  absl::Status derived_status = s;
  if (absl::IsCancelled(s) || absl::IsAborted(s)) {
    derived_status = StatusGroup::MakeDerived(s);
  }

  local_.StartAbort(derived_status);

  bool status_ok = false;
  {
    mutex_lock l(mu_);
    status_ok = status_.ok();
    if (status_ok) {
      status_ = derived_status;
    }
  }

  if (status_ok) {
    // Aborts all active RecvTensor calls.
    mutex_lock l(calls_mu_);
    // Invoking callbacks while holding the lock, to block concurrent
    // DeregisterCall(). Once DeregisterCall() returned, the caller may release
    // resources needed by the callback.
    for (auto& it : calls_) {
      for (auto& bucket : it.second->buckets) {
        mutex_lock l(bucket.mu);
        for (auto& call : bucket.calls) {
          call->StartAbort(derived_status);
        }
        bucket.calls.clear();
      }
    }
    calls_.clear();
  }
}

void BaseRemoteRendezvous::CancelledByManager(CancellationManager* cm) {
  // Abort all the RecvTensor calls associated with thie cancellation
  // manager.
  mutex_lock l(calls_mu_);
  auto it = calls_.find(cm);
  if (it != calls_.end()) {
    // Invoking callbacks while holding the lock, to block concurrent
    // DeregisterCall(). Once DeregisterCall() returned, the caller may release
    // resources needed by the callback.
    for (auto& bucket : it->second->buckets) {
      mutex_lock l(bucket.mu);
      for (auto& call : bucket.calls) {
        call->StartAbort(
            errors::Cancelled("RecvFromRemoteAsync is cancelled."));
      }
    }
    calls_.erase(it);
  }
}

void BaseRemoteRendezvous::RegisterCall(BaseRecvTensorCall* call,
                                        const Rendezvous::Args& args) {
  CancellationManager* cm = args.cancellation_manager;
  {
    tf_shared_lock l(mu_);
    if (!status_.ok()) {
      call->StartAbort(status_);
      return;
    }
  }

  int hash = HashCall(call) % num_shards_;
  bool buckets_found = false;
  bool already_cancelled = false;
  {
    tf_shared_lock l(calls_mu_);
    if (cm != nullptr) {
      already_cancelled = cm->IsCancelled();
    }
    auto it = calls_.find(cm);
    buckets_found = it != calls_.end();
    if (buckets_found && !already_cancelled) {
      // Fast path for Cancellation manager that has been managed by this class.
      it->second->num_calls.fetch_add(1);
      auto& bucket = it->second->buckets[hash];
      mutex_lock l(bucket.mu);
      bool emplaced = bucket.calls.emplace(call).second;
      CHECK(emplaced);  // Crash OK.
      return;
    }
  }
  if (!buckets_found && !already_cancelled) {
    mutex_lock l(calls_mu_);
    auto it = calls_.find(cm);
    if (it == calls_.end()) {
      CancellationToken token = CancellationManager::kInvalidToken;
      if (cm != nullptr) {
        token = cm->get_cancellation_token();
        already_cancelled = !cm->RegisterCallback(
            token,
            std::bind(&BaseRemoteRendezvous::CancelledByManager, this, cm));
      }
      if (!already_cancelled) {
        it = calls_
                 .emplace(cm, std::make_unique<PendingCalls>(
                                  token, 0, num_shards_, GetNewRef(this)))
                 .first;
      }
    }
    DCHECK(it != calls_.end());
    if (!already_cancelled) {
      it->second->num_calls.fetch_add(1);
      auto& bucket = it->second->buckets[hash];
      mutex_lock bucket_lock(bucket.mu);
      bool emplaced = bucket.calls.emplace(call).second;
      CHECK(emplaced);  // Crash OK.
    }
  }

  if (already_cancelled) {
    call->StartAbort(errors::Cancelled("RecvFromRemoteAsync is cancelled."));
  }
}

void BaseRemoteRendezvous::DeregisterCall(BaseRecvTensorCall* call,
                                          const Rendezvous::Args& args) {
  int hash = HashCall(call) % num_shards_;
  auto cm = args.cancellation_manager;
  bool is_last_call = false;
  {
    tf_shared_lock l(calls_mu_);
    auto it = calls_.find(cm);
    if (it != calls_.end()) {
      auto& bucket = it->second->buckets[hash];
      bool removed = false;
      {
        mutex_lock l(bucket.mu);
        removed = bucket.calls.erase(call);
      }
      if (removed) {
        is_last_call = it->second->num_calls.fetch_sub(1) == 1;
      }
    }
  }
  if (is_last_call) {
    // Holds an exclusive lock so that all updating to the buckets are done, and
    // num_calls is accurate.
    mutex_lock l(calls_mu_);
    auto it = calls_.find(cm);
    if (it != calls_.end() && it->second->num_calls == 0) {
      if (cm != nullptr) {
        cm->TryDeregisterCallback(it->second->token);
      }
      calls_.erase(it);
    }
  }
}

BaseRemoteRendezvous::DeferredCall::DeferredCall(
    const ParsedKey& parsed, DoneCallback done,
    tsl::core::RefCountPtr<Rendezvous> rendez)
    : parsed(parsed), done(std::move(done)), rendezvous(std::move(rendez)) {}

}  // end namespace tensorflow
