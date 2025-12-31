// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/client/rpc_helper.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/user_context_registry.h"
#include "xla/python/ifrt/user_context_status_util.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/prof_util.h"
#include "xla/python/ifrt_proxy/common/test_utils.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_to_from_proto.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace ifrt {
namespace proxy {

namespace {

constexpr absl::Duration kPeriodicFlushInterval = absl::Microseconds(50);

// Thread-safe data structure for holding batched operations.
class BatchedOps {
 public:
  using BatchOperation = RpcHelper::BatchOperation;

  void Add(BatchOperation op, ArrayHandle handle) {
    absl::MutexLock l(mu_);
    batched_[op].push_back(handle);
  }

  struct IfrtRequests {
    std::unique_ptr<IfrtRequest> delete_req;
    std::unique_ptr<IfrtRequest> destruct_req;
  };

  IfrtRequests Consume() {
    IfrtRequests result;
    absl::MutexLock l(mu_);
    if (!batched_[BatchOperation::kDeleteArray].empty()) {
      result.delete_req = std::make_unique<IfrtRequest>();
      for (const auto& arr_handle : batched_[BatchOperation::kDeleteArray]) {
        result.delete_req->mutable_delete_array_request()->add_array_handle(
            arr_handle.handle);
      }
      batched_[BatchOperation::kDeleteArray].clear();
    }
    if (!batched_[BatchOperation::kDestructArray].empty()) {
      result.destruct_req = std::make_unique<IfrtRequest>();
      for (const auto& arr_handle : batched_[BatchOperation::kDestructArray]) {
        result.destruct_req->mutable_destruct_array_request()->add_array_handle(
            arr_handle.handle);
      }
      batched_[BatchOperation::kDestructArray].clear();
    }
    return result;
  }

 private:
  absl::Mutex mu_;
  std::array<std::vector<ArrayHandle>, BatchOperation::kSentinelDoNotUse>
      batched_ ABSL_GUARDED_BY(mu_);
};

// Tracks user contexts referenced by the proxy server.
class UserContextsReferencedByProxyServer {
 public:
  // Registers a user context referenced by the proxy server. If a user context
  // with the same ID is already registered, increments its ref count.
  void RegisterUserContext(TrackedUserContextRef tracked_user_context) {
    if (tracked_user_context == nullptr) {
      return;
    }
    absl::MutexLock l(mu_);
    UserContextId user_context_id = tracked_user_context->user_context()->Id();
    auto [it, inserted] = user_contexts_.insert(
        {user_context_id,
         TrackedUserContextRefWithRefCount{std::move(tracked_user_context)}});
    ++it->second.ref_count;
  }

  // Schedules to decrement the ref count of user contexts referenced by the
  // proxy server once all responses before/on `seq_num` have been processed.
  // Once the ref count reaches 0, the user context is unregistered.
  void ScheduleToUnregisterUserContexts(
      int64_t seq_num, std::vector<UserContextId> user_context_ids) {
    absl::MutexLock l(mu_);
    pending_unregistration_.insert({seq_num, std::move(user_context_ids)});
  }

  // Informs that a response with the given `seq_num` has been processed. User
  // context unregistration will happen for all responses with a range
  // [`seq_num`, ...,  N] that is contiguous and stop at any gap indicating a
  // response that has not been processed yet.
  void DoneProcessingResponse(int64_t seq_num) {
    // Collect user contexts to erase outside of the lock when we return from
    // this method.
    std::vector<xla::ifrt::TrackedUserContextRef> user_contexts_to_erase;

    absl::MutexLock l(mu_);
    pending_seq_nums_.insert(seq_num);
    while (!pending_seq_nums_.empty() &&
           *pending_seq_nums_.begin() == next_seq_num_to_process_) {
      auto node = pending_unregistration_.extract(next_seq_num_to_process_);
      for (UserContextId user_context_id : node.mapped()) {
        auto it = user_contexts_.find(UserContextId(user_context_id));
        CHECK(it != user_contexts_.end());
        if (--it->second.ref_count == 0) {
          user_contexts_to_erase.push_back(
              std::move(it->second.tracked_user_context));
          user_contexts_.erase(it);
        }
      }
      pending_seq_nums_.erase(pending_seq_nums_.begin());
      ++next_seq_num_to_process_;
    }
  }

 private:
  struct TrackedUserContextRefWithRefCount {
    TrackedUserContextRef tracked_user_context;
    // Multiple instances of UserContext may be created on the proxy server
    // for the same `UserContextId`. To ensure that we keep the user context
    // alive as long as any of its instances are alive on the proxy server,
    // we keep track of the number of references to each `UserContextId` on
    // the proxy server.
    int ref_count = 0;
  };

  absl::Mutex mu_;
  absl::flat_hash_map<UserContextId, TrackedUserContextRefWithRefCount>
      user_contexts_ ABSL_GUARDED_BY(mu_);
  int64_t next_seq_num_to_process_ ABSL_GUARDED_BY(mu_) = 0;
  absl::btree_set<int64_t> pending_seq_nums_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<int64_t, std::vector<UserContextId>>
      pending_unregistration_ ABSL_GUARDED_BY(mu_);
};

}  // namespace

// Batches any requested operations and flushes them periodically in the
// background, and allows sending other requested operations immediately.
// Immediate operations are guaranteed to be sent after all previously enqueued
// batched operations.
class RpcHelper::Batcher {
 public:
  explicit Batcher(std::shared_ptr<ClientSession> session)
      : session_(std::move(session)),
        user_contexts_referenced_by_proxy_server_(
            std::make_shared<UserContextsReferencedByProxyServer>()) {
    thread_pool_.emplace(tsl::Env::Default(), "IfrtProxyRpcHelperBatcher",
                         /*num_threads=*/1);
    thread_pool_->Schedule(absl::bind_front(&Batcher::PeriodicFlusher, this));
  }

  // Sends the given request immediately after sending any batched operations
  // that have been previously enqueued.
  tsl::Future<ClientSession::Response> Immediate(
      std::unique_ptr<IfrtRequest> request) {
    absl::MutexLock l(mu_);
    if (finished_) {
      LOG(WARNING) << "After RpcHelper::Finish(): " << request->DebugString();
      return tsl::Future<ClientSession::Response>(
          absl::FailedPreconditionError("RpcHelper::Finish() already called."));
    }
    Flush();
    return session_->Enqueue(std::move(request));
  }

  // Enqueues an operation to be sent later. Guaranteed to not be blocked by the
  // underlying transport.
  void Batch(BatchOperation op, ArrayHandle handle) {
    batched_.Add(op, handle);
  }

  // Asks the underlying transport to terminate.
  void Finish(absl::Status s) {
    LOG(INFO) << "RpcHelper::Batcher::Finish() starting: " << s;
    {
      absl::MutexLock l(mu_);
      finished_ = true;
      auto remaining = batched_.Consume();
      if (remaining.delete_req != nullptr) {
        LOG(WARNING) << "RpcHelper::Batch: Finish() called while there are "
                        "still batched delete operations";
      }
      if (remaining.destruct_req != nullptr) {
        LOG(WARNING) << "RpcHelper::Batch: Finish() called while there are "
                        "still batched destruct operations";
      }
    }
    LOG(INFO) << "RpcHelper::Batcher::Finish(): resetting thread_pool_.";
    thread_pool_.reset();
    LOG(INFO) << "RpcHelper::Batcher::Finish(): calling session_->Finish().";
    session_->Finish(s);
    LOG(INFO) << "RpcHelper::Batcher::Finish(): done.";
  }

  std::shared_ptr<UserContextsReferencedByProxyServer>
  user_contexts_referenced_by_proxy_server() {
    return user_contexts_referenced_by_proxy_server_;
  }

 private:
  void PeriodicFlusher() {
    while (true) {
      absl::SleepFor(kPeriodicFlushInterval);
      absl::MutexLock l(mu_);
      if (finished_) {
        return;
      }
      {
        bool periodic_flush_paused = false;
        TestHookCall(TestHookName::kRpcBatcherPausePeriodicFlush,
                     &periodic_flush_paused);
        if (periodic_flush_paused) {
          continue;
        }
      }
      tsl::profiler::TraceMe traceme("proxy_periodic_flush");
      Flush();
    }
  }

  // Sends all enqueued batched operations.
  void Flush() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto reqs = batched_.Consume();
    if (reqs.delete_req != nullptr) {
      XFlowHelper x_flow_helper("batch_delete");
      auto traceme = x_flow_helper.Span<XFlowHelper::kSend>();
      VLOG(3) << "Sending req: " << reqs.delete_req->ShortDebugString();
      session_->Enqueue(std::move(reqs.delete_req))
          .OnReady(
              absl::bind_front(HandleBatchResponse, session_, x_flow_helper));
    }
    if (reqs.destruct_req != nullptr) {
      XFlowHelper x_flow_helper("batch_destruct");
      auto traceme = x_flow_helper.Span<XFlowHelper::kSend>();
      VLOG(3) << "Sending req: " << reqs.destruct_req->ShortDebugString();
      session_->Enqueue(std::move(reqs.destruct_req))
          .OnReady(
              absl::bind_front(HandleBatchResponse, session_, x_flow_helper));
    }
  }

  // Handles a response from the server of a previous batched operation;
  // bad responses are logged but otherwise ignored. The method is static since
  // it can be called in the background after RpcHelper::Batcher is destroyed.
  static void HandleBatchResponse(
      std::shared_ptr<ClientSession> session, XFlowHelper x_flow_helper,
      absl::StatusOr<std::shared_ptr<IfrtResponse>> r) {
    if (!r.ok()) {
      x_flow_helper.InstantActivity<XFlowHelper::kRecv>();
      LOG(WARNING) << "Batched response from ifrt proxy server: " << r.status();
      return;
    }
    VLOG(3) << "Got response: " << r.value()->ShortDebugString();
    if (r.value()->has_delete_array_response()) {
      auto traceme = x_flow_helper.Span<XFlowHelper::kRecvSend>();
      auto ifrt_req = std::make_unique<IfrtRequest>();
      ifrt_req->mutable_check_future_request()->set_future_handle(
          r.value()->delete_array_response().deletion_future_handle());
      VLOG(3) << "Sending req: " << ifrt_req->ShortDebugString();
      session->Enqueue(std::move(ifrt_req))
          .OnReady(
              absl::bind_front(HandleBatchResponse, session, x_flow_helper));
    } else if (r.value()->has_destruct_array_response() ||
               r.value()->has_check_future_response()) {
      x_flow_helper.InstantActivity<XFlowHelper::kRecv>();
    } else {
      LOG(ERROR) << "Unrecognized response from server for batched request: "
                 << (*r)->DebugString();
    }
  }

  const std::shared_ptr<ClientSession> session_;
  // Tracks user contexts referenced by the proxy server. Uses a shared pointer
  // because the reference to `UserContextsReferencedByProxyServer` is captured
  // in OnReady callbacks and may outlive the `RpcHelper::Batcher` in a certain
  // situation (e.g., when the proxy client is destroyed).
  const std::shared_ptr<UserContextsReferencedByProxyServer>
      user_contexts_referenced_by_proxy_server_;

  BatchedOps batched_;

  absl::Mutex mu_;
  bool finished_ ABSL_GUARDED_BY(mu_) = false;
  std::optional<tsl::thread::ThreadPool> thread_pool_;
};

// DoRpc is a templated function that implements the logic of all RPC-wrapping
// functions of `RpcHelper`, such as `RpcHelper::MakeArrayFromHostBuffer()`.
//
// `profiling_name` needs to be a string literal.
template <typename Req, typename Resp>
tsl::Future<std::shared_ptr<Resp>> DoRpc(RpcHelper::Batcher* batcher,
                                         void (IfrtRequest::*set_req)(Req*),
                                         Resp* (IfrtResponse::*get_resp)(),
                                         bool (IfrtResponse::*has_resp)() const,
                                         std::unique_ptr<Req> req,
                                         absl::string_view profiling_name) {
  auto ifrt_req = std::make_unique<IfrtRequest>();
  (ifrt_req.get()->*set_req)(req.release());

  XFlowHelper x_flow_helper(profiling_name);
  auto traceme = x_flow_helper.Span<XFlowHelper::kSend>();

  std::shared_ptr<UserContextsReferencedByProxyServer>
      user_contexts_referenced_by_proxy_server =
          batcher->user_contexts_referenced_by_proxy_server();

  const UserContextRef& user_context = UserContextScope::current();
  if (user_context != nullptr && user_context->Id() != UserContextId(0)) {
    ifrt_req->mutable_request_metadata()->set_user_context_id(
        user_context->Id().value());
    TrackedUserContextRef tracked_user_context =
        UserContextRegistry::Get().Register(user_context);
    user_contexts_referenced_by_proxy_server->RegisterUserContext(
        std::move(tracked_user_context));
  }

  auto [promise, future] = tsl::MakePromise<std::shared_ptr<Resp>>();
  auto on_ready = [promise = std::move(promise),
                   user_contexts_referenced_by_proxy_server =
                       std::move(user_contexts_referenced_by_proxy_server),
                   has_resp, get_resp, profiling_name, x_flow_helper](
                      absl::StatusOr<std::shared_ptr<IfrtResponse>> r) mutable {
    if (!r.ok()) {
      VLOG(3) << profiling_name << " response: " << r.status();
      LOG_EVERY_N_SEC(ERROR, 10)
          << "Connection to IFRT proxy server was terminated: " << r.status();
      promise.Set(absl::UnavailableError(
          absl::StrCat("Connection to IFRT proxy server was terminated: ",
                       r.status().ToString())));
      return;
    }
    VLOG(3) << "Got response: " << r.value()->ShortDebugString();
    auto result = [&](std::shared_ptr<IfrtResponse> r)
        -> absl::StatusOr<std::shared_ptr<Resp>> {
      auto traceme = x_flow_helper.Span<XFlowHelper::kRecv>();

      if (!r->has_response_metadata()) {
        return absl::InternalError(absl::StrCat(
            "IFRT server sent a message without metadata: ", r->DebugString()));
      }

      const absl::Status metadata_status = xla::ifrt::ReattachUserContextRefs(
          tsl::StatusFromProto(r->response_metadata().status()));
      const bool has_expected_response = (r.get()->*has_resp)();
      const auto has_some_response =
          r->response_case() != IfrtResponse::RESPONSE_NOT_SET;

      if (metadata_status.ok() && !has_some_response) {
        return absl::InternalError(absl::StrCat(
            "OK response with no actual response set: ", r->DebugString()));
      }

      if (!has_expected_response && has_some_response) {
        return absl::InternalError(absl::StrCat(
            "Response with wrong type (expected ",
            Resp::GetDescriptor()->name(), "): ", r->DebugString()));
      }

      // If the metadata_status is not-OK, according to ifrt_service.proto,
      // there may be an error _instead_ of an actual response value. So, check
      // if an actual response value exists, and if so return it irrespective of
      // what the metadata_status says.
      absl::StatusOr<std::shared_ptr<Resp>> result;
      if (!has_some_response) {
        result = std::move(metadata_status);
      } else {
        const int64_t seq_num = r->response_metadata().seq_num();

        std::vector<UserContextId> user_context_ids;
        user_context_ids.reserve(
            r->response_metadata().destroyed_user_context_ids_size());
        for (uint64_t user_context_id :
             r->response_metadata().destroyed_user_context_ids()) {
          user_context_ids.push_back(UserContextId(user_context_id));
        }
        user_contexts_referenced_by_proxy_server
            ->ScheduleToUnregisterUserContexts(seq_num,
                                               std::move(user_context_ids));

        result = std::shared_ptr<Resp>(
            new Resp(*std::move((r.get()->*get_resp)())),
            [user_contexts_referenced_by_proxy_server, seq_num](Resp* resp) {
              delete resp;
              user_contexts_referenced_by_proxy_server->DoneProcessingResponse(
                  seq_num);
            });
      }
      return result;
    }(*std::move(r));

    if (!result.ok()) {
      LOG(WARNING) << profiling_name << ": " << result.status();
    }
    promise.Set(std::move(result));
  };
  VLOG(3) << ifrt_req->ShortDebugString();
  batcher->Immediate(std::move(ifrt_req)).OnReady(std::move(on_ready));

  return std::move(future);
}

#define RPC(METHOD, PROPERTY)                                                 \
  RpcHelper::ResponseFuture<METHOD##Response> RpcHelper::METHOD(              \
      std::unique_ptr<METHOD##Request> req) {                                 \
    return DoRpc(                                                             \
        batcher_.get(), &IfrtRequest::set_allocated_##PROPERTY##_request,     \
        &IfrtResponse::mutable_##PROPERTY##_response,                         \
        &IfrtResponse::has_##PROPERTY##_response, std::move(req), #PROPERTY); \
  }

RPC(Init, init);
RPC(GetDefaultDeviceAssignment, get_default_device_assignment);
RPC(CheckFuture, check_future);
RPC(CheckValueReady, check_value_ready);
RPC(MakeArrayFromHostBuffer, make_array_from_host_buffer);
RPC(MakeArraysFromHostBufferShards, make_arrays_from_host_buffer_shards);
RPC(MakeErrorArrays, make_error_arrays);
RPC(AssembleArrayFromSingleDeviceArrays,
    assemble_array_from_single_device_arrays);
RPC(RemapArrays, remap_arrays);
RPC(ReshardArrays, reshard_arrays);
RPC(DisassembleIntoSingleDeviceArrays, disassemble_into_single_device_arrays);
RPC(CopyToHostBuffer, copy_to_host_buffer);
RPC(IsArrayDeleted, is_array_deleted);
RPC(DestructArray, destruct_array)
RPC(CopyArrays, copy_arrays);
RPC(FullyReplicatedShard, fully_replicated_shard);
RPC(DeleteArray, delete_array);
RPC(Compile, compile);
RPC(LoadedExecutableMetadata, loaded_executable_metadata);
RPC(LoadedExecutableMpmdMetadata, loaded_executable_mpmd_metadata);
RPC(LoadedExecutableCostAnalysis, loaded_executable_cost_analysis);
RPC(LoadedExecutableMpmdCostAnalysis, loaded_executable_mpmd_cost_analysis);
RPC(LoadedExecutableHumanReadableProgramText,
    loaded_executable_human_readable_program_text);
RPC(LoadedExecutableExecute, loaded_executable_execute);
RPC(LoadedExecutableFetchExecuteResult, loaded_executable_fetch_execute_result);
RPC(LoadedExecutableDelete, loaded_executable_delete);
RPC(LoadedExecutableIsDeleted, loaded_executable_is_deleted);
RPC(LoadedExecutableDestruct, loaded_executable_destruct);
RPC(LoadedHostCallbackPoll, loaded_host_callback_poll);
RPC(LoadedHostCallbackReturn, loaded_host_callback_return);
RPC(GetDefaultLayout, get_default_layout);

tsl::Future<> RpcHelper::CheckFuture(uint64_t handle) {
  auto req = std::make_unique<CheckFutureRequest>();
  req->set_future_handle(handle);

  auto [promise, future] = tsl::MakePromise<>();
  CheckFuture(std::move(req))
      .OnReady([promise = std::move(promise)](
                   absl::StatusOr<std::shared_ptr<CheckFutureResponse>>
                       response) mutable { promise.Set(response.status()); });

  return std::move(future);
}

RpcHelper::RpcHelper(IfrtProxyVersion version,
                     std::shared_ptr<ClientSession> session)
    : batcher_(std::make_unique<Batcher>(std::move(session))),
      version_(std::move(version)) {}

RpcHelper::~RpcHelper() {
  LOG(INFO) << "RpcHelper::~RpcHelper() starting.";
  Disconnect();
  LOG(INFO) << "RpcHelper::~RpcHelper() done.";
}

void RpcHelper::Batch(BatchOperation op, ArrayHandle handle) {
  return batcher_->Batch(op, handle);
}

void RpcHelper::Disconnect() {
  batcher_->Finish(absl::CancelledError(
      "Disconnected by client [via RpcHelper::Disconnect()]"));
}

uint64_t RpcHelper::NextHandle() {
  uint64_t result = next_handle_.fetch_add(1, std::memory_order_relaxed);
  CHECK_LT(result, kServerGeneratedHandlesMinValue);
  return result;
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
