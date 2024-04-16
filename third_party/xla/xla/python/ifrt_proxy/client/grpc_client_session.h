/*
 * Copyright 2023 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_GRPC_CLIENT_SESSION_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_GRPC_CLIENT_SESSION_H_

#include <functional>
#include <memory>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/client_callback.h"
#include "grpcpp/support/sync_stream.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.grpc.pb.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "tsl/platform/threadpool.h"
#include "tsl/platform/unbounded_work_queue.h"

namespace xla {
namespace ifrt {
namespace proxy {

// `GrpcClientSession` implements the client side of an `IfrtSession`
// stream(ing RPC) and allows users to enqueue `IfrtRequest`s on the
// stream and register callbacks for when `IfrtResponse`s are received.
class GrpcClientSession : public ClientSession {
 public:
  // `StreamTerminatedCallback` represents a function that will be called when
  // the underlying streaming RPC is terminated permanently. The callback may be
  // invoked by the "primary" thread and with various mutex locks held, so the
  // callback should both return soon and not block on any events (deadlocks may
  // happen otherwise).
  using StreamTerminatedCallback = std::function<void(absl::Status)>;

  // Returns an instantiation of GrpcClientSession on the given `stub`.
  // `stream_terminated_cb` is guaranteed to be called exactly once (unless the
  // process terminates beforehand). It is guaranteed that no registered
  // `ResponseCallback` (see below) will be called after `stream_terminated_cb`.
  static std::shared_ptr<GrpcClientSession> Create(
      std::shared_ptr<grpc::GrpcIfrtService::StubInterface> stub,
      GrpcIfrtSessionMetadata metadata,
      StreamTerminatedCallback stream_terminated_cb);

  Future<Response> Enqueue(std::unique_ptr<IfrtRequest> request) override;

  // `ResponseCallback` represents a function that can be invoked when
  // `ClientSession` receives an `IfrtResponse`. May be invoked by the "primary"
  // thread and with various mutex locks held.
  using ResponseCallback = std::function<void(Response)>;

  absl::Status Enqueue(std::unique_ptr<IfrtRequest> req,
                       ResponseCallback callback);

  // Terminates the `GrpcClientSession` if it has not already been terminated.
  // Waits until `stream_terminated_cb` returns.
  void Finish(const absl::Status& client_status) override;

  // Not copyable (or moveable)
  GrpcClientSession(const GrpcClientSession&) = delete;
  GrpcClientSession& operator=(const GrpcClientSession&) = delete;

  // Calls `Finish()`. Also waits for the destruction of
  // `user_futures_work_queue_` (see below) and thus can block on user-level
  // callbacks.
  ~GrpcClientSession() override;

 private:
  class ResponseCallbackTable;

  GrpcClientSession(std::shared_ptr<grpc::GrpcIfrtService::StubInterface> stub,
                    std::unique_ptr<::grpc::ClientContext> context,
                    StreamTerminatedCallback stream_terminated_cb);

  // Repeatedly waits for a `IfrtResponse` message to arrive; for each message,
  // looks up the corresponding callback registered in `response_callbacks_` and
  // invokes it inline.
  void ReadLoop();

  // Thread-safe table that logically maps from RequestMetadata.OpId to
  // ResponseCallback.
  const std::unique_ptr<ResponseCallbackTable> response_callbacks_;

  // Thread that invokes `ReadLoop()`.
  std::unique_ptr<tsl::thread::ThreadPool> reader_thread_;

  // A notification (waited on by `Finish()`) for when `ReadLoop()` exits.
  absl::Notification reader_thread_stopped_;

  // Set by `Finish()`, respected by `Enqueue()` calls.
  bool writes_stopped_ ABSL_GUARDED_BY(writer_mu_) = false;

  // A mutex that ensures serialization between various `Enqueue()` calls, since
  // only one thread is allowed to write to the gRPC stream at a time.
  absl::Mutex writer_mu_;

  // Ensures logic inside `Finish()` is internally called only once.
  absl::once_flag finish_once_;

  // References to gRPC objects used to read and write to the stream.
  const std::shared_ptr<grpc::GrpcIfrtService::StubInterface> stub_;
  const std::unique_ptr<::grpc::ClientContext> context_;
  const std::unique_ptr<
      ::grpc::ClientReaderWriterInterface<IfrtRequest, IfrtResponse>>
      stream_;

  const StreamTerminatedCallback stream_terminated_cb_;

  // Threadpool used to perform `Future<>::Promise::Set()` for Futures returned
  // to callers of `Enqueue(std::unique_ptr<IfrtRequest> request)`. We do this
  // because `Set()` may block on arbitrary `OnReady` callbacks set by those
  // callers.
  std::unique_ptr<tsl::UnboundedWorkQueue> user_futures_work_queue_;
};

// Creates a gRPC stub that connects to `server_address`. It can be used for
// `GrpcClientSession`. The same stub can be reused across multiple sessions.
std::shared_ptr<grpc::GrpcIfrtService::StubInterface> CreateGrpcStub(
    absl::string_view server_address);

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_GRPC_CLIENT_SESSION_H_
