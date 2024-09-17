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

#include "xla/python/ifrt_proxy/client/grpc_client_session.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "grpc/grpc.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "xla/pjrt/distributed/util.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/common/grpc_credentials.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.grpc.pb.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/threadpool.h"
#include "tsl/platform/unbounded_work_queue.h"

namespace xla {
namespace ifrt {
namespace proxy {

using OpId = int64_t;

// Logically equivalent to a map<OpId, ResponseCallback>, but thread-safe and
// with various convenience functions.
class GrpcClientSession::ResponseCallbackTable {
 public:
  absl::Status Add(OpId op_id, ResponseCallback callback) {
    absl::MutexLock l(&mu_);
    const bool inserted = table_.insert({op_id, std::move(callback)}).second;
    if (!inserted) {
      return absl::AlreadyExistsError(
          absl::StrCat("Op id ", op_id, " already exists"));
    }
    return absl::OkStatus();
  }

  std::optional<ResponseCallback> Pop(OpId op_id) {
    absl::MutexLock l(&mu_);
    auto it = table_.find(op_id);
    if (it == table_.end()) {
      return std::nullopt;
    }
    auto cb = std::move(it->second);
    table_.erase(it);
    return std::move(cb);
  }

  absl::flat_hash_map<OpId, ResponseCallback> PopAll() {
    absl::flat_hash_map<OpId, ResponseCallback> result;
    absl::MutexLock l(&mu_);
    result = std::move(table_);
    table_ = absl::flat_hash_map<OpId, ResponseCallback>();
    return result;
  }

 private:
  absl::Mutex mu_;
  absl::flat_hash_map<OpId, ResponseCallback> table_ ABSL_GUARDED_BY(mu_);
};

std::shared_ptr<GrpcClientSession> GrpcClientSession::Create(
    std::shared_ptr<grpc::GrpcIfrtService::StubInterface> stub,
    GrpcIfrtSessionMetadata metadata,
    StreamTerminatedCallback stream_terminated_cb) {
  auto context = std::make_unique<::grpc::ClientContext>();
  context->AddMetadata("ifrt-proxy-grpc-ifrt-session-metadata-bin",
                       metadata.SerializeAsString());
  std::shared_ptr<GrpcClientSession> result(new GrpcClientSession(
      std::move(stub), std::move(context), std::move(stream_terminated_cb)));
  return result;
}

GrpcClientSession::GrpcClientSession(
    std::shared_ptr<grpc::GrpcIfrtService::StubInterface> stub,
    std::unique_ptr<::grpc::ClientContext> context,
    StreamTerminatedCallback stream_terminated_cb)
    : response_callbacks_(std::make_unique<ResponseCallbackTable>()),
      reader_thread_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), "ifrt_proxy_client_grpc_reader",
          /*num_threads=*/1)),
      stub_(std::move(stub)),
      context_(std::move(context)),
      stream_(stub_->IfrtSession(context_.get())),
      stream_terminated_cb_(std::move(stream_terminated_cb)),
      user_futures_work_queue_(std::make_unique<tsl::UnboundedWorkQueue>(
          tsl::Env::Default(), "GrpcClientSessionUserFuturesWorkQueue")) {
  reader_thread_->Schedule(
      absl::bind_front(&GrpcClientSession::ReadLoop, this));
}

Future<std::shared_ptr<IfrtResponse>> GrpcClientSession::Enqueue(
    std::unique_ptr<IfrtRequest> request) {
  auto promise = Future<std::shared_ptr<IfrtResponse>>::CreatePromise();
  absl::Status status = Enqueue(
      std::move(request),
      [promise, queue = user_futures_work_queue_.get()](
          absl::StatusOr<std::shared_ptr<IfrtResponse>> response) mutable {
        queue->Schedule([promise = std::move(promise),
                         response = std::move(response)]() mutable -> void {
          promise.Set(std::move(response));
        });
      });
  if (!status.ok()) {
    user_futures_work_queue_->Schedule([promise, status]() mutable -> void {
      promise.Set(std::move(status));
    });
  }
  return Future<std::shared_ptr<IfrtResponse>>(std::move(promise));
}

absl::Status GrpcClientSession::Enqueue(std::unique_ptr<IfrtRequest> req,
                                        ResponseCallback callback) {
  const OpId op_id = req->request_metadata().op_id();

  absl::MutexLock l(&writer_mu_);
  if (writes_stopped_) {
    return absl::FailedPreconditionError(
        "GrpcClientSession: writes no longer allowed.");
  }

  TF_RETURN_IF_ERROR(response_callbacks_->Add(op_id, std::move(callback)));

  if (!stream_->Write(*req)) {
    CHECK(response_callbacks_->Pop(op_id).has_value());
    return absl::UnknownError("GrpcClientSession: writing to stream failed.");
  }

  return absl::OkStatus();
}

void GrpcClientSession::ReadLoop() {
  while (true) {
    auto read_buffer = std::make_unique<IfrtResponse>();
    if (!stream_->Read(read_buffer.get())) {
      LOG(INFO) << "GrpcClientSession: reader loop is exiting.";
      break;
    }

    const OpId op_id = read_buffer->response_metadata().op_id();
    std::optional<ResponseCallback> callback = response_callbacks_->Pop(op_id);

    if (callback.has_value()) {
      VLOG(1) << "GrpcClientSession: Issuing callback for " << op_id;
      (*callback)(std::move(read_buffer));
      VLOG(1) << "GrpcClientSession: Done with callback for " << op_id;
    } else {
      LOG(ERROR) << "Received response with no remaining registered callback: "
                 << read_buffer->DebugString();
    }
  }

  reader_thread_stopped_.Notify();
  Finish(absl::OkStatus());
}

void GrpcClientSession::Finish(const absl::Status& client_status) {
  LOG(INFO) << "GrpcClientSession: Finish() called with client status "
            << client_status;

  absl::call_once(finish_once_, [&] {
    context_->TryCancel();

    LOG(INFO) << "GrpcClientSession: Waiting for reader thread to stop.";
    reader_thread_stopped_.WaitForNotification();

    auto finish_stream_and_get_server_status = [&]() -> absl::Status {
      LOG(INFO) << "GrpClientSession: Attempting to call stream->Finish()";
      absl::MutexLock l(&writer_mu_);
      // Note: stream_->Finish() counts as a write, and needs to be serialized
      // with stream->Write().
      LOG(INFO) << "GrpClientSession: Attempting to call stream->Finish(), "
                   "mutex acquired";
      absl::Status server_status = xla::FromGrpcStatus(stream_->Finish());
      LOG(INFO) << "GrpClientSession: stream->Finish() returned server status "
                << server_status;

      CHECK(!writes_stopped_);
      writes_stopped_ = true;

      return server_status;
    };

    absl::Status combined_status = finish_stream_and_get_server_status();
    combined_status.Update(client_status);

    auto all_callbacks = response_callbacks_->PopAll();
    for (auto& [_, cb] : all_callbacks) {
      if (combined_status.ok()) {
        cb(absl::AbortedError("Finish(OK) called."));
      } else {
        cb(combined_status);
      }
    }

    LOG(INFO) << "GrpClientSession::Finish(): calling terminated cb with "
              << combined_status;
    stream_terminated_cb_(combined_status);
  });
}

GrpcClientSession::~GrpcClientSession() {
  GrpcClientSession::Finish(absl::CancelledError("~GrpcClientSession called."));
  reader_thread_.reset();  // Wait until the reader thread exits.
  LOG(INFO) << "Deleting GrpcClientSession.user_futures_work_queue_ ...";
  user_futures_work_queue_.reset();
  LOG(INFO) << "Deleted GrpcClientSession.user_futures_work_queue_.";
}

std::shared_ptr<grpc::GrpcIfrtService::StubInterface> CreateGrpcStub(
    absl::string_view server_address) {
  ::grpc::ChannelArguments args;
  // Remove message size limit to accommodate large messages exchanged during
  // model compilation.
  args.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, -1);
  args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, -1);
  std::shared_ptr<::grpc::Channel> channel = ::grpc::CreateCustomChannel(
      std::string(server_address), GetClientCredentials(), args);
  VLOG(0) << "  Established channel.";
  CHECK(channel != nullptr);

  std::shared_ptr<grpc::GrpcIfrtService::StubInterface> stub =
      grpc::GrpcIfrtService::NewStub(channel);
  VLOG(0) << "  Created stub.";
  CHECK(stub != nullptr);

  return stub;
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
