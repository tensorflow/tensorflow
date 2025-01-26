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

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/log_sink_registry.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "grpc/support/time.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "grpcpp/support/sync_stream.h"
#include "xla/python/ifrt_proxy/client/version.h"
#include "xla/python/ifrt_proxy/common/grpc_credentials.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.grpc.pb.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/test_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace proxy {

namespace {

using ::testing::Not;
using ::tsl::testing::IsOk;

// Sufficient time for all processing (that are not explicitly waiting for
// further input) to have finished.
constexpr absl::Duration kSufficientTime = absl::Seconds(5);

GrpcIfrtSessionMetadata Metadata() {
  GrpcIfrtSessionMetadata metadata;
  metadata.mutable_version()->set_protocol_version(kClientMaxVersion);
  return metadata;
}

absl::Status TestError() { return absl::UnknownError("test error"); }

struct Queue : public TestQueue<absl::Status> {
  Queue() : TestQueue<absl::Status>(kSufficientTime) {}
};

// Checks that the input is a list of zero-or-more OK statuses followed by
// zero-or-more NOT-OK statuses. Succeeds for {OK, NOT_OK, NOT_OK}, but fails
// for {OK, NOT_OK, OK}.
void ExpectHeadAndTail(
    std::vector<std::variant<absl::StatusOr<Queue*>, absl::Status>> var_list) {
  std::vector<absl::Status> status_list;
  for (const auto& v : var_list) {
    if (std::holds_alternative<absl::StatusOr<Queue*>>(v)) {
      status_list.push_back(std::get<absl::StatusOr<Queue*>>(v).status());
    } else {
      status_list.push_back(std::get<absl::Status>(v));
    }
  }
  bool seen_not_ok = false;
  std::string str;
  for (const auto& s : status_list) {
    absl::StrAppend(&str, "\n", s.ToString(), "\n-----\n");
  }
  for (const auto& s : status_list) {
    if (!s.ok()) seen_not_ok = true;
    if (seen_not_ok) {
      EXPECT_THAT(s, Not(IsOk())) << str;
    }
  }
}

using ServerStream = ::grpc::ServerReaderWriter<IfrtResponse, IfrtRequest>;
using SessionAction = bool;
constexpr SessionAction kContinueSession = true;
constexpr SessionAction kStopSession = false;
using OnSessionStart = std::function<SessionAction()>;
using OnReqReceived =
    std::function<SessionAction(const IfrtRequest&, ServerStream*)>;

// A simple implementation of IfrtService with various test-hooks.
class SimpleIfrtService : public grpc::GrpcIfrtService::Service {
 public:
  SimpleIfrtService(OnReqReceived on_req_received,
                    OnSessionStart on_session_start)
      : on_req_received_(std::move(on_req_received)),
        on_session_start_(std::move(on_session_start)) {}

  ::grpc::Status IfrtSession(::grpc::ServerContext* context,
                             ServerStream* stream) override {
    if (on_session_start_ && on_session_start_() == kStopSession) {
      return ::grpc::Status::OK;
    }

    {
      absl::MutexLock l(&mu_);
      CHECK(contexts_.insert(context).second);
    }

    while (true) {
      IfrtRequest request;
      LOG(INFO) << "Server: waiting on Read().";
      if (!stream->Read(&request)) {
        LOG(INFO) << "Server: Read() returned false.";
        break;
      }
      LOG(INFO) << "Server: Read() returned true.";
      if (!on_req_received_) {
        IfrtResponse response;
        response.mutable_response_metadata()->set_op_id(
            request.request_metadata().op_id());
        stream->Write(response);
      } else if (on_req_received_(request, stream) == kStopSession) {
        break;
      }
    }
    {
      absl::MutexLock l(&mu_);
      CHECK_EQ(contexts_.erase(context), 1);
    }

    LOG(INFO) << "Finishing IFRT session";
    return ::grpc::Status::OK;
  }

  void CancelAllServerSessions() {
    absl::MutexLock l(&mu_);
    for (const auto& context : contexts_) {
      context->TryCancel();
    }
  }

 private:
  const OnReqReceived on_req_received_;
  const OnSessionStart on_session_start_;

  // Keeps track of `::grpc::ServerContext` for all ongoing sessions.
  absl::Mutex mu_;
  absl::flat_hash_set<::grpc::ServerContext*> contexts_ ABSL_GUARDED_BY(mu_);
};

// Encapsulates objects related to a client and server instance of
// `grpc::GrpcIfrtService`.
class ClientAndServer {
 public:
  explicit ClientAndServer(OnReqReceived on_req_received = nullptr,
                           OnSessionStart on_session_start = nullptr) {
    std::string address =
        absl::StrCat("localhost:", tsl::testing::PickUnusedPortOrDie());
    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(address, GetServerCredentials());
    ifrt_service_ =
        std::make_unique<SimpleIfrtService>(on_req_received, on_session_start);
    builder.RegisterService(ifrt_service_.get());
    server_ = builder.BuildAndStart();

    LOG(INFO) << "Server started and listening on " << address;
    absl::FlushLogSinks();

    std::shared_ptr<::grpc::Channel> channel =
        ::grpc::CreateChannel(address, GetClientCredentials());
    channel->WaitForConnected(gpr_time_add(
        gpr_now(GPR_CLOCK_REALTIME), gpr_time_from_seconds(10, GPR_TIMESPAN)));
    LOG(INFO) << "conn_state = " << channel->GetState(/*try_to_connect=*/false);

    auto stub = grpc::GrpcIfrtService::NewStub(channel);
    CHECK(stub != nullptr);

    client_session_ = GrpcClientSession::Create(
        std::move(stub), Metadata(), [this](absl::Status s) {
          client_finished_q_.Push(s);
          client_finished_notification_.Notify();
        });

    client_finished_q_.AllowNonEmptyDestruction(/*allow=*/true);
  }

  void StopServer() {
    ifrt_service_->CancelAllServerSessions();
    server_->Shutdown();
    server_->Wait();
  }

  ~ClientAndServer() {
    StopServer();
    client_session_->Finish(absl::CancelledError("~ClientAndServer"));
    client_finished_notification_.WaitForNotificationWithTimeout(
        kSufficientTime);
    CHECK(client_finished_notification_.HasBeenNotified());
  }

  GrpcClientSession* client_session() { return client_session_.get(); }

  Queue* client_finished_q() { return &client_finished_q_; }

  absl::StatusOr<Queue*> SendSimpleRequest() {
    owned_queues_.push_back(std::make_unique<Queue>());
    Queue* q = owned_queues_.back().get();

    auto req = std::make_unique<IfrtRequest>();
    TF_RETURN_IF_ERROR(client_session_->Enqueue(
        std::move(req), [q](absl::StatusOr<GrpcClientSession::Response> resp) {
          q->Push(resp.status());
        }));

    return q;
  }

 private:
  std::vector<std::unique_ptr<Queue>> owned_queues_;
  Queue client_finished_q_;
  absl::Notification client_finished_notification_;
  std::shared_ptr<GrpcClientSession> client_session_;

  std::unique_ptr<::grpc::Server> server_;
  std::unique_ptr<SimpleIfrtService> ifrt_service_;
};

TEST(GrpcClientSessionTest, HappyCaseOneRequestWithServerTermination) {
  ClientAndServer cs;

  TF_ASSERT_OK_AND_ASSIGN(Queue * response_q, cs.SendSimpleRequest());

  EXPECT_THAT(response_q->Pop(), IsOk());

  EXPECT_EQ(cs.client_finished_q()->PopOrTimeout(), std::nullopt);

  cs.StopServer();
  EXPECT_THAT(cs.client_finished_q()->Pop(), Not(IsOk()));
}

TEST(GrpcClientSessionTest, HappyCaseTwoRequestsWithClientFinish) {
  ClientAndServer cs;

  TF_ASSERT_OK_AND_ASSIGN(Queue * response_q_1, cs.SendSimpleRequest());
  TF_ASSERT_OK_AND_ASSIGN(Queue * response_q_2, cs.SendSimpleRequest());

  EXPECT_THAT(response_q_1->Pop(), IsOk());
  EXPECT_THAT(response_q_2->Pop(), IsOk());

  EXPECT_EQ(cs.client_finished_q()->PopOrTimeout(), std::nullopt);

  cs.client_session()->Finish(TestError());
  EXPECT_THAT(cs.client_finished_q()->Pop(), Not(IsOk()));
}

TEST(GrpcClientSessionTest, ServerFinishesDuringFirstRead) {
  ClientAndServer cs(
      /*on_req_received=*/[](auto, auto) { return kStopSession; });

  TF_ASSERT_OK_AND_ASSIGN(Queue * response_q_1, cs.SendSimpleRequest());
  EXPECT_THAT(response_q_1->Pop(), Not(IsOk()));

  absl::StatusOr<Queue*> response_q_2 = cs.SendSimpleRequest();
  EXPECT_THAT(response_q_2.status(), Not(IsOk()));

  EXPECT_THAT(cs.client_finished_q()->Pop(), Not(IsOk()));
}

TEST(GrpcClientSessionTest, ServerFinishesDuringConstruction) {
  ClientAndServer cs(/*on_req_received=*/nullptr,
                     /*on_session_start=*/[]() { return kStopSession; });

  absl::StatusOr<Queue*> response_q_1 = cs.SendSimpleRequest();
  absl::StatusOr<Queue*> response_q_2 = cs.SendSimpleRequest();

  ExpectHeadAndTail({response_q_1, response_q_2});
  if (response_q_1.ok()) EXPECT_THAT(response_q_1.value()->Pop(), Not(IsOk()));
  if (response_q_2.ok()) EXPECT_THAT(response_q_2.value()->Pop(), Not(IsOk()));

  EXPECT_THAT(cs.client_finished_q()->Pop(), Not(IsOk()));
}

TEST(GrpcClientSessionTest, ClientFinishesAfterServerConsumesFirstRequest) {
  std::atomic<GrpcClientSession*> session_ptr;
  ClientAndServer cs(
      /*on_req_received=*/[session_ptr = &session_ptr](auto, auto) {
        session_ptr->load()->Finish(TestError());
        return kContinueSession;
      });
  session_ptr.store(cs.client_session());

  TF_ASSERT_OK_AND_ASSIGN(Queue * response_q_1, cs.SendSimpleRequest());
  EXPECT_THAT(response_q_1->Pop(), Not(IsOk()));

  absl::StatusOr<Queue*> response_q_2 = cs.SendSimpleRequest();
  EXPECT_THAT(response_q_2.status(), Not(IsOk()));

  EXPECT_THAT(cs.client_finished_q()->Pop(), Not(IsOk()));
}

TEST(GrpcClientSessionTest, ClientFinishesAfterServerWritesFirstResponse) {
  std::atomic<GrpcClientSession*> session_ptr;
  ClientAndServer cs(
      /*on_req_received=*/[session_ptr = &session_ptr](const IfrtRequest& r,
                                                       ServerStream* s) {
        IfrtResponse response;
        response.mutable_response_metadata()->set_op_id(
            r.request_metadata().op_id());
        s->Write(response);
        session_ptr->load()->Finish(TestError());
        return kContinueSession;
      });
  session_ptr.store(cs.client_session());

  TF_ASSERT_OK_AND_ASSIGN(Queue * response_q_1, cs.SendSimpleRequest());
  absl::StatusOr<Queue*> response_q_2 = cs.SendSimpleRequest();

  // The client may or may not terminate before the first response arrives.
  response_q_1->Pop().IgnoreError();

  // The client may or may not terminate before the second request could be
  // enqueued. If it could be enqueued, the client will die without the server
  // sending the corresponding response.
  if (response_q_2.ok()) {
    EXPECT_THAT(response_q_2.value()->Pop(), Not(IsOk()));
  }

  EXPECT_THAT(cs.client_finished_q()->Pop(), Not(IsOk()));
}

TEST(GrpcClientSessionTest, ClientFinishesDuringServerConstruction) {
  std::atomic<GrpcClientSession*> session_ptr;
  absl::Notification init_done;
  ClientAndServer cs(/*on_req_received=*/nullptr,
                     /*on_session_start=*/[session_ptr = &session_ptr,
                                           init_done = &init_done]() {
                       init_done->WaitForNotification();
                       session_ptr->load()->Finish(TestError());
                       return kContinueSession;
                     });
  session_ptr.store(cs.client_session());
  init_done.Notify();

  absl::StatusOr<Queue*> response_q_1 = cs.SendSimpleRequest();
  absl::StatusOr<Queue*> response_q_2 = cs.SendSimpleRequest();

  if (response_q_1.ok()) {
    EXPECT_THAT(response_q_1.value()->Pop(), Not(IsOk()));
  }
  if (response_q_2.ok()) {
    EXPECT_THAT(response_q_2.value()->Pop(), Not(IsOk()));
  }

  ExpectHeadAndTail({response_q_1, response_q_2});

  EXPECT_THAT(cs.client_finished_q()->Pop(), Not(IsOk()));
}

TEST(GrpcClientSessionTest, MethodsAfterFinishReturnError) {
  ClientAndServer cs;

  TF_ASSERT_OK_AND_ASSIGN(Queue * response_q_1, cs.SendSimpleRequest());
  cs.client_session()->Finish(TestError());

  EXPECT_THAT(cs.SendSimpleRequest(), Not(IsOk()));

  response_q_1->AllowNonEmptyDestruction(/*allow=*/true);
}

TEST(GrpcClientSessionTest, ReceivingBadIfrtResponseDoesNotCrash) {
  ClientAndServer cs(
      /*on_req_received=*/[](const IfrtRequest& r, ServerStream* s) mutable {
        IfrtResponse resp;
        resp.mutable_response_metadata()->set_op_id(2000);
        s->Write(resp);
        resp.mutable_response_metadata()->set_op_id(
            r.request_metadata().op_id());
        s->Write(resp);
        return kContinueSession;
      });

  TF_ASSERT_OK_AND_ASSIGN(Queue * response_q, cs.SendSimpleRequest());

  EXPECT_THAT(response_q->Pop(), IsOk());
}

TEST(GrpcClientSessionTest, BadInitialChannelFailsPromptly) {
  std::string address =
      absl::StrCat("localhost:", tsl::testing::PickUnusedPortOrDie());

  std::shared_ptr<::grpc::Channel> channel =
      ::grpc::CreateChannel(address, GetClientCredentials());

  std::unique_ptr<grpc::GrpcIfrtService::StubInterface> stub =
      grpc::GrpcIfrtService::NewStub(channel);
  EXPECT_TRUE(stub != nullptr);

  auto session_finished = std::make_shared<Queue>();
  auto session = GrpcClientSession::Create(
      std::move(stub), Metadata(),
      [session_finished](absl::Status s) { session_finished->Push(s); });

  EXPECT_THAT(session_finished->Pop(), Not(IsOk()));
}

}  // namespace

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
