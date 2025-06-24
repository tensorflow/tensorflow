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

#include <functional>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/log_entry.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/client_context.h"
#include "xla/pjrt/distributed/util.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt_proxy/client/client.h"
#include "xla/python/ifrt_proxy/client/global_flags.h"
#include "xla/python/ifrt_proxy/client/grpc_client_session.h"
#include "xla/python/ifrt_proxy/client/grpc_host_buffer.h"
#include "xla/python/ifrt_proxy/client/registry.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/client/version.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace proxy {

#define CONN_UPDATE_LOG(msg)                                              \
  do {                                                                    \
    LOG(INFO) << msg;                                                     \
    if (options.on_connection_update) {                                   \
      options.on_connection_update(absl::StrCat(absl::Now(), ": ", msg)); \
    }                                                                     \
  } while (false)

namespace {

// Attempts to establish a session to the proxy-server and returns a `Client`
// based on the session if successful. `on_disconnect` will be invoked exactly
// once if this function returns successfully, and not invoked if this function
// returns a non-OK status.
absl::StatusOr<std::unique_ptr<Client>> AttemptConnection(
    absl::string_view server_address, int attempt_no,
    const ClientConnectionOptions& options) {
  std::unique_ptr<RpcHelper> rpc_helper;
  auto init_response_promise =
      Future<std::shared_ptr<InitResponse>>::CreatePromise();

  // TODO(b/266635130): Move gRPC stub creation to be outside of `Client` so
  // that we can pass mock `ClientSession` to the client.
  auto control_path_stub = CreateGrpcStub(server_address);

  auto session_disconnect_cb =
      [init_response =
           Future<std::shared_ptr<InitResponse>>(init_response_promise),
       on_disconnect = options.on_disconnect,
       attempt_no](absl::Status s) mutable {
        // If the `rpc_helper->Init().OnReady(cb)` statement below has returned,
        // the callback cb in that statement (which sets `init_response`) is
        // guaranteed by `GrpcClientSession::Create()` to be called before
        // `session_disconnect_cb`.
        // TODO(madthanu): The above statement is false (even if we wanted to,
        // we cannot meaningfully enforce or document the guarantee of
        // the returned Future's OnReady being called before another callback),
        // although the exact way init_response_promise is set below makes it
        // work most of the time.
        if (init_response.IsReady() && init_response.Await().ok()) {
          // If the init RPC has already completed successfully, we have
          // already or will be returning OK from the `AttemptConnection` call.
          LOG(WARNING) << "IFRT proxy server disconnected: " << s;
          if (on_disconnect != nullptr) {
            on_disconnect(s);
          }
        } else {
          // Otherwise, we are going to return an error from
          // `AttemptConnection`. So do not invoke `on_disconnect`.
          LOG(INFO) << "GrpcClientSession attempt " << attempt_no
                    << " failed: " << s;
        }
      };

  GrpcIfrtSessionMetadata metadata;
  {
    GrpcGetVersionRequest request;
    request.mutable_min_version()->set_protocol_version(kClientMinVersion);
    request.mutable_max_version()->set_protocol_version(kClientMaxVersion);

    ::grpc::ClientContext context;
    GrpcGetVersionResponse response;
    TF_RETURN_IF_ERROR(xla::FromGrpcStatus(
        control_path_stub->GetVersion(&context, request, &response)));

    CHECK_GE(response.version().protocol_version(), kClientMinVersion);
    CHECK_LE(response.version().protocol_version(), kClientMaxVersion);
    *metadata.mutable_version() = response.version();
  }
  *metadata.mutable_initialization_data() =
      options.initialization_data.ToProto();

  auto session = GrpcClientSession::Create(control_path_stub, metadata,
                                           session_disconnect_cb);
  rpc_helper =
      std::make_unique<RpcHelper>(metadata.version(), std::move(session));

  CONN_UPDATE_LOG(absl::StrCat("Sending InitRequest and waiting for ",
                               "response (attempt ", attempt_no, ")."));

  // TODO(b/282757875): Use a separate Request that will indicate quickly
  // whether the grpc_client<->grpc_server session has been established or
  // not, instead of combining it with the Request that will fetch device
  // information (which can take a while, depending on the IFRT backend).
  rpc_helper->Init(std::make_unique<InitRequest>())
      .OnReady([&](auto resp) mutable { init_response_promise.Set(resp); });

  TF_ASSIGN_OR_RETURN(
      auto init_response,
      Future<std::shared_ptr<InitResponse>>(init_response_promise).Await());

  bool reuse_control_path_stub_for_data_path =
      GetGlobalClientFlags()->synchronous_host_buffer_store ||
      (metadata.version().protocol_version() < 10);
  auto data_path_stub = reuse_control_path_stub_for_data_path
                            ? control_path_stub
                            : CreateGrpcStub(server_address);

  auto host_buffer_store = std::make_unique<GrpcClientHostBufferStore>(
      data_path_stub, metadata.version(), init_response->session_id());
  rpc_helper->set_host_buffer_store(std::move(host_buffer_store));

  return Client::Create(std::move(rpc_helper), std::move(*init_response));
}

absl::StatusOr<std::unique_ptr<Client>> CreateGrpcClient(
    absl::string_view server_address, const ClientConnectionOptions& options) {
  absl::Time start_time = absl::Now();
  absl::Status last_status;
  for (int i = 0; absl::Now() - start_time < options.connection_timeout; ++i) {
    CONN_UPDATE_LOG(absl::StrCat("Connecting to IFRT proxy server at ",
                                 server_address, ", attempt #", i, "..."));
    absl::StatusOr<std::unique_ptr<Client>> result =
        AttemptConnection(server_address, i, options);
    if (result.ok()) {
      CONN_UPDATE_LOG(absl::StrCat("Connected to IFRT proxy server on ",
                                   "attempt #", i, "."));
      return result;
    }
    last_status = result.status();
    CONN_UPDATE_LOG(absl::StrCat("Connection to IFRT proxy server attempt #", i,
                                 "failed: ", last_status.ToString()));
    absl::SleepFor(absl::Seconds(1));
  }

  // We want to prepend a human-friendly error message to status before
  // returning.
  auto err_msg =
      absl::StrCat("Unable to establish connection to ifrt_proxy server, ",
                   "please check provided address '", server_address,
                   "'; detailed error: ", last_status.message());
  CONN_UPDATE_LOG(err_msg);
  return tsl::errors::CreateWithUpdatedMessage(last_status, err_msg);
}

}  // namespace

bool register_client_factory =
    ([] { RegisterClientFactory("grpc", CreateGrpcClient); }(), true);

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
