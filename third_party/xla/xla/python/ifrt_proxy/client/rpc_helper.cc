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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "tsl/platform/random.h"
#include "tsl/platform/status_to_from_proto.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"
#include "tsl/profiler/utils/xplane_schema.h"

namespace xla {
namespace ifrt {
namespace proxy {

namespace {

using ::tsl::profiler::XFlow;

// XFlowHelper makes it easier to create trace spans with a flow between them.
// Typical usage:
//
// XFlowHelper flow("my_request");
// ...
//
// auto response_handler = [flow](ResponseMsg msg) {
//   flow.InstantActivity<kRecv>();
//   LOG(INFO) << "Received response: " << msg;
// }
//
// {
//   auto request_span = flow.Span<kSend>();
//   auto request_protobuf = CreateRequestProtobuf();
//   transport.Send(request_protobuf, response_handler);
// }
//
//
class XFlowHelper {
 public:
  explicit XFlowHelper(absl::string_view name)
      : xflow_id_(tsl::random::New64() >> 8 /*XFlow IDs are 56 bits*/),
        name_(name) {}

  typedef enum { kSend, kRecv, kRecvSend } Direction;

  template <Direction D>
  tsl::profiler::TraceMe Span() const {
    return tsl::profiler::TraceMe([xflow_id = xflow_id_, name = name_] {
      return Encode<D>(xflow_id, name);
    });
  }

  template <Direction D>
  void InstantActivity() const {
    return tsl::profiler::TraceMe::InstantActivity(
        [xflow_id = xflow_id_, name = name_] {
          return Encode<D>(xflow_id, name);
        });
  }

 private:
  template <Direction D>
  static std::string Encode(uint64_t xflow_id, absl::string_view name) {
    static constexpr absl::string_view flow_dir_str =
        D == kSend ? "send" : (D == kRecv ? "recv" : "recv_send");
    const XFlow flow(xflow_id, D == kRecvSend ? XFlow::kFlowInOut
                                              : (D == kRecv ? XFlow::kFlowIn
                                                            : XFlow::kFlowOut));
    return tsl::profiler::TraceMeEncode(
        name, {{"dir", flow_dir_str}, {"flow", flow.ToStatValue()}});
  };

  const uint64_t xflow_id_;
  const absl::string_view name_;
};

}  // namespace

// DoRpc is a templated function that implements the logic of all RPC-wrapping
// functions of `RpcHelper`, such as `RpcHelper::MakeArrayFromHostBuffer()`.
template <typename Req, typename Resp>
Future<std::shared_ptr<Resp>> DoRpc(ClientSession* session,
                                    void (IfrtRequest::*set_req)(Req*),
                                    Resp* (IfrtResponse::*get_resp)(),
                                    bool (IfrtResponse::*has_resp)() const,
                                    std::unique_ptr<Req> req,
                                    absl::string_view profiling_name) {
  auto ifrt_req = std::make_unique<IfrtRequest>();
  (ifrt_req.get()->*set_req)(req.release());

  XFlowHelper x_flow_helper(profiling_name);
  auto traceme = x_flow_helper.Span<XFlowHelper::kSend>();

  auto promise = Future<std::shared_ptr<Resp>>::CreatePromise();
  auto on_ready = [promise, has_resp, get_resp, x_flow_helper](
                      absl::StatusOr<std::shared_ptr<IfrtResponse>> r) mutable {
    auto traceme = x_flow_helper.Span<XFlowHelper::kRecv>();
    if (!r.ok()) {
      LOG_EVERY_N_SEC(ERROR, 10)
          << "Connection to IFRT proxy server was terminated: " << r.status();
      promise.Set(absl::UnavailableError(
          absl::StrCat("Connection to IFRT proxy server was terminated: ",
                       r.status().ToString())));
      return;
    }

    std::shared_ptr<IfrtResponse> response = *std::move(r);
    if (!response->has_response_metadata()) {
      promise.Set(absl::InternalError(
          absl::StrCat("IFRT server sent a message without metadata: ",
                       response->DebugString())));
      return;
    }

    const absl::Status metadata_status =
        tsl::StatusFromProto(response->response_metadata().status());
    const bool has_expected_response = (response.get()->*has_resp)();
    const auto has_some_response =
        response->response_case() != IfrtResponse::RESPONSE_NOT_SET;

    if (metadata_status.ok() && !has_some_response) {
      promise.Set(absl::InternalError(
          absl::StrCat("OK response with no actual response set: ",
                       response->DebugString())));
      return;
    }

    if (!has_expected_response && has_some_response) {
      promise.Set(absl::InternalError(absl::StrCat(
          "Response with wrong type (expected ", Resp::GetDescriptor()->name(),
          "): ", response->DebugString())));
      return;
    }

    // If the metadata_status is not-OK, according to ifrt_service.proto,
    // there may be an error _instead_ of an actual response value. So, check if
    // an actual response value exists, and if so return it irrespective of what
    // the metadata_status says.
    if (!has_some_response) {
      promise.Set(metadata_status);
    } else {
      promise.Set(
          std::make_shared<Resp>(*std::move((response.get()->*get_resp)())));
    }
  };
  session->Enqueue(std::move(ifrt_req)).OnReady(on_ready);

  return Future<std::shared_ptr<Resp>>(promise);
}

#define RPC(METHOD, PROPERTY)                                                 \
  RpcHelper::ResponseFuture<METHOD##Response> RpcHelper::METHOD(              \
      std::unique_ptr<METHOD##Request> req) {                                 \
    return DoRpc(                                                             \
        session_.get(), &IfrtRequest::set_allocated_##PROPERTY##_request,     \
        &IfrtResponse::mutable_##PROPERTY##_response,                         \
        &IfrtResponse::has_##PROPERTY##_response, std::move(req), #PROPERTY); \
  }

RPC(Init, init);
RPC(GetDefaultDeviceAssignment, get_default_device_assignment);
RPC(CheckFuture, check_future);
RPC(CheckValueReady, check_value_ready);
RPC(MakeArrayFromHostBuffer, make_array_from_host_buffer);
RPC(AssembleArrayFromSingleDeviceArrays,
    assemble_array_from_single_device_arrays);
RPC(RemapArrays, remap_arrays);
RPC(DisassembleIntoSingleDeviceArrays, disassemble_into_single_device_arrays);
RPC(CopyToHostBuffer, copy_to_host_buffer);
RPC(IsArrayDeleted, is_array_deleted);
RPC(DestructArray, destruct_array)
RPC(CopyArrays, copy_arrays);
RPC(Reshard, reshard);
RPC(FullyReplicatedShard, fully_replicated_shard);
RPC(DeleteArray, delete_array);
RPC(Compile, compile);
RPC(LoadedExecutableMetadata, loaded_executable_metadata);
RPC(LoadedExecutableExecute, loaded_executable_execute);
RPC(LoadedExecutableDelete, loaded_executable_delete);
RPC(LoadedExecutableIsDeleted, loaded_executable_is_deleted);
RPC(LoadedExecutableDestruct, loaded_executable_destruct);
RPC(LoadedHostCallbackPoll, loaded_host_callback_poll);
RPC(LoadedHostCallbackReturn, loaded_host_callback_return);

Future<> RpcHelper::CheckFuture(uint64_t handle) {
  auto req = std::make_unique<CheckFutureRequest>();
  req->set_future_handle(handle);

  auto promise = Future<>::CreatePromise();
  CheckFuture(std::move(req))
      .OnReady(
          [promise](absl::StatusOr<std::shared_ptr<CheckFutureResponse>>
                        response) mutable { promise.Set(response.status()); });

  return Future<>(std::move(promise));
}

void RpcHelper::Disconnect() {
  session_->Finish(absl::CancelledError("Disconnected by client"));
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
