/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime/send_recv.h"

#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"

namespace xla {
namespace gpu {

using xla::runtime::AggregateAttrDef;
using xla::runtime::AggregateAttrEncoding;
using xla::runtime::CustomCall;
using xla::runtime::CustomCallAttrEncodingSet;
using xla::runtime::Dictionary;
using xla::runtime::StridedMemrefView;
using xla::runtime::Tagged;
using xla::runtime::TypeIDNameRegistry;

namespace mhlo = ::mlir::mhlo;

//===----------------------------------------------------------------------===//
// Structs for encoding send/recv operations attributes.
//===----------------------------------------------------------------------===//

struct ChannelHandle {
  int64_t handle;
  int64_t type;
};

}  // namespace gpu

//===----------------------------------------------------------------------===//
// Register send/recv attributes decoding with the Xla runtime.
//===----------------------------------------------------------------------===//

namespace runtime {

XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(xla::gpu::ChannelHandle,
                                             AggregateMember<int64_t>("handle"),
                                             AggregateMember<int64_t>("type"));

}  // namespace runtime

//===----------------------------------------------------------------------===//
// Type names for encoded attributes.
//===----------------------------------------------------------------------===//

namespace gpu {

void RegisterSendRecvTypeIdNames(TypeIDNameRegistry& registry) {
  registry.Register<Tagged<ChannelHandle>>("__type_id_channel_handle");
}

//===----------------------------------------------------------------------===//
// Encoding from MHLO attributes to Xla runtime aggregate attributes.
//===----------------------------------------------------------------------===//

void PopulateSendRecvAttrEncoding(CustomCallAttrEncodingSet& encoding) {
  {  // --- Encode `mhlo::ChannelHandleAttr`.
    using Attr = mhlo::ChannelHandleAttr;
    encoding.Add<AggregateAttrEncoding<Attr, ChannelHandle>>(
        encoding, AggregateAttrDef<Attr>()
                      .Add("handle", &Attr::getHandle)
                      .Add("type", &Attr::getType));
  }
}

//===----------------------------------------------------------------------===//
// Send/Recv custom call implementation.
//===----------------------------------------------------------------------===//

static std::vector<float>* storage = new std::vector<float>(4);

static absl::StatusOr<int32_t> GetRecvChannel(Dictionary frontend_attrs) {
  auto str = frontend_attrs.get<std::string_view>("_xla_dcn_recv_channel");

  int32_t recv_channel;
  if (failed(str) || !absl::SimpleAtoi(*str, &recv_channel))
    return absl::InternalError(
        "Failed to get receive channel id from the frontend attributes");

  return recv_channel;
}

static absl::Status SendImpl(const ServiceExecutableRunOptions* run_options,
                             StridedMemrefView arg, ChannelHandle channel,
                             bool is_host_transfer, Dictionary frontend_attrs) {
  // For now we only support transfers between the device and the host.
  if (!is_host_transfer)
    return absl::InvalidArgumentError(
        "Device to device communication operations are not supported");

  // Get the corresponding receive channel id.
  auto recv_channel = GetRecvChannel(frontend_attrs);
  if (!recv_channel.ok()) return recv_channel.status();

  VLOG(3) << "Send buffer to host: channel=" << channel.handle
          << "; recv_channel=" << *recv_channel;

  return absl::UnimplementedError("Send operation is not implemented");
}

static absl::Status RecvImpl(const ServiceExecutableRunOptions* run_options,
                             StridedMemrefView arg, ChannelHandle channel,
                             bool is_host_transfer, Dictionary frontend_attrs) {
  // For now we only support transfers between the device and the host.
  if (!is_host_transfer)
    return absl::InvalidArgumentError(
        "Device to device communication operations are not supported");

  VLOG(3) << "Receive buffer from host: channel=" << channel.handle;

  return absl::UnimplementedError("Recv operation is not implemented");
}

static absl::Status SendDoneImpl(const ServiceExecutableRunOptions* run_options,
                                 ChannelHandle channel, bool is_host_transfer) {
  return absl::UnimplementedError("SendDone operation is not implemented");
}

static absl::Status RecvDoneImpl(const ServiceExecutableRunOptions* run_options,
                                 ChannelHandle channel, bool is_host_transfer) {
  return absl::UnimplementedError("RecvDone operation is not implemented");
}

//===----------------------------------------------------------------------===//
// Send/Recv custom calls bindings and registration.
//===----------------------------------------------------------------------===//

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Send, FunctionWrapper<SendImpl>(), checks,
    CustomCall::Bind("xla.gpu.send")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<StridedMemrefView>()
        .Attr<ChannelHandle>("channel_handle")
        .Attr<bool>("is_host_transfer")
        .Attr<Dictionary>("frontend_attributes"));

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Recv, FunctionWrapper<RecvImpl>(), checks,
    CustomCall::Bind("xla.gpu.recv")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<StridedMemrefView>()
        .Attr<ChannelHandle>("channel_handle")
        .Attr<bool>("is_host_transfer")
        .Attr<Dictionary>("frontend_attributes"));

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    SendDone, FunctionWrapper<SendDoneImpl>(), checks,
    CustomCall::Bind("xla.gpu.send_done")
        .UserData<const ServiceExecutableRunOptions*>()
        .Attr<ChannelHandle>("channel_handle")
        .Attr<bool>("is_host_transfer"));

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    RecvDone, FunctionWrapper<RecvDoneImpl>(), checks,
    CustomCall::Bind("xla.gpu.recv_done")
        .UserData<const ServiceExecutableRunOptions*>()
        .Attr<ChannelHandle>("channel_handle")
        .Attr<bool>("is_host_transfer"));

//===----------------------------------------------------------------------===//

// Registers XLA Gpu runtime Send/Recv custom calls.
void RegisterSendRecvCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("xla.gpu.send", Send);
  registry.Register("xla.gpu.recv", Recv);
  registry.Register("xla.gpu.send_done", SendDone);
  registry.Register("xla.gpu.recv_done", RecvDone);
}

}  // namespace gpu
}  // namespace xla
