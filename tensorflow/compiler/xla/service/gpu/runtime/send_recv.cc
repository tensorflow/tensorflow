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
#include "tensorflow/compiler/xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"

namespace xla {
namespace gpu {

using absl::InvalidArgumentError;

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

static absl::Status SendImpl(const ServiceExecutableRunOptions* run_options,
                             StridedMemrefView arg, ChannelHandle channel,
                             bool is_host_transfer, Dictionary frontend_attrs) {
  VLOG(3) << "Send buffer:"
          << " channel=" << channel.handle
          << " is_host_transfer=" << is_host_transfer;

  // For now we only support transfers between the device and the host.
  if (!is_host_transfer)
    return InvalidArgumentError(
        "Device to device communication operations are not supported");

  // Send buffer to a handler registered with the run options.
  if (auto* send = run_options->run_options().send_device_memory_function())
    return ToAbslStatus((*send)(channel.handle, run_options->stream(),
                                ToShape(arg), GetDeviceAddress(arg)));

  return InvalidArgumentError("SendDeviceMemoryFunction is not available");
}

static absl::Status RecvImpl(const ServiceExecutableRunOptions* run_options,
                             StridedMemrefView arg, ChannelHandle channel,
                             bool is_host_transfer, Dictionary frontend_attrs) {
  VLOG(3) << "Receive buffer:"
          << " channel=" << channel.handle
          << " is_host_transfer=" << is_host_transfer;

  // For now we only support transfers between the device and the host.
  if (!is_host_transfer)
    return InvalidArgumentError(
        "Device to device communication operations are not supported");

  // Recv buffer from a handler registered with the run options.
  if (auto* recv = run_options->run_options().recv_device_memory_function()) {
    auto dst = GetDeviceAddress(arg);
    return ToAbslStatus(
        (*recv)(channel.handle, run_options->stream(), ToShape(arg), &dst));
  }

  return InvalidArgumentError("RecvDeviceMemoryFunction is not available");
}

static absl::Status SendDoneImpl(const ServiceExecutableRunOptions* run_options,
                                 ChannelHandle channel, bool is_host_transfer) {
  // TODO(ezhulenev): Currently `SendDeviceMemoryFunction` is a blocking
  // function that completes data transfer operation. We have to make it
  // non-blocking to be able to overlap compute with data transfer.
  return absl::OkStatus();
}

static absl::Status RecvDoneImpl(const ServiceExecutableRunOptions* run_options,
                                 ChannelHandle channel, bool is_host_transfer) {
  // TODO(ezhulenev): Currently `RecvDeviceMemoryFunction` is a blocking
  // function that completes data transfer operation. We have to make it
  // non-blocking to be able to overlap compute with data transfer.
  return absl::OkStatus();
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
