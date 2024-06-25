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

#include "xla/python/ifrt_proxy/server/host_callback.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/bind_front.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/host_callback.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt_proxy/common/proto_util.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/python/pjrt_ifrt/xla_host_callback.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace ifrt {
namespace proxy {

RemoteLoadedHostCallbackQueue::~RemoteLoadedHostCallbackQueue() { Close(); }

absl::Status RemoteLoadedHostCallbackQueue::Push(ExecutionRequest request) {
  absl::MutexLock l(&mu_);
  if (closed_) {
    return absl::CancelledError(
        "RemoteLoadedHostCallback has stopped accepting new execution "
        "requests");
  }
  requests_.push_back(std::move(request));
  return absl::OkStatus();
}

std::optional<RemoteLoadedHostCallbackQueue::ExecutionRequest>
RemoteLoadedHostCallbackQueue::Pop() {
  auto not_empty = [this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return !requests_.empty() || closed_;
  };
  absl::MutexLock l(&mu_, absl::Condition(&not_empty));
  if (closed_) {
    return std::nullopt;
  }
  ExecutionRequest request = std::move(requests_.front());
  requests_.pop_front();
  return request;
}

void RemoteLoadedHostCallbackQueue::Close() {
  std::deque<ExecutionRequest> requests;
  {
    absl::MutexLock l(&mu_);
    if (!closed_) {
      requests.swap(requests_);
    }
    closed_ = true;
  }
  for (auto& request : requests) {
    request.status.Set(absl::CancelledError(
        "RemoteLoadedHostCallback execution has been cancelled"));
  }
}

absl::StatusOr<tsl::RCReference<RemoteLoadedHostCallback>>
RemoteLoadedHostCallback::CreateFromSerialized(
    xla::ifrt::Client* client, absl::string_view serialized,
    std::shared_ptr<RemoteLoadedHostCallbackQueue> queue) {
  xla::ifrt::XlaHostCallbackProto proto;
  if (!proto.ParseFromString(AsProtoStringData(serialized))) {
    return absl::DataLossError(
        "Unable to deserialize RemoteLoadedHostCallback");
  }

  auto from_proto =
      [](const auto& arg_protos) -> std::vector<xla::HostCallbackArgInfo> {
    std::vector<xla::HostCallbackArgInfo> args;
    args.reserve(arg_protos.size());
    for (const xla::ifrt::XlaHostCallbackProto::ArgInfo& arg_proto :
         arg_protos) {
      xla::HostCallbackArgInfo& arg = args.emplace_back();
      arg.channel_id = static_cast<uint16_t>(arg_proto.channel_id());
      arg.shape = xla::Shape(arg_proto.shape());
    }
    return args;
  };

  return tsl::MakeRef<RemoteLoadedHostCallback>(
      client, from_proto(proto.operands()), from_proto(proto.results()),
      std::move(queue));
}

RemoteLoadedHostCallback::RemoteLoadedHostCallback(
    xla::ifrt::Client* client, std::vector<xla::HostCallbackArgInfo> operands,
    std::vector<xla::HostCallbackArgInfo> results,
    std::shared_ptr<RemoteLoadedHostCallbackQueue> queue)
    : llvm::RTTIExtends<RemoteLoadedHostCallback,
                        PjRtHostSendAndRecvLoadedHostCallback>(
          client,
          [&]() {
            auto xla_host_callback = std::make_unique<xla::HostCallback>();
            xla_host_callback->operands = std::move(operands);
            xla_host_callback->results = std::move(results);
            xla_host_callback->callback =
                absl::bind_front(&RemoteLoadedHostCallback::Execute, this);
            return xla_host_callback;
          }()),
      queue_(std::move(queue)) {}

RemoteLoadedHostCallback::~RemoteLoadedHostCallback() {
  if (queue_ != nullptr) {
    queue_->Close();
  }
}

absl::Status RemoteLoadedHostCallback::Execute(void** result_ptrs,
                                               void** operand_ptrs) {
  if (queue_ == nullptr) {
    return absl::FailedPreconditionError(
        "RemoteLoadedHostCallback without queue cannot be executed");
  }

  RemoteLoadedHostCallbackQueue::ExecutionRequest request;

  auto to_buffer =
      [&](absl::Span<const xla::HostCallbackArgInfo> args, void** ptrs,
          std::vector<RemoteLoadedHostCallbackQueue::Buffer>& buffers) {
        buffers.reserve(args.size());
        for (int i = 0; i < args.size(); ++i) {
          const int64_t size = xla::ShapeUtil::ByteSizeOf(args[i].shape);
          buffers.push_back(
              RemoteLoadedHostCallbackQueue::Buffer{ptrs[i], size});
        }
      };
  to_buffer(host_callback().operands, operand_ptrs, request.operands);
  to_buffer(host_callback().results, result_ptrs, request.results);

  request.status = Future<>::CreatePromise();
  Future<> status(request.status);

  // Enqueue the execution request. `IfrtBackend` retrieves this by calling
  // `PopExecutionRequest` and fulfills the `results` promise.
  TF_RETURN_IF_ERROR(queue_->Push(std::move(request)));

  // Block until the execution finishes and return its status.
  return status.Await();
}

absl::StatusOr<std::string> RemoteLoadedHostCallback::Serialize() const {
  xla::ifrt::XlaHostCallbackProto proto;

  auto to_proto = [](absl::Span<const xla::HostCallbackArgInfo> args,
                     auto* args_proto) {
    args_proto->Reserve(args.size());
    for (const auto& arg : args) {
      auto* arg_proto = args_proto->Add();
      arg_proto->set_channel_id(arg.channel_id);
      *arg_proto->mutable_shape() = arg.shape.ToProto();
    }
  };
  to_proto(host_callback().operands, proto.mutable_operands());
  to_proto(host_callback().results, proto.mutable_results());

  return proto.SerializeAsString();
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
