/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/runtime/stream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/utility/utility.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/tsl/platform/random.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"

namespace tensorflow {
namespace tfrt_stub {

absl::StatusOr<std::optional<StreamCallbackId>> CreateStreamCallbackId(
    absl::string_view model_name, mlir::ModuleOp module) {
  mlir::Builder builder(module.getContext());

  // Inject information about the callback to `tf.PwStreamResults` ops. The
  // attribute names must match `PwStreamResult` op's implementation.

  std::vector<mlir::TF::PwStreamResultsOp> ops;
  module->walk([&](mlir::TF::PwStreamResultsOp op) { ops.push_back(op); });

  if (ops.empty()) {
    return std::nullopt;
  }

  auto& stream_interface = GetGlobalStreamCallbackRegistry().stream_interface();

  auto controller_address = stream_interface.controller_address();
  auto controller_address_attr = builder.getStringAttr(controller_address);

  auto model_name_attr = builder.getStringAttr(model_name);

  // We use int64_t instead of uint64_t returned by `New64()` because
  // TensorFlow doesn't support uint64 attributes.
  const StreamCallbackId callback_id(
      static_cast<int64_t>(tsl::random::New64()));
  auto callback_id_attr = builder.getI64IntegerAttr(callback_id.id);

  for (auto op : ops) {
    op->setAttr("_controller_address", controller_address_attr);
    op->setAttr("_model_name", model_name_attr);
    op->setAttr("_callback_id", callback_id_attr);
  }

  return callback_id;
}

absl::StatusOr<ScopedStreamCallback> StreamCallbackRegistry::Register(
    absl::string_view model_name, StreamCallbackId callback_id, StepId step_id,
    absl::AnyInvocable<
        void(absl::flat_hash_map<std::string, tensorflow::Tensor>)>
        callback) {
  absl::MutexLock l(&mu_);

  const auto [it, inserted] =
      stream_callbacks_.insert({std::make_pair(callback_id, step_id), nullptr});
  if (!inserted) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Stream callback ", callback_id, " @ ", step_id, " already exists"));
  }

  it->second = std::make_unique<CallbackState>();
  it->second->thread = absl::WrapUnique(tsl::Env::Default()->StartThread(
      tensorflow::ThreadOptions(),
      /*name=*/absl::StrCat("stream_handler_", callback_id, "_", step_id),
      [model_name = std::string(model_name), callback_id, step_id,
       callback = std::move(callback), state = it->second.get(),
       this]() mutable {
        StreamedResult result;
        while (state->channel.Read(result)) {
          absl::Duration dequeue_latency = absl::Now() - result.enqueued_time;
          interface_->RecordDequeueLatency(model_name, dequeue_latency);

          tsl::profiler::TraceMe trace_me("StreamCallbackInvocation");
          trace_me.AppendMetadata([&]() {
            return tsl::profiler::TraceMeEncode({
                {"callback_id", callback_id.id},
                {"step_id", step_id.id},
            });
          });

          absl::Time start_time = absl::Now();
          callback(std::move(result.tensors));
          interface_->RecordCallbackLatency(model_name,
                                            absl::Now() - start_time);
        }
      }));

  return ScopedStreamCallback(this, callback_id, step_id);
}

absl::Status StreamCallbackRegistry::Write(StreamCallbackId callback_id,
                                           StepId step_id,
                                           StreamedResult result) {
  absl::MutexLock lock(&mu_);
  auto iter = stream_callbacks_.find({callback_id, step_id});
  if (iter == stream_callbacks_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Stream callback ", callback_id, " @ ", step_id,
        " does not exist; this usually indicates that a streaming signature "
        "was called by a non-streaming request"));
  }

  auto* state = iter->second.get();
  DCHECK(state);
  return state->channel.Write(std::move(result));
}

std::unique_ptr<StreamCallbackRegistry::CallbackState>
StreamCallbackRegistry::Unregister(StreamCallbackId callback_id,
                                   StepId step_id) {
  absl::MutexLock l(&mu_);
  const auto it = stream_callbacks_.find({callback_id, step_id});
  if (it == stream_callbacks_.end()) {
    return nullptr;
  }
  auto state = std::move(it->second);
  stream_callbacks_.erase(it);
  return state;
}

ScopedStreamCallback::ScopedStreamCallback(ScopedStreamCallback&& other)
    : registry_(other.registry_),
      callback_id_(other.callback_id_),
      step_id_(other.step_id_) {
  other.callback_id_ = std::nullopt;
  other.step_id_ = StepId::GetInvalidStepId();
}

ScopedStreamCallback& ScopedStreamCallback::operator=(
    ScopedStreamCallback&& other) {
  Unregister();

  registry_ = other.registry_;
  callback_id_ = other.callback_id_;
  step_id_ = other.step_id_;
  other.callback_id_ = std::nullopt;
  other.step_id_ = StepId::GetInvalidStepId();

  return *this;
}

void ScopedStreamCallback::Unregister() {
  if (!callback_id_.has_value()) {
    return;
  }

  tsl::profiler::TraceMe trace_me("ScopedStreamCallback::Unregister");
  trace_me.AppendMetadata([&]() {
    return tsl::profiler::TraceMeEncode({
        {"callback_id", callback_id_->id},
        {"step_id", step_id_.id},
    });
  });

  DCHECK(registry_);
  auto state = registry_->Unregister(*callback_id_, step_id_);
  DCHECK(state);

  // At this point, it is safe to close the channel.
  state->channel.Close();

  // Wait until the stream handler finishes.
  state->thread.reset();

  callback_id_.reset();
}

StreamInterfaceFactory& GetGlobalStreamInterfaceFactory() {
  static auto* stream_interface_factory = new StreamInterfaceFactory;
  return *stream_interface_factory;
}

StreamCallbackRegistry& GetGlobalStreamCallbackRegistry() {
  static auto* stream_callback_registry =
      new StreamCallbackRegistry(GetGlobalStreamInterfaceFactory()
                                     .CreateControllerStreamInterface()
                                     .value());
  return *stream_callback_registry;
}

}  // namespace tfrt_stub
}  // namespace tensorflow
