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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_SEND_RECV_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_SEND_RECV_H_

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/stream_executor/event.h"

namespace xla {
namespace gpu {

// Registers XLA Gpu runtime Send/Recv custom calls.
void RegisterSendRecvCustomCalls(runtime::DirectCustomCallRegistry& registry);

// Register type names for communication attributes defined by MHLO dialect.
void RegisterSendRecvTypeIdNames(runtime::TypeIDNameRegistry& registry);

// Adds attributes encoding for Send/Recv custom calls
void PopulateSendRecvAttrEncoding(runtime::CustomCallAttrEncodingSet& encoding);

//===----------------------------------------------------------------------===//
// Support for running asynchronous Send/Recv SendDone/RecvDone operations.
//===----------------------------------------------------------------------===//

class SendRecvEvents {
 public:
  absl::Status PushEvent(int32_t handle, tsl::AsyncValueRef<se::Event> event);
  absl::StatusOr<tsl::AsyncValueRef<se::Event>> PopEvent(int32_t handle);

 private:
  absl::Mutex mutex_;
  absl::flat_hash_map<int, tsl::AsyncValueRef<se::Event>> events_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_SEND_RECV_H_
