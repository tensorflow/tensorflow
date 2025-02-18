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

#ifndef XLA_PYTHON_IFRT_PROXY_SERVER_HOST_CALLBACK_H_
#define XLA_PYTHON_IFRT_PROXY_SERVER_HOST_CALLBACK_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/host_callback.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {
namespace proxy {

// Command queue interface between `RemoteLoadedHostCallback` and `IfrtBackend`.
// Responsible for keeping track of in-flight execution requests.
class RemoteLoadedHostCallbackQueue {
 public:
  struct Buffer {
    void* data;
    int64_t size;
  };

  // Encapsulates a host buffer execution. Operand and result buffers are
  // pre-allocated and the caller is expected to fill them in-place before
  // fulfilling the `status` promise.
  struct ExecutionRequest {
    std::vector<Buffer> operands;
    std::vector<Buffer> results;
    Future<>::Promise status;
  };

  ~RemoteLoadedHostCallbackQueue();

  // Pushes a new execution request to the queue. Returns an error if the queue
  // has already been closed.
  absl::Status Push(ExecutionRequest request);

  // Blocks until this host callback queue has at least one pending execution
  // and returns its information needed to perform execution. Returns nullopt if
  // the request queue has already been closed by `Close()`.
  std::optional<ExecutionRequest> Pop();

  // Closes this request queue. After this call, all pending executions are
  // unblocked with an error and no more executions can be enqueued.
  void Close();

 private:
  absl::Mutex mu_;
  bool closed_ ABSL_GUARDED_BY(mu_) = false;
  std::deque<ExecutionRequest> requests_ ABSL_GUARDED_BY(mu_);
};

// Host callback that delegates its execution to an external executor. The
// executor waits for execution requests to be enqueued to the given
// `RemoteLoadedHostCallbackQueue` and returns results after execution by
// fulfilling the returned promise.
//
// This class is thread-safe.
//
// Note: The current implementation inherits from PjRt's host callback
// implementation. Even though this is a violation of the IFRT proxy's layering
// principle, it is unavoidable right now because the base `LoadedHostCallback`
// in IFRT has no associated execution semantics. For now, the IFRT proxy
// focuses on supporting host callbacks on PjRt-like IFRT implementations.
class RemoteLoadedHostCallback
    : public llvm::RTTIExtends<RemoteLoadedHostCallback,
                               PjRtHostSendAndRecvLoadedHostCallback> {
 public:
  // Creates from a serialized string returned by `Serialize()`.
  static absl::StatusOr<tsl::RCReference<RemoteLoadedHostCallback>>
  CreateFromSerialized(xla::ifrt::Client* client, absl::string_view serialized,
                       std::shared_ptr<RemoteLoadedHostCallbackQueue> queue);

  // Create from operand/result specs.
  RemoteLoadedHostCallback(
      xla::ifrt::Client* client, std::vector<xla::HostCallbackArgInfo> operands,
      std::vector<xla::HostCallbackArgInfo> results,
      std::shared_ptr<RemoteLoadedHostCallbackQueue> queue);

  ~RemoteLoadedHostCallback() override;

  // Serializes the remote host callback instance. The returned string can be
  // deserialized into `RmeoteLoadedHostCallback` using `CreateFromSerialized`.
  absl::StatusOr<std::string> Serialize() const override;

 private:
  // Implements the interface required by `xla::HostCallback`.
  absl::Status Execute(void** result_ptrs, void** operand_ptrs);

  std::shared_ptr<RemoteLoadedHostCallbackQueue> queue_;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_SERVER_HOST_CALLBACK_H_
