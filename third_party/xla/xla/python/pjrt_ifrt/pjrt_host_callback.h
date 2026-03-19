/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_PJRT_IFRT_PJRT_HOST_CALLBACK_H_
#define XLA_PYTHON_PJRT_IFRT_PJRT_HOST_CALLBACK_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/host_callback.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// Wrapper of a PjRt `xla::HostCallback` that uses XLA host send and recv. This
// object is expected to be passed to the compiler when creating
// `xla::ifrt::PjRtLoadedExecutable`.
//
// `PjRtLoadedHostCallback` does not support serialization by default, but it
// may be implemented by subclassing it.
//
// TODO(hyeontaek): Update the comment (compiler to client) after splitting
// compilation and loading.
class PjRtHostSendAndRecvLoadedHostCallback
    : public llvm::RTTIExtends<PjRtHostSendAndRecvLoadedHostCallback,
                               LoadedHostCallback> {
 public:
  PjRtHostSendAndRecvLoadedHostCallback(
      Client* client, std::unique_ptr<xla::HostCallback> host_callback)
      : client_(client), host_callback_(std::move(host_callback)) {}

  const xla::HostCallback& host_callback() const { return *host_callback_; }

  // LoadedHostCallback implementation.

  ~PjRtHostSendAndRecvLoadedHostCallback() override = default;

  Client* client() const override { return client_; }

  absl::StatusOr<std::string> Serialize() const override;

  static char ID;  // NOLINT

 private:
  template <typename T, typename... Args>
  friend tsl::RCReference<T> tsl::MakeRef(Args&&... args);

  Client* client_;
  std::unique_ptr<xla::HostCallback> host_callback_;
};

// Wrapper of an opaque callable that is loaded into FFI's ExecutionContext
// during execution.
class PjRtFfiLoadedHostCallback
    : public llvm::RTTIExtends<PjRtFfiLoadedHostCallback, LoadedHostCallback> {
 public:
  explicit PjRtFfiLoadedHostCallback(Client* client, void* callable)
      : client_(client), callable_(callable) {}

  ~PjRtFfiLoadedHostCallback() override = default;

  Client* client() const override { return client_; }

  void* callable() const { return callable_; };

  absl::StatusOr<std::string> Serialize() const override;

  static char ID;  // NOLINT

 private:
  Client* client_;
  void* callable_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_HOST_CALLBACK_H_
