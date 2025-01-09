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

#ifndef XLA_PYTHON_IFRT_PROXY_SERVER_HOST_BUFFER_H_
#define XLA_PYTHON_IFRT_PROXY_SERVER_HOST_BUFFER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace xla {
namespace ifrt {
namespace proxy {

// Keeps host buffers transferred from the client so that `IfrtBackend` can
// access them when requests with pointers to host buffers arrive.
//
// We expect one `HostBufferStore` to exist per session (i.e., per `IfrtBackend`
// instance) so that host buffers are cleaned up on session termination.
class HostBufferStore {
 public:
  // Stores the data associated with the given handle. Returns an error if the
  // handle already exists.
  absl::Status Store(uint64_t handle, std::string data);

  // Retrieves the data associated with the handle. Returns an error if the
  // handle does not exist within the given timeout or if `Shutdown()` is
  // called.
  absl::StatusOr<std::shared_ptr<const std::string>> Lookup(
      uint64_t handle, absl::Duration timeout = absl::ZeroDuration());

  // Deletes the host buffer associated with the handle. Returns an error if the
  // handle does not exist.
  absl::Status Delete(uint64_t handle);

  // Deletes all handles and permanently prevents addition of any new handles.
  void Shutdown(std::string reason);

  ~HostBufferStore() { Shutdown("HostBufferStore is being destroyed"); }

 private:
  absl::Mutex mu_;
  absl::flat_hash_map<uint64_t, std::shared_ptr<const std::string>> buffers_
      ABSL_GUARDED_BY(mu_);
  std::optional<std::string> shutdown_msg_ ABSL_GUARDED_BY(mu_);
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_SERVER_HOST_BUFFER_H_
