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

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_HOST_BUFFER_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_HOST_BUFFER_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/future.h"

namespace xla {
namespace ifrt {
namespace proxy {

class ClientHostBufferStore {
 public:
  virtual ~ClientHostBufferStore() = default;

  // Stores the data associated with the given handle. Returns an error if the
  // handle already exists.
  virtual Future<> Store(uint64_t handle, absl::string_view data) = 0;

  // Stores the data associated with the given handle. Returns an error if the
  // handle already exists.
  // TODO(b/315023499) Find a way to increase the chunk size
  virtual Future<> Store(uint64_t handle, const absl::Cord& data) = 0;

  // Retrieves the data associated with the handle. Returns an error if the
  // handle does not exist.
  virtual Future<absl::Cord> Lookup(uint64_t handle) = 0;

  // Deletes the host buffer associated with the handle. Returns an error if the
  // handle does not exist.
  virtual Future<> Delete(uint64_t handle) = 0;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_HOST_BUFFER_H_
