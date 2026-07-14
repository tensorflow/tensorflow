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

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_REGISTRY_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_REGISTRY_H_

#include <functional>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"

namespace xla {
namespace ifrt {
namespace proxy {

struct ClientConnectionOptions {
  // Timeout for establishing the connection.
  absl::Duration connection_timeout = absl::Minutes(2);

  // A callback that (if it is not set to nullptr) will be called if there was a
  // successful connection to the proxy server, but there was a later
  // disconnect. The callback may be called synchronously from a thread that
  // performs various important activities, and therefore should not block on
  // any events (or deadlocks may happen).
  std::function<void(absl::Status)> on_disconnect = nullptr;

  // Captures logs related to establishing the connection. Logs may be generated
  // synchronously from a thread that performs various important activities,
  // so the function should not block (or deadlocks may happen).
  std::function<void(absl::string_view)> on_connection_update = nullptr;

  // Runtime specific initialization data.
  AttributeMap initialization_data = AttributeMap({});
};

// Registers a new factory for client backend implementation. Crashes if the
// same backend name is registered more than once.
void RegisterClientFactory(
    absl::string_view transport_name,
    std::function<absl::StatusOr<std::unique_ptr<xla::ifrt::Client>>(
        absl::string_view address, const ClientConnectionOptions& options)>
        factory);

// Creates a client for the given backend target. The backend target string must
// be in the form of `<backend-type>:<backend-address>`.
absl::StatusOr<std::unique_ptr<xla::ifrt::Client>> CreateClient(
    absl::string_view proxy_server_address,
    const ClientConnectionOptions& options = ClientConnectionOptions());

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_REGISTRY_H_
