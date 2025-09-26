/* Copyright 2020 Google LLC

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

#ifndef XLA_PJRT_DISTRIBUTED_CLIENT_H_
#define XLA_PJRT_DISTRIBUTED_CLIENT_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "tsl/platform/env.h"

namespace tsl {
class CoordinationServiceAgent;
}  // namespace tsl

namespace xla {

class DistributedRuntimeClient {
 public:
  struct Options {
    // This node's global ID. Required.
    int32_t node_id = -1;

    // Environment used for starting threads.
    tsl::Env* env = tsl::Env::Default();

    // RPC timeout used for RPC that don't have their own timeouts.
    absl::Duration rpc_timeout = absl::Seconds(120);

    // Time period for which Connect() should be retried. The client will keep
    // trying to open the initial connection for this period, even if any
    // individual Connect() RPC fails. May be zero, in which case Connect() will
    // only be attempted once.
    absl::Duration init_timeout = absl::ZeroDuration();

    // How long to wait for all nodes to call Shutdown(). If the timeout
    // expires, then shutdown() reports an error and returns control.
    absl::Duration shutdown_timeout = absl::Minutes(5);

    // The duration after which the service concludes a client has vanished if
    // it hasn't received any heartbeats from the client.
    absl::Duration heartbeat_timeout = absl::Seconds(100);

    // Callback invoked by the client when notification of a missing heartbeat
    // is reported by the coordinator, or we have not heard from the coordinator
    // recently. `coordinator_reported_failure` is true in the former case.
    // Exposed so tests can override this behavior to something non-fatal.
    std::function<void(absl::Status)> missed_heartbeat_callback =
        [](const absl::Status& status) {
          LOG(QFATAL) << "Terminating process because the JAX distributed "
                         "service detected fatal errors. This most likely "
                         "indicates that another task died; see the other task "
                         "logs for more details. Disable Python buffering, "
                         "i.e. `python -u`, to be sure to see all the "
                         "previous output. "
                         "absl::Status: "
                      << status;
        };

    // For testing. Should the client explicitly Shutdown() on destruction?
    bool shutdown_on_destruction = true;

    // Whether the client should send a request to wait for error from the
    // coordination service at the startup.
    // TODO(b/355706798): eventually remove this option.
    bool poll_for_error_from_service_at_startup = true;

    // If true, a multi-controller JAX job can continue even if this client
    // fails. Otherwise, the job will fail when the task failes.
    bool recoverable = false;
  };

  virtual ~DistributedRuntimeClient() = default;

  // Connects to the master, and blocks until all clients have successfully
  // connected.
  // Not thread-safe, i.e., calls to Connect()/Shutdown() must be serialized by
  // some other means.
  virtual absl::Status Connect() = 0;

  // Reports to the master that the client is ready to shutdown, and blocks
  // until all clients are ready to shutdown or the shutdown timeout expires.
  // Not thread-safe.
  virtual absl::Status Shutdown() = 0;

  // The following APIs are thread-safe.

  // Key-value store API.
  // There are no concurrency guarantees. To avoid a race / impose an ordering
  // on potentially concurrent ops (e.g. set, delete), use WaitAtBarrier().
  virtual absl::StatusOr<std::string> BlockingKeyValueGet(
      absl::string_view key, absl::Duration timeout) = 0;

  // Returns `NotFoundError` immediately if the key is not found.
  virtual absl::StatusOr<std::string> KeyValueTryGet(absl::string_view key) = 0;

  // Get all key-value pairs under a directory (key).
  // A value is considered to be in the directory if its key is prefixed with
  // the directory.
  // This is not a blocking call. If no keys are found, an empty vector is
  // returned immediately.
  virtual absl::StatusOr<std::vector<std::pair<std::string, std::string>>>
  KeyValueDirGet(absl::string_view key) = 0;

  virtual absl::Status KeyValueSet(absl::string_view key,
                                   absl::string_view value) = 0;
  virtual absl::Status KeyValueSet(absl::string_view key,
                                   absl::string_view value,
                                   bool allow_overwrite) = 0;

  // Delete the key-value. If the key is a directory, recursively clean
  // up all key-values under the directory.
  virtual absl::Status KeyValueDelete(absl::string_view key) = 0;

  // Blocks until all nodes (or the ones specified in `nodes`) are at the
  // barrier or the barrier times out. `barrier_id` should be unique across
  // barriers.
  virtual absl::Status WaitAtBarrier(
      std::string barrier_id, absl::Duration timeout,
      std::optional<absl::Span<const int32_t>> nodes) = 0;

  // Returns the subset of live nodes. See CoordinationService.GetAliveTasks for
  // detailed semantics.
  virtual absl::StatusOr<std::vector<int32_t>> GetLiveNodes(
      absl::Span<const int32_t> nodes) = 0;

  // Returns pointer to coordination service agent, or InternalError if the
  // client does not use coordination service.
  virtual absl::StatusOr<tsl::CoordinationServiceAgent*>
  GetCoordinationServiceAgent() = 0;
};

// Creates a distributed runtime client.
std::unique_ptr<DistributedRuntimeClient> GetDistributedRuntimeClient(
    std::shared_ptr<::grpc::Channel> channel,
    const DistributedRuntimeClient::Options& options);

std::shared_ptr<KeyValueStoreInterface> GetDistributedKeyValueStore(
    std::shared_ptr<DistributedRuntimeClient> client, std::string key_prefix);

}  // namespace xla

#endif  // XLA_PJRT_DISTRIBUTED_CLIENT_H_
