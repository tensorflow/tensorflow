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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_CLIENT_H_

#include <functional>
#include <memory>
#include <string>

#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.grpc.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/env.h"

namespace xla {

class DistributedRuntimeClient {
 public:
  struct Options {
    // This node's global ID. Required.
    int32_t node_id = -1;

    // Environment used for starting threads.
    tensorflow::Env* env = tensorflow::Env::Default();

    // RPC timeout used for RPC that don't have their own timeouts.
    absl::Duration rpc_timeout = absl::Seconds(120);

    // Time period for which Connect() should be retried. The client will keep
    // trying to open the initial connection for this period, even if any
    // individual Connect() RPC fails. May be zero, in which case Connect() will
    // only be attempted once.
    absl::Duration init_timeout = absl::ZeroDuration();

    // How long to wait for all nodes to call Shutdown(). If the timeout
    // expires, then shutdown() reports an error and returns control.
    absl::Duration shutdown_timeout = absl::Seconds(60);

    // Interval at which the client should send heartbeat RPCs to the
    // coordinator.
    absl::Duration heartbeat_interval = absl::Seconds(10);

    // How many failed heartbeat RPCs may fail due to a possibly-ephemeral
    // reason before we decide the coordinator has vanished and that we should
    // shut down.
    int max_missing_heartbeats = 10;

    // Callback invoked by the client when notification of a missing heartbeat
    // is reported by the coordinator, or we have not heard from the coordinator
    // recently. `coordinator_reported_failure` is true in the former case.
    // Exposed so tests can override this behavior to something non-fatal.
    std::function<void(xla::Status, bool coordinator_reported_failure)>
        missed_heartbeat_callback =
            [](xla::Status status, bool coordinator_reported_failure) {
              if (coordinator_reported_failure) {
                LOG(QFATAL)
                    << "Terminating process because the coordinator detected "
                       "missing heartbeats. This most likely indicates that "
                       "another task died; see the other task logs for more "
                       "details. Status: "
                    << status;
              } else {
                LOG(QFATAL)
                    << "Terminating process because of missing heartbeat "
                       "response from the coordinator. This most likely "
                       "indicates that the coordinator task died; see the "
                       "coordinator's task logs for more details. Status: "
                    << status;
              }
            };

    // For testing. Should the client explicitly Shutdown() on destruction?
    bool shutdown_on_destruction = true;
  };

  virtual ~DistributedRuntimeClient() {}

  // Connects to the master, and blocks until all clients have successfully
  // connected.
  // Not thread-safe, i.e., calls to Connect()/Shutdown()/EnumerateDevices()
  // must be serialized by some other means.
  virtual xla::Status Connect() = 0;

  // Reports to the master that the client is ready to shutdown, and blocks
  // until all clients are ready to shutdown or the shutdown timeout expires.
  // Not thread-safe.
  virtual xla::Status Shutdown() = 0;

  // Blocking enumeration of global devices. Used by the GPU platform.
  // Not thread-safe.
  virtual xla::Status EnumerateDevices(
      const LocalTopologyProto& local_topology,
      GlobalTopologyProto* global_topology) = 0;

  // The following APIs are thread-safe.
  virtual xla::StatusOr<std::string> BlockingKeyValueGet(
      std::string key, absl::Duration timeout) = 0;

  virtual xla::Status KeyValueSet(std::string key, std::string value) = 0;

  // The following APIs are not implemented by default.
  // TODO(b/233099439): Implement in default distributed runtime service.
  virtual xla::Status KeyValueDelete(std::string key) = 0;

  // Blocks until all nodes are at the barrier or the barrier times out.
  // `barrier_id` should be unique across barriers. Once the barrier has passed
  // or failed, subsequent calls will not block, and immediately respond with
  // the previous result.
  // TODO(b/233099439): Implement in default distributed runtime service.
  virtual xla::Status WaitAtBarrier(std::string barrier_id,
                                    absl::Duration timeout) = 0;
};

// Creates a distributed runtime client.
std::unique_ptr<DistributedRuntimeClient> GetDistributedRuntimeClient(
    std::shared_ptr<::grpc::Channel> channel,
    const DistributedRuntimeClient::Options& options,
    bool use_coordination_service);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_CLIENT_H_
