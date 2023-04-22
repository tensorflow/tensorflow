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

#include <memory>

#include "grpcpp/channel.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.grpc.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/env.h"

namespace xla {

class DistributedRuntimeClient {
 public:
  struct Options {
    // This node's global ID. Required.
    int32 node_id = -1;

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
  DistributedRuntimeClient(std::shared_ptr<::grpc::Channel> channel,
                           const Options& options);
  explicit DistributedRuntimeClient(std::shared_ptr<::grpc::Channel> channel)
      : DistributedRuntimeClient(channel, Options()) {}
  ~DistributedRuntimeClient();

  // Connects to the master, and blocks until all clients have successfully
  // connected.
  // Not thread-safe, i.e., calls to Connect()/Shutdown()/EnumerateDevices()
  // must be serialized by some other means.
  xla::Status Connect();

  // Reports to the master that the client is ready to shutdown, and blocks
  // until all clients are ready to shutdown or the shutdown timeout expires.
  // Not thread-safe.
  xla::Status Shutdown();

  // Blocking enumeration of global devices. Used by the GPU platform.
  // Not thread-safe.
  xla::Status EnumerateDevices(const LocalTopologyProto& local_topology,
                               GlobalTopologyProto* global_topology);

  // The following APIs are thread-safe.
  xla::StatusOr<std::string> BlockingKeyValueGet(std::string key,
                                                 absl::Duration timeout);

  xla::Status KeyValueSet(std::string key, std::string value);

 private:
  // Entry point for the heartbeat thread.
  void HeartbeatLoop();

  const std::unique_ptr<grpc::DistributedRuntimeService::Stub> stub_;
  const Options options_;

  // Possible states of the client.
  // The only legal transitions are downwards in the order below. i.e., there is
  // no way to reopen a closed client.
  enum class State {
    // The client has not yet connected to the server, i.e., had a Connect()
    // RPC succeed.
    kNotConnected,

    // The client is connected to the server and as far as we are aware the
    // connection is healthy.
    kConnected,

    // The client is in the process of shutting down, i.e., Shutdown() has been
    // called.
    kShuttingDown,

    // The client has shut down its server connection, either due to an error
    // or due to an explicit shutdown.
    kClosed,
  };

  static absl::string_view StateToString(State state);

  // state_ is protected by a mutex because the heartbeat thread needs to look
  // at it.
  absl::Mutex mu_;
  State state_ ABSL_GUARDED_BY(mu_) = State::kNotConnected;

  // A unique session ID, assigned by the server during Connect().
  uint64 session_id_;

  // Notification that tells the heartbeat thread to stop running.
  absl::Notification stop_heartbeats_;

  // Thread responsible for performing heartbeats.
  std::unique_ptr<tensorflow::Thread> heartbeat_thread_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_CLIENT_H_
