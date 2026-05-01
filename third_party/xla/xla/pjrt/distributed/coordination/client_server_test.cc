/* Copyright 2020 The OpenXLA Authors.

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

#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/barrier.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "xla/pjrt/distributed/coordination/coordination_client.h"
#include "xla/pjrt/distributed/coordination/coordination_service.h"
#include "xla/pjrt/distributed/coordination/coordination_service.pb.h"
#include "xla/pjrt/distributed/coordination/coordination_service_agent.h"
#include "xla/pjrt/distributed/coordination/grpc_coordination_client.h"
#include "xla/pjrt/distributed/coordination/grpc_coordination_service_impl.h"
#include "xla/tsl/distributed_runtime/rpc/async_service_interface.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"

namespace xla {
namespace {
using ::testing::AnyOf;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

constexpr absl::Duration kBarrierTimeout = absl::Milliseconds(200);
constexpr absl::Duration kHeartbeatTimeout = absl::Seconds(3);
constexpr char kBarrierId[] = "barrier_id";

// Note: b/169705709: no protobuf matchers in OSS.
MATCHER_P2(IsKvEntry, key, value, "") {
  return key == arg.key() && value == arg.value();
}

std::string DebugString(const CoordinationService::Config& config) {
  return absl::StrFormat(
      "CoordinationService::Config {\n"
      "  cluster_register_timeout: %s\n"
      "  cluster_register_with_barrier: %v\n"
      "  heartbeat_timeout: %s\n"
      "  num_tasks: %d\n"
      "  shutdown_barrier_timeout: %s\n"
      "  recoverable: %v\n"
      "}",
      absl::FormatDuration(config.cluster_register_timeout),
      config.cluster_register_with_barrier,
      absl::FormatDuration(config.heartbeat_timeout), config.num_tasks,
      absl::FormatDuration(config.shutdown_barrier_timeout),
      config.recoverable);
}

std::string DebugString(const CoordinationServiceAgent::Config& config) {
  return absl::StrFormat(
      "CoordinationServiceAgent::Config {\n"
      "  cluster_register_timeout: %s\n"
      "  heartbeat_timeout: %s\n"
      "  shutdown_barrier_timeout: %s\n"
      "  agent_destruction_without_shutdown: %v\n"
      "  poll_for_error_from_service_at_startup: %v\n"
      "}",
      absl::FormatDuration(config.cluster_register_timeout),
      absl::FormatDuration(config.heartbeat_timeout),
      absl::FormatDuration(config.shutdown_barrier_timeout),
      config.agent_destruction_without_shutdown,
      config.poll_for_error_from_service_at_startup);
}

class ClientServerTest : public ::testing::Test {
 public:
  std::unique_ptr<CoordinationServiceAgent> GetClient(
      int node_id,
      CoordinationServiceAgent::Config config = DefaultAgentConfig(),
      tsl::StatusCallback error_fn = [](const absl::Status& status) {
        LOG(ERROR) << "Agent hit an error: " << status;
      }) {
    VLOG(1) << "Getting client " << node_id << " with config:\n"
            << DebugString(config);

    CHECK(server_ != nullptr);
    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
        service_address_, grpc::InsecureChannelCredentials());

    std::unique_ptr<CoordinationClient> leader_client;
    leader_client.reset(NewGrpcCoordinationClient(channel));

    auto coord_agent = CoordinationServiceAgent::Create(
        tsl::Env::Default(), node_id, config, std::move(leader_client),
        std::move(error_fn));
    if (!coord_agent.ok()) {
      LOG(ERROR) << "Coordination agent failed to initialize: "
                 << coord_agent.status();
      return nullptr;
    }
    return *std::move(coord_agent);
  }

  void StartService(CoordinationService::Config config = DefaultServiceConfig(),
                    int port = -1) {
    VLOG(1) << "Starting service with config:\n" << DebugString(config);

    if (port == -1) {
      port = tsl::testing::PickUnusedPortOrDie();
    }
    grpc::ServerBuilder builder;
    service_address_ = absl::StrCat("[::]:", port);
    builder.AddListeningPort(service_address_,
                             grpc::InsecureServerCredentials());
    // Set up the actual coordination service (where all the real logic
    // lives).
    coord_service_ =
        std::make_unique<CoordinationService>(tsl::Env::Default(), config);
    // Set up threads and RPC service.
    coord_compute_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "CoordinationServiceRpcHandler",
        /*num_threads=*/4);
    coord_rpc_service_ = std::make_unique<GrpcCoordinationServiceImpl>(
        coord_compute_pool_.get(), &builder);
    auto* grpc_coord_service =
        static_cast<GrpcCoordinationServiceImpl*>(coord_rpc_service_.get());
    grpc_coord_service->SetCoordinationServiceInstance(coord_service_.get());
    // Start the server.
    server_ = builder.BuildAndStart();
    // Only start RPC loop after the service is live.
    coord_rpc_thread_.reset(tsl::Env::Default()->StartThread(
        tsl::ThreadOptions(), "CoordinationServiceHandleRPCsLoop",
        [service = coord_rpc_service_.get()] { service->HandleRPCsLoop(); }));
  }

  void StopService() {
    if (!server_) {
      // Service has already stopped.
      return;
    }
    server_->Shutdown(absl::ToChronoTime(absl::Now() + absl::Seconds(5)));
    server_->Wait();

    // Service object must be destroyed to clear all pending RPCs before
    // shutting down the RPC service and the server.
    coord_service_ = nullptr;

    // Shut down all the service objects.
    static_cast<GrpcCoordinationServiceImpl*>(coord_rpc_service_.get())
        ->SetCoordinationServiceInstance(nullptr);
    coord_rpc_service_->Shutdown();
    coord_rpc_thread_ = nullptr;
    coord_rpc_service_ = nullptr;
    coord_compute_pool_ = nullptr;

    // Destroy the server.
    server_ = nullptr;
  }

  void TearDown() override { StopService(); }

 protected:
  static CoordinationService::Config DefaultServiceConfig() {
    CoordinationService::Config config;
    config.cluster_register_timeout = absl::Seconds(2);
    config.cluster_register_with_barrier = true;
    config.heartbeat_timeout = kHeartbeatTimeout;
    config.num_tasks = 1;
    config.shutdown_barrier_timeout = absl::Seconds(2);
    config.recoverable = false;
    return config;
  }

  static CoordinationServiceAgent::Config DefaultAgentConfig() {
    CoordinationServiceAgent::Config config;
    config.cluster_register_timeout = absl::Seconds(3);
    config.heartbeat_timeout = kHeartbeatTimeout;
    config.shutdown_barrier_timeout = absl::Seconds(3);
    config.agent_destruction_without_shutdown = false;
    config.poll_for_error_from_service_at_startup = true;
    return config;
  }

  // Runs f(0), f(1), ..., f(num_clients - 1) concurrently on separate threads
  // and returns their statuses.
  std::vector<absl::Status> RunClients(
      int num_clients, absl::AnyInvocable<absl::Status(int node_id)> f) {
    std::vector<absl::Status> statuses(num_clients);
    {
      tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                          num_clients);
      for (int i = 0; i < num_clients; ++i) {
        thread_pool.Schedule([&, i]() { statuses[i] = f(i); });
      }
    }
    return statuses;
  }

 private:
  std::string service_address_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<CoordinationService> coord_service_;
  std::unique_ptr<tsl::thread::ThreadPool> coord_compute_pool_;
  std::unique_ptr<tsl::AsyncServiceInterface> coord_rpc_service_;
  std::unique_ptr<tsl::Thread> coord_rpc_thread_;
};

TEST_F(ClientServerTest, ConnectAndShutdownAreBarriers) {
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);

  absl::Mutex mu;
  int connect_count = 0;
  int shutdown_count = 0;

  absl::Barrier barrier(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);

    // Allow the threads to call Connect one-by-one in order.
    auto my_connect_turn = [&]() {
      mu.AssertHeld();
      return connect_count == node_id;
    };
    {
      absl::MutexLock lock(mu);
      mu.Await(absl::Condition(&my_connect_turn));
      ++connect_count;
    }
    TF_RETURN_IF_ERROR(client->Connect());
    // Verify that all of the threads have called Connect() by the time we get
    // here.
    {
      absl::MutexLock lock(mu);
      if (connect_count != num_nodes) {
        return absl::InternalError(absl::StrCat(
            "Connect count is ", connect_count, " but expected ", num_nodes));
      }
    }

    // Similarly for shutting down.
    auto my_shutdown_turn = [&]() {
      mu.AssertHeld();
      return shutdown_count == node_id;
    };
    {
      absl::MutexLock lock(mu);
      mu.Await(absl::Condition(&my_shutdown_turn));
      ++shutdown_count;
    }
    TF_RETURN_IF_ERROR(client->Shutdown());
    {
      absl::MutexLock lock(mu);
      if (shutdown_count != num_nodes) {
        return absl::InternalError(absl::StrCat(
            "Shutdown count is ", shutdown_count, " but expected ", num_nodes));
      }
    }

    return absl::OkStatus();
  };

  std::vector<absl::Status> statuses = RunClients(num_nodes, thread_fn);
  EXPECT_THAT(statuses, Each(absl_testing::IsOk()));
}

TEST_F(ClientServerTest, ClientsTerminateShutdownIfAnyClientGoesAway) {
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);

  auto thread_fn = [&](int node_id) -> absl::Status {
    CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
    agent_config.agent_destruction_without_shutdown = (node_id == 0);
    auto client = GetClient(node_id, agent_config);

    TF_RETURN_IF_ERROR(client->Connect());

    if (node_id == 0) {
      return absl::OkStatus();
    }

    // The call to Shutdown() should be interrupted if a worker stops issuing
    // heartbeats.
    return client->Shutdown();
  };

  std::vector<absl::Status> statuses = RunClients(num_nodes, thread_fn);
  TF_EXPECT_OK(statuses[0]);
  for (int i = 1; i < num_nodes; ++i) {
    EXPECT_THAT(
        statuses[i],
        AnyOf(
            // Shutdown barrier took too long and failed.
            absl_testing::StatusIs(absl::StatusCode::kInternal,
                                   HasSubstr("timed out")),
            // Heartbeat timeout sets node into error state, failing shutdown
            // barrier.
            absl_testing::StatusIs(absl::StatusCode::kInternal,
                                   HasSubstr("heartbeat timeout")),
            // Agent polled error first, and so Shutdown()
            // fails because agent is already in error.
            absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition)));
  }
}

TEST_F(ClientServerTest, ClientsShutdownSuccessfully) {
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);

    TF_RETURN_IF_ERROR(client->Connect());
    return client->Shutdown();
    // The error polling request will be cancelled automatically when the
    // client is shutting down.
  };

  std::vector<absl::Status> statuses = RunClients(num_nodes, thread_fn);
  EXPECT_THAT(statuses, Each(absl_testing::IsOk()));
}

TEST_F(ClientServerTest, MissedHeartbeatCallbackIsExecutedIfAnyClientGoesAway) {
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);

  auto thread_fn = [&](int node_id) -> absl::Status {
    absl::Notification shutdown;
    CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
    agent_config.agent_destruction_without_shutdown = (node_id == 0);
    auto client =
        GetClient(node_id, agent_config,
                  /*error_fn=*/[&shutdown](const absl::Status& status) {
                    shutdown.Notify();
                  });

    TF_RETURN_IF_ERROR(client->Connect());

    if (node_id == 0) {
      return absl::OkStatus();
    }
    shutdown.WaitForNotification();
    return absl::OkStatus();
  };

  std::vector<absl::Status> statuses = RunClients(num_nodes, thread_fn);
  EXPECT_THAT(statuses, Each(absl_testing::IsOk()));
}

TEST_F(ClientServerTest, ShutdownErrorIsPropagatedToClients) {
  int num_nodes = 2;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);
  std::vector<absl::Status> statuses = {
      absl::UnknownError("Uninitialized status."),
      absl::UnknownError("Uninitialized status.")};
  absl::Notification shutdown;

  auto thread_fn = [&](int node_id) -> absl::Status {
    CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
    agent_config.cluster_register_timeout = absl::Seconds(2);
    agent_config.shutdown_barrier_timeout = absl::Seconds(2);
    auto client = GetClient(
        node_id, agent_config,
        /*error_fn=*/[&statuses, node_id](const absl::Status& status) {
          statuses[node_id] = status;
        });

    TF_RETURN_IF_ERROR(client->Connect());

    if (node_id == 0) {
      // Shut down early.
      client = nullptr;
      shutdown.Notify();
    } else {
      // Block until shutdown barrier times out.
      shutdown.WaitForNotification();
    }
    return absl::OkStatus();
  };

  RunClients(num_nodes, thread_fn);
  EXPECT_THAT(statuses[0], absl_testing::StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(statuses[1], absl_testing::StatusIs(absl::StatusCode::kInternal));
}

TEST_F(ClientServerTest, ClientsTerminateIfServiceGoesAway) {
#if defined(ADDRESS_SANITIZER)
  GTEST_SKIP()
      << "This test is known to produce memory leaks due to ungraceful "
         "termination of the RPC server despite having pending connections.";
#endif
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);

  absl::Barrier barrier(num_nodes + 1);

  auto thread_fn = [&](int node_id) -> absl::Status {
    absl::Notification shutdown;
    CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
    agent_config.cluster_register_timeout = absl::Seconds(10);
    agent_config.shutdown_barrier_timeout = absl::Seconds(10);
    auto client = GetClient(
        /*node_id=*/node_id, agent_config,
        /*error_fn=*/[&shutdown](const absl::Status& status) {
          shutdown.Notify();
        });

    TF_RETURN_IF_ERROR(client->Connect());

    barrier.Block();
    shutdown.WaitForNotification();

    TF_RETURN_IF_ERROR(client->Shutdown());
    return absl::OkStatus();
  };

  std::vector<absl::Status> statuses(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
    barrier.Block();
    StopService();
  }
  for (int i = 0; i < num_nodes; ++i) {
    EXPECT_EQ(statuses[i].code(), tsl::error::FAILED_PRECONDITION);
  }
}

// We should eventually connect, even if some clients are late to show up.
TEST_F(ClientServerTest, LateClientsAreOk) {
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);

  absl::Barrier barrier(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
    agent_config.cluster_register_timeout = absl::Seconds(20);
    agent_config.shutdown_barrier_timeout = absl::Seconds(20);
    auto client = GetClient(node_id, agent_config);

    barrier.Block();
    absl::SleepFor(absl::Milliseconds(200) * node_id);
    TF_RETURN_IF_ERROR(client->Connect());
    TF_RETURN_IF_ERROR(client->Shutdown());
    return absl::OkStatus();
  };

  std::vector<absl::Status> statuses = RunClients(num_nodes, thread_fn);
  EXPECT_THAT(statuses, Each(absl_testing::IsOk()));
}

// We should eventually time out if a client does not show up.
TEST_F(ClientServerTest, ConnectEventuallyTimesOutIfAClientDoesNotShowUp) {
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);

    TF_RETURN_IF_ERROR(client->Connect());
    TF_RETURN_IF_ERROR(client->Shutdown());
    return absl::OkStatus();
  };

  // Note: one fewer thread than 'num_nodes'.
  std::vector<absl::Status> statuses(num_nodes - 1);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes - 1; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }
  for (int i = 0; i < num_nodes - 1; ++i) {
    EXPECT_EQ(statuses[i].code(), tsl::error::DEADLINE_EXCEEDED);
  }
}

// After init, ML program will run. If a client restarts, the ML program will
// have an inconsistent state. To recover from this, ALL clients need to
// restart.
TEST_F(ClientServerTest, ClientRestart_AfterConnect_Fails) {
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.cluster_register_timeout = absl::Seconds(5);
  config.shutdown_barrier_timeout = absl::Seconds(5);
  config.num_tasks = num_nodes;
  StartService(config);
  absl::Notification n;

  auto thread_fn = [&](int node_id) -> absl::Status {
    CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
    agent_config.cluster_register_timeout = absl::Seconds(5);
    agent_config.shutdown_barrier_timeout = absl::Seconds(5);
    auto client = GetClient(node_id, agent_config);

    TF_RETURN_IF_ERROR(client->Connect());
    // All clients have successfully connected at this point.
    // Simulate client restart by creating a new client.
    if (node_id == 2) {
      client = nullptr;
      auto restarted_client = GetClient(node_id, agent_config);
      auto status = restarted_client->Connect();
      n.Notify();
      return status;
    }
    n.WaitForNotification();
    TF_RETURN_IF_ERROR(client->Shutdown());
    return absl::OkStatus();
  };

  std::vector<absl::Status> statuses = RunClients(num_nodes, thread_fn);
  // Errors should have been propagated to the clients, and thus the shutdown
  // call will fail with `FailedPrecondition` since the tasks are already in
  // error.
  EXPECT_THAT(statuses[0],
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(statuses[1],
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
  // This client was restarted, so its connection attempt will be aborted.
  EXPECT_THAT(statuses[2], absl_testing::StatusIs(absl::StatusCode::kAborted));
}

// If a client restarts during init, it can silently reconnect because no
// stateful operations have run yet, so the program state is still valid.
TEST_F(ClientServerTest, ClientRestart_DuringConnect_Succeeds) {
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.cluster_register_timeout = absl::Seconds(5);
  config.shutdown_barrier_timeout = absl::Seconds(5);
  config.num_tasks = num_nodes;
  StartService(config);
  absl::Notification previous_node_2_connecting, node_2_restarted;

  std::vector<absl::Status> statuses(num_nodes + 1);
  auto thread_fn = [&](int node_id) -> absl::Status {
    bool restarted_node_2 = false;
    if (node_id == 3) {
      restarted_node_2 = true;
      node_id = 2;  // This is the restarted client.
    }
    CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
    agent_config.cluster_register_timeout = absl::Seconds(5);
    agent_config.shutdown_barrier_timeout = absl::Seconds(5);
    auto client = GetClient(node_id, agent_config);

    // Overall timeline:
    // 1. Node 0, 2 connects.
    // 2. Node 2 restarts and connects.
    // 3. Node 1 connects.
    // 4. All attempts succeed, except the initial node 2 connection attempt.
    if (node_id == 0) {
      TF_RETURN_IF_ERROR(client->Connect());
      TF_RETURN_IF_ERROR(client->Shutdown());
      return absl::OkStatus();
    } else if (node_id == 1) {
      node_2_restarted.WaitForNotification();
      absl::SleepFor(absl::Seconds(1));  // Give time for node 2 to connect.
      TF_RETURN_IF_ERROR(client->Connect());
      TF_RETURN_IF_ERROR(client->Shutdown());
      return absl::OkStatus();
    } else if (node_id == 2 && !restarted_node_2) {
      previous_node_2_connecting.Notify();
      return client->Connect();  // Stale attempt, should fail.
    } else {
      // Restarted node 2.
      previous_node_2_connecting.WaitForNotification();
      absl::SleepFor(absl::Seconds(1));  // Give time for node 2 to connect.
      node_2_restarted.Notify();
      TF_RETURN_IF_ERROR(client->Connect());
      TF_RETURN_IF_ERROR(client->Shutdown());
      return absl::OkStatus();
    }
  };
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes + 1);

    for (int i = 0; i < num_nodes + 1; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }
  EXPECT_THAT(statuses[0], absl_testing::StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(statuses[1], absl_testing::StatusIs(absl::StatusCode::kOk));
  // This was the initial connection attempt that should be aborted.
  EXPECT_THAT(statuses[2],
              absl_testing::StatusIs(absl::StatusCode::kAlreadyExists));
  // This was the restarted client which should silently reconnect.
  EXPECT_THAT(statuses[3], absl_testing::StatusIs(absl::StatusCode::kOk));
}

TEST_F(ClientServerTest, WaitAtBarrier_Succeed) {
  int num_nodes = 2;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    TF_RETURN_IF_ERROR(client->Connect());

    TF_RETURN_IF_ERROR(client->WaitAtBarrier("barrier_1", kBarrierTimeout, {}));
    TF_RETURN_IF_ERROR(client->WaitAtBarrier("barrier_2", kBarrierTimeout, {}));

    TF_RETURN_IF_ERROR(client->Shutdown());
    return absl::OkStatus();
  };

  std::vector<absl::Status> statuses = RunClients(num_nodes, thread_fn);
  EXPECT_THAT(statuses, Each(absl_testing::IsOk()));
}

TEST_F(ClientServerTest, WaitAtBarrier_Timeout) {
  int num_nodes = 2;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);
  absl::Notification n;

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    TF_RETURN_IF_ERROR(client->Connect());

    // Node 1 waits for barrier to time out before proceeding.
    if (node_id == 1) {
      n.WaitForNotification();
    }
    absl::Status barrier_status =
        client->WaitAtBarrier("barrier_1", kBarrierTimeout, {});
    // Node 0 notifies that barrier has already timed out.
    if (node_id == 0) {
      n.Notify();
    }
    TF_RETURN_IF_ERROR(barrier_status);

    TF_RETURN_IF_ERROR(client->Shutdown());
    return absl::OkStatus();
  };

  std::vector<absl::Status> statuses = RunClients(num_nodes, thread_fn);
  for (int i = 0; i < num_nodes; ++i) {
    // Co-ordination service returns the status of the previous barrier
    // failure without waiting for the thread to time out.
    EXPECT_EQ(statuses[i].code(), tsl::error::DEADLINE_EXCEEDED)
        << " node id: " << i;
  }
}

TEST_F(ClientServerTest, WaitAtBarrier_TimeoutWithDifferentBarrierId) {
  int num_nodes = 2;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    TF_RETURN_IF_ERROR(client->Connect());

    std::string barrier_id;
    if (node_id == 0) {
      barrier_id = "barrier_0";
    } else if (node_id == 1) {
      barrier_id = "barrier_1";
    }
    TF_RETURN_IF_ERROR(client->WaitAtBarrier(barrier_id, kBarrierTimeout, {}));

    TF_RETURN_IF_ERROR(client->Shutdown());
    return absl::OkStatus();
  };

  std::vector<absl::Status> statuses = RunClients(num_nodes, thread_fn);
  for (int i = 0; i < num_nodes; ++i) {
    EXPECT_EQ(statuses[i].code(), tsl::error::DEADLINE_EXCEEDED)
        << " node id: " << i;
  }
}

TEST_F(ClientServerTest, WaitAtBarrier_ReuseSameId_Succeeds) {
  int num_nodes = 2;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    TF_RETURN_IF_ERROR(client->Connect());

    TF_RETURN_IF_ERROR(client->WaitAtBarrier("barrier_1", kBarrierTimeout, {}));
    TF_RETURN_IF_ERROR(client->WaitAtBarrier("barrier_2", kBarrierTimeout, {}));
    TF_RETURN_IF_ERROR(client->WaitAtBarrier("barrier_1", kBarrierTimeout, {}));
    TF_RETURN_IF_ERROR(client->WaitAtBarrier("barrier_2", kBarrierTimeout, {}));

    TF_RETURN_IF_ERROR(client->Shutdown());
    return absl::OkStatus();
  };

  std::vector<absl::Status> statuses = RunClients(num_nodes, thread_fn);
  EXPECT_THAT(statuses, Each(absl_testing::IsOk()));
}

TEST_F(ClientServerTest, WaitAtBarrier_RestartAndBarrierAgain_Fails) {
  int num_nodes = 2;
  // Allow clients to connect by themselves so restarted client can connect
  // and try barrier again.
  CoordinationService::Config config = DefaultServiceConfig();
  config.cluster_register_with_barrier = false;
  config.shutdown_barrier_timeout = absl::ZeroDuration();
  config.num_tasks = num_nodes;
  StartService(config);
  absl::Status barrier_status;
  absl::Notification n;

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    TF_RETURN_IF_ERROR(client->Connect());

    // Complete barrier 3 times (simulate job progress).
    for (int i = 0; i < 3; ++i) {
      TF_RETURN_IF_ERROR(
          client->WaitAtBarrier("barrier_1", kBarrierTimeout, {}));
    }
    if (node_id == 1) {
      client = nullptr;  // Simulate client restart.
      auto restarted_client = GetClient(1);
      TF_RETURN_IF_ERROR(restarted_client->Connect());
      // This should fail! This variable is checked after the thread pool is
      // destroyed.
      barrier_status =
          restarted_client->WaitAtBarrier("barrier_1", kBarrierTimeout, {});
      n.Notify();
    }
    // Client 0 should only be destroyed after we get the barrier result.
    n.WaitForNotification();
    return absl::OkStatus();
  };

  RunClients(num_nodes, thread_fn);
  EXPECT_THAT(barrier_status,
              absl_testing::StatusIs(absl::StatusCode::kInternal,
                                     HasSubstr("restarted")));
}

TEST_F(ClientServerTest,
       WaitAtBarrier_TimeoutThenOkay_StragglingTaskGetsSameError) {
  int num_nodes = 2;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);
  absl::Notification n, n_2;
  absl::Status status_0, status_0_new, status_1, status_1_new;
  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    TF_RETURN_IF_ERROR(client->Connect());
    if (node_id == 0) {
      status_0 = client->WaitAtBarrier("barrier_1", kBarrierTimeout, {});
      n.Notify();
      n_2.WaitForNotification();
      status_0_new = client->WaitAtBarrier("barrier_1", kBarrierTimeout, {});
    } else {
      n.WaitForNotification();  // Block until node 0's barrier times out.
      status_1 = client->WaitAtBarrier("barrier_1", kBarrierTimeout, {});
      n_2.Notify();
      status_1_new = client->WaitAtBarrier("barrier_1", kBarrierTimeout, {});
    }
    return absl::OkStatus();
  };

  RunClients(num_nodes, thread_fn);
  // Both nodes should get the same error.
  EXPECT_THAT(status_0,
              absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_THAT(status_1,
              absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
  // Next barrier call is okay.
  TF_EXPECT_OK(status_0_new);
  TF_EXPECT_OK(status_1_new);
}

TEST_F(ClientServerTest,
       WaitAtBarrier_QuickTaskStartBarrierTwice_LateTaskGetsSlowError) {
  int num_nodes = 2;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);
  absl::Notification n;
  absl::Status status_0, status_0_new, status_1;
  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    TF_RETURN_IF_ERROR(client->Connect());
    TF_RETURN_IF_ERROR(client->WaitAtBarrier("barrier_1", kBarrierTimeout, {}));
    TF_RETURN_IF_ERROR(client->WaitAtBarrier("barrier_1", kBarrierTimeout, {}));
    if (node_id == 0) {
      // Let each barrier time out.
      status_0 = client->WaitAtBarrier("barrier_1", kBarrierTimeout, {});
      status_0_new = client->WaitAtBarrier("barrier_1", kBarrierTimeout, {});
      n.Notify();
    } else {
      // Block until node 0's barriers times out.
      n.WaitForNotification();
      status_1 = client->WaitAtBarrier("barrier_1", kBarrierTimeout, {});
    }
    return absl::OkStatus();
  };

  RunClients(num_nodes, thread_fn);
  // Both barriers from node 0 should time out.
  EXPECT_THAT(status_0,
              absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_THAT(status_0_new,
              absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
  // Next barrier call from node 1 gets barrier counter mismatch error.
  EXPECT_THAT(status_1, absl_testing::StatusIs(absl::StatusCode::kInternal,
                                               HasSubstr("too quick / slow")));
}

TEST_F(ClientServerTest, WaitAtBarrierSubset_Succeeds) {
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);
  absl::Notification n0, n1;

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    TF_RETURN_IF_ERROR(client->Connect());

    if (node_id != 2) {
      TF_RETURN_IF_ERROR(
          client->WaitAtBarrier("barrier_1", kBarrierTimeout, {0, 1}));
    }

    TF_RETURN_IF_ERROR(client->Shutdown());
    return absl::OkStatus();
  };

  std::vector<absl::Status> statuses = RunClients(num_nodes, thread_fn);
  EXPECT_THAT(statuses, Each(absl_testing::IsOk()));
}

TEST_F(ClientServerTest, WaitAtBarrier_DifferentSubset_Fails) {
  int num_nodes = 2;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);
  absl::Notification n;
  absl::Status status_0, status_1 = absl::UnknownError("Uninitialized error.");

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    TF_RETURN_IF_ERROR(client->Connect());
    if (node_id == 0) {
      status_0 = client->WaitAtBarrier("barrier_1", kBarrierTimeout, {0});
      n.Notify();
    } else {
      n.WaitForNotification();
      // Same barrier id, but specifies different tasks.
      status_1 = client->WaitAtBarrier("barrier_1", kBarrierTimeout, {1});
    }
    return absl::OkStatus();
  };

  RunClients(num_nodes, thread_fn);
  // First barrier call succeeds.
  TF_EXPECT_OK(status_0);
  // Second barrier call with different task args fails.
  EXPECT_THAT(status_1,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Conflicting tasks specified")));
}

TEST_F(ClientServerTest, CancelNonExistentBarrier_Fails) {
  int num_nodes = 1;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);
  auto client = GetClient(0);
  TF_ASSERT_OK(client->Connect());

  EXPECT_THAT(client->CancelBarrier("non_existent_barrier"),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(ClientServerTest,
       WaitAtBarrierSubsetNonParticipatingProcessAttempts_Fails) {
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);
  absl::Notification n;
  absl::Barrier barrier(num_nodes + 1);

  // Timeline:
  // 1. Node 1, 2 joins barrier.
  // 2. Barrier fails because node 2 is unexpected.
  // 3. Node 0 joins barrier, but fails because barrier already failed.

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    TF_RETURN_IF_ERROR(client->Connect());

    // Node 0 will be notified only after the barrier has failed and will thus
    // fail too.
    if (node_id == 0) {
      n.WaitForNotification();
    }
    auto status = client->WaitAtBarrier("barrier_1", kBarrierTimeout, {0, 1});
    // Node 1 will fail in the barrier because non-participating node 2 also
    // calls it.
    if (node_id == 1) {
      n.Notify();
    }
    // Not calling `Shutdown` because the client will have already returned
    // error in the previous call to `WaitAtBarrier` for all 3 nodes. In the
    // error state, calling `Shutdown` is undefined behavior.
    return status;
  };

  std::vector<absl::Status> statuses(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() {
        statuses[i] = thread_fn(i);
        barrier.Block();
      });
    }

    // Block until the threads have finished execution.
    barrier.Block();

    for (int i = 0; i < num_nodes; ++i) {
      EXPECT_EQ(statuses[i].code(), tsl::error::INVALID_ARGUMENT)
          << " node id: " << i << " status: " << statuses[i].message();
    }
  }
}

TEST_F(ClientServerTest, GetAliveTasks_Succeed) {
  const int num_nodes = 2;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    TF_RETURN_IF_ERROR(client->Connect());
    absl::StatusOr<std::vector<CoordinationServiceAgent::AliveTask>>
        alive_tasks = client->GetAliveTasks({0, 1});
    if (!alive_tasks.ok()) {
      return alive_tasks.status();
    }
    TF_RETURN_IF_ERROR(client->Shutdown());
    return absl::OkStatus();
  };

  std::vector<absl::Status> statuses = RunClients(num_nodes, thread_fn);
  EXPECT_THAT(statuses, Each(absl_testing::IsOk()));
}

TEST_F(ClientServerTest, GetKeyValueDir) {
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = 1;
  StartService(config);
  auto client = GetClient(/*node_id=*/0);
  TF_ASSERT_OK(client->Connect());
  TF_ASSERT_OK(client->InsertKeyValue("test_dir/sub_dir/1", "1"));
  TF_ASSERT_OK(client->InsertKeyValue("test_dir/sub_dir/2", "2"));
  TF_ASSERT_OK(client->InsertKeyValue("test_dir/3", "3"));
  TF_ASSERT_OK(client->InsertKeyValue("test", "4"));  // Not in a directory.

  auto results = client->GetKeyValueDir("test_dir/");

  TF_ASSERT_OK(results.status());
  auto kvs = results.value();

  EXPECT_THAT(kvs, UnorderedElementsAre(IsKvEntry("test_dir/sub_dir/1", "1"),
                                        IsKvEntry("test_dir/sub_dir/2", "2"),
                                        IsKvEntry("test_dir/3", "3")));
}

TEST_F(ClientServerTest, InsertKeyValue_Duplicate_Fails) {
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = 1;
  StartService(config);
  auto client = GetClient(/*node_id=*/0);
  TF_ASSERT_OK(client->Connect());
  TF_ASSERT_OK(client->InsertKeyValue("test_key", "original_value"));
  EXPECT_TRUE(
      absl::IsAlreadyExists(client->InsertKeyValue("test_key", "never_added")));
  auto result = client->GetKeyValue("test_key", absl::Milliseconds(100));
  TF_ASSERT_OK(result.status());
  EXPECT_EQ(result.value(), "original_value");
}

TEST_F(ClientServerTest, InsertKeyValue_Duplicate_Overwrites) {
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = 1;
  StartService(config);
  auto client = GetClient(/*node_id=*/0);
  TF_ASSERT_OK(client->Connect());
  TF_ASSERT_OK(client->InsertKeyValue("test_key", "original_value"));
  TF_EXPECT_OK(client->InsertKeyValue("test_key", "overwritten_value",
                                      /*allow_overwrite=*/true));
  auto result = client->GetKeyValue("test_key", absl::Milliseconds(100));
  TF_ASSERT_OK(result.status());
  EXPECT_EQ(result.value(), "overwritten_value");
}

TEST_F(ClientServerTest, DeleteKeyValue) {
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = 1;
  StartService(config);
  auto client = GetClient(/*node_id=*/0);
  TF_ASSERT_OK(client->Connect());
  TF_ASSERT_OK(client->InsertKeyValue("to_be_deleted", "deleted"));
  TF_ASSERT_OK(client->InsertKeyValue("to_be_kept", "kept"));

  auto results = client->DeleteKeyValue("to_be_deleted");

  TF_EXPECT_OK(results);
  auto deleted_kv =
      client->GetKeyValue("to_be_deleted", absl::Milliseconds(200));
  // We time out from attempting to retrieve a deleted key.
  EXPECT_EQ(deleted_kv.status().code(), tsl::error::DEADLINE_EXCEEDED);
  // Other key should still exist.
  auto kept_kv = client->GetKeyValue("to_be_kept", absl::Milliseconds(200));
  TF_ASSERT_OK(kept_kv.status());
  EXPECT_EQ(kept_kv.value(), "kept");
}

TEST_F(ClientServerTest, DeleteKeyValue_Directory) {
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = 1;
  StartService(config);
  auto client = GetClient(/*node_id=*/0);
  TF_ASSERT_OK(client->Connect());
  TF_ASSERT_OK(client->InsertKeyValue("test_dir/sub_dir/1", "1"));
  TF_ASSERT_OK(client->InsertKeyValue("test_dir/sub_dir/2", "2"));
  TF_ASSERT_OK(client->InsertKeyValue("test_dir/3", "3"));

  auto results = client->DeleteKeyValue("test_dir/");

  TF_EXPECT_OK(results);
  auto kvs = client->GetKeyValueDir("test_dir/");
  TF_ASSERT_OK(kvs.status());
  EXPECT_THAT(kvs.value(), IsEmpty());
}

// This prevents a regression found in b/380359918 where original error
// messages are hidden because the RPC layer cannot send long error messages.
TEST_F(ClientServerTest, BarrierTimeout_ManyLateTasks_ReturnsCorrectError) {
  CoordinationService::Config config = DefaultServiceConfig();
  config.cluster_register_timeout = absl::Seconds(1);
  config.shutdown_barrier_timeout = absl::Seconds(1);
  config.cluster_register_with_barrier = false;
  config.num_tasks = 100;
  StartService(config);
  auto client = GetClient(/*node_id=*/0);
  TF_ASSERT_OK(client->Connect());

  // Blocks until the barrier times out.
  auto status =
      client->WaitAtBarrier("test_barrier", absl::Milliseconds(100), {});

  EXPECT_THAT(status,
              absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST_F(ClientServerTest, Dtor_CancelsOngoingGetKeyValueAndBarrier) {
  CoordinationService::Config config = DefaultServiceConfig();
  config.cluster_register_with_barrier = false;
  config.num_tasks = 2;
  StartService(config);
  CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
  agent_config.cluster_register_timeout = absl::Seconds(2);
  agent_config.shutdown_barrier_timeout = absl::Seconds(2);
  agent_config.agent_destruction_without_shutdown = true;
  auto client = GetClient(/*node_id=*/0, agent_config);
  TF_ASSERT_OK(client->Connect());
  absl::Status barrier_status, get_key_value_status;
  client->WaitAtBarrierAsync(
      "barrier", absl::Seconds(2), {},
      [&barrier_status](absl::Status s) { barrier_status = s; });
  client->GetKeyValueAsync(
      "test_key",
      [&get_key_value_status](const absl::StatusOr<std::string>& s) {
        get_key_value_status = s.status();
      });

  // Destroy client.
  client = nullptr;

  // Pending RPCs should be cancelled.
  EXPECT_EQ(barrier_status.code(), tsl::error::CANCELLED);
  EXPECT_EQ(get_key_value_status.code(), tsl::error::CANCELLED);
  // Unsure why, but this avoids tsan races surrounding RPC handler's mutex
  // during dtor.
  absl::SleepFor(absl::Seconds(1));
}

TEST_F(ClientServerTest, RecoverableClientNeverJoins_InitialConnectFails) {
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.recoverable = true;
  config.num_tasks = num_nodes;
  StartService(config);

  absl::Notification node_2_joins_late;
  std::vector<absl::Status> statuses(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
    agent_config.cluster_register_timeout = absl::Seconds(2);
    agent_config.shutdown_barrier_timeout = absl::Seconds(2);
    auto client = GetClient(node_id, agent_config);
    if (node_id == 0) {
      statuses[0] = client->Connect();
    } else if (node_id == 1) {
      statuses[1] = client->Connect();
      // Connect should time out and fail because node 2 is not connected yet.
      node_2_joins_late.Notify();
    } else {
      node_2_joins_late.WaitForNotification();
      statuses[2] = client->Connect();
    }
    return absl::OkStatus();
  };

  RunClients(num_nodes, thread_fn);
  // Even though node 2 is recoverable, initial connect still fails if it is
  // late.
  for (int i = 0; i < num_nodes; ++i) {
    // 1) Connect() starts a barrier.
    // 2) Barrier times out, returning `DeadlineExceeded` error.
    // 3) The failed barrier sets the entire cluster to an error state.
    // Subsequent connect attempts will return `Aborted` error due to this
    // error state.
    EXPECT_THAT(statuses[i], absl_testing::StatusIs(
                                 AnyOf(absl::StatusCode::kDeadlineExceeded,
                                       absl::StatusCode::kAborted)))
        << i;
  }
}

TEST_F(ClientServerTest, NonrecoverableClientDies_ErrorPropagated) {
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_nodes;
  StartService(config);
  absl::Notification shutdown;
  std::vector<absl::Status> statuses(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
    agent_config.cluster_register_timeout = absl::Seconds(2);
    agent_config.shutdown_barrier_timeout = absl::Seconds(2);
    agent_config.agent_destruction_without_shutdown = true;
    auto client = GetClient(node_id, agent_config,
                            // Nodes 1, 2 are recoverable.
                            /*error_fn=*/
                            [&statuses, node_id](const absl::Status& status) {
                              statuses[node_id] = status;
                            });
    TF_RETURN_IF_ERROR(client->Connect());
    if (node_id == 0) {
      // Non-recoverable node crashed unexpectedly.
      return absl::OkStatus();
    }
    // Other nodes will get heartbeat timeouts.
    absl::SleepFor(kHeartbeatTimeout * 2);
    return absl::OkStatus();
  };

  RunClients(num_nodes, thread_fn);
  EXPECT_THAT(statuses,
              ElementsAre(
                  // Node 0 died unexpectedly, so no error function invoked.
                  absl_testing::StatusIs(absl::StatusCode::kOk),
                  // Node 1, 2 get error notification that node 0 died.
                  absl_testing::StatusIs(absl::StatusCode::kUnavailable),
                  absl_testing::StatusIs(absl::StatusCode::kUnavailable)));
}

TEST_F(ClientServerTest, RecoverableClientDies_NoErrorPropagated) {
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.recoverable = true;
  config.num_tasks = num_nodes;
  StartService(config);
  absl::Notification shutdown;
  std::vector<absl::Status> statuses(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
    agent_config.cluster_register_timeout = absl::Seconds(2);
    agent_config.shutdown_barrier_timeout = absl::Seconds(2);
    agent_config.agent_destruction_without_shutdown = true;
    auto client = GetClient(node_id, agent_config,
                            // Nodes 1, 2 are recoverable.
                            /*error_fn=*/
                            [&statuses, node_id](const absl::Status& status) {
                              statuses[node_id] = status;
                            });
    TF_RETURN_IF_ERROR(client->Connect());
    if (node_id == 1) {
      // Recoverable node crashed unexpectedly.
      return absl::OkStatus();
    }
    // Other nodes will get heartbeat timeouts.
    absl::SleepFor(kHeartbeatTimeout * 2);
    return absl::OkStatus();
  };

  RunClients(num_nodes, thread_fn);
  // Nobody noticed that node 1 (recoverable) died.
  TF_EXPECT_OK(statuses[0]);
  TF_EXPECT_OK(statuses[1]);
  TF_EXPECT_OK(statuses[2]);
}

TEST_F(ClientServerTest, RecoverableClient_RestartsAndStartsBarrier) {
  int num_nodes = 2;
  CoordinationService::Config config = DefaultServiceConfig();
  config.recoverable = true;
  config.num_tasks = num_nodes;
  StartService(config);
  absl::Notification node_1_joins_barrier;
  absl::Status status_0, status_0_shutdown, status_1, status_1_shutdown;
  status_0 = status_0_shutdown = status_1 = status_1_shutdown =
      absl::UnknownError("Uninitialized error.");

  auto thread_fn = [&](int node_id) -> absl::Status {
    CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
    agent_config.cluster_register_timeout = absl::Seconds(2);
    agent_config.shutdown_barrier_timeout = absl::Seconds(2);
    auto client = GetClient(node_id, agent_config);
    TF_RETURN_IF_ERROR(client->Connect());
    // This increments the internal barrier counter, and checks if the
    // recoverable node can start a new barrier later (despite having a reset
    // counter on the client-side)
    TF_RETURN_IF_ERROR(
        client->WaitAtBarrier(kBarrierId, absl::Seconds(10), {}));
    TF_RETURN_IF_ERROR(
        client->WaitAtBarrier(kBarrierId, absl::Seconds(10), {}));
    TF_RETURN_IF_ERROR(
        client->WaitAtBarrier(kBarrierId, absl::Seconds(10), {}));
    // Timeline:
    // 1. Node 0 restarts.
    // 2. Node 0 starts a new barrier.
    // 3. Node 1 joins barrier.
    // 4. End-of-job barrier.
    if (node_id == 0) {
      // Restart the client.
      client = nullptr;
      auto restarted_client = GetClient(/*node_id=*/0, agent_config);
      TF_RETURN_IF_ERROR(restarted_client->Connect());

      restarted_client->WaitAtBarrierAsync(
          kBarrierId, absl::Seconds(10), {},
          [&status_0](const absl::Status& status) { status_0 = status; });
      // This makes sure that the underlying async RPC layer sends the request
      // before we let node 1 join the barrier.
      absl::SleepFor(absl::Seconds(1));
      // Node 1 should join the barrier after the new barrier is initiated.
      node_1_joins_barrier.Notify();
      // Check that the cluster is healthy at the end of the test.
      status_0_shutdown = restarted_client->WaitAtBarrier(
          "before_shutdown", absl::Seconds(10), {});
    } else {
      node_1_joins_barrier.WaitForNotification();
      status_1 = client->WaitAtBarrier(kBarrierId, absl::Seconds(10), {});
      // Check that the cluster is healthy at the end of the test.
      status_1_shutdown =
          client->WaitAtBarrier("before_shutdown", absl::Seconds(10), {});
    }
    return absl::OkStatus();
  };

  RunClients(num_nodes, thread_fn);
  TF_EXPECT_OK(status_0);
  TF_EXPECT_OK(status_0_shutdown);
  TF_EXPECT_OK(status_1);
  TF_EXPECT_OK(status_1_shutdown);
}

TEST_F(ClientServerTest, ServiceIncarnationMismatch) {
  const int port = tsl::testing::PickUnusedPortOrDie();
  const int num_clients = 9;
  CoordinationService::Config config = DefaultServiceConfig();
  config.num_tasks = num_clients + 1;
  config.cluster_register_with_barrier = false;
  // Set a long heartbeat timeout so that no heartbeat fires during the service
  // restart and poisons the clients with UNAVAILABLE errors.
  config.heartbeat_timeout = absl::Seconds(60);
  StartService(config, port);

  // Create a separate client for each RPC so that error_fn is triggered
  // independently for each one.
  std::vector<absl::Status> errors(num_clients);
  std::vector<std::unique_ptr<CoordinationServiceAgent>> clients(num_clients);
  CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
  agent_config.agent_destruction_without_shutdown = true;
  agent_config.heartbeat_timeout = absl::Seconds(60);
  agent_config.poll_for_error_from_service_at_startup = false;
  for (int i = 0; i < num_clients; ++i) {
    clients[i] =
        GetClient(i, agent_config,
                  [&errors, i](const absl::Status& s) { errors[i] = s; });
    TF_ASSERT_OK(clients[i]->Connect());
  }

  // Create a probe client to wait for the channel to reconnect.
  auto probe = GetClient(num_clients, agent_config);
  TF_ASSERT_OK(probe->Connect());

  // Restart the service.
  StopService();
  StartService(config, port);

  // Wait for the gRPC channel to reconnect by polling with a lightweight RPC.
  // TryGetKeyValue is a good probe because it returns immediately.
  while (true) {
    auto status = probe->TryGetKeyValue("probe");
    if (status.status().code() != absl::StatusCode::kUnavailable) {
      break;
    }
    absl::SleepFor(absl::Milliseconds(100));
  }

  // All RPCs should fail because the clients have the old service incarnation.
  auto has_wrong_service = absl_testing::StatusIs(
      absl::StatusCode::kInternal, HasSubstr("wrong service incarnation"));

  int i = 0;
  EXPECT_THAT(clients[i]->GetKeyValue("key").status(), has_wrong_service);
  EXPECT_THAT(errors[i], has_wrong_service);
  ++i;

  EXPECT_THAT(clients[i]->TryGetKeyValue("key").status(), has_wrong_service);
  EXPECT_THAT(errors[i], has_wrong_service);
  ++i;

  EXPECT_THAT(clients[i]->InsertKeyValue("key", "value"), has_wrong_service);
  EXPECT_THAT(errors[i], has_wrong_service);
  ++i;

  EXPECT_THAT(clients[i]->DeleteKeyValue("key"), has_wrong_service);
  EXPECT_THAT(errors[i], has_wrong_service);
  ++i;

  EXPECT_THAT(clients[i]->IncrementKeyValue("key", 1).status(),
              has_wrong_service);
  EXPECT_THAT(errors[i], has_wrong_service);
  ++i;

  EXPECT_THAT(clients[i]->GetKeyValueDir("key").status(), has_wrong_service);
  EXPECT_THAT(errors[i], has_wrong_service);
  ++i;

  EXPECT_THAT(clients[i]->WaitAtBarrier("barrier", absl::Seconds(1), {}),
              has_wrong_service);
  EXPECT_THAT(errors[i], has_wrong_service);
  ++i;

  EXPECT_THAT(clients[i]->GetAliveTasks({0}).status(), has_wrong_service);
  EXPECT_THAT(errors[i], has_wrong_service);
  ++i;

  EXPECT_THAT(clients[i]->WatchTasks({}).status(), has_wrong_service);
  EXPECT_THAT(errors[i], has_wrong_service);
  ++i;
}

struct RecoverableTestParams {
  // This determines if the restarted clients will invoke shutdown before going
  // away. This checks if the service logic will not be corrupted even if the
  // client does not gracefully disconnect (i.e. param set to false).
  bool recoverable_node_shutdown_on_destruction;
  // This inserts errors to transition the cluster into error states (from the
  // service's POV), which may result in different code paths or error codes.
  bool inject_heartbeat_errors;
  // This exercises barrier counter validation logic. On the agent side, a
  // restarted client will have its counter reset. Thus, the service needs to
  // let the recoverable node join despite a mismatched counter.
  bool pass_multiple_barriers_before_test;
};

class RecoverableTest
    : public ClientServerTest,
      public ::testing::WithParamInterface<RecoverableTestParams> {};

TEST_P(RecoverableTest, RecoverableClient_RestartsAndJoinsBarrier) {
  const RecoverableTestParams params = GetParam();
  int num_nodes = 2;
  CoordinationService::Config config = DefaultServiceConfig();
  config.recoverable = true;
  config.num_tasks = num_nodes;
  StartService(config);
  absl::Notification restart_now;
  absl::Status status_0, status_0_shutdown, status_1, status_1_shutdown;
  status_0 = status_0_shutdown = status_1 = status_1_shutdown =
      absl::UnknownError("Uninitialized error.");

  auto thread_fn = [&](int node_id) {
    CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
    agent_config.cluster_register_timeout = absl::Seconds(2);
    agent_config.shutdown_barrier_timeout = absl::Seconds(2);
    agent_config.agent_destruction_without_shutdown =
        params.recoverable_node_shutdown_on_destruction ? false
                                                        : (node_id == 1);
    auto client = GetClient(node_id, agent_config);
    TF_ASSERT_OK(client->Connect());
    if (params.pass_multiple_barriers_before_test) {
      // This increments the internal barrier counter, and checks if the
      // recoverable node can join the barrier later (despite having a reset
      // counter on the client-side)
      TF_ASSERT_OK(client->WaitAtBarrier(kBarrierId, absl::Seconds(10), {}));
      TF_ASSERT_OK(client->WaitAtBarrier(kBarrierId, absl::Seconds(10), {}));
      TF_ASSERT_OK(client->WaitAtBarrier(kBarrierId, absl::Seconds(10), {}));
    }
    // Timeline:
    // 1. Node 0 joins barrier.
    // 2. Node 1 restarts.
    // 3. Node 1 joins barrier.
    // 4. End-of-job barrier.
    if (node_id == 0) {
      // Usually, if a non-recoverable node goes away, the barrier will fail
      // immediately. However, if the node is recoverable, the barrier will
      // wait for the node to restart.
      client->WaitAtBarrierAsync(
          kBarrierId, absl::Seconds(10), {},
          [&status_0](const absl::Status& status) { status_0 = status; });
      // This makes sure that the underlying async RPC layer sends the request
      // before we let node 1 restart.
      absl::SleepFor(absl::Seconds(1));
      // Node 1 should restart after the barrier is initiated.
      restart_now.Notify();
      // Check that the cluster is healthy at the end of the test.
      status_0_shutdown =
          client->WaitAtBarrier("before_shutdown", absl::Seconds(10), {});
    } else {
      restart_now.WaitForNotification();
      // Restart the client.
      client = nullptr;
      if (params.inject_heartbeat_errors) {
        absl::SleepFor(2 * kHeartbeatTimeout);
      }
      agent_config.agent_destruction_without_shutdown = false;
      auto restarted_client = GetClient(/*node_id=*/1, agent_config);
      TF_ASSERT_OK(restarted_client->Connect());
      status_1 =
          restarted_client->WaitAtBarrier(kBarrierId, absl::Seconds(10), {});
      // Check that the cluster is healthy at the end of the test.
      status_1_shutdown = restarted_client->WaitAtBarrier(
          "before_shutdown", absl::Seconds(10), {});
    }
  };

  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { thread_fn(i); });
    }
  }
  TF_EXPECT_OK(status_0);
  TF_EXPECT_OK(status_0_shutdown);
  TF_EXPECT_OK(status_1);
  TF_EXPECT_OK(status_1_shutdown);
}

TEST_P(RecoverableTest,
       RecoverableClient_JoinsBarrierThenRestartsAndJoinsBarrierAgain) {
  const RecoverableTestParams params = GetParam();
  int num_nodes = 3;
  CoordinationService::Config config = DefaultServiceConfig();
  config.recoverable = true;
  config.num_tasks = num_nodes;
  StartService(config);
  absl::Notification node_0_restarts, node_2_joins_barrier_last;
  absl::Status status_0_before_restart, status_0_after_restart,
      status_0_shutdown, status_1, status_1_shutdown, status_2,
      status_2_shutdown;
  status_0_before_restart = status_0_after_restart = status_0_shutdown =
      status_1 = status_1_shutdown = status_2 = status_2_shutdown =
          absl::UnknownError("Uninitialized error.");

  auto thread_fn = [&](int node_id) {
    CoordinationServiceAgent::Config agent_config = DefaultAgentConfig();
    agent_config.cluster_register_timeout = absl::Seconds(2);
    agent_config.shutdown_barrier_timeout = absl::Seconds(2);
    agent_config.agent_destruction_without_shutdown =
        params.recoverable_node_shutdown_on_destruction ? false
                                                        : (node_id == 0);
    auto client = GetClient(node_id, agent_config);
    TF_ASSERT_OK(client->Connect());
    if (params.pass_multiple_barriers_before_test) {
      // This increments the internal barrier counter, and checks if the
      // recoverable node can join the barrier later (despite having a reset
      // counter on the client-side)
      TF_ASSERT_OK(client->WaitAtBarrier(kBarrierId, absl::Seconds(10), {}));
      TF_ASSERT_OK(client->WaitAtBarrier(kBarrierId, absl::Seconds(10), {}));
      TF_ASSERT_OK(client->WaitAtBarrier(kBarrierId, absl::Seconds(10), {}));
    }
    // Timeline:
    // 1. Node 0, 1 joins barrier.
    // 2. Node 0 restarts.
    // 3. Node 0 joins barrier again.
    // 4. Node 2 joins barrier.
    // 5. End-of-job barrier.
    if (node_id == 0) {
      // This is the key difference: client invokes the barrier **first**,
      // then restarts and joins barrier again. This means that service has to
      // deal with the barrier state from a stale client.
      client->WaitAtBarrierAsync(
          kBarrierId, absl::Seconds(10), {},
          [&status_0_before_restart](const absl::Status& status) {
            status_0_before_restart = status;
          });
      node_0_restarts.WaitForNotification();
      // Restart the client.
      client = nullptr;
      if (params.inject_heartbeat_errors) {
        absl::SleepFor(2 * kHeartbeatTimeout);
      }
      agent_config.agent_destruction_without_shutdown = false;
      auto restarted_client = GetClient(/*node_id=*/0, agent_config);
      TF_ASSERT_OK(restarted_client->Connect());
      restarted_client->WaitAtBarrierAsync(
          kBarrierId, absl::Seconds(10), {},
          [&status_0_after_restart](const absl::Status& status) {
            status_0_after_restart = status;
          });
      node_2_joins_barrier_last.Notify();
      status_0_shutdown = restarted_client->WaitAtBarrier(
          "before_shutdown", absl::Seconds(10), {});
    } else if (node_id == 1) {
      client->WaitAtBarrierAsync(
          kBarrierId, absl::Seconds(10), {},
          [&status_1](const absl::Status& status) { status_1 = status; });
      // This makes sure that the underlying async RPC layer sends the request
      // before node 0 restarts.
      absl::SleepFor(absl::Seconds(1));
      node_0_restarts.Notify();
      status_1_shutdown =
          client->WaitAtBarrier("before_shutdown", absl::Seconds(10), {});
    } else {
      node_2_joins_barrier_last.WaitForNotification();
      status_2 = client->WaitAtBarrier(kBarrierId, absl::Seconds(10), {});
      status_2_shutdown =
          client->WaitAtBarrier("before_shutdown", absl::Seconds(10), {});
    }
  };

  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { thread_fn(i); });
    }
  }
  // If node 0 restarts before joining the barrier, the barrier should be
  // cancelled.
  EXPECT_THAT(status_0_before_restart,
              absl_testing::StatusIs(absl::StatusCode::kCancelled));
  TF_EXPECT_OK(status_0_after_restart);
  TF_EXPECT_OK(status_0_shutdown);
  TF_EXPECT_OK(status_1);
  TF_EXPECT_OK(status_1_shutdown);
  TF_EXPECT_OK(status_2);
  TF_EXPECT_OK(status_2_shutdown);
}

INSTANTIATE_TEST_SUITE_P(
    RecoverableTestSuite, RecoverableTest,
    // Full matrix of test parameters.
    ::testing::Values(
        // Basic tests: agent will invoke shutdown on destruction.
        RecoverableTestParams{
            /*recoverable_node_shutdown_on_destruction=*/true,
            /*inject_heartbeat_errors=*/true,
            /*pass_multiple_barriers_before_test=*/false,
        },
        RecoverableTestParams{/*recoverable_node_shutdown_on_destruction=*/true,
                              /*inject_heartbeat_errors=*/false,
                              /*pass_multiple_barriers_before_test=*/false},
        // Check if service can handle restarted clients that do not
        // gracefully disconnect.
        RecoverableTestParams{
            /*recoverable_node_shutdown_on_destruction=*/false,
            /*inject_heartbeat_errors=*/true,
            /*pass_multiple_barriers_before_test=*/false},
        // Hard test case: agent does not invoke shutdown, and doesn't trigger
        // heartbeat errors either.
        RecoverableTestParams{
            /*recoverable_node_shutdown_on_destruction=*/false,
            /*inject_heartbeat_errors=*/false,
            /*pass_multiple_barriers_before_test=*/false},
        // Same as above, but with multiple barriers to exercise counter
        // validation logic.
        RecoverableTestParams{
            /*recoverable_node_shutdown_on_destruction=*/true,
            /*inject_heartbeat_errors=*/true,
            /*pass_multiple_barriers_before_test=*/true,
        },
        RecoverableTestParams{/*recoverable_node_shutdown_on_destruction=*/true,
                              /*inject_heartbeat_errors=*/false,
                              /*pass_multiple_barriers_before_test=*/true},
        RecoverableTestParams{
            /*recoverable_node_shutdown_on_destruction=*/false,
            /*inject_heartbeat_errors=*/true,
            /*pass_multiple_barriers_before_test=*/true},
        // Hardest test case: agent does not invoke shutdown or trigger
        // heartbeat errors. It silently restarts and re-connects.
        RecoverableTestParams{
            /*recoverable_node_shutdown_on_destruction=*/false,
            /*inject_heartbeat_errors=*/false,
            /*pass_multiple_barriers_before_test=*/true}));
}  // namespace
}  // namespace xla
