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
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
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
#include "xla/tsl/distributed_runtime/coordination/coordination_client.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/distributed_runtime/rpc/async_service_interface.h"
#include "xla/tsl/distributed_runtime/rpc/coordination/grpc_coordination_client.h"
#include "xla/tsl/distributed_runtime/rpc/coordination/grpc_coordination_service_impl.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace tsl {
namespace {
using ::tensorflow::CoordinationServiceConfig;
using ::testing::AnyOf;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;
using ::tsl::testing::StatusIs;

constexpr absl::Duration kBarrierTimeout = absl::Milliseconds(200);

tensorflow::CoordinatedTask GetTask(int node_id) {
  tensorflow::CoordinatedTask task;
  task.set_task_id(node_id);
  task.set_job_name("agent");
  return task;
}

// Note: b/169705709: no protobuf matchers in OSS.
MATCHER_P2(IsKvEntry, key, value, "") {
  return key == arg.key() && value == arg.value();
}

class ClientServerTest : public ::testing::Test {
 public:
  CoordinationServiceConfig GetConfig(
      absl::Duration init_and_shutdown_timeout,
      bool shutdown_on_destruction = true,
      bool cluster_register_with_barrier = true,
      bool cluster_shutdown_with_barrier = true) {
    // Set config.
    tensorflow::CoordinationServiceConfig config;
    config.set_service_type("standalone");
    config.set_service_leader("/job:agent/task:0");
    config.set_cluster_register_timeout_in_ms(
        absl::ToInt64Milliseconds(init_and_shutdown_timeout));
    config.set_heartbeat_timeout_in_ms(
        absl::ToInt64Milliseconds(absl::Seconds(3)));
    if (cluster_shutdown_with_barrier) {
      config.set_shutdown_barrier_timeout_in_ms(
          absl::ToInt64Milliseconds(init_and_shutdown_timeout));
    }
    config.set_agent_destruction_without_shutdown(!shutdown_on_destruction);
    // TODO(b/369222279): Add more test cases that exercise TF behaviour (no
    // barrier).
    config.set_cluster_register_with_barrier(cluster_register_with_barrier);
    config.set_poll_for_error_from_service_at_startup(true);
    return config;
  }

  CoordinationServiceConfig GetServiceConfig(
      int num_nodes, absl::Duration init_and_shutdown_timeout,
      bool cluster_register_with_barrier, bool cluster_shutdown_with_barrier) {
    auto config =
        GetConfig(init_and_shutdown_timeout,
                  /*shutdown_on_destruction=*/true,
                  cluster_register_with_barrier, cluster_shutdown_with_barrier);
    tensorflow::CoordinatedJob* job =
        config.mutable_coordinated_job_list()->Add();
    job->set_name("agent");
    job->set_num_tasks(num_nodes);
    auto service = tsl::CoordinationServiceInterface::EnableCoordinationService(
        Env::Default(), config, /*cache=*/nullptr);
    return config;
  }

  std::unique_ptr<CoordinationServiceAgent> GetClient(
      int node_id, absl::Duration init_and_shutdown_timeout = absl::Seconds(3),
      bool shutdown_on_destruction = true,
      StatusCallback error_fn = [](const absl::Status& status) {
        LOG(ERROR) << "Agent hit an error: " << status;
      }) {
    assert(server_ != nullptr);
    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
        service_address_, grpc::InsecureChannelCredentials());

    std::unique_ptr<tsl::CoordinationClient> leader_client;
    leader_client.reset(tsl::NewGrpcCoordinationClient(channel));

    auto coord_agent = tsl::CreateCoordinationServiceAgent();
    CoordinationServiceConfig config =
        GetConfig(init_and_shutdown_timeout, shutdown_on_destruction);
    const absl::Status status =
        coord_agent->Initialize(tsl::Env::Default(), "agent", node_id, config,
                                std::move(leader_client), std::move(error_fn));
    if (!status.ok()) {
      LOG(ERROR) << "Coordination agent failed to initialize: " << status;
    }
    return coord_agent;
  }

  void StartService(int num_nodes,
                    absl::Duration init_and_shutdown_timeout = absl::Seconds(2),
                    bool cluster_register_with_barrier = true,
                    bool cluster_shutdown_with_barrier = true) {
    auto config = GetServiceConfig(num_nodes, init_and_shutdown_timeout,
                                   cluster_register_with_barrier,
                                   cluster_shutdown_with_barrier);

    int port = tsl::testing::PickUnusedPortOrDie();
    grpc::ServerBuilder builder;
    service_address_ = absl::StrCat("[::]:", port);
    builder.AddListeningPort(service_address_,
                             grpc::InsecureServerCredentials());
    // Set up the actual coordination service (where all the real logic
    // lives).
    coord_service_ =
        tsl::CoordinationServiceInterface::EnableCoordinationService(
            Env::Default(), config, /*cache=*/nullptr);
    // Set up threads and RPC service.
    coord_compute_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        Env::Default(), "CoordinationServiceRpcHandler",
        /*num_threads=*/4);
    coord_rpc_service_ = std::make_unique<tsl::GrpcCoordinationServiceImpl>(
        coord_compute_pool_.get(), &builder);
    auto* grpc_coord_service = static_cast<tsl::GrpcCoordinationServiceImpl*>(
        coord_rpc_service_.get());
    grpc_coord_service->SetCoordinationServiceInstance(coord_service_.get());
    // Start the server.
    server_ = builder.BuildAndStart();
    // Only start RPC loop after the service is live.
    coord_rpc_thread_.reset(Env::Default()->StartThread(
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
    static_cast<tsl::GrpcCoordinationServiceImpl*>(coord_rpc_service_.get())
        ->SetCoordinationServiceInstance(nullptr);
    coord_rpc_service_->Shutdown();
    coord_rpc_thread_ = nullptr;
    coord_rpc_service_ = nullptr;
    coord_compute_pool_ = nullptr;

    // Destroy the server.
    server_ = nullptr;
  }

  void TearDown() override { StopService(); }

 private:
  std::string service_address_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<tsl::CoordinationServiceInterface> coord_service_;
  std::unique_ptr<tsl::thread::ThreadPool> coord_compute_pool_;
  std::unique_ptr<tsl::AsyncServiceInterface> coord_rpc_service_;
  std::unique_ptr<tsl::Thread> coord_rpc_thread_;
};

TEST_F(ClientServerTest, ConnectAndShutdownAreBarriers) {
  int num_nodes = 3;
  StartService(num_nodes);

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
      absl::MutexLock lock(&mu);
      mu.Await(absl::Condition(&my_connect_turn));
      ++connect_count;
    }
    TF_RETURN_IF_ERROR(client->Connect());
    // Verify that all of the threads have called Connect() by the time we get
    // here.
    {
      absl::MutexLock lock(&mu);
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
      absl::MutexLock lock(&mu);
      mu.Await(absl::Condition(&my_shutdown_turn));
      ++shutdown_count;
    }
    TF_RETURN_IF_ERROR(client->Shutdown());
    {
      absl::MutexLock lock(&mu);
      if (shutdown_count != num_nodes) {
        return absl::InternalError(absl::StrCat(
            "Shutdown count is ", shutdown_count, " but expected ", num_nodes));
      }
    }

    return absl::OkStatus();
  };

  std::vector<absl::Status> statuses(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }
  for (int i = 0; i < num_nodes; ++i) {
    TF_EXPECT_OK(statuses[i]);
  }
}

TEST_F(ClientServerTest, ClientsTerminateShutdownIfAnyClientGoesAway) {
  int num_nodes = 3;
  StartService(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id,
                            /*init_and_shutdown_timeout=*/absl::Seconds(3),
                            /*shutdown_on_destruction=*/node_id != 0);

    TF_RETURN_IF_ERROR(client->Connect());

    if (node_id == 0) {
      return absl::OkStatus();
    }

    // The call to Shutdown() should be interrupted if a worker stops issuing
    // heartbeats.
    return client->Shutdown();
  };

  std::vector<absl::Status> statuses(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }
  TF_EXPECT_OK(statuses[0]);
  for (int i = 1; i < num_nodes; ++i) {
    EXPECT_THAT(
        statuses[i],
        AnyOf(
            // Shutdown barrier took too long and failed.
            StatusIs(absl::StatusCode::kInternal, HasSubstr("timed out")),
            // Agent polled error first, and so Shutdown()
            // fails because agent is already in error.
            StatusIs(absl::StatusCode::kFailedPrecondition)));
  }
}

TEST_F(ClientServerTest, ClientsShutdownSuccessfully) {
  int num_nodes = 3;
  StartService(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);

    TF_RETURN_IF_ERROR(client->Connect());
    return client->Shutdown();
    // The error polling request will be cancelled automatically when the
    // client is shutting down.
  };

  std::vector<absl::Status> statuses(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }
  for (int i = 0; i < num_nodes; ++i) {
    TF_EXPECT_OK(statuses[i]);
  }
}

TEST_F(ClientServerTest, MissedHeartbeatCallbackIsExecutedIfAnyClientGoesAway) {
  int num_nodes = 3;
  StartService(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    absl::Notification shutdown;
    auto client = GetClient(
        node_id,
        /*init_and_shutdown_timeout=*/absl::Seconds(3),
        /*shutdown_on_destruction=*/node_id != 0,
        /*error_fn=*/[&](const absl::Status& status) { shutdown.Notify(); });

    TF_RETURN_IF_ERROR(client->Connect());

    if (node_id == 0) {
      return absl::OkStatus();
    }
    shutdown.WaitForNotification();
    return absl::OkStatus();
  };

  std::vector<absl::Status> statuses(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }
  for (int i = 0; i < num_nodes; ++i) {
    TF_EXPECT_OK(statuses[i]);
  }
}

TEST_F(ClientServerTest, ShutdownErrorIsPropagatedToClients) {
  int num_nodes = 2;
  StartService(num_nodes);
  std::vector<absl::Status> statuses = {
      absl::UnknownError("Uninitialized status."),
      absl::UnknownError("Uninitialized status.")};
  absl::Notification shutdown;

  auto thread_fn = [&](int node_id) {
    auto client = GetClient(
        node_id,
        /*init_and_shutdown_timeout=*/absl::Seconds(2),
        /*shutdown_on_destruction=*/true,
        /*error_fn=*/[&statuses, node_id](const absl::Status& status) {
          statuses[node_id] = status;
        });

    TF_ASSERT_OK(client->Connect());

    if (node_id == 0) {
      // Shut down early.
      client = nullptr;
      shutdown.Notify();
    } else {
      // Block until shutdown barrier times out.
      shutdown.WaitForNotification();
    }
  };

  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { thread_fn(i); });
    }
  }
  EXPECT_THAT(statuses[0], StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(statuses[1], StatusIs(absl::StatusCode::kInternal));
}

TEST_F(ClientServerTest, ClientsTerminateIfServiceGoesAway) {
#if defined(ADDRESS_SANITIZER)
  GTEST_SKIP()
      << "This test is known to produce memory leaks due to ungraceful "
         "termination of the RPC server despite having pending connections.";
#endif
  int num_nodes = 3;
  StartService(num_nodes);

  absl::Barrier barrier(num_nodes + 1);

  auto thread_fn = [&](int node_id) -> absl::Status {
    absl::Notification shutdown;
    auto client = GetClient(
        /*node_id=*/node_id,
        /*init_and_shutdown_timeout=*/absl::Seconds(10),
        /*shutdown_on_destruction=*/true,
        /*error_fn=*/[&](const absl::Status& status) { shutdown.Notify(); });

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
  StartService(num_nodes);

  absl::Barrier barrier(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id,
                            /*init_and_shutdown_timeout=*/absl::Seconds(20));

    barrier.Block();
    absl::SleepFor(absl::Milliseconds(200) * node_id);
    TF_RETURN_IF_ERROR(client->Connect());
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
  }
  for (int i = 0; i < num_nodes; ++i) {
    TF_EXPECT_OK(statuses[i]);
  }
}

// We should eventually time out if a client does not show up.
TEST_F(ClientServerTest, ConnectEventuallyTimesOutIfAClientDoesNotShowUp) {
  int num_nodes = 3;
  StartService(num_nodes);

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
  StartService(num_nodes,
               /*init_and_shutdown_timeout=*/absl::Seconds(5));
  absl::Notification n;

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client =
        GetClient(node_id, /*init_and_shutdown_timeout=*/absl::Seconds(5));

    TF_RETURN_IF_ERROR(client->Connect());
    // All clients have successfully connected at this point.
    // Simulate client restart by creating a new client.
    if (node_id == 2) {
      client = nullptr;
      auto restarted_client =
          GetClient(node_id, /*init_and_shutdown_timeout=*/absl::Seconds(5));
      auto status = restarted_client->Connect();
      n.Notify();
      return status;
    }
    n.WaitForNotification();
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
  }
  // Errors should have been propagated to the clients, and thus the shutdown
  // call will fail with `FailedPrecondition` since the tasks are already in
  // error.
  EXPECT_THAT(statuses[0], StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(statuses[1], StatusIs(absl::StatusCode::kFailedPrecondition));
  // This client was restarted, so its connection attempt will be aborted.
  EXPECT_THAT(statuses[2], StatusIs(absl::StatusCode::kAborted));
}

// If a client restarts during init, it can silently reconnect because no
// stateful operations have run yet, so the program state is still valid.
TEST_F(ClientServerTest, ClientRestart_DuringConnect_Succeeds) {
  int num_nodes = 3;
  StartService(num_nodes,
               /*init_and_shutdown_timeout=*/absl::Seconds(5));
  absl::Notification previous_node_2_connecting, node_2_restarted;

  std::vector<absl::Status> statuses(num_nodes + 1);
  auto thread_fn = [&](int node_id) -> absl::Status {
    bool restarted_node_2 = false;
    if (node_id == 3) {
      restarted_node_2 = true;
      node_id = 2;  // This is the restarted client.
    }
    auto client =
        GetClient(node_id, /*init_and_shutdown_timeout=*/absl::Seconds(5));

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
  EXPECT_THAT(statuses[0], StatusIs(absl::StatusCode::kOk));
  EXPECT_THAT(statuses[1], StatusIs(absl::StatusCode::kOk));
  // This was the initial connection attempt that should be aborted.
  EXPECT_THAT(statuses[2], StatusIs(absl::StatusCode::kAlreadyExists));
  // This was the restarted client which should silently reconnect.
  EXPECT_THAT(statuses[3], StatusIs(absl::StatusCode::kOk));
}

TEST_F(ClientServerTest, WaitAtBarrier_Succeed) {
  int num_nodes = 2;
  StartService(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    TF_RETURN_IF_ERROR(client->Connect());

    TF_RETURN_IF_ERROR(client->WaitAtBarrier("barrier_1", kBarrierTimeout, {}));
    TF_RETURN_IF_ERROR(client->WaitAtBarrier("barrier_2", kBarrierTimeout, {}));

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
  }
  for (int i = 0; i < num_nodes; ++i) {
    TF_EXPECT_OK(statuses[i]);
  }
}

TEST_F(ClientServerTest, WaitAtBarrier_Timeout) {
  int num_nodes = 2;
  StartService(num_nodes);
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

  std::vector<absl::Status> statuses(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }
  for (int i = 0; i < num_nodes; ++i) {
    // Co-ordination service returns the status of the previous barrier
    // failure without waiting for the thread to time out.
    EXPECT_EQ(statuses[i].code(), tsl::error::DEADLINE_EXCEEDED)
        << " node id: " << i;
  }
}

TEST_F(ClientServerTest, WaitAtBarrier_TimeoutWithDifferentBarrierId) {
  int num_nodes = 2;
  StartService(num_nodes);

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

  std::vector<absl::Status> statuses(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }
  for (int i = 0; i < num_nodes; ++i) {
    EXPECT_EQ(statuses[i].code(), tsl::error::DEADLINE_EXCEEDED)
        << " node id: " << i;
  }
}

TEST_F(ClientServerTest, WaitAtBarrier_ReuseSameId_Succeeds) {
  int num_nodes = 2;
  StartService(num_nodes);

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

  std::vector<absl::Status> statuses(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }
  for (int i = 0; i < num_nodes; ++i) {
    TF_EXPECT_OK(statuses[i]);
  }
}

TEST_F(ClientServerTest, WaitAtBarrier_RestartAndBarrierAgain_Fails) {
  int num_nodes = 2;
  // Allow clients to connect by themselves so restarted client can connect and
  // try barrier again.
  StartService(num_nodes, /*init_and_shutdown_timeout=*/absl::Seconds(2),
               /*cluster_register_with_barrier=*/false,
               /*cluster_shutdown_with_barrier=*/false);
  absl::Status barrier_status;
  absl::Notification n;

  auto thread_fn = [&](int node_id) {
    auto client = GetClient(node_id);
    TF_ASSERT_OK(client->Connect());

    // Complete barrier 3 times (simulate job progress).
    for (int i = 0; i < 3; ++i) {
      TF_ASSERT_OK(client->WaitAtBarrier("barrier_1", kBarrierTimeout, {}));
    }
    if (node_id == 1) {
      client = nullptr;  // Simulate client restart.
      auto restarted_client = GetClient(1);
      TF_ASSERT_OK(restarted_client->Connect());
      // This should fail! This variable is checked after the thread pool is
      // destroyed.
      barrier_status =
          restarted_client->WaitAtBarrier("barrier_1", kBarrierTimeout, {});
      n.Notify();
    }
    // Client 0 should only be destroyed after we get the barrier result.
    n.WaitForNotification();
  };

  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { thread_fn(i); });
    }
  }
  EXPECT_THAT(barrier_status,
              StatusIs(absl::StatusCode::kInternal, HasSubstr("restarted")));
}

TEST_F(ClientServerTest,
       WaitAtBarrier_TimeoutThenOkay_StragglingTaskGetsSameError) {
  int num_nodes = 2;
  StartService(num_nodes);
  absl::Notification n, n_2;
  absl::Status status_0, status_0_new, status_1, status_1_new;
  auto thread_fn = [&](int node_id) {
    auto client = GetClient(node_id);
    TF_ASSERT_OK(client->Connect());
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
  };

  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { thread_fn(i); });
    }
  }
  // Both nodes should get the same error.
  EXPECT_THAT(status_0, StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_THAT(status_1, StatusIs(absl::StatusCode::kDeadlineExceeded));
  // Next barrier call is okay.
  TF_EXPECT_OK(status_0_new);
  TF_EXPECT_OK(status_1_new);
}

TEST_F(ClientServerTest,
       WaitAtBarrier_QuickTaskStartBarrierTwice_LateTaskGetsSlowError) {
  int num_nodes = 2;
  StartService(num_nodes);
  absl::Notification n;
  absl::Status status_0, status_0_new, status_1;
  auto thread_fn = [&](int node_id) {
    auto client = GetClient(node_id);
    TF_ASSERT_OK(client->Connect());
    TF_ASSERT_OK(client->WaitAtBarrier("barrier_1", kBarrierTimeout, {}));
    TF_ASSERT_OK(client->WaitAtBarrier("barrier_1", kBarrierTimeout, {}));
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
  };

  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { thread_fn(i); });
    }
  }
  // Both barriers from node 0 should time out.
  EXPECT_THAT(status_0, StatusIs(absl::StatusCode::kDeadlineExceeded));
  EXPECT_THAT(status_0_new, StatusIs(absl::StatusCode::kDeadlineExceeded));
  // Next barrier call from node 1 gets barrier counter mismatch error.
  EXPECT_THAT(status_1, StatusIs(absl::StatusCode::kInternal,
                                 HasSubstr("too quick / slow")));
}

TEST_F(ClientServerTest, WaitAtBarrierSubset_Succeeds) {
  int num_nodes = 3;
  StartService(num_nodes);
  absl::Notification n0, n1;

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    TF_RETURN_IF_ERROR(client->Connect());

    if (node_id != 2) {
      TF_RETURN_IF_ERROR(client->WaitAtBarrier("barrier_1", kBarrierTimeout,
                                               {GetTask(0), GetTask(1)}));
    }

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
    for (int i = 0; i < num_nodes; ++i) {
      TF_EXPECT_OK(statuses[i]);
    }
  }
}

TEST_F(ClientServerTest, WaitAtBarrier_DifferentSubset_Fails) {
  int num_nodes = 2;
  StartService(num_nodes);
  absl::Notification n;
  absl::Status status_0, status_1 = absl::UnknownError("Uninitialized error.");

  auto thread_fn = [&](int node_id) {
    auto client = GetClient(node_id);
    TF_ASSERT_OK(client->Connect());
    if (node_id == 0) {
      status_0 =
          client->WaitAtBarrier("barrier_1", kBarrierTimeout, {GetTask(0)});
      n.Notify();
    } else {
      n.WaitForNotification();
      // Same barrier id, but specifies different tasks.
      status_1 =
          client->WaitAtBarrier("barrier_1", kBarrierTimeout, {GetTask(1)});
    }
  };

  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { thread_fn(i); });
    }
  }
  // First barrier call succeeds.
  TF_EXPECT_OK(status_0);
  // Second barrier call with different task args fails.
  EXPECT_THAT(status_1, StatusIs(absl::StatusCode::kInvalidArgument,
                                 HasSubstr("Conflicting tasks specified")));
}

TEST_F(ClientServerTest, CancelNonExistentBarrier_Fails) {
  int num_nodes = 1;
  StartService(num_nodes);
  auto client = GetClient(0);
  TF_ASSERT_OK(client->Connect());

  EXPECT_THAT(client->CancelBarrier("non_existent_barrier"),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(ClientServerTest,
       WaitAtBarrierSubsetNonParticipatingProcessAttempts_Fails) {
  int num_nodes = 3;
  StartService(num_nodes);
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
    auto status = client->WaitAtBarrier("barrier_1", kBarrierTimeout,
                                        {GetTask(0), GetTask(1)});
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

TEST_F(ClientServerTest, GetKeyValueDir) {
  StartService(/*num_nodes=*/1);
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
  StartService(/*num_nodes=*/1);
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
  StartService(/*num_nodes=*/1);
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
  StartService(/*num_nodes=*/1);
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
  StartService(/*num_nodes=*/1);
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

TEST_F(ClientServerTest, Dtor_CancelsOngoingGetKeyValueAndBarrier) {
  // Set 2 nodes with no register barrier to allowing pending barrier RPC.
  StartService(/*num_nodes=*/2, /*init_and_shutdown_timeout=*/absl::Seconds(2),
               /*cluster_register_with_barrier=*/false);
  auto client = GetClient(/*node_id=*/0,
                          /*init_and_shutdown_timeout=*/absl::Seconds(2),
                          /*shutdown_on_destruction=*/false);
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

}  // namespace
}  // namespace tsl
