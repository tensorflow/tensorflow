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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
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
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/support/channel_arguments.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/distributed/topology_util.h"
#include "xla/protobuf_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace {
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using tsl::testing::StatusIs;

constexpr absl::Duration kHeartbeatInterval = absl::Milliseconds(500);
constexpr int kMaxMissingHeartbeats = 5;
constexpr absl::Duration kBarrierTimeout = absl::Milliseconds(200);

class ClientServerTest : public testing::Test {
 public:
  std::shared_ptr<DistributedRuntimeClient> GetClient(
      int node_id, DistributedRuntimeClient::Options client_options = {},
      std::shared_ptr<::grpc::Channel> channel = nullptr) {
    client_options.node_id = node_id;
    // Set a small heartbeat interval for quicker tests.
    client_options.heartbeat_interval = kHeartbeatInterval;
    client_options.max_missing_heartbeats = kMaxMissingHeartbeats;
    if (channel == nullptr) {
      channel = coord_service_->server()->InProcessChannel(
          ::grpc::ChannelArguments());
    }
    return GetDistributedRuntimeClient(channel, client_options);
  }

  void StartService(int num_nodes,
                    CoordinationServiceImpl::Options service_options = {}) {
    int port = tsl::testing::PickUnusedPortOrDie();
    service_address_ = absl::StrCat("[::]:", port);

    service_options.num_nodes = num_nodes;
    // Set a small heartbeat interval for quicker tests.
    service_options.heartbeat_interval = kHeartbeatInterval;
    service_options.max_missing_heartbeats = kMaxMissingHeartbeats;

    // Set up and register service on the gRPC server.
    coord_service_ = DistributedRuntimeService::Get(
                         service_address_, ::grpc::InsecureServerCredentials(),
                         service_options)
                         .value();
  }

  std::string service_address() { return service_address_; }

  void StopService() { coord_service_ = nullptr; }

 private:
  std::unique_ptr<DistributedRuntimeService> coord_service_;
  std::string service_address_ = "";
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
      TF_RET_CHECK(connect_count == num_nodes);
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
      TF_RET_CHECK(shutdown_count == num_nodes);
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

TEST_F(ClientServerTest, ConnectAndEnumerateDevices) {
  StartService(/*num_nodes=*/2);

  std::string host_0_boot_id = "foo";
  std::string host_1_boot_id = "bar";
  std::vector<LocalTopologyProto> locals(2);
  locals[0].set_node_id(0);
  locals[1].set_node_id(1);
  locals[0].set_boot_id(host_0_boot_id);
  locals[1].set_boot_id(host_1_boot_id);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(0);
  DeviceProto* d2 = locals[0].add_devices();
  d2->set_local_device_ordinal(707);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);

  GlobalTopologyProto expected_topology;
  auto* node0 = expected_topology.add_nodes();
  auto* node1 = expected_topology.add_nodes();
  *node0 = locals[0];
  node0->set_boot_id(host_0_boot_id);
  node0->mutable_devices(0)->set_global_device_id(0);
  node0->mutable_devices(1)->set_global_device_id(1);
  node0->mutable_devices(2)->set_global_device_id(2);
  node0->mutable_devices(0)->set_slice_index(0);
  node0->mutable_devices(1)->set_slice_index(0);
  node0->mutable_devices(2)->set_slice_index(0);
  *node1 = locals[1];
  node1->set_boot_id(host_1_boot_id);
  node1->mutable_devices(0)->set_global_device_id(3);
  node1->mutable_devices(0)->set_slice_index(1);

  // Used to ensure that thread0's client sends their device after thread1's
  // client. This ensures that devices are sent out of turn (compared to their
  // node ids).
  absl::Notification n;
  auto thread0_fn = [&]() -> absl::Status {
    auto client = GetClient(/*node_id=*/0);
    GlobalTopologyProto topology;
    TF_RETURN_IF_ERROR(client->Connect());
    // Wait until second thread sends their device info to the service. This
    // tests that devices are set in the order of their node ids even if they
    // are sent out of turn.
    n.WaitForNotification();
    // Sleep a short while for the other thread to send their device info first.
    absl::SleepFor(absl::Seconds(1));

    auto kv_store = GetDistributedKeyValueStore(client, /*key_prefix=*/"");
    TF_RETURN_IF_ERROR(
        ExchangeTopologies("cuda", /*node_id=*/0, /*num_nodes=*/2,
                           /*get_local_topology_timeout=*/absl::Minutes(1),
                           /*get_global_topology_timeout=*/absl::Minutes(1),
                           kv_store.get(), locals[0], &topology,
                           /*assign_global_device_ids=*/true));
    TF_RET_CHECK(
        xla::protobuf_util::ProtobufEquals(topology, expected_topology))
        << topology.DebugString();
    TF_RETURN_IF_ERROR(client->KeyValueSet("key1", "value1"));
    TF_ASSIGN_OR_RETURN(
        std::string value,
        client->BlockingKeyValueGet("key2", absl::InfiniteDuration()));
    TF_RET_CHECK(value == "value2");
    return absl::OkStatus();
  };
  auto thread1_fn = [&]() -> absl::Status {
    auto client = GetClient(/*node_id=*/1);
    GlobalTopologyProto topology;
    TF_RETURN_IF_ERROR(client->Connect());
    // Unblock the first thread after sending device info to the service. This
    // tests that devices are set in the order of their node ids even if they
    // are sent out of turn.
    // We cannot send the notification after the call since there is a barrier
    // within the call that would cause a deadlock.
    n.Notify();
    auto kv_store = GetDistributedKeyValueStore(client, /*key_prefix=*/"");
    TF_RETURN_IF_ERROR(ExchangeTopologies(
        "cuda", /*node_id=*/1, /*num_nodes=*/2,
        /*get_local_topology_timeout=*/absl::Minutes(1),
        /*get_global_topology_timeout=*/absl::Minutes(1), kv_store.get(),
        locals[1], &topology, /*assign_global_device_ids=*/true));
    TF_RET_CHECK(
        xla::protobuf_util::ProtobufEquals(topology, expected_topology))
        << topology.DebugString();
    TF_ASSIGN_OR_RETURN(
        std::string value,
        client->BlockingKeyValueGet("key1", absl::InfiniteDuration()));
    TF_RET_CHECK(value == "value1");
    TF_RETURN_IF_ERROR(client->KeyValueSet("key2", "value2"));
    return absl::OkStatus();
  };

  std::vector<std::function<absl::Status()>> functions = {thread0_fn,
                                                          thread1_fn};
  std::vector<absl::Status> statuses(functions.size());
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        functions.size());
    for (int i = 0; i < functions.size(); ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = functions[i](); });
    }
  }
  TF_EXPECT_OK(statuses[0]);
  TF_EXPECT_OK(statuses[1]);
}

// Make sure device list is ordered by 0,1,...,10 instead of 0,1,10,2,...,9.
TEST_F(ClientServerTest, EnumerateElevenDevices) {
  int num_nodes = 11;
  StartService(num_nodes);
  std::vector<LocalTopologyProto> locals(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    locals[i].set_node_id(i);
    // Two unique boot_id, one per host.
    locals[i].set_boot_id(absl::StrCat("test_boot_id_", i % 2));
    auto device = locals[i].add_devices();
    // Split local devices across two hosts.
    int ordinal = i % (num_nodes / 2);
    device->set_local_device_ordinal(ordinal);
    device->set_name("test_device");
    device->set_vendor("test_vendor");
  }
  GlobalTopologyProto expected_topology;
  for (int i = 0; i < num_nodes; ++i) {
    auto* node = expected_topology.add_nodes();
    *node = locals[i];
    node->mutable_devices(0)->set_global_device_id(i);
    node->mutable_devices(0)->set_slice_index(i % 2);
  }

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    GlobalTopologyProto topology;
    TF_RETURN_IF_ERROR(client->Connect());
    auto kv_store = GetDistributedKeyValueStore(client, /*key_prefix=*/"");
    TF_RETURN_IF_ERROR(ExchangeTopologies(
        "cuda", /*node_id=*/node_id, num_nodes,
        /*get_local_topology_timeout=*/absl::Minutes(1),
        /*get_global_topology_timeout=*/absl::Minutes(1), kv_store.get(),
        locals[node_id], &topology, /*assign_global_device_ids=*/true));
    TF_RET_CHECK(
        xla::protobuf_util::ProtobufEquals(topology, expected_topology))
        << topology.DebugString();
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

// Setting `init_timeout` to 0 means that the client should attempt connection
// only once, but the client should still wait a short while for other tasks.
TEST_F(ClientServerTest, ZeroInitTimeoutShouldStillWaitForOtherTasks) {
  int num_nodes = 2;
  StartService(num_nodes);

  absl::Barrier barrier(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.init_timeout = absl::ZeroDuration();
    auto client = GetClient(node_id, client_options);

    // Node 0 will connect to the service immediately, but still wait for the
    // straggling node 1.
    if (node_id == 1) {
      absl::SleepFor(absl::Seconds(5));
    }
    TF_RETURN_IF_ERROR(client->Connect());

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

TEST_F(ClientServerTest,
       ClientsTerminateShutdownIfAnyClientGoesAway_WithoutErrorPolling) {
  int num_nodes = 3;
  StartService(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.shutdown_on_destruction = node_id != 0;
    client_options.poll_for_error_from_service_at_startup = false;
    client_options.missed_heartbeat_callback = [&](absl::Status status) {};
    auto client = GetClient(node_id, client_options);

    TF_RETURN_IF_ERROR(client->Connect());

    if (node_id == 0) {
      return absl::OkStatus();
    }

    // The call to Shutdown() should be interrupted if a worker stops issuing
    // heartbeats.
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
  TF_EXPECT_OK(statuses[0]);
  for (int i = 1; i < num_nodes; ++i) {
    // Other nodes will be placed into ERROR state when the service informs
    // them of node 0's missing heartbeat failure.
    // agent->Shutdown() may lead into two different error codes depending on
    // the timing of the call:
    // 1. Internal: node turns into ERROR state during the shutdown call.
    // 2. Failed Precondition: node is already in ERROR state before the
    // shutdown call (note: agent will still stop sending heartbeats).
    EXPECT_TRUE(absl::IsInternal(statuses[i]) ||
                absl::IsFailedPrecondition(statuses[i]));
  }
}

TEST_F(ClientServerTest, ClientsTerminateShutdownIfAnyClientGoesAway) {
  int num_nodes = 3;
  StartService(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.shutdown_on_destruction = node_id != 0;
    client_options.missed_heartbeat_callback = [&](absl::Status status) {};
    auto client = GetClient(node_id, client_options);

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
    // The error type depends on whether the node turns into ERROR state during
    // or before the shutdown call.
    EXPECT_TRUE(absl::IsInternal(statuses[i]) ||
                absl::IsFailedPrecondition(statuses[i]));
  }
}

TEST_F(ClientServerTest, ClientsShutdownSuccessfully) {
  int num_nodes = 3;
  StartService(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.shutdown_on_destruction = true;
    client_options.missed_heartbeat_callback = [&](absl::Status status) {};
    auto client = GetClient(node_id, client_options);

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
    DistributedRuntimeClient::Options client_options;
    client_options.shutdown_on_destruction = (node_id != 0);
    absl::Notification shutdown;
    client_options.missed_heartbeat_callback = [&](absl::Status status) {
      shutdown.Notify();
    };
    auto client = GetClient(node_id, client_options);

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

TEST_F(ClientServerTest,
       ClientsReceiveMissedHeartbeatIfAnyClientGoesAway_WithoutErrorPolling) {
  int num_nodes = 3;
  StartService(num_nodes);

  auto thread_fn = [&](int node_id) -> absl::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.shutdown_on_destruction = (node_id != 0);
    absl::Notification shutdown;
    client_options.missed_heartbeat_callback = [&](absl::Status status) {
      shutdown.Notify();
    };
    client_options.poll_for_error_from_service_at_startup = false;
    auto client = GetClient(node_id, client_options);

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
    DistributedRuntimeClient::Options client_options;
    client_options.rpc_timeout = absl::Seconds(1);
    client_options.shutdown_timeout = absl::Seconds(10);
    absl::Notification shutdown;
    client_options.missed_heartbeat_callback = [&](absl::Status status) {
      shutdown.Notify();
    };
    auto channel = GetDistributedRuntimeClientChannel(
        service_address(), ::grpc::InsecureChannelCredentials());
    auto client = GetClient(node_id, client_options, channel);

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
    DistributedRuntimeClient::Options client_options;
    client_options.init_timeout = absl::Seconds(20);
    client_options.rpc_timeout = absl::Milliseconds(200);
    auto client = GetClient(node_id, client_options);

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
  absl::Duration timeout = absl::Milliseconds(100);
  CoordinationServiceImpl::Options service_options;
  service_options.cluster_register_timeout = timeout;
  service_options.shutdown_timeout = timeout;
  StartService(num_nodes, service_options);

  auto thread_fn = [&](int node_id) -> absl::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.init_timeout = timeout;
    client_options.rpc_timeout = timeout;
    // Overwrite the default error callback which invokes LOG(QFATAL).
    client_options.missed_heartbeat_callback = [](absl::Status status) {
      LOG(ERROR) << "Distributed client has missing heartbeats: " << status;
    };
    auto client = GetClient(node_id, client_options);

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
  absl::Duration timeout = absl::Seconds(5);
  CoordinationServiceImpl::Options service_options;
  service_options.cluster_register_timeout = timeout;
  service_options.shutdown_timeout = timeout;
  StartService(num_nodes, service_options);
  absl::Notification n;

  auto thread_fn = [&](int node_id) -> absl::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.init_timeout = timeout;
    client_options.rpc_timeout = timeout;
    // Overwrite the default error callback which invokes LOG(QFATAL).
    client_options.missed_heartbeat_callback = [](absl::Status status) {
      LOG(ERROR) << "Distributed client has missing heartbeats: " << status;
    };
    auto client = GetClient(node_id, client_options);

    TF_RETURN_IF_ERROR(client->Connect());
    // All clients have successfully connected at this point.
    // Simulate client restart by creating a new client.
    if (node_id == 2) {
      client = nullptr;
      auto restarted_client = GetClient(node_id, client_options);
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
  absl::Duration timeout = absl::Seconds(5);
  CoordinationServiceImpl::Options service_options;
  service_options.cluster_register_timeout = timeout;
  service_options.shutdown_timeout = timeout;
  StartService(num_nodes, service_options);
  absl::Notification previous_node_2_connecting, node_2_restarted;

  std::vector<absl::Status> statuses(num_nodes + 1);
  auto thread_fn = [&](int node_id) -> absl::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.init_timeout = timeout;
    client_options.rpc_timeout = timeout;
    // Overwrite the default error callback which invokes LOG(QFATAL).
    client_options.missed_heartbeat_callback = [](absl::Status status) {
      LOG(ERROR) << "Distributed client has missing heartbeats: " << status;
    };
    bool restarted_node_2 = false;
    if (node_id == 3) {
      restarted_node_2 = true;
      node_id = 2;  // This is the restarted client.
    }
    auto client = GetClient(node_id, client_options);

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

    TF_RETURN_IF_ERROR(
        client->WaitAtBarrier("barrier_1", kBarrierTimeout, std::nullopt));
    TF_RETURN_IF_ERROR(
        client->WaitAtBarrier("barrier_2", kBarrierTimeout, std::nullopt));

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
        client->WaitAtBarrier("barrier_1", kBarrierTimeout, std::nullopt);
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
    TF_RETURN_IF_ERROR(
        client->WaitAtBarrier(barrier_id, kBarrierTimeout, std::nullopt));

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

TEST_F(ClientServerTest, WaitAtBarrierSubset_Succeeds) {
  int num_nodes = 3;
  StartService(num_nodes);
  absl::Notification n0, n1;

  auto thread_fn = [&](int node_id) -> absl::Status {
    auto client = GetClient(node_id);
    TF_RETURN_IF_ERROR(client->Connect());

    if (node_id != 2) {
      TF_RETURN_IF_ERROR(client->WaitAtBarrier(
          "barrier_1", kBarrierTimeout, absl::Span<const int32_t>{0, 1}));
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
                                        absl::Span<const int32_t>{0, 1});
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

TEST_F(ClientServerTest, KeyValueDirGet) {
  StartService(/*num_nodes=*/1);
  auto client = GetClient(/*node_id=*/0);
  TF_ASSERT_OK(client->Connect());
  TF_ASSERT_OK(client->KeyValueSet("test_dir/sub_dir/1", "1"));
  TF_ASSERT_OK(client->KeyValueSet("test_dir/sub_dir/2", "2"));
  TF_ASSERT_OK(client->KeyValueSet("test_dir/3", "3"));
  TF_ASSERT_OK(client->KeyValueSet("test", "4"));  // Not in a directory.

  auto results = client->KeyValueDirGet("test_dir/");

  TF_ASSERT_OK(results.status());
  auto kvs = results.value();

  EXPECT_THAT(kvs, UnorderedElementsAre(Pair("test_dir/sub_dir/1", "1"),
                                        Pair("test_dir/sub_dir/2", "2"),
                                        Pair("test_dir/3", "3")));
}

TEST_F(ClientServerTest, KeyValueSet_Duplicate_Fails) {
  StartService(/*num_nodes=*/1);
  auto client = GetClient(/*node_id=*/0);
  TF_ASSERT_OK(client->Connect());
  TF_ASSERT_OK(client->KeyValueSet("test_key", "original_value"));
  EXPECT_TRUE(
      absl::IsAlreadyExists(client->KeyValueSet("test_key", "never_added")));
  auto result =
      client->BlockingKeyValueGet("test_key", absl::Milliseconds(100));
  TF_ASSERT_OK(result.status());
  EXPECT_EQ(result.value(), "original_value");
}

TEST_F(ClientServerTest, KeyValueSet_Duplicate_Overwrites) {
  StartService(/*num_nodes=*/1);
  auto client = GetClient(/*node_id=*/0);
  TF_ASSERT_OK(client->Connect());
  TF_ASSERT_OK(client->KeyValueSet("test_key", "original_value"));
  TF_EXPECT_OK(client->KeyValueSet("test_key", "overwritten_value",
                                   /*allow_overwrite=*/true));
  auto result =
      client->BlockingKeyValueGet("test_key", absl::Milliseconds(100));
  TF_ASSERT_OK(result.status());
  EXPECT_EQ(result.value(), "overwritten_value");
}

TEST_F(ClientServerTest, KeyValueDelete) {
  StartService(/*num_nodes=*/1);
  auto client = GetClient(/*node_id=*/0);
  TF_ASSERT_OK(client->Connect());
  TF_ASSERT_OK(client->KeyValueSet("to_be_deleted", "deleted"));
  TF_ASSERT_OK(client->KeyValueSet("to_be_kept", "kept"));

  auto results = client->KeyValueDelete("to_be_deleted");

  TF_EXPECT_OK(results);
  auto deleted_kv =
      client->BlockingKeyValueGet("to_be_deleted", absl::Milliseconds(200));
  // We time out from attempting to retrieve a deleted key.
  EXPECT_EQ(deleted_kv.status().code(), tsl::error::DEADLINE_EXCEEDED);
  // Other key should still exist.
  auto kept_kv =
      client->BlockingKeyValueGet("to_be_kept", absl::Milliseconds(200));
  TF_ASSERT_OK(kept_kv.status());
  EXPECT_EQ(kept_kv.value(), "kept");
}

TEST_F(ClientServerTest, KeyValueDelete_Directory) {
  StartService(/*num_nodes=*/1);
  auto client = GetClient(/*node_id=*/0);
  TF_ASSERT_OK(client->Connect());
  TF_ASSERT_OK(client->KeyValueSet("test_dir/sub_dir/1", "1"));
  TF_ASSERT_OK(client->KeyValueSet("test_dir/sub_dir/2", "2"));
  TF_ASSERT_OK(client->KeyValueSet("test_dir/3", "3"));

  auto results = client->KeyValueDelete("test_dir/");

  TF_EXPECT_OK(results);
  auto kvs = client->KeyValueDirGet("test_dir/");
  TF_ASSERT_OK(kvs.status());
  EXPECT_THAT(kvs.value(), IsEmpty());
}

TEST_F(ClientServerTest, UseCompression) {
  StartService(/*num_nodes=*/1);

  // Sanity check that the client can connect with compression enabled.
  auto channel = GetDistributedRuntimeClientChannel(
      service_address(), ::grpc::InsecureChannelCredentials(),
      /*use_compression=*/true);
  auto client = GetClient(/*node_id=*/0, {}, channel);

  TF_ASSERT_OK(client->Connect());
  TF_ASSERT_OK(client->KeyValueSet("foo/bar/1", "1"));
  TF_ASSERT_OK(client->Shutdown());
}

}  // namespace
}  // namespace xla
