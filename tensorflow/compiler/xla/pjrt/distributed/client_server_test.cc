/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "grpcpp/grpcpp.h"
#include "absl/synchronization/barrier.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/pjrt/distributed/client.h"
#include "tensorflow/compiler/xla/pjrt/distributed/distributed.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.pb.h"
#include "tensorflow/compiler/xla/pjrt/distributed/service.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace xla {
namespace {

TEST(ClientServerTest, ConnectAndShutdownAreBarriers) {
  int num_nodes = 3;
  DistributedRuntimeServiceImpl::Options service_options;
  service_options.num_nodes = num_nodes;
  DistributedRuntimeServiceImpl service(service_options);
  ::grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();

  absl::Mutex mu;
  int connect_count = 0;
  int shutdown_count = 0;

  absl::Barrier barrier(num_nodes);

  auto thread_fn = [&](int node_id) -> xla::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.node_id = node_id;
    DistributedRuntimeClient client(
        server->InProcessChannel(::grpc::ChannelArguments()), client_options);
    GlobalTopologyProto topology;

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
    TF_RETURN_IF_ERROR(client.Connect());
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
    TF_RETURN_IF_ERROR(client.Shutdown());
    {
      absl::MutexLock lock(&mu);
      TF_RET_CHECK(shutdown_count == num_nodes);
    }

    return xla::Status::OK();
  };

  std::vector<xla::Status> statuses(num_nodes);
  {
    tensorflow::thread::ThreadPool thread_pool(tensorflow::Env::Default(),
                                               "test_threads", num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }
  for (int i = 0; i < num_nodes; ++i) {
    TF_EXPECT_OK(statuses[i]);
  }
}

TEST(ClientServerTest, ConnectAndEnumerateDevices) {
  DistributedRuntimeServiceImpl::Options service_options;
  service_options.num_nodes = 2;
  DistributedRuntimeServiceImpl service(service_options);
  ::grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();

  std::vector<LocalTopologyProto> locals(2);
  locals[0].set_node_id(0);
  locals[1].set_node_id(1);
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
  node0->mutable_devices(0)->set_global_device_id(0);
  node0->mutable_devices(1)->set_global_device_id(1);
  node0->mutable_devices(2)->set_global_device_id(2);
  *node1 = locals[1];
  node1->mutable_devices(0)->set_global_device_id(3);

  auto thread0_fn = [&]() -> xla::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.node_id = 0;
    DistributedRuntimeClient client(
        server->InProcessChannel(::grpc::ChannelArguments()), client_options);
    GlobalTopologyProto topology;
    TF_RETURN_IF_ERROR(client.Connect());
    TF_RETURN_IF_ERROR(client.EnumerateDevices(locals[0], &topology));
    TF_RET_CHECK(
        xla::protobuf_util::ProtobufEquals(topology, expected_topology))
        << topology.DebugString();
    TF_RETURN_IF_ERROR(client.KeyValueSet("key1", "value1"));
    TF_ASSIGN_OR_RETURN(
        std::string value,
        client.BlockingKeyValueGet("key2", absl::InfiniteDuration()));
    TF_RET_CHECK(value == "value2");
    return xla::Status::OK();
  };
  auto thread1_fn = [&]() -> xla::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.node_id = 1;
    DistributedRuntimeClient client(
        server->InProcessChannel(::grpc::ChannelArguments()), client_options);
    GlobalTopologyProto topology;
    TF_RETURN_IF_ERROR(client.Connect());
    TF_RETURN_IF_ERROR(client.EnumerateDevices(locals[1], &topology));
    TF_RET_CHECK(
        xla::protobuf_util::ProtobufEquals(topology, expected_topology))
        << topology.DebugString();
    TF_ASSIGN_OR_RETURN(
        std::string value,
        client.BlockingKeyValueGet("key1", absl::InfiniteDuration()));
    TF_RET_CHECK(value == "value1");
    TF_RETURN_IF_ERROR(client.KeyValueSet("key2", "value2"));
    return xla::Status::OK();
  };

  std::vector<std::function<xla::Status()>> functions = {thread0_fn,
                                                         thread1_fn};
  std::vector<xla::Status> statuses(functions.size());
  {
    tensorflow::thread::ThreadPool thread_pool(
        tensorflow::Env::Default(), "test_threads", functions.size());
    for (int i = 0; i < functions.size(); ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = functions[i](); });
    }
  }
  TF_EXPECT_OK(statuses[0]);
  TF_EXPECT_OK(statuses[1]);
}

TEST(ClientServerTest, ClientsTerminateShutdownIfAnyClientGoesAway) {
  int num_nodes = 3;
  DistributedRuntimeServiceImpl::Options service_options;
  service_options.num_nodes = num_nodes;
  service_options.heartbeat_interval = absl::Milliseconds(500);
  service_options.max_missing_heartbeats = 2;
  DistributedRuntimeServiceImpl service(service_options);
  ::grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();

  auto thread_fn = [&](int node_id) -> xla::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.node_id = node_id;
    client_options.heartbeat_interval = service_options.heartbeat_interval;
    client_options.max_missing_heartbeats = 2;
    client_options.shutdown_on_destruction = false;
    client_options.missed_heartbeat_callback =
        [&](xla::Status status, bool coordinator_initiated) {};
    DistributedRuntimeClient client(
        server->InProcessChannel(::grpc::ChannelArguments()), client_options);
    GlobalTopologyProto topology;

    TF_RETURN_IF_ERROR(client.Connect());

    if (node_id == 0) {
      return xla::Status::OK();
    }

    // The call to Shutdown() should be interrupted if a worker stops issuing
    // heartbeats.
    TF_RETURN_IF_ERROR(client.Shutdown());
    return xla::Status::OK();
  };

  std::vector<xla::Status> statuses(num_nodes);
  {
    tensorflow::thread::ThreadPool thread_pool(tensorflow::Env::Default(),
                                               "test_threads", num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }
  TF_EXPECT_OK(statuses[0]);
  for (int i = 1; i < num_nodes; ++i) {
    EXPECT_EQ(statuses[i].code(), tensorflow::error::ABORTED);
  }
}

TEST(ClientServerTest, ClientsReceiveMissedHeartbeatIfAnyClientGoesAway) {
  int num_nodes = 3;
  DistributedRuntimeServiceImpl::Options service_options;
  service_options.num_nodes = num_nodes;
  service_options.heartbeat_interval = absl::Milliseconds(500);
  service_options.max_missing_heartbeats = 2;
  DistributedRuntimeServiceImpl service(service_options);
  ::grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();

  auto thread_fn = [&](int node_id) -> xla::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.node_id = node_id;
    client_options.heartbeat_interval = service_options.heartbeat_interval;
    client_options.max_missing_heartbeats = 2;
    client_options.shutdown_on_destruction = (node_id != 0);
    absl::Notification shutdown;
    client_options.missed_heartbeat_callback = [&](xla::Status status,
                                                   bool coordinator_initiated) {
      shutdown.Notify();
    };
    DistributedRuntimeClient client(
        server->InProcessChannel(::grpc::ChannelArguments()), client_options);
    GlobalTopologyProto topology;

    TF_RETURN_IF_ERROR(client.Connect());

    if (node_id == 0) {
      return xla::Status::OK();
    }
    shutdown.WaitForNotification();
    return xla::Status::OK();
  };

  std::vector<xla::Status> statuses(num_nodes);
  {
    tensorflow::thread::ThreadPool thread_pool(tensorflow::Env::Default(),
                                               "test_threads", num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }
  for (int i = 0; i < num_nodes; ++i) {
    TF_EXPECT_OK(statuses[i]);
  }
}

// TODO(phawkins): find out why this test fails in CI but works locally.
TEST(ClientServerTest, DISABLED_ClientsTerminateIfServiceGoesAway) {
  int num_nodes = 3;
  DistributedRuntimeServiceImpl::Options service_options;
  service_options.num_nodes = num_nodes;
  service_options.heartbeat_interval = absl::Milliseconds(500);
  service_options.max_missing_heartbeats = 2;
  // We use a socket connection for this test case because the in-process API
  // does not react well to the server being told to shutdown while there are
  // active clients.
  int port = tensorflow::testing::PickUnusedPortOrDie();
  auto server_credentials = ::grpc::InsecureServerCredentials();
  TF_ASSERT_OK_AND_ASSIGN(
      auto service,
      DistributedRuntimeService::Get(absl::StrCat("[::]:", port),
                                     server_credentials, service_options));

  absl::Barrier barrier(num_nodes + 1);

  auto thread_fn = [&](int node_id) -> xla::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.node_id = node_id;
    client_options.heartbeat_interval = service_options.heartbeat_interval;
    client_options.max_missing_heartbeats = 2;
    client_options.rpc_timeout = absl::Seconds(1);
    absl::Notification shutdown;
    client_options.missed_heartbeat_callback = [&](xla::Status status,
                                                   bool coordinator_initiated) {
      shutdown.Notify();
    };
    std::shared_ptr<::grpc::ChannelCredentials> creds =
        ::grpc::InsecureChannelCredentials();
    std::shared_ptr<::grpc::Channel> channel =
        ::grpc::CreateChannel(absl::StrCat("dns:///localhost:", port), creds);
    auto client =
        std::make_unique<DistributedRuntimeClient>(channel, client_options);
    GlobalTopologyProto topology;

    TF_RETURN_IF_ERROR(client->Connect());

    barrier.Block();
    shutdown.WaitForNotification();

    TF_RETURN_IF_ERROR(client->Shutdown());
    return xla::Status::OK();
  };

  std::vector<xla::Status> statuses(num_nodes);
  {
    tensorflow::thread::ThreadPool thread_pool(tensorflow::Env::Default(),
                                               "test_threads", num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
    barrier.Block();
    service = nullptr;
  }
  for (int i = 0; i < num_nodes; ++i) {
    EXPECT_EQ(statuses[i].code(), tensorflow::error::DEADLINE_EXCEEDED)
        << statuses[i];
  }
}

// We should eventually connect, even if some clients are late to show up.
TEST(ClientServerTest, LateClientsAreOk) {
  int num_nodes = 3;
  DistributedRuntimeServiceImpl::Options service_options;
  service_options.num_nodes = num_nodes;
  DistributedRuntimeServiceImpl service(service_options);
  ::grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();

  absl::Barrier barrier(num_nodes);

  auto thread_fn = [&](int node_id) -> xla::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.node_id = node_id;
    client_options.init_timeout = absl::Milliseconds(20000);
    client_options.rpc_timeout = absl::Milliseconds(200);
    DistributedRuntimeClient client(
        server->InProcessChannel(::grpc::ChannelArguments()), client_options);
    GlobalTopologyProto topology;

    barrier.Block();
    absl::SleepFor(absl::Milliseconds(200) * node_id);
    TF_RETURN_IF_ERROR(client.Connect());
    TF_RETURN_IF_ERROR(client.Shutdown());
    return xla::Status::OK();
  };

  std::vector<xla::Status> statuses(num_nodes);
  {
    tensorflow::thread::ThreadPool thread_pool(tensorflow::Env::Default(),
                                               "test_threads", num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }
  for (int i = 0; i < num_nodes; ++i) {
    TF_EXPECT_OK(statuses[i]);
  }
}

// We should eventually time out if a client does not show up.
TEST(ClientServerTest, ConnectEventuallyTimesOutIfAClientDoesNotShowUp) {
  int num_nodes = 3;
  DistributedRuntimeServiceImpl::Options service_options;
  service_options.num_nodes = num_nodes;
  DistributedRuntimeServiceImpl service(service_options);
  ::grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();

  auto thread_fn = [&](int node_id) -> xla::Status {
    DistributedRuntimeClient::Options client_options;
    client_options.node_id = node_id;
    client_options.init_timeout = absl::Milliseconds(500);
    client_options.rpc_timeout = absl::Milliseconds(200);
    DistributedRuntimeClient client(
        server->InProcessChannel(::grpc::ChannelArguments()), client_options);
    GlobalTopologyProto topology;

    TF_RETURN_IF_ERROR(client.Connect());
    TF_RETURN_IF_ERROR(client.Shutdown());
    return xla::Status::OK();
  };

  // Note: one fewer thread than 'num_nodes'.
  std::vector<xla::Status> statuses(num_nodes - 1);
  {
    tensorflow::thread::ThreadPool thread_pool(tensorflow::Env::Default(),
                                               "test_threads", num_nodes);
    for (int i = 0; i < num_nodes - 1; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }
  for (int i = 0; i < num_nodes - 1; ++i) {
    EXPECT_EQ(statuses[i].code(), tensorflow::error::DEADLINE_EXCEEDED);
  }
}

}  // namespace
}  // namespace xla
