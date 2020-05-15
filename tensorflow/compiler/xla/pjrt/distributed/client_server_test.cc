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

#include "grpcpp/security/server_credentials.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/pjrt/distributed/client.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.pb.h"
#include "tensorflow/compiler/xla/pjrt/distributed/service.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool.h"

namespace xla {
namespace {

TEST(ClientServerTest, ConnectToServer) {
  DistributedRuntimeServiceImpl service(/*num_nodes=*/2);
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
    DistributedRuntimeClient client(
        server->InProcessChannel(::grpc::ChannelArguments()));
    GlobalTopologyProto topology;
    TF_RETURN_IF_ERROR(client.Connect(locals[0], &topology));
    TF_RET_CHECK(
        xla::protobuf_util::ProtobufEquals(topology, expected_topology));
    TF_RETURN_IF_ERROR(client.KeyValueSet("key1", "value1"));
    TF_ASSIGN_OR_RETURN(
        std::string value,
        client.BlockingKeyValueGet("key2", absl::InfiniteDuration()));
    TF_RET_CHECK(value == "value2");
    return xla::Status::OK();
  };
  auto thread1_fn = [&]() -> xla::Status {
    DistributedRuntimeClient client(
        server->InProcessChannel(::grpc::ChannelArguments()));
    GlobalTopologyProto topology;
    TF_RETURN_IF_ERROR(client.Connect(locals[1], &topology));
    TF_RET_CHECK(
        xla::protobuf_util::ProtobufEquals(topology, expected_topology));
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

}  // namespace
}  // namespace xla
