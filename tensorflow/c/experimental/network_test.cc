/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/network.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <memory>
#include <string>

#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/experimental/network_internal.h"
#include "tensorflow/c/experimental/rendezvous.h"
#include "tensorflow/c/experimental/rendezvous_internal.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {

bool accept_functionA(const char* protocol_name) {
  return strcmp(protocol_name, "grpc+A") == 0;
}

bool accept_functionB(const char* protocol_name) {
  return strcmp(protocol_name, "grpc+B") == 0;
}

struct SomeServerData {
  bool server_started = false;
};

struct SomeRendezvousData {
  int test = 0;
};

void* init_function(const TF_GrpcServer* server, TF_Status* status) {
  SomeServerData* server_data = new SomeServerData();
  TF_SetStatus(status, TF_OK, "");
  return server_data;
}

void start_function(const TF_GrpcServer* server, void* context,
                    TF_Status* status) {
  auto* server_data = static_cast<SomeServerData*>(context);
  server_data->server_started = true;
  TF_SetStatus(status, TF_OK, "");
}

void stop_function(const TF_GrpcServer* server, void* context,
                   TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
}

void join_function(const TF_GrpcServer* server, void* context,
                   TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
}

void delete_function(void* context) {
  auto* server_data = static_cast<SomeServerData*>(context);
  delete server_data;
}

void* rendezvous_init_function(void* server_context) {
  return new SomeRendezvousData();
}

void Deallocator(void* data, size_t, void* arg) {
  tensorflow::cpu_allocator()->DeallocateRaw(data);
  *reinterpret_cast<bool*>(arg) = true;
}

void receive_from_remote_async_function(TF_ParsedKey* key,
                                        TF_RendezvousArgs* args,
                                        TF_RendezvousDoneCallback* callback,
                                        void* context) {
  // Create dummy tensor
  const int num_bytes = 6 * sizeof(float);
  float* values =
      reinterpret_cast<float*>(tensorflow::cpu_allocator()->AllocateRaw(
          EIGEN_MAX_ALIGN_BYTES, num_bytes));
  int64_t dims[] = {2, 3};
  bool deallocator_called = false;
  auto* tensor = TF_NewTensor(TF_FLOAT, dims, 2, values, num_bytes,
                              &Deallocator, &deallocator_called);
  callback->tensor = tensor;
  auto* tf_status = TF_NewStatus();
  TF_SetStatus(tf_status, TF_OK, "");
  callback->status = tf_status;
  TF_RendezvousDone(callback);
  TF_DeleteStatus(tf_status);
  TF_DeleteTensor(tensor);
}

void rendezvous_delete_function(void* context) {
  auto* rendezvous_data = static_cast<SomeRendezvousData*>(context);
  delete rendezvous_data;
}

tensorflow::ServerDef GetServerDef(const string& protocol,
                                   const string& job_name, int num_tasks) {
  tensorflow::ServerDef server_def;
  server_def.set_protocol(protocol);
  server_def.set_job_name(job_name);
  server_def.set_task_index(0);
  tensorflow::ClusterDef* cluster_def = server_def.mutable_cluster();
  tensorflow::JobDef* job_def = cluster_def->add_job();
  job_def->set_name(job_name);
  for (int i = 0; i < num_tasks; i++) {
    int port = tensorflow::testing::PickUnusedPortOrDie();
    job_def->mutable_tasks()->insert(
        {i, tensorflow::strings::StrCat("localhost:", port)});
  }
  return server_def;
}

class NetworksTest : public ::testing::Test {
 public:
  ~NetworksTest() override {}

  SomeServerData* GetServerData(CGrpcServer* server) {
    EXPECT_NE(server->context_, nullptr);
    return static_cast<SomeServerData*>(server->context_);
  }
};

Rendezvous::ParsedKey Key(const string& sender, const uint64 incarnation,
                          const string& receiver, const string& name) {
  Rendezvous::ParsedKey result;
  CHECK(
      Rendezvous::ParseKey(Rendezvous::CreateKey(sender, incarnation, receiver,
                                                 name, FrameAndIter(0, 0)),
                           &result)
          .ok());
  return result;
}

void InitializeRendezvous(GrpcServer* grpc_server, ServerDef* server_def,
                          RemoteRendezvous* remote_rendezvous) {
  int rendezvous_id = 0;
  auto session_name = tensorflow::strings::StrCat("test_", rendezvous_id);
  TF_EXPECT_OK(grpc_server->worker_env()->session_mgr->CreateSession(
      session_name, *server_def, true));

  std::shared_ptr<tensorflow::WorkerSession> worker_session;
  TF_EXPECT_OK(grpc_server->worker_env()->session_mgr->WorkerSessionForSession(
      session_name, &worker_session));

  TF_EXPECT_OK(remote_rendezvous->Initialize(worker_session.get()));
}

TEST_F(NetworksTest, TestStartServer) {
  auto* rendezvous_builder = TF_NewRemoteRendezvousBuilder(
      rendezvous_init_function, receive_from_remote_async_function,
      rendezvous_delete_function);

  TF_Status* tf_status = TF_NewStatus();
  TF_GrpcServerFactory* factory = TF_NewGrpcServerFactory(
      accept_functionA, init_function, start_function, stop_function,
      join_function, delete_function, rendezvous_builder);
  TF_RegisterGrpcServerFactory("testfactoryA", factory);

  ServerDef server_def = GetServerDef("grpc+A", "localhost", 1);
  std::unique_ptr<ServerInterface> server;
  TF_EXPECT_OK(NewServer(server_def, &server));
  auto* grpc_server = static_cast<CGrpcServer*>(server.get());
  auto* server_data = GetServerData(grpc_server);
  ASSERT_FALSE(server_data->server_started);

  TF_EXPECT_OK(server->Start());
  ASSERT_TRUE(server_data->server_started);

  TF_DeleteStatus(tf_status);
  TF_DeleteGrpcServerFactory(factory);
  TF_DeleteRemoteRendezvousBuilder(rendezvous_builder);
  // TODO(annarev): find a clean way to shutdown server.
  server.release();
}

TEST_F(NetworksTest, TestReceiveData) {
  auto* rendezvous_builder = TF_NewRemoteRendezvousBuilder(
      rendezvous_init_function, receive_from_remote_async_function,
      rendezvous_delete_function);

  TF_Status* tf_status = TF_NewStatus();
  TF_GrpcServerFactory* factory = TF_NewGrpcServerFactory(
      accept_functionB, init_function, start_function, stop_function,
      join_function, delete_function, rendezvous_builder);
  TF_RegisterGrpcServerFactory("testfactoryB", factory);

  ServerDef server_def = GetServerDef("grpc+B", "localhost", 1);
  std::unique_ptr<ServerInterface> server;
  TF_EXPECT_OK(NewServer(server_def, &server));
  auto* grpc_server = static_cast<CGrpcServer*>(server.get());

  TF_EXPECT_OK(server->Start());
  auto* rendezvous_mgr = grpc_server->worker_env()->rendezvous_mgr;
  auto* remote_rendezvous = rendezvous_mgr->Find(0);

  auto key = Key("/job:localhost/replica:1/task:2/device:CPU:0", 1,
                 "/job:localhost/replica:0/task:0/device:CPU:0", "test");
  Rendezvous::Args args;
  bool done_callback_called = false;
  auto* done_callback_called_ptr = &done_callback_called;
  absl::Notification notification;
  auto* notification_ptr = &notification;

  InitializeRendezvous(grpc_server, &server_def, remote_rendezvous);
  remote_rendezvous->RecvAsync(
      key, args,
      [done_callback_called_ptr, notification_ptr](
          const Status&, const Rendezvous::Args&, const Rendezvous::Args&,
          const Tensor&, const bool) mutable {
        *done_callback_called_ptr = true;
        notification_ptr->Notify();
      });
  notification.WaitForNotificationWithTimeout(absl::Seconds(10));
  ASSERT_EQ(done_callback_called, true);

  TF_DeleteStatus(tf_status);
  TF_DeleteGrpcServerFactory(factory);
  TF_DeleteRemoteRendezvousBuilder(rendezvous_builder);
  // Server doesn't have a clean shutdown.
  server.release();
}

}  // namespace tensorflow
