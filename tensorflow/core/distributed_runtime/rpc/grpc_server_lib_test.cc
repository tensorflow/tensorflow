/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/server_lib.h"

#include "tensorflow/core/distributed_runtime/rpc/grpc_session.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// Tests that a server can be cleanly started, stopped, and joined
// when no calls are made against the server.
TEST(Server, StopAfterNoop) {
  ServerDef def;
  def.set_protocol("grpc");
  def.set_job_name("localhost");
  def.set_task_index(0);
  JobDef* job_def = def.mutable_cluster()->add_job();
  job_def->set_name("localhost");
  (*job_def->mutable_tasks())[0] =
      strings::StrCat("localhost:", testing::PickUnusedPortOrDie());
  std::unique_ptr<ServerInterface> svr;
  TF_EXPECT_OK(NewServer(def, &svr));
  TF_EXPECT_OK(svr->Start());
  TF_EXPECT_OK(svr->Stop());
  TF_EXPECT_OK(svr->Join());
}

// Tests that a server can be cleanly started, stopped, and joined
// when a simple call is made against the server.
TEST(Server, StopAfterCall) {
  ServerDef def;
  def.set_protocol("grpc");
  def.set_job_name("localhost");
  def.set_task_index(0);
  JobDef* job_def = def.mutable_cluster()->add_job();
  job_def->set_name("localhost");
  int port = testing::PickUnusedPortOrDie();
  (*job_def->mutable_tasks())[0] = strings::StrCat("localhost:", port);
  std::unique_ptr<ServerInterface> svr;
  TF_EXPECT_OK(NewServer(def, &svr));
  TF_EXPECT_OK(svr->Start());
  {
    SessionOptions options;
    options.target = strings::StrCat("grpc://localhost:", port);
    std::unique_ptr<GrpcSession> sess(new GrpcSession(options));
    const std::vector<DeviceAttributes> devices = sess->ListDevices();
    EXPECT_GT(devices.size(), 0);
  }
  TF_EXPECT_OK(svr->Stop());
  TF_EXPECT_OK(svr->Join());
}

}  // namespace tensorflow
