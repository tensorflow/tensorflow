/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/session_mgr.h"

#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class SessionMgrTest : public ::testing::Test {
 protected:
  SessionMgrTest()
      : mgr_(&env_, "/job:mnist/replica:0/task:0",
             std::unique_ptr<WorkerCacheInterface>(),
             factory_),
        legacy_session_(mgr_.WorkerSessionForSession("novel_session_id")) {}

  WorkerEnv env_;
  SessionMgr::WorkerCacheFactory factory_ =
      [](const ServerDef& server_def, WorkerCacheInterface** worker_cache) {
        *worker_cache = nullptr;  // Set to null to make debugging easier.
        return Status::OK();
      };
  SessionMgr mgr_;
  WorkerSession* legacy_session_;
};

TEST_F(SessionMgrTest, CreateSessionSimple) {
  ServerDef server_def;
  string session_handle = "test_session_handle";
  TF_EXPECT_OK(mgr_.CreateSession(session_handle, server_def));
  WorkerSession* session = mgr_.WorkerSessionForSession(session_handle);
  EXPECT_NE(nullptr, session) << "Session for " << session_handle << "was null";
  EXPECT_NE(mgr_.LegacySession(), session);
  TF_EXPECT_OK(mgr_.DeleteSession(session_handle));
}

TEST_F(SessionMgrTest, LegacySession) {
  ServerDef server_def;
  string session_handle = "";
  WorkerSession* session = mgr_.WorkerSessionForSession(session_handle);
  EXPECT_EQ(mgr_.LegacySession(), session);

  TF_EXPECT_OK(mgr_.DeleteSession(session_handle));
}

TEST_F(SessionMgrTest, WorkerNameFromServerDef) {
  ServerDef server_def;
  server_def.set_job_name("worker");
  server_def.set_task_index(3);
  string worker_name = SessionMgr::WorkerNameFromServerDef(server_def);
  EXPECT_EQ("/job:worker/replica:0/task:3", worker_name);
}

TEST_F(SessionMgrTest, DeleteLegacySession) {
  TF_EXPECT_OK(mgr_.DeleteSession("legacy_session"));
}

}  // namespace tensorflow
