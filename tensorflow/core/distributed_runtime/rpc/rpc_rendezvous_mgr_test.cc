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

#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"

#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// string -> Tensor<string>
Tensor V(const string& content) {
  Tensor tensor(DT_STRING, TensorShape({}));
  tensor.scalar<string>()() = content;
  return tensor;
}

// Tensor<string> -> string
string V(const Tensor& tensor) {
  CHECK_EQ(tensor.dtype(), DT_STRING);
  CHECK(TensorShapeUtils::IsScalar(tensor.shape()));
  return tensor.scalar<string>()();
}

TEST(RpcRendezvousMgrTest, LocalSendRecv) {
  WorkerEnv env;
  env.env = Env::Default();
  env.worker_name = "/job:mnist/replica:1/task:2";
  RpcRendezvousMgr rmgr(&env);
  const int64 step_id = 123;
  const string key = Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0));
  {
    Rendezvous* rendez = rmgr.Find(step_id);
    core::ScopedUnref unref(rendez);
    Rendezvous::Args args;
    TF_ASSERT_OK(rendez->Send(key, args, V("peach"), false));
  }
  {
    Tensor val(DT_FLOAT);
    bool val_dead = false;
    TF_ASSERT_OK(rmgr.RecvLocal(step_id, key, &val, &val_dead));
    EXPECT_EQ(V(val), "peach");
  }
  rmgr.Cleanup(step_id);
}

TEST(RpcRendezvousMgrTest, LocalAbort) {
  WorkerEnv env;
  env.env = Env::Default();
  env.worker_name = "/job:mnist/replica:1/task:2";
  RpcRendezvousMgr rmgr(&env);
  const string key = Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0));
  {  // Explicit Abort().
    const int64 step_id = 123;
    Rendezvous* rendez = rmgr.Find(step_id);
    core::ScopedUnref unref(rendez);
    SchedClosure([env, rendez]() {
      env.env->SleepForMicroseconds(100 * 1000);
      rendez->StartAbort(errors::Aborted(""));
    });
    Tensor val(DT_STRING);
    bool val_dead = false;
    Rendezvous::Args args;
    EXPECT_TRUE(errors::IsAborted(rendez->Recv(key, args, &val, &val_dead)));
  }
  {  // Cleanup causes Abort().
    const int64 step_id = 321;
    Rendezvous* rendez = rmgr.Find(step_id);
    core::ScopedUnref unref(rendez);
    SchedClosure([env, &rmgr, step_id]() {
      env.env->SleepForMicroseconds(100 * 1000);
      rmgr.Cleanup(step_id);
    });
    Tensor val(DT_STRING);
    bool val_dead = false;
    Rendezvous::Args args;
    EXPECT_TRUE(errors::IsAborted(rendez->Recv(key, args, &val, &val_dead)));
  }
}

TEST(RpcRendezvousMgrTest, CleanupAll) {
  WorkerEnv env;
  env.env = Env::Default();
  env.worker_name = "/job:mnist/replica:1/task:2";
  RpcRendezvousMgr rmgr(&env);
  const string key = Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0));
  {
    const int64 step_id = 123;
    Rendezvous* rendez = rmgr.Find(step_id);
    core::ScopedUnref unref(rendez);
    Rendezvous::Args args;
    TF_ASSERT_OK(rendez->Send(key, args, V("peach"), false));
    rmgr.CleanupAll();
    Tensor val(DT_STRING);
    bool val_dead = false;
    EXPECT_TRUE(errors::IsAborted(rendez->Recv(key, args, &val, &val_dead)));
  }
}

class DummyDeviceContext : public DeviceContext {
 public:
  explicit DummyDeviceContext(int stream_id) : stream_id_(stream_id) {}
  ~DummyDeviceContext() override {}
  int stream_id() const { return stream_id_; }

 private:
  const int stream_id_;
};

TEST(RpcRendezvousMgrTest, TransferDummyDeviceContext) {
  DummyDeviceContext* dc = new DummyDeviceContext(123);

  WorkerEnv env;
  env.env = Env::Default();
  env.worker_name = "/job:mnist/replica:1/task:2";
  RpcRendezvousMgr rmgr(&env);
  const int64 step_id = 123;
  const string key = Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0));
  {
    Rendezvous* rendez = rmgr.Find(step_id);
    core::ScopedUnref unref(rendez);
    Rendezvous::Args args;
    args.device_context = dc;
    TF_ASSERT_OK(rendez->Send(key, args, V("peach"), false));
  }
  {
    Notification n;
    rmgr.RecvLocalAsync(
        step_id, key, [&n](const Status& s, const Rendezvous::Args send_args,
                           const Rendezvous::Args recv_args, const Tensor& val,
                           bool is_dead) {
          auto send_dev_context =
              static_cast<DummyDeviceContext*>(send_args.device_context);
          CHECK_EQ(123, send_dev_context->stream_id());
          CHECK_EQ(V(val), "peach");
          n.Notify();
        });
    n.WaitForNotification();
  }
  rmgr.Cleanup(step_id);
  dc->Unref();
}

// NOTE: Remote Send/Recv is better tested in worker_test.cc

}  // namespace tensorflow
