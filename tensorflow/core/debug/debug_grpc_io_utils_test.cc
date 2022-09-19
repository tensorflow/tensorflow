/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/debug/debug_graph_utils.h"
#include "tensorflow/core/debug/debug_grpc_testlib.h"
#include "tensorflow/core/debug/debug_io_utils.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {

class GrpcDebugTest : public ::testing::Test {
 protected:
  struct ServerData {
    int port;
    string url;
    std::unique_ptr<test::TestEventListenerImpl> server;
    std::unique_ptr<thread::ThreadPool> thread_pool;
  };

  void SetUp() override {
    ClearEnabledWatchKeys();
    SetUpInProcessServer(&server_data_, 0);
  }

  void TearDown() override { TearDownInProcessServer(&server_data_); }

  void SetUpInProcessServer(ServerData* server_data,
                            int64_t server_start_delay_micros) {
    server_data->port = testing::PickUnusedPortOrDie();
    server_data->url = strings::StrCat("grpc://localhost:", server_data->port);
    server_data->server.reset(new test::TestEventListenerImpl());

    server_data->thread_pool.reset(
        new thread::ThreadPool(Env::Default(), "test_server", 1));
    server_data->thread_pool->Schedule(
        [server_data, server_start_delay_micros]() {
          Env::Default()->SleepForMicroseconds(server_start_delay_micros);
          server_data->server->RunServer(server_data->port);
        });
  }

  void TearDownInProcessServer(ServerData* server_data) {
    server_data->server->StopServer();
    server_data->thread_pool.reset();
  }

  void ClearEnabledWatchKeys() { DebugGrpcIO::ClearEnabledWatchKeys(); }

  const int64_t GetChannelConnectionTimeoutMicros() {
    return DebugGrpcIO::channel_connection_timeout_micros_;
  }

  void SetChannelConnectionTimeoutMicros(const int64_t timeout) {
    DebugGrpcIO::channel_connection_timeout_micros_ = timeout;
  }

  ServerData server_data_;
};

TEST_F(GrpcDebugTest, ConnectionTimeoutWorks) {
  // Use a short timeout so the test won't take too long.
  const int64_t kOriginalTimeoutMicros = GetChannelConnectionTimeoutMicros();
  const int64_t kShortTimeoutMicros = 500 * 1000;
  SetChannelConnectionTimeoutMicros(kShortTimeoutMicros);
  ASSERT_EQ(kShortTimeoutMicros, GetChannelConnectionTimeoutMicros());

  const string& kInvalidGrpcUrl =
      strings::StrCat("grpc://localhost:", testing::PickUnusedPortOrDie());
  Tensor tensor(DT_FLOAT, TensorShape({1, 1}));
  tensor.flat<float>()(0) = 42.0;
  Status publish_status = DebugIO::PublishDebugTensor(
      DebugNodeKey("/job:localhost/replica:0/task:0/cpu:0", "foo_tensor", 0,
                   "DebugIdentity"),
      tensor, Env::Default()->NowMicros(), {kInvalidGrpcUrl});
  SetChannelConnectionTimeoutMicros(kOriginalTimeoutMicros);
  TF_ASSERT_OK(DebugIO::CloseDebugURL(kInvalidGrpcUrl));

  ASSERT_FALSE(publish_status.ok());
  const string expected_error_msg = strings::StrCat(
      "Failed to connect to gRPC channel at ", kInvalidGrpcUrl.substr(7),
      " within a timeout of ", kShortTimeoutMicros / 1e6, " s");
  ASSERT_NE(string::npos,
            publish_status.error_message().find(expected_error_msg));
}

TEST_F(GrpcDebugTest, ConnectionToDelayedStartingServerWorks) {
  ServerData server_data;
  // Server start will be delayed for 1 second.
  SetUpInProcessServer(&server_data, 1 * 1000 * 1000);

  Tensor tensor(DT_FLOAT, TensorShape({1, 1}));
  tensor.flat<float>()(0) = 42.0;
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "foo_tensor", 0, "DebugIdentity");
  Status publish_status = DebugIO::PublishDebugTensor(
      kDebugNodeKey, tensor, Env::Default()->NowMicros(), {server_data.url});
  ASSERT_TRUE(publish_status.ok());
  TF_ASSERT_OK(DebugIO::CloseDebugURL(server_data.url));

  ASSERT_EQ(1, server_data.server->node_names.size());
  ASSERT_EQ(1, server_data.server->output_slots.size());
  ASSERT_EQ(1, server_data.server->debug_ops.size());
  EXPECT_EQ(kDebugNodeKey.device_name, server_data.server->device_names[0]);
  EXPECT_EQ(kDebugNodeKey.node_name, server_data.server->node_names[0]);
  EXPECT_EQ(kDebugNodeKey.output_slot, server_data.server->output_slots[0]);
  EXPECT_EQ(kDebugNodeKey.debug_op, server_data.server->debug_ops[0]);
  TearDownInProcessServer(&server_data);
}

TEST_F(GrpcDebugTest, SendSingleDebugTensorViaGrpcTest) {
  Tensor tensor(DT_FLOAT, TensorShape({1, 1}));
  tensor.flat<float>()(0) = 42.0;
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "foo_tensor", 0, "DebugIdentity");
  TF_ASSERT_OK(DebugIO::PublishDebugTensor(
      kDebugNodeKey, tensor, Env::Default()->NowMicros(), {server_data_.url}));
  TF_ASSERT_OK(DebugIO::CloseDebugURL(server_data_.url));

  // Verify that the expected debug tensor sending happened.
  ASSERT_EQ(1, server_data_.server->node_names.size());
  ASSERT_EQ(1, server_data_.server->output_slots.size());
  ASSERT_EQ(1, server_data_.server->debug_ops.size());
  EXPECT_EQ(kDebugNodeKey.device_name, server_data_.server->device_names[0]);
  EXPECT_EQ(kDebugNodeKey.node_name, server_data_.server->node_names[0]);
  EXPECT_EQ(kDebugNodeKey.output_slot, server_data_.server->output_slots[0]);
  EXPECT_EQ(kDebugNodeKey.debug_op, server_data_.server->debug_ops[0]);
}

TEST_F(GrpcDebugTest, SendDebugTensorWithLargeStringAtIndex0ViaGrpcTest) {
  Tensor tensor(DT_STRING, TensorShape({1, 1}));
  tensor.flat<tstring>()(0) = string(5000 * 1024, 'A');
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "foo_tensor", 0, "DebugIdentity");
  const Status status = DebugIO::PublishDebugTensor(
      kDebugNodeKey, tensor, Env::Default()->NowMicros(), {server_data_.url});
  ASSERT_FALSE(status.ok());
  ASSERT_NE(status.error_message().find("string value at index 0 from debug "
                                        "node foo_tensor:0:DebugIdentity does "
                                        "not fit gRPC message size limit"),
            string::npos);
  TF_ASSERT_OK(DebugIO::CloseDebugURL(server_data_.url));
}

TEST_F(GrpcDebugTest, SendDebugTensorWithLargeStringAtIndex1ViaGrpcTest) {
  Tensor tensor(DT_STRING, TensorShape({1, 2}));
  tensor.flat<tstring>()(0) = "A";
  tensor.flat<tstring>()(1) = string(5000 * 1024, 'A');
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "foo_tensor", 0, "DebugIdentity");
  const Status status = DebugIO::PublishDebugTensor(
      kDebugNodeKey, tensor, Env::Default()->NowMicros(), {server_data_.url});
  ASSERT_FALSE(status.ok());
  ASSERT_NE(status.error_message().find("string value at index 1 from debug "
                                        "node foo_tensor:0:DebugIdentity does "
                                        "not fit gRPC message size limit"),
            string::npos);
  TF_ASSERT_OK(DebugIO::CloseDebugURL(server_data_.url));
}

TEST_F(GrpcDebugTest, SendMultipleDebugTensorsSynchronizedViaGrpcTest) {
  const int32_t kSends = 4;

  // Prepare the tensors to sent.
  std::vector<Tensor> tensors;
  for (int i = 0; i < kSends; ++i) {
    Tensor tensor(DT_INT32, TensorShape({1, 1}));
    tensor.flat<int>()(0) = i * i;
    tensors.push_back(tensor);
  }

  thread::ThreadPool* tp =
      new thread::ThreadPool(Env::Default(), "grpc_debug_test", kSends);

  mutex mu;
  Notification all_done;
  int tensor_count TF_GUARDED_BY(mu) = 0;
  std::vector<Status> statuses TF_GUARDED_BY(mu);

  const std::vector<string> urls({server_data_.url});

  // Set up the concurrent tasks of sending Tensors via an Event stream to the
  // server.
  auto fn = [this, &mu, &tensor_count, &tensors, &statuses, &all_done,
             &urls]() {
    int this_count;
    {
      mutex_lock l(mu);
      this_count = tensor_count++;
    }

    // Different concurrent tasks will send different tensors.
    const uint64 wall_time = Env::Default()->NowMicros();
    Status publish_status = DebugIO::PublishDebugTensor(
        DebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                     strings::StrCat("synchronized_node_", this_count), 0,
                     "DebugIdentity"),
        tensors[this_count], wall_time, urls);

    {
      mutex_lock l(mu);
      statuses.push_back(publish_status);
      if (this_count == kSends - 1 && !all_done.HasBeenNotified()) {
        all_done.Notify();
      }
    }
  };

  // Schedule the concurrent tasks.
  for (int i = 0; i < kSends; ++i) {
    tp->Schedule(fn);
  }

  // Wait for all client tasks to finish.
  all_done.WaitForNotification();
  delete tp;

  // Close the debug gRPC stream.
  Status close_status = DebugIO::CloseDebugURL(server_data_.url);
  ASSERT_TRUE(close_status.ok());

  // Check all statuses from the PublishDebugTensor calls().
  for (const Status& status : statuses) {
    TF_ASSERT_OK(status);
  }

  // One prep tensor plus kSends concurrent tensors are expected.
  ASSERT_EQ(kSends, server_data_.server->node_names.size());
  for (size_t i = 0; i < server_data_.server->node_names.size(); ++i) {
    std::vector<string> items =
        str_util::Split(server_data_.server->node_names[i], '_');
    int tensor_index;
    strings::safe_strto32(items[2], &tensor_index);

    ASSERT_EQ(TensorShape({1, 1}),
              server_data_.server->debug_tensors[i].shape());
    ASSERT_EQ(tensor_index * tensor_index,
              server_data_.server->debug_tensors[i].flat<int>()(0));
  }
}

TEST_F(GrpcDebugTest, SendDebugTensorsThroughMultipleRoundsUsingGrpcGating) {
  // Prepare the tensor to send.
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "test_namescope/test_node", 0,
                                   "DebugIdentity");
  Tensor tensor(DT_INT32, TensorShape({1, 1}));
  tensor.flat<int>()(0) = 42;

  const std::vector<string> urls({server_data_.url});
  for (int i = 0; i < 3; ++i) {
    server_data_.server->ClearReceivedDebugData();
    const uint64 wall_time = Env::Default()->NowMicros();

    // On the 1st send (i == 0), gating is disabled, so data should be sent.
    // On the 2nd send (i == 1), gating is enabled, and the server has enabled
    //   the watch key in the previous send, so data should be sent.
    // On the 3rd send (i == 2), gating is enabled, but the server has disabled
    //   the watch key in the previous send, so data should not be sent.
    const bool enable_gated_grpc = (i != 0);
    TF_ASSERT_OK(DebugIO::PublishDebugTensor(kDebugNodeKey, tensor, wall_time,
                                             urls, enable_gated_grpc));

    server_data_.server->RequestDebugOpStateChangeAtNextStream(
        i == 0 ? EventReply::DebugOpStateChange::READ_ONLY
               : EventReply::DebugOpStateChange::DISABLED,
        kDebugNodeKey);

    // Close the debug gRPC stream.
    Status close_status = DebugIO::CloseDebugURL(server_data_.url);
    ASSERT_TRUE(close_status.ok());

    // Check dumped files according to the expected gating results.
    if (i < 2) {
      ASSERT_EQ(1, server_data_.server->node_names.size());
      ASSERT_EQ(1, server_data_.server->output_slots.size());
      ASSERT_EQ(1, server_data_.server->debug_ops.size());
      EXPECT_EQ(kDebugNodeKey.device_name,
                server_data_.server->device_names[0]);
      EXPECT_EQ(kDebugNodeKey.node_name, server_data_.server->node_names[0]);
      EXPECT_EQ(kDebugNodeKey.output_slot,
                server_data_.server->output_slots[0]);
      EXPECT_EQ(kDebugNodeKey.debug_op, server_data_.server->debug_ops[0]);
    } else {
      ASSERT_EQ(0, server_data_.server->node_names.size());
    }
  }
}

TEST_F(GrpcDebugTest, SendDebugTensorsThroughMultipleRoundsUnderReadWriteMode) {
  // Prepare the tensor to send.
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "test_namescope/test_node", 0,
                                   "DebugIdentity");
  Tensor tensor(DT_INT32, TensorShape({1, 1}));
  tensor.flat<int>()(0) = 42;

  const std::vector<string> urls({server_data_.url});
  for (int i = 0; i < 3; ++i) {
    server_data_.server->ClearReceivedDebugData();
    const uint64 wall_time = Env::Default()->NowMicros();

    // On the 1st send (i == 0), gating is disabled, so data should be sent.
    // On the 2nd send (i == 1), gating is enabled, and the server has enabled
    //   the watch key in the previous send (READ_WRITE), so data should be
    //   sent. In this iteration, the server response with a EventReply proto to
    //   unblock the debug node.
    // On the 3rd send (i == 2), gating is enabled, but the server has disabled
    //   the watch key in the previous send, so data should not be sent.
    const bool enable_gated_grpc = (i != 0);
    TF_ASSERT_OK(DebugIO::PublishDebugTensor(kDebugNodeKey, tensor, wall_time,
                                             urls, enable_gated_grpc));

    server_data_.server->RequestDebugOpStateChangeAtNextStream(
        i == 0 ? EventReply::DebugOpStateChange::READ_WRITE
               : EventReply::DebugOpStateChange::DISABLED,
        kDebugNodeKey);

    // Close the debug gRPC stream.
    Status close_status = DebugIO::CloseDebugURL(server_data_.url);
    ASSERT_TRUE(close_status.ok());

    // Check dumped files according to the expected gating results.
    if (i < 2) {
      ASSERT_EQ(1, server_data_.server->node_names.size());
      ASSERT_EQ(1, server_data_.server->output_slots.size());
      ASSERT_EQ(1, server_data_.server->debug_ops.size());
      EXPECT_EQ(kDebugNodeKey.device_name,
                server_data_.server->device_names[0]);
      EXPECT_EQ(kDebugNodeKey.node_name, server_data_.server->node_names[0]);
      EXPECT_EQ(kDebugNodeKey.output_slot,
                server_data_.server->output_slots[0]);
      EXPECT_EQ(kDebugNodeKey.debug_op, server_data_.server->debug_ops[0]);
    } else {
      ASSERT_EQ(0, server_data_.server->node_names.size());
    }
  }
}

TEST_F(GrpcDebugTest, TestGateDebugNodeOnEmptyEnabledSet) {
  ASSERT_FALSE(DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity",
                                            {"grpc://localhost:3333"}));

  // file:// debug URLs are not subject to grpc gating.
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen(
      "foo:0:DebugIdentity", {"grpc://localhost:3333", "file:///tmp/tfdbg_1"}));
}

TEST_F(GrpcDebugTest, TestGateDebugNodeOnNonEmptyEnabledSet) {
  const string kGrpcUrl1 = "grpc://localhost:3333";
  const string kGrpcUrl2 = "grpc://localhost:3334";

  DebugGrpcIO::SetDebugNodeKeyGrpcState(
      kGrpcUrl1, "foo:0:DebugIdentity",
      EventReply::DebugOpStateChange::READ_ONLY);
  DebugGrpcIO::SetDebugNodeKeyGrpcState(
      kGrpcUrl1, "bar:0:DebugIdentity",
      EventReply::DebugOpStateChange::READ_ONLY);

  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("foo:1:DebugIdentity", {kGrpcUrl1}));
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("foo:1:DebugNumericSummary", {kGrpcUrl1}));
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("qux:0:DebugIdentity", {kGrpcUrl1}));
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity", {kGrpcUrl1}));
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("bar:0:DebugIdentity", {kGrpcUrl1}));

  // Wrong grpc:// debug URLs.
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity", {kGrpcUrl2}));
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("bar:0:DebugIdentity", {kGrpcUrl2}));

  // file:// debug URLs are not subject to grpc gating.
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("qux:0:DebugIdentity",
                                           {"file:///tmp/tfdbg_1", kGrpcUrl1}));
}

TEST_F(GrpcDebugTest, TestGateDebugNodeOnMultipleEmptyEnabledSets) {
  const string kGrpcUrl1 = "grpc://localhost:3333";
  const string kGrpcUrl2 = "grpc://localhost:3334";
  const string kGrpcUrl3 = "grpc://localhost:3335";

  DebugGrpcIO::SetDebugNodeKeyGrpcState(
      kGrpcUrl1, "foo:0:DebugIdentity",
      EventReply::DebugOpStateChange::READ_ONLY);
  DebugGrpcIO::SetDebugNodeKeyGrpcState(
      kGrpcUrl2, "bar:0:DebugIdentity",
      EventReply::DebugOpStateChange::READ_ONLY);

  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity", {kGrpcUrl1}));
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("bar:0:DebugIdentity", {kGrpcUrl2}));
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity", {kGrpcUrl2}));
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("bar:0:DebugIdentity", {kGrpcUrl1}));
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity", {kGrpcUrl3}));
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("bar:0:DebugIdentity", {kGrpcUrl3}));
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity",
                                           {kGrpcUrl1, kGrpcUrl2}));
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("bar:0:DebugIdentity",
                                           {kGrpcUrl1, kGrpcUrl2}));
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity",
                                           {kGrpcUrl1, kGrpcUrl3}));
  ASSERT_FALSE(DebugIO::IsDebugNodeGateOpen("bar:0:DebugIdentity",
                                            {kGrpcUrl1, kGrpcUrl3}));
}

TEST_F(GrpcDebugTest, TestGateDebugNodeOnNonEmptyEnabledSetAndEmptyURLs) {
  DebugGrpcIO::SetDebugNodeKeyGrpcState(
      "grpc://localhost:3333", "foo:0:DebugIdentity",
      EventReply::DebugOpStateChange::READ_ONLY);

  std::vector<string> debug_urls_1;
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("foo:1:DebugIdentity", debug_urls_1));
}

TEST_F(GrpcDebugTest, TestGateCopyNodeOnEmptyEnabledSet) {
  const string kGrpcUrl1 = "grpc://localhost:3333";
  const string kWatch1 = "foo:0:DebugIdentity";

  ASSERT_FALSE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch1, kGrpcUrl1, true)}));
  ASSERT_TRUE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch1, kGrpcUrl1, false)}));

  // file:// debug URLs are not subject to grpc gating.
  ASSERT_TRUE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec("foo:0:DebugIdentity", kGrpcUrl1, true),
       DebugWatchAndURLSpec("foo:0:DebugIdentity", "file:///tmp/tfdbg_1",
                            false)}));
}

TEST_F(GrpcDebugTest, TestGateCopyNodeOnNonEmptyEnabledSet) {
  const string kGrpcUrl1 = "grpc://localhost:3333";
  const string kGrpcUrl2 = "grpc://localhost:3334";
  const string kWatch1 = "foo:0:DebugIdentity";
  const string kWatch2 = "foo:1:DebugIdentity";
  DebugGrpcIO::SetDebugNodeKeyGrpcState(
      kGrpcUrl1, kWatch1, EventReply::DebugOpStateChange::READ_ONLY);

  ASSERT_TRUE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch1, kGrpcUrl1, true)}));

  ASSERT_FALSE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch1, kGrpcUrl2, true)}));
  ASSERT_TRUE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch1, kGrpcUrl2, false)}));

  ASSERT_FALSE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch2, kGrpcUrl1, true)}));
  ASSERT_TRUE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch2, kGrpcUrl1, false)}));

  ASSERT_TRUE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch1, kGrpcUrl1, true),
       DebugWatchAndURLSpec(kWatch1, kGrpcUrl2, true)}));
  ASSERT_TRUE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch1, kGrpcUrl1, true),
       DebugWatchAndURLSpec(kWatch2, kGrpcUrl2, true)}));
}

}  // namespace tensorflow
