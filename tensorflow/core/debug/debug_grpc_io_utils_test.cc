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
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace {

class GrpcDebugTest : public ::testing::Test {
 protected:
  bool SetUpServer() {
    // Obtain port number for the test server.
    int port = testing::PickUnusedPortOrDie();

    server_client_pair.reset(new test::GrpcTestServerClientPair(port));

    // Launch a debug test server in a subprocess.
    const string test_server_bin = strings::StrCat(
        testing::TensorFlowSrcRoot(), "/core/debug/debug_test_server_main");
    const std::vector<string> argv(
        {test_server_bin,
         strings::Printf("%d", server_client_pair->server_port),
         server_client_pair->dump_root});
    subprocess_ = testing::CreateSubProcess(argv);

    return subprocess_->Start();
  }

  void TearDownServer() {
    // Stop the test server subprocess.
    subprocess_->Kill(9);

    // Clean up server dump directory.
    int64 undeleted_files = -1;
    int64 undeleted_dirs = -1;
    Env::Default()->DeleteRecursively(server_client_pair->dump_root,
                                      &undeleted_files, &undeleted_dirs);

    ASSERT_EQ(0, undeleted_files);
    ASSERT_EQ(0, undeleted_dirs);
  }

  std::unique_ptr<test::GrpcTestServerClientPair> server_client_pair;

 private:
  std::shared_ptr<SubProcess> subprocess_;
};

TEST_F(GrpcDebugTest, AttemptToSendToNonexistentGrpcAddress) {
  Tensor tensor(DT_FLOAT, TensorShape({1, 1}));
  tensor.flat<float>()(0) = 42.0;

  const string kInvalidGrpcUrl = "grpc://0.0.0.0:0";

  // Attempt to publish debug tensor to the invalid URL should lead to a non-OK
  // Status.
  Status publish_status = DebugIO::PublishDebugTensor(
      "foo_tensor", "DebugIdentity", tensor, Env::Default()->NowMicros(),
      {kInvalidGrpcUrl});
  ASSERT_FALSE(publish_status.ok());
  ASSERT_NE(
      string::npos,
      publish_status.error_message().find(
          "Channel at the following gRPC address is not ready: 0.0.0.0:0"));

  DebugIO::CloseDebugURL(kInvalidGrpcUrl);
}

TEST_F(GrpcDebugTest, SendSingleDebugTensorViaGrpcTest) {
  // Start the server process.
  ASSERT_TRUE(SetUpServer());

  // Poll the server with Event stream requests until first success.
  ASSERT_TRUE(server_client_pair->PollTillFirstRequestSucceeds());

  // Verify that the expected dump file exists.
  std::vector<string> dump_files;
  Env::Default()->GetChildren(server_client_pair->dump_root, &dump_files);

  ASSERT_EQ(1, dump_files.size());
  ASSERT_EQ(0, dump_files[0].find("prep_node_0_DebugIdentity_"));

  TearDownServer();
}

TEST_F(GrpcDebugTest, SendMultipleDebugTensorsSynchronizedViaGrpcTest) {
  const int kSends = 4;

  // Start the server process.
  ASSERT_TRUE(SetUpServer());

  // Prepare the tensors to sent.
  std::vector<Tensor> tensors;
  for (int i = 0; i < kSends; ++i) {
    Tensor tensor(DT_INT32, TensorShape({1, 1}));
    tensor.flat<int>()(0) = i * i;
    tensors.push_back(tensor);
  }

  // Poll the server with Event stream requests until first success.
  ASSERT_TRUE(server_client_pair->PollTillFirstRequestSucceeds());

  thread::ThreadPool* tp =
      new thread::ThreadPool(Env::Default(), "grpc_debug_test", kSends);

  mutex mu;
  Notification all_done;
  int tensor_count GUARDED_BY(mu) = 0;
  std::vector<Status> statuses GUARDED_BY(mu);

  const std::vector<string> urls({server_client_pair->test_server_url});

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
        strings::StrCat("synchronized_node_", this_count, ":0"),
        "DebugIdentity", tensors[this_count], wall_time, urls);

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
  Status close_status =
      DebugIO::CloseDebugURL(server_client_pair->test_server_url);
  ASSERT_TRUE(close_status.ok());

  // Check all statuses from the PublishDebugTensor calls().
  for (const Status& status : statuses) {
    TF_ASSERT_OK(status);
  }

  // Load the dump files generated by the server upon receiving the tensors
  // via the Event stream.
  std::vector<string> dump_files;
  Env::Default()->GetChildren(server_client_pair->dump_root, &dump_files);

  // One prep tensor plus kSends concurrent tensors are expected.
  ASSERT_EQ(1 + kSends, dump_files.size());

  // Verify the content of the dumped tensors (in Event proto files).
  for (const string& dump_file : dump_files) {
    if (dump_file.find("prep_node") == 0) {
      continue;
    }

    std::vector<string> items = str_util::Split(dump_file, '_');
    int tensor_index;
    strings::safe_strto32(items[2], &tensor_index);

    const string file_path =
        io::JoinPath(server_client_pair->dump_root, dump_file);

    Event event;
    TF_ASSERT_OK(ReadEventFromFile(file_path, &event));

    const TensorProto& tensor_proto = event.summary().value(0).tensor();
    Tensor tensor(tensor_proto.dtype());
    ASSERT_TRUE(tensor.FromProto(tensor_proto));

    // Verify the content of the tensor sent via the Event stream.
    ASSERT_EQ(TensorShape({1, 1}), tensor.shape());
    ASSERT_EQ(tensor_index * tensor_index, tensor.flat<int>()(0));
  }

  TearDownServer();
}

}  // namespace
}  // namespace tensorflow
