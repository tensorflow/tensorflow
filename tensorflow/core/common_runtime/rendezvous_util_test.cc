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
#include "tensorflow/core/common_runtime/rendezvous_util.h"

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class RendezvousUtilTest : public ::testing::Test {
 public:
  RendezvousUtilTest() { rendez_ = NewLocalRendezvous(); }

  ~RendezvousUtilTest() override { rendez_->Unref(); }

  Rendezvous* rendez_;
};

// string -> Tensor<string>
Tensor V(const string& content) {
  Tensor tensor(DT_STRING, TensorShape({}));
  tensor.scalar<tstring>()() = content;
  return tensor;
}

// Tensor<string> -> string
string V(const Tensor& tensor) {
  CHECK_EQ(tensor.dtype(), DT_STRING);
  CHECK(TensorShapeUtils::IsScalar(tensor.shape()));
  return tensor.scalar<tstring>()();
}

string MakeStringKey(const string& name) {
  return Rendezvous::CreateKey(
      "/job:localhost/replica:0/task:0/device:CPU:0", 0,
      "/job:localhost/replica:0/task:0/device:GPU:0", name, FrameAndIter(0, 0));
}

TEST_F(RendezvousUtilTest, SendBeforeRecv) {
  // Fire off sends before receive the tensors.
  TF_ASSERT_OK(SendTensorsToRendezvous(
      rendez_, nullptr, {}, {MakeStringKey("hello1"), MakeStringKey("hello2")},
      {V("hello1"), V("hello2")}));

  Notification n;
  std::vector<Tensor> received_keys;
  RecvOutputsFromRendezvousAsync(
      rendez_, nullptr, {}, {MakeStringKey("hello1"), MakeStringKey("hello2")},
      &received_keys, [&n](const Status& status) { n.Notify(); });
  n.WaitForNotification();

  EXPECT_EQ(2, received_keys.size());
  EXPECT_EQ("hello1", V(received_keys[0]));
  EXPECT_EQ("hello2", V(received_keys[1]));
}

TEST_F(RendezvousUtilTest, RecvBeforeSend) {
  // Fire off recvs, wait for a notification in the callback.
  Notification n;
  std::vector<Tensor> received_keys;
  RecvOutputsFromRendezvousAsync(
      rendez_, nullptr, {}, {MakeStringKey("hello1"), MakeStringKey("hello2")},
      &received_keys, [&n](const Status& status) { n.Notify(); });

  TF_ASSERT_OK(SendTensorsToRendezvous(
      rendez_, nullptr, {}, {MakeStringKey("hello1"), MakeStringKey("hello2")},
      {V("hello1"), V("hello2")}));

  n.WaitForNotification();

  EXPECT_EQ(2, received_keys.size());
  EXPECT_EQ("hello1", V(received_keys[0]));
  EXPECT_EQ("hello2", V(received_keys[1]));
}

}  // namespace
}  // namespace tensorflow
