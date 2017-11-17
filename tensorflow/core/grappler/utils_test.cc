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

#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class UtilsTest : public ::testing::Test {
 protected:
  NodeDef CreateConcatOffsetNode() const {
    const string gdef_ascii = R"EOF(
name: "gradients/InceptionV3/Mixed_7c/Branch_1/concat_v2_grad/ConcatOffset"
op: "ConcatOffset"
input: "InceptionV3/Mixed_7c/Branch_1/concat_v2/axis"
input: "gradients/InceptionV3/Mixed_7c/Branch_1/concat_v2_grad/Shape"
input: "gradients/InceptionV3/Mixed_7c/Branch_1/concat_v2_grad/Shape_1"
attr {
  key: "N"
  value {
    i: 2
  }
}
    )EOF";
    NodeDef node;
    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &node));
    return node;
  }

  NodeDef CreateDequeueNode() const {
    const string gdef_ascii = R"EOF(
name: "Train/TrainInput/input_producer_Dequeue"
op: "QueueDequeueV2"
input: "Train/TrainInput/input_producer"
attr {
  key: "component_types"
  value {
    list {
      type: DT_INT32
    }
  }
}
attr {
  key: "timeout_ms"
  value {
    i: -1
  }
}
    )EOF";
    NodeDef node;
    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &node));
    return node;
  }

  NodeDef CreateFusedBatchNormNode() const {
    const string gdef_ascii = R"EOF(
name: "InceptionV3/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm"
op: "FusedBatchNorm"
input: "InceptionV3/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm"
input: "InceptionV3/Conv2d_1a_3x3/BatchNorm/gamma/read"
input: "InceptionV3/Conv2d_1a_3x3/BatchNorm/beta/read"
input: "InceptionV3/Conv2d_1a_3x3/BatchNorm/Const"
input: "InceptionV3/Conv2d_1a_3x3/BatchNorm/Const_1"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "data_format"
  value {
    s: "NHWC"
  }
}
attr {
  key: "epsilon"
  value {
    f: 0.001
  }
}
attr {
  key: "is_training"
  value {
    b: true
  }
}
    )EOF";
    NodeDef node;
    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &node));
    return node;
  }
};

TEST_F(UtilsTest, NodeName) {
  EXPECT_EQ("abc", NodeName("abc"));
  EXPECT_EQ("abc", NodeName("^abc"));
  EXPECT_EQ("abc", NodeName("abc:0"));
  EXPECT_EQ("abc", NodeName("^abc:0"));

  EXPECT_EQ("abc/def", NodeName("abc/def"));
  EXPECT_EQ("abc/def", NodeName("^abc/def"));
  EXPECT_EQ("abc/def", NodeName("abc/def:1"));
  EXPECT_EQ("abc/def", NodeName("^abc/def:1"));

  EXPECT_EQ("abc/def0", NodeName("abc/def0"));
  EXPECT_EQ("abc/def0", NodeName("^abc/def0"));
  EXPECT_EQ("abc/def0", NodeName("abc/def0:0"));
  EXPECT_EQ("abc/def0", NodeName("^abc/def0:0"));

  EXPECT_EQ("abc/def_0", NodeName("abc/def_0"));
  EXPECT_EQ("abc/def_0", NodeName("^abc/def_0"));
  EXPECT_EQ("abc/def_0", NodeName("abc/def_0:3"));
  EXPECT_EQ("abc/def_0", NodeName("^abc/def_0:3"));

  EXPECT_EQ("abc/def_0", NodeName("^abc/def_0:3214"));
}

TEST_F(UtilsTest, NodePosition) {
  EXPECT_EQ(2, NodePosition("abc:2"));
  EXPECT_EQ(123, NodePosition("abc:123"));
  EXPECT_EQ(-1, NodePosition("^abc:123"));
  EXPECT_EQ(-1, NodePosition("^abc"));
  EXPECT_EQ(0, NodePosition(""));
}

TEST_F(UtilsTest, AddNodeNamePrefix) {
  EXPECT_EQ("OPTIMIZED/abc", AddPrefixToNodeName("abc", "OPTIMIZED"));
  EXPECT_EQ("^OPTIMIZED/abc", AddPrefixToNodeName("^abc", "OPTIMIZED"));
  EXPECT_EQ("OPTIMIZED/", AddPrefixToNodeName("", "OPTIMIZED"));
}

TEST_F(UtilsTest, ExecuteWithTimeout) {
  std::unique_ptr<thread::ThreadPool> thread_pool(
      new thread::ThreadPool(Env::Default(), "ExecuteWithTimeout", 2));

  // This should run till the end.
  ASSERT_TRUE(ExecuteWithTimeout(
      []() {  // Do nothing.
      },
      1000 /* timeout_in_ms */, thread_pool.get()));

  // This should time out.
  Notification notification;
  ASSERT_FALSE(ExecuteWithTimeout(
      [&notification]() { notification.WaitForNotification(); },
      1 /* timeout_in_ms */, thread_pool.get()));
  // Make sure to unblock the thread.
  notification.Notify();

  // This should run till the end.
  ASSERT_TRUE(ExecuteWithTimeout([]() { sleep(1); }, 0 /* timeout_in_ms */,
                                 thread_pool.get()));

  // Deleting before local variables go off the stack.
  thread_pool.reset();
}

TEST_F(UtilsTest, NumOutputs) {
  EXPECT_EQ(2, NumOutputs(CreateConcatOffsetNode()));
  EXPECT_EQ(5, NumOutputs(CreateFusedBatchNormNode()));
  EXPECT_EQ(1, NumOutputs(CreateDequeueNode()));
}

TEST(AsControlDependency, BasicTest) {
  NodeDef node;
  node.set_name("foo");
  EXPECT_EQ("^foo", AsControlDependency(node));
  EXPECT_EQ("^foo", AsControlDependency(node.name()));
  EXPECT_EQ("^foo", AsControlDependency("^foo"));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
