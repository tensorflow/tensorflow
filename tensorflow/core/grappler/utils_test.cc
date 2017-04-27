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
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class UtilsTest : public ::testing::Test {};

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

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
