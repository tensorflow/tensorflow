/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/process_util.h"

#include <limits>
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

TEST(ProcessUtilTest, NumThreads) {
  SessionOptions opts;
  opts.config.set_inter_op_parallelism_threads(10);
  EXPECT_EQ(10, NumInterOpThreadsFromSessionOptions(opts));
}

TEST(ProcessUtilTest, ThreadPool) {
  SessionOptions opts;
  opts.config.set_inter_op_parallelism_threads(10);

  thread::ThreadPool* pool = NewThreadPoolFromSessionOptions(opts);
  EXPECT_EQ(10, pool->NumThreads());
  delete pool;
}
TEST(ProcessUtilTest, ValidThreadCountsAreAccepted) {
  SessionOptions options;


  options.config.set_inter_op_parallelism_threads(4);
  options.config.set_intra_op_parallelism_threads(4);

  int32_t inter_threads = NumInterOpThreadsFromSessionOptions(options);
  EXPECT_EQ(inter_threads, 4);

  thread::ThreadPool* pool = NewThreadPoolFromSessionOptions(options, 4);
  EXPECT_NE(pool, nullptr);
  delete pool;
}

TEST(ProcessUtilTest, ZeroThreadCountMeansAutoDetect) {
  SessionOptions options;

  // 0 means auto-detect
  options.config.set_inter_op_parallelism_threads(0);
  options.config.set_intra_op_parallelism_threads(0);

  int32_t inter_threads = NumInterOpThreadsFromSessionOptions(options);
  EXPECT_GT(inter_threads, 0);

  thread::ThreadPool* pool = NewThreadPoolFromSessionOptions(options, 0);
  EXPECT_NE(pool, nullptr);
  delete pool;
}

TEST(ProcessUtilTest, NegativeThreadCountIsClamped) {
  SessionOptions options;

  // Negative values should be clamped to 0 (auto-detect)
  options.config.set_inter_op_parallelism_threads(-5);

  int32_t inter_threads = NumInterOpThreadsFromSessionOptions(options);
  EXPECT_GT(inter_threads, 0);
}

TEST(ProcessUtilTest, IntMaxThreadCountIsClamped) {
  SessionOptions options;

  // INT_MAX should be clamped to safe limit
  options.config.set_inter_op_parallelism_threads(std::numeric_limits<int32_t>::max());
  options.config.set_intra_op_parallelism_threads(std::numeric_limits<int32_t>::max());

  int32_t inter_threads = NumInterOpThreadsFromSessionOptions(options);
  EXPECT_LE(inter_threads, 1024);
  EXPECT_GT(inter_threads, 0);

  // Should not crash when creating thread pool
  thread::ThreadPool* pool = NewThreadPoolFromSessionOptions(
      options, std::numeric_limits<int32_t>::max());
  EXPECT_NE(pool, nullptr);
  delete pool;
}

TEST(ProcessUtilTest, VeryLargeThreadCountIsClamped) {
  SessionOptions options;

  // Large value should be clamped
  options.config.set_inter_op_parallelism_threads(1000000);

  int32_t inter_threads = NumInterOpThreadsFromSessionOptions(options);
  EXPECT_LE(inter_threads, 1024);
  EXPECT_GT(inter_threads, 0);

  thread::ThreadPool* pool = NewThreadPoolFromSessionOptions(options, 1000000);
  EXPECT_NE(pool, nullptr);
  delete pool;
}

TEST(ProcessUtilTest, ThreadPoolCreationDoesNotSegfault) {
  SessionOptions options;
  options.config.set_inter_op_parallelism_threads(2147483647);  // INT_MAX
  options.config.set_intra_op_parallelism_threads(2147483647);  // INT_MAX

  // This should not segfault
  thread::ThreadPool* pool = NewThreadPoolFromSessionOptions(options, 2147483647);
  EXPECT_NE(pool, nullptr);
  delete pool;
}

}  // namespace
}  // namespace tensorflow
