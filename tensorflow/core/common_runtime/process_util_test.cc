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

#include "tensorflow/core/platform/test.h"

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

}  // anonymous namespace
}  // namespace tensorflow
