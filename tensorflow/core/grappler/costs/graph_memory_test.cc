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

#include "tensorflow/core/grappler/costs/graph_memory.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class GraphMemoryTest : public ::testing::Test {};

TEST_F(GraphMemoryTest, Basic) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {{"CPU:0"}});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphMemory memory(item);
  Status s = memory.InferStatically();
  TF_CHECK_OK(s);
  EXPECT_EQ(240, memory.GetWorstCaseMemoryUsage());
  EXPECT_EQ(80, memory.GetBestCaseMemoryUsage());
}

TEST_F(GraphMemoryTest, UnknownBatchSize) {
  TrivialTestGraphInputYielder fake_input(4, 1, -1, false, {{"CPU:0"}});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphMemory memory(item);
  Status s = memory.InferStatically();
  TF_CHECK_OK(s);
  EXPECT_EQ(24, memory.GetWorstCaseMemoryUsage());
  EXPECT_EQ(8, memory.GetBestCaseMemoryUsage());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
