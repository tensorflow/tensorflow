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

#include "tensorflow/core/grappler/costs/cost_estimator.h"

#include <gtest/gtest.h>
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

TEST(CostEstimatorTest, CombineCosts) {
  Costs c = Costs::ZeroCosts();
  c.execution_time = Costs::NanoSeconds(1);
  c.compute_time = Costs::NanoSeconds(2);
  c.memory_time = Costs::NanoSeconds(3);
  c.intermediate_memory_time = Costs::NanoSeconds(4);
  c.intermediate_memory_read_time = Costs::NanoSeconds(5);
  c.intermediate_memory_write_time = Costs::NanoSeconds(6);
  c.hbm_read_time = Costs::NanoSeconds(7);
  c.hbm_write_time = Costs::NanoSeconds(8);
  c.hbm_read_time_noderate = Costs::NanoSeconds(9);
  c.hbm_write_time_noderate = Costs::NanoSeconds(10);
  c.max_memory = 1;
  c.max_per_op_buffers = 2;
  c.max_per_op_streaming = 3;
  c.num_ops_total = 1;
  c.inaccurate = false;
  c.num_ops_with_unknown_shapes = 0;

  Costs sum = CombineCosts(c, c);

  EXPECT_EQ(sum.execution_time, Costs::NanoSeconds(2));
  EXPECT_EQ(sum.compute_time, Costs::NanoSeconds(4));
  EXPECT_EQ(sum.memory_time, Costs::NanoSeconds(6));
  EXPECT_EQ(sum.intermediate_memory_time, Costs::NanoSeconds(8));
  EXPECT_EQ(sum.intermediate_memory_read_time, Costs::NanoSeconds(10));
  EXPECT_EQ(sum.intermediate_memory_write_time, Costs::NanoSeconds(12));
  EXPECT_EQ(sum.hbm_read_time, Costs::NanoSeconds(14));
  EXPECT_EQ(sum.hbm_write_time, Costs::NanoSeconds(16));
  EXPECT_EQ(sum.hbm_read_time_noderate, Costs::NanoSeconds(18));
  EXPECT_EQ(sum.hbm_write_time_noderate, Costs::NanoSeconds(20));
  EXPECT_EQ(sum.max_memory, 2);
  EXPECT_EQ(sum.max_per_op_buffers, 2);
  EXPECT_EQ(sum.max_per_op_streaming, 3);
  EXPECT_EQ(sum.num_ops_total, 2);
  EXPECT_FALSE(sum.inaccurate);
  EXPECT_EQ(sum.num_ops_with_unknown_shapes, 0);
}

TEST(CostEstimatorTest, MultiplyCosts) {
  Costs c = Costs::ZeroCosts();
  c.execution_time = Costs::NanoSeconds(1);
  c.compute_time = Costs::NanoSeconds(2);
  c.memory_time = Costs::NanoSeconds(3);
  c.intermediate_memory_time = Costs::NanoSeconds(4);
  c.intermediate_memory_read_time = Costs::NanoSeconds(5);
  c.intermediate_memory_write_time = Costs::NanoSeconds(6);
  c.hbm_read_time_noderate = Costs::NanoSeconds(7);
  c.hbm_write_time_noderate = Costs::NanoSeconds(8);
  c.max_memory = 1;
  c.max_per_op_buffers = 2;
  c.max_per_op_streaming = 3;
  c.num_ops_total = 1;
  c.inaccurate = false;
  c.num_ops_with_unknown_shapes = 0;

  Costs product = MultiplyCosts(c, 10);

  EXPECT_EQ(product.execution_time, Costs::NanoSeconds(10));
  EXPECT_EQ(product.compute_time, Costs::NanoSeconds(20));
  EXPECT_EQ(product.memory_time, Costs::NanoSeconds(30));
  EXPECT_EQ(product.intermediate_memory_time, Costs::NanoSeconds(40));
  EXPECT_EQ(product.intermediate_memory_read_time, Costs::NanoSeconds(50));
  EXPECT_EQ(product.intermediate_memory_write_time, Costs::NanoSeconds(60));
  EXPECT_EQ(product.hbm_read_time_noderate, Costs::NanoSeconds(70));
  EXPECT_EQ(product.hbm_write_time_noderate, Costs::NanoSeconds(80));
  EXPECT_EQ(product.max_memory, 1);
  EXPECT_EQ(product.max_per_op_buffers, 2);
  EXPECT_EQ(product.max_per_op_streaming, 3);
  EXPECT_EQ(product.num_ops_total, 1);
  EXPECT_FALSE(product.inaccurate);
  EXPECT_EQ(product.num_ops_with_unknown_shapes, 0);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
