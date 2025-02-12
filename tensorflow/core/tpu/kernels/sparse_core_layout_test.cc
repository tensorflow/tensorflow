/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/sparse_core_layout.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/tpu/kernels/sparse_core_layout.pb.h"

namespace tensorflow {
namespace tpu {
namespace {

using ::testing::EqualsProto;
using ::testing::proto::Partially;
using ::testing::status::IsOkAndHolds;

TEST(SparseCoreLayoutStacker, StacksTwoTablesAndPads) {
  SparseCoreLayoutStacker stacker(2);
  ASSERT_OK(stacker.AddTable("table1", 100, 6, "stack1", 10));
  ASSERT_OK(stacker.AddTable("table2", 50, 5, "stack1", 10));
  EXPECT_THAT(stacker.GetLayouts(), IsOkAndHolds(EqualsProto(R"pb(
                tables {
                  table_name: 'table1'
                  stacked_table_name: 'table1_table2'
                  num_sparse_cores: 8
                  num_partitions: 2
                  total_rows_per_sparse_core_shard: 24  # = (128 + 64) / 8
                  unsharded_shape: [ 100, 6 ]
                  unsharded_padded_shape: [ 128, 8 ]
                  sparse_core_shard_row_offset: 0
                  sparse_core_shard_rotation: 0
                }
                tables {
                  table_name: 'table2'
                  stacked_table_name: 'table1_table2'
                  num_sparse_cores: 8
                  num_partitions: 2
                  total_rows_per_sparse_core_shard: 24
                  unsharded_shape: [ 50, 5 ]
                  unsharded_padded_shape: [ 64, 8 ]
                  sparse_core_shard_row_offset: 16  # = 128/8
                  sparse_core_shard_rotation: 4
                }
              )pb")));
}

TEST(SparseCoreLayoutStacker, RespectsDisableStacking) {
  SparseCoreLayoutStacker stacker(2);
  stacker.SetStackingEnabled(false);
  ASSERT_OK(stacker.AddTable("table1", 100, 6, "stack1", 10));
  ASSERT_OK(stacker.AddTable("table2", 50, 5, "stack1", 10));
  EXPECT_THAT(stacker.GetLayouts(), IsOkAndHolds(EqualsProto(R"pb(
                tables {
                  table_name: 'table1'
                  stacked_table_name: 'table1'
                  num_sparse_cores: 8
                  num_partitions: 2
                  total_rows_per_sparse_core_shard: 16  # = 128 / 8
                  unsharded_shape: [ 100, 6 ]
                  unsharded_padded_shape: [ 128, 8 ]
                  sparse_core_shard_row_offset: 0
                  sparse_core_shard_rotation: 0
                }
                tables {
                  table_name: 'table2'
                  stacked_table_name: 'table2'
                  num_sparse_cores: 8
                  num_partitions: 2
                  total_rows_per_sparse_core_shard: 8  # = 64/8
                  unsharded_shape: [ 50, 5 ]
                  unsharded_padded_shape: [ 64, 8 ]
                  sparse_core_shard_row_offset: 0
                  sparse_core_shard_rotation: 0
                }
              )pb")));
}

TEST(SparseCoreLayoutStacker, RespectsActivationMemLimit) {
  SparseCoreLayoutStacker stacker(2);
  stacker.SetActivationMemoryBytesLimit(16384 + 1);

  // Here there are several identical tables with an activation memory limit of
  //    sizeof (float) * 8 * 1024 = 8192 per table.
  // It should be able to fit the first two into a stack but not the third.
  ASSERT_OK(stacker.AddTable("table1", 128, 8, "stack1", 1024));
  ASSERT_OK(stacker.AddTable("table2", 128, 8, "stack1", 1024));
  ASSERT_OK(stacker.AddTable("table3", 128, 8, "stack1", 1024));
  ASSERT_OK(stacker.AddTable("table4", 128, 8, "stack1", 1024));
  ASSERT_OK(stacker.AddTable("table5", 128, 8, "stack1", 1024));
  EXPECT_THAT(
      stacker.GetLayouts(), IsOkAndHolds(Partially(EqualsProto(R"pb(
        tables { table_name: 'table1' stacked_table_name: 'table1_table2' }
        tables { table_name: 'table2' stacked_table_name: 'table1_table2' }
        tables { table_name: 'table3' stacked_table_name: 'table3_table4' }
        tables { table_name: 'table4' stacked_table_name: 'table3_table4' }
        tables { table_name: 'table5' stacked_table_name: 'table5' }
      )pb"))));
}

TEST(SparseCoreLayoutStacker, RespectsVariableShardLimit) {
  SparseCoreLayoutStacker stacker(2);
  stacker.SetVariableShardBytesLimit(4096 + 1);

  // Here there are several identical tables that contribute
  //    sizeof (float) * 8 * 128 / 2 = 2048 bytes to each shard.
  // It should be able to fit the first two into a stack but not the third.
  ASSERT_OK(stacker.AddTable("table1", 128, 8, "stack1", 1024));
  ASSERT_OK(stacker.AddTable("table2", 128, 8, "stack1", 1024));
  ASSERT_OK(stacker.AddTable("table3", 128, 8, "stack1", 1024));
  ASSERT_OK(stacker.AddTable("table4", 128, 8, "stack1", 1024));
  ASSERT_OK(stacker.AddTable("table5", 128, 8, "stack1", 1024));
  EXPECT_THAT(
      stacker.GetLayouts(), IsOkAndHolds(Partially(EqualsProto(R"pb(
        tables { table_name: 'table1' stacked_table_name: 'table1_table2' }
        tables { table_name: 'table2' stacked_table_name: 'table1_table2' }
        tables { table_name: 'table3' stacked_table_name: 'table3_table4' }
        tables { table_name: 'table4' stacked_table_name: 'table3_table4' }
        tables { table_name: 'table5' stacked_table_name: 'table5' }
      )pb"))));
}

TEST(SparseCoreLayoutStacker, RespectsRowLimit) {
  SparseCoreLayoutStacker stacker(2);
  // Disable the other limits.
  stacker.SetActivationMemoryBytesLimit(0);
  stacker.SetVariableShardBytesLimit(0);

  // Here there are several identical tables that contribute 2^30 rows. Since
  // the default row limit is 2^31-1, they should not be able to stack.
  ASSERT_OK(stacker.AddTable("table1", 1 << 29, 8, "stack1", 1024));
  ASSERT_OK(stacker.AddTable("table2", 1 << 29, 8, "stack1", 1024));
  ASSERT_OK(stacker.AddTable("table3", 1 << 29, 8, "stack1", 1024));
  ASSERT_OK(stacker.AddTable("table4", 1 << 29, 8, "stack1", 1024));
  EXPECT_THAT(stacker.GetLayouts(), IsOkAndHolds(Partially(EqualsProto(R"pb(
                tables {
                  table_name: 'table1'
                  stacked_table_name: 'table1_table2_table3'
                }
                tables {
                  table_name: 'table2'
                  stacked_table_name: 'table1_table2_table3'
                }
                tables {
                  table_name: 'table3'
                  stacked_table_name: 'table1_table2_table3'
                }
                tables { table_name: 'table4' stacked_table_name: 'table4' }
              )pb"))));
}

TEST(SparseCoreLayoutStacker, RespectsTableLimit) {
  SparseCoreLayoutStacker stacker(2);
  // Disable the other limits.
  stacker.SetActivationMemoryBytesLimit(0);
  stacker.SetVariableShardBytesLimit(0);

  // Max of 2 tables per stack. Without this, all the tables would go in the
  // same stack.
  stacker.SetStackingTableLimit(2);

  ASSERT_OK(stacker.AddTable("table1", 128, 8, "stack1", 1024));
  ASSERT_OK(stacker.AddTable("table2", 128, 8, "stack1", 1024));
  ASSERT_OK(stacker.AddTable("table3", 128, 8, "stack1", 1024));
  ASSERT_OK(stacker.AddTable("table4", 128, 8, "stack1", 1024));
  EXPECT_THAT(
      stacker.GetLayouts(), IsOkAndHolds(Partially(EqualsProto(R"pb(
        tables { table_name: 'table1' stacked_table_name: 'table1_table2' }
        tables { table_name: 'table2' stacked_table_name: 'table1_table2' }
        tables { table_name: 'table3' stacked_table_name: 'table3_table4' }
        tables { table_name: 'table4' stacked_table_name: 'table3_table4' }
      )pb"))));
}

}  // namespace
}  // namespace tpu
}  // namespace tensorflow
