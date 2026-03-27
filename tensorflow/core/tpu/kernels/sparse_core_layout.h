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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_LAYOUT_H_
#define TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_LAYOUT_H_

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/tpu/kernels/sparse_core_layout.pb.h"

namespace tensorflow::tpu {

// A class to figure out which tables to stack.
class SparseCoreLayoutStacker {
 public:
  // Constructor.  Arguments:
  //   num_partitions: How many shards the sparse core shards are concatenated
  //     into (usually one per TPU chip).
  //       NOTE: As of Q4 2023, SPMD is not supported by the sparse core python
  //       libraries so we don't support it here.
  //   sparse_cores_per_partition: Number of sparsecore per partition
  //   disable_table_stacking: Should not stack tables.
  explicit SparseCoreLayoutStacker(int num_partitions,
                                   bool disable_table_stacking = false,
                                   int sparse_cores_per_partition = 4);

  // Change various limits. You must call these before calling Addtable.
  void SetActivationMemoryBytesLimit(int64_t activation_mem_bytes_limit) {
    CHECK(stacks_by_group_.empty()) << "must call before AddTable";
    activation_mem_bytes_limit_ = activation_mem_bytes_limit;
  }
  void SetVariableShardBytesLimit(int64_t variable_shard_bytes_limit) {
    CHECK(stacks_by_group_.empty()) << "must call before AddTable";
    variable_shard_bytes_limit_ = variable_shard_bytes_limit;
  }
  void SetStackingEnabled(bool stacking_enabled) {
    CHECK(stacks_by_group_.empty()) << "must call before AddTable";
    stacking_enabled_ = stacking_enabled;
  }
  void SetStackingRowLimit(int64_t row_limit) {
    CHECK(stacks_by_group_.empty()) << "must call before AddTable";
    row_limit_ = row_limit;
  }
  void SetStackingTableLimit(int table_limit) {
    CHECK(stacks_by_group_.empty()) << "must call before AddTable";
    table_limit_ = table_limit;
  }

  // Add a new table.  Arguments:
  //   table_name: How this table will be referred to.
  //   table_height: The number of rows.
  //   table_width: The number of columns in the input layer. For storage, this
  //     will be rounded up to a multiple of eight, but the padding columns will
  //     be stripped off when fed into the rest of the model.
  //   group: An arbitrary identifier that should be derived from the optimizer
  //     and hyperparameters. Only tables with the same group and rounded
  //     table_width can be stacked. The actual contents of this field are not
  //     particularly meaningful except they are used to construct the
  //     stack_name field in the SparseCoreTableLayout.
  //   output_samples: How many times a row from this table will have to be
  //     returned per batch. This is ordinarily the batch size unless we lay out
  //     several values from the same example in a sequence, or if multiple
  //     features share the same table.
  //
  // Be sure you call AddTable in a deterministic order; the details of the
  // stacking will depend on the order you call AddTable.
  absl::Status AddTable(absl::string_view table_name, int64_t table_height,
                        int64_t table_width, absl::string_view group,
                        int64_t output_samples);

  // Get the information about each table out.
  absl::StatusOr<SparseCoreTableLayouts> GetLayouts();

 private:
  struct TableStack {
    // A name we give the stack while we're constructing it. The name will be
    // overridden later to be equal to the names of the tables.
    std::string temporary_name;
    int64_t padded_width = 0;
    int64_t unsharded_height = 0;
    int64_t total_activation_mem_bytes = 0;
    int64_t total_variable_shard_bytes = 0;

    // While we're filling out this structure, we can't fill out all the fields
    // in the SparseCoreTableLayout; we fill out as many of them as we can.
    std::vector<SparseCoreTableLayout> incomplete_tables;
  };

  const int num_partitions_;
  const int sparse_cores_per_partition_;
  const int num_sparse_cores_;

  bool stacking_enabled_ = true;
  int64_t activation_mem_bytes_limit_ = 0;
  int64_t variable_shard_bytes_limit_ = 0;
  // Sparse core ops use signed int for row numbers so we had better not stack
  // beyond this limit.
  int64_t row_limit_ = (1LL << 31) - 1;

  // The maximum number of tables in any stack.
  int table_limit_ = std::numeric_limits<int>::max();

  // All the stacks that we currently know about. Note that we use a btree_map
  // rather than a flat_hash_map so the resulting order is deterministic as long
  // as we are called in a deterministic order. Key is (padded_width, group).
  absl::btree_map<std::pair<int64_t, std::string>, std::vector<TableStack>>
      stacks_by_group_;
};

}  // namespace tensorflow::tpu

#endif  // TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_LAYOUT_H_
