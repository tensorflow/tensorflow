/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_OP_PROFILE_BUILDER_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_OP_PROFILE_BUILDER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_profile.pb.h"

namespace tensorflow {
namespace profiler {

struct OpProfileOptions {
  bool group_by_program = true;
  bool group_by_deduplicated_name = true;
  int children_per_node = 100;
};

// The structure of an op profile tree may looks like below:
// 1. group "by_program"
// - It starts from the root node, named as "by_program", and this node does
// not show up in op profile.
// - The children of root node is a list of hlo program node, named as the
// program/module name (eg. cluster.xx).
// - The children of a program node is hlo op category node, named as the
// category name (eg. data formatting).
// - The children of a category node is a list of op node or deduplicated
// group node:
//   - For op that has duplicates, the child will be a deduplicated node,
// named like "copy.1111 and its deduplicate(s)". Its children will be all op
// nodes that are deduplicated.
//   - For op that does not have duplicates, the child will be an op node
// under the op category (eg. copy.2222).
//
// Example path: "by_program" -> "main(...)"
// -> "data_formatting" -> "copy.12345 and its duplicate(s) -> "copy.12345"
//
// 2. group "by_category"
// Similarly to how the `by_program` op profile tree is constructed,
// `by_category` just removed the "program_node" layer:
// - It starts from the root node, named as "by_category", this node also does
// not show up in op profile.
// - The children of root node is a list of op category node, everything below
// is similar to above.
// - ...
//
// Example path: "by_category" -> "data_formatting" -> "copy.12345 and its
// duplicate(s) -> "copy.12345"
//
// How the op profile metrics are calculated:
// 1. For parent node in the nested structure like root node and program node:
// - time_ps will be accumulated from the self_time of all op nodes under it
// (might still be off a bit if the parent node has self_time, more details in
// b/333608397#comment5)
// - flops and memory access will only be accumulated from leaf op node under
// it to avoid double counting
// - unable to get occurrences of program executions now
// 2. For conceptual horizontal grouping node (eg.category, deduplicated)
// - all op_metris fields will be accumulated from leaf op node only in the
// group, to avoid double counting
class OpProfileBuilder {
 public:
  OpProfileBuilder(const OpProfileOptions& options, op_profile::Node* root,
                   const tensorflow::protobuf::Map<uint64_t, std::string>*
                       program_name_map = nullptr);

  // Accumulate the op_metrics to the op_profile node tree
  void AddOp(const OpMetrics& op_metrics);

  // Finalize the op_profile proto in a few steps (inter-dependent):
  // 1. Reset time_ps for root node for more precise total time
  // 2. Loop over the node to op_metrics map, populate corresponding op_metrics
  // to the node.metrics
  // 3. `SortAndPruneChildren` given query param `op_profile_limit`
  // 4. `FinalizeDeduplicatedNodes` by coping the first op node data to the
  // deduplicated node
  void Finalize(double peak_gigaflops_per_second_per_core,
                std::vector<double> peak_mem_gibibytes_per_second_per_core,
                uint64_t total_time_ps);

 private:
  struct Category {
    op_profile::Node* node;
    absl::flat_hash_map<std::string, op_profile::Node*> deduplicated_nodes;
  };

  struct Program {
    op_profile::Node* node;
    absl::flat_hash_map<std::string, Category> categories;
  };

  std::string GenerateProgramName(uint64_t program_id) const;

  // Adds and returns a node for op_metrics.
  // If op_metrics corresponds to a fusion, adds children to the node for the
  // fused instructions.
  // If deduplicated_node is not null, adds the node under it.
  // Otherwise, if category is not null, adds the node under category.
  // Otherwise, adds the node under root.
  op_profile::Node* AddOpNode(const OpMetrics& op_metrics,
                              Category* category = nullptr,
                              op_profile::Node* deduplicated_node = nullptr);

  // Returns a node for op_metrics.deduplicated_name().
  // Adds a node to the tree if necessary.
  op_profile::Node* LookupOrAddDeduplicatedNode(const OpMetrics& op_metrics,
                                                Category* category);

  // Returns a node for op_metrics.category().
  // Adds a node to the tree if necessary.
  // If program is not null, the category node is added under program.
  // Otherwise, the category node is added under root.
  Category* LookupOrAddCategoryNode(const OpMetrics& op_metrics,
                                    Program* program);

  // Returns a node for op_metrics.hlo_module_id().
  // Adds a node to the Node tree if necessary.
  Program* LookupOrAddProgramNode(const OpMetrics& op_metrics);

  OpProfileOptions options_;
  op_profile::Node* root_;

  // Map to look up and aggregate OpMetrics.
  absl::node_hash_map<op_profile::Node*, OpMetrics> metrics_;

  // Maps to look up if a category / program / deduplicated node has
  // already been added to the tree.
  absl::flat_hash_map<uint64_t, Program> programs_map_;
  absl::flat_hash_map<std::string, Category> category_map_;

  // Map to look up program names by id.
  const tensorflow::protobuf::Map<uint64_t, std::string>* program_name_map_ =
      nullptr;
};
}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_OP_PROFILE_BUILDER_H_
