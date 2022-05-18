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

class OpProfileBuilder {
 public:
  OpProfileBuilder(const OpProfileOptions& options, op_profile::Node* root,
                   const tensorflow::protobuf::Map<uint64_t, std::string>*
                       program_name_map = nullptr);

  void AddOp(const OpMetrics& op_metrics);

  void Finalize(double peak_gigaflops_per_second_per_core,
                double peak_gibibytes_per_second_per_core,
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
