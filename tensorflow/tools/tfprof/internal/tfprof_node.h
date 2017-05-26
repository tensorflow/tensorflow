/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_NODE_H_
#define THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_NODE_H_

#include <map>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/tools/tfprof/internal/tfprof_options.h"
#include "tensorflow/tools/tfprof/tfprof_log.pb.h"

namespace tensorflow {
namespace tfprof {

class TFGraphNode {
 public:
  TFGraphNode(const NodeDef* node)
      : node_(node),
        code_(nullptr),
        step_stat_(nullptr),
        all_start_micros_(0),
        latest_end_rel_micros_(0),
        requested_bytes_(0),
        host_temp_bytes_(0),
        host_persistent_bytes_(0),
        accelerator_temp_bytes_(0),
        accelerator_persistent_bytes_(0),
        allocator_bytes_in_use_(0),
        float_ops_(0),
        op_(node->op()) {
    for (const auto& attr : node->attr()) {
      // TODO(xpan): Also consider _output_shapes.
      if (attr.first != "shape" || !attr.second.has_shape()) continue;
      if (!shape_.empty()) {
        fprintf(stderr, "Found duplicated shapes!\n");
        continue;
      }
      std::vector<int64> shape_vec;
      if (attr.second.shape().dim_size() == 0 &&
          !attr.second.shape().unknown_rank()) {
        // Scalar parameter with empty shape but known rank.
        shape_vec.push_back(1);
      } else {
        for (const auto& d : attr.second.shape().dim()) {
          shape_vec.push_back(d.size());
        }
      }
      update_shape(shape_vec);
    }
    op_types_.insert(node->op());
  }

  void AddInput(TFGraphNode* input, int64 output_idx) {
    inputs_[input->name()] = input;
    output_idx_[input->name()] = output_idx;
  }

  void AddOpType(const string& op_type) { op_types_.insert(op_type); }

  void AddStepStat(const string& device, const NodeExecStats* step_stat);

  void AddFloatOps(int64 float_ops) { float_ops_ = float_ops; }

  void AddCode(const CodeDef* code) { code_ = code; }

  const string& name() const { return node_->name(); }
  const string& op() const { return op_; }
  const NodeDef* node_def() { return node_; }

  const NodeExecStats* step_stats() const { return step_stat_; }

  const std::map<string, TFGraphNode*>& inputs() const { return inputs_; }
  const std::map<string, int64>& output_idx() const { return output_idx_; }

  // This is time spent in kernel execution.
  int64 kernel_exec_micros() const {
    if (!step_stat_) return 0;
    int64 total = 0;
    for (const auto& execs : gpu_kernel_execs_) {
      for (const auto& exec : execs.second) {
        total += exec.second;
      }
    }
    if (total > 0) return total;

    // If there is no gpu kernel time, fall back to assume it runs on cpu.
    for (const auto& execs : op_execs_) {
      for (const auto& exec : execs.second) {
        total += exec.second;
      }
    }
    return total;
  }

  int64 all_start_micros() const { return all_start_micros_; }
  int64 latest_end_rel_micros() const { return latest_end_rel_micros_; }
  const std::map<string, std::vector<std::pair<int64, int64>>>& op_execs()
      const {
    return op_execs_;
  }

  int64 requested_bytes() const { return requested_bytes_; }
  int64 accelerator_temp_bytes() const { return accelerator_temp_bytes_; }
  int64 host_temp_bytes() const { return host_temp_bytes_; }
  int64 accelerator_persistent_bytes() const {
    return accelerator_persistent_bytes_;
  }
  int64 host_persistent_bytes() const { return host_persistent_bytes_; }
  const std::map<int64, std::pair<int64, uint64>>& output_bytes() const {
    return output_bytes_;
  }
  int64 allocator_bytes_in_use() const { return allocator_bytes_in_use_; }

  int64 float_ops() const { return float_ops_; }
  const CodeDef* code() { return code_; }
  string canonical_device() const { return canonical_device_; }
  string host_device() const { return host_device_; }
  std::set<string> devices() const { return devices_; }
  const std::set<string>& op_types() const { return op_types_; }

  const std::vector<int64>& shape() const { return shape_; }

 private:
  void update_shape(const std::vector<int64>& shape) { shape_ = shape; }

  std::map<string, TFGraphNode*> inputs_;
  std::map<string, int64> output_idx_;

  const NodeDef* node_;
  const CodeDef* code_;
  const NodeExecStats* step_stat_;

  std::vector<int64> shape_;
  std::set<string> op_types_;

  // The earliest/latest time including scheduling and kernel execution.
  int64 all_start_micros_;
  int64 latest_end_rel_micros_;
  // device -> vector of {op_start_micros, op_kernel_exec_micros} pairs.
  std::map<string, std::vector<std::pair<int64, int64>>> gpu_kernel_execs_;
  std::map<string, std::vector<std::pair<int64, int64>>> op_execs_;

  // /j:#/t:#/r:#/device:#. A canonical device name without extra suffix.
  string canonical_device_;
  // The host device name.
  string host_device_;
  // All devices the op is associated with (e.g. gpu:0 (scheduling),
  // gpu:0:stream:xx (kernel exec), cpu:0 host)
  std::set<string> devices_;

  // Total output bytes requested by the op.
  int64 requested_bytes_;
  // Total temporary bytes allocated and released by the op.
  int64 host_temp_bytes_;
  // Total persistent bytes (e.g. variable) allocated by the op.
  int64 host_persistent_bytes_;
  int64 accelerator_temp_bytes_;
  int64 accelerator_persistent_bytes_;
  // The total number of bytes currently allocated by the allocator if >0.
  int64 allocator_bytes_in_use_;
  // output_idx -> {output_bytes, memory_ptr}
  std::map<int64, std::pair<int64, uint64>> output_bytes_;

  int64 float_ops_;

  string op_;
};

class TFMultiGraphNode {
 public:
  TFMultiGraphNode(const string& name)
      : name_(name),
        kernel_exec_micros_(0),
        requested_bytes_(0),
        float_ops_(0) {}

  bool SnapshotNodes(const std::vector<string>& type_regexes) {
    kernel_exec_micros_ = 0;
    requested_bytes_ = 0;
    float_ops_ = 0;
    op_types_.clear();
    shapes_.clear();
    devices_.clear();
    snapshot_nodes_.clear();

    std::map<string, std::vector<const TFGraphNode*>> nodes =
        pick_nodes(type_regexes);

    if (nodes.empty()) {
      return (type_regexes.size() == 1 && type_regexes[0] == ".*");
    }

    std::set<string> visits;
    for (const auto& entry : nodes) {
      op_types_.insert(entry.first);

      for (const TFGraphNode* node : entry.second) {
        if (visits.find(node->name()) != visits.end()) continue;
        visits.insert(node->name());

        kernel_exec_micros_ += node->kernel_exec_micros();
        requested_bytes_ += node->requested_bytes();
        float_ops_ += node->float_ops();
        if (node->shape().size() > 0) {
          shapes_.push_back(node->shape());
        }
        devices_.insert(node->canonical_device());
        snapshot_nodes_[node->name()] = node;
      }
    }
    return true;
  }

  void AddGraphNode(const TFGraphNode* node) {
    if (nodes_.find(node->name()) != nodes_.end()) {
      return;
    }
    nodes_[node->name()] = node;
    for (const string& type : node->op_types()) {
      nodes_by_type_[type].push_back(node);
    }
  }

  const std::map<string, const TFGraphNode*>& graph_nodes() const {
    return snapshot_nodes_;
  }

  void AddChildren(const string& name) {
    if (children_.find(name) != children_.end()) {
      return;
    }
    children_[name].reset(new TFMultiGraphNode(name));
  }
  const std::map<string, std::unique_ptr<TFMultiGraphNode>>& children() const {
    return children_;
  }

  const string& name() const { return name_; }

  int64 kernel_exec_micros() const { return kernel_exec_micros_; }

  int64 requested_bytes() const { return requested_bytes_; }

  int64 float_ops() const { return float_ops_; }

  const std::set<string>& devices() const { return devices_; }

  const std::set<string>& op_types() const { return op_types_; }

  const std::vector<std::vector<int64>>& shapes() const { return shapes_; }

 private:
  std::map<string, std::vector<const TFGraphNode*>> pick_nodes(
      const std::vector<string>& type_regexes) {
    if (type_regexes.empty()) {
      return {};
    }
    if (type_regexes.size() == 1 && type_regexes[0] == ".*") {
      return nodes_by_type_;
    }
    std::map<string, std::vector<const TFGraphNode*>> ret;
    for (const string& regex : type_regexes) {
      for (const auto& n : nodes_by_type_) {
        if (RE2::FullMatch(n.first, regex)) {
          ret[n.first] = n.second;
        }
      }
    }
    return ret;
  }

  const string name_;
  // Snapshot micros based on type_regexes
  std::set<string> op_types_;
  int64 kernel_exec_micros_;
  int64 requested_bytes_;
  int64 float_ops_;
  std::set<string> devices_;
  std::vector<std::vector<int64>> shapes_;
  std::map<string, const TFGraphNode*> snapshot_nodes_;

  // Overall data held by the TFMultiGraphNode.
  std::map<string, const TFGraphNode*> nodes_;
  std::map<string, std::vector<const TFGraphNode*>> nodes_by_type_;
  std::map<string, std::unique_ptr<TFMultiGraphNode>> children_;
};
}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_NODE_H_
