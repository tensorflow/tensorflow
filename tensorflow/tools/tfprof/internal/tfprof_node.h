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
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/errors.h"
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
        op_start_micros_(0),
        op_schedule_micros_(0),
        kernel_compute_micros_(0),
        all_spent_micros_(0),
        requested_bytes_(0),
        float_ops_(0) {
    if (!node) return;

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
    device_ = node->device();
  }

  TFGraphNode() : TFGraphNode(nullptr) {}

  void AddInput(TFGraphNode* input) { inputs_[input->name()] = input; }

  void AddOpType(const string& op_type) { op_types_.insert(op_type); }

  void AddStepStat(const string& device, const NodeExecStats* step_stat);

  // Add CostGraphDef::Node.
  void AddNodeStat(const CostGraphDef::Node* cost_node);

  void AddFloatOps(int64 float_ops) { float_ops_ = float_ops; }

  void AddCode(const CodeDef* code) { code_ = code; }

  const string& name() const { return node_->name(); }
  const NodeDef* node_def() { return node_; }
  const std::map<string, TFGraphNode*>& inputs() const { return inputs_; }
  int64 op_start_micros() { return op_start_micros_; }
  // This is time spent in Op::Compute(), which is GPU kernel schedule time.
  // Currently not used.
  int64 op_schedule_micros() { return op_schedule_micros_; }
  // This is time spent in kernel execution.
  int64 kernel_compute_micros() const { return kernel_compute_micros_; }
  int64 all_spent_micros() { return all_spent_micros_; }
  int64 requested_bytes() const { return requested_bytes_; }
  int64 float_ops() const { return float_ops_; }
  const CodeDef* code() { return code_; }
  string device() const { return device_; }
  const std::set<string>& op_types() const { return op_types_; }

  const std::vector<int64>& shape() const { return shape_; }

 private:
  void update_shape(const std::vector<int64>& shape) { shape_ = shape; }

  std::map<string, TFGraphNode*> inputs_;
  const NodeDef* node_;
  const CodeDef* code_;
  const NodeExecStats* step_stat_;

  std::vector<int64> shape_;
  std::set<string> op_types_;
  string device_;
  int64 op_start_micros_;
  int64 op_schedule_micros_;
  int64 kernel_compute_micros_;
  int64 all_spent_micros_;
  int64 requested_bytes_;
  int64 float_ops_;
};

class TFCodeNode {
 public:
  TFCodeNode(const string& trace)
      : trace_(trace),
        kernel_compute_micros_(0),
        requested_bytes_(0),
        float_ops_(0) {}

  void AddGraphNode(const TFGraphNode* node) {
    if (nodes_.find(node->name()) != nodes_.end()) {
      return;
    }
    nodes_[node->name()] = node;

    kernel_compute_micros_ += node->kernel_compute_micros();
    requested_bytes_ += node->requested_bytes();
    float_ops_ += node->float_ops();
    op_types_.insert(node->op_types().begin(), node->op_types().end());
    if (node->shape().size() > 0) {
      shapes_.push_back(node->shape());
    }
    if (!node->device().empty()) {
      devices_.insert(node->device());
    }
  }
  const std::map<string, const TFGraphNode*>& graph_nodes() const {
    return nodes_;
  }

  void AddChildren(const string& trace) {
    if (children_.find(trace) != children_.end()) {
      return;
    }
    children_[trace].reset(new TFCodeNode(trace));
  }
  std::map<string, std::unique_ptr<TFCodeNode>>& children() {
    return children_;
  }

  const string& name() const { return trace_; }

  int64 kernel_compute_micros() const { return kernel_compute_micros_; }

  int64 requested_bytes() const { return requested_bytes_; }

  int64 float_ops() const { return float_ops_; }

  const std::set<string>& devices() const { return devices_; }

  const std::set<string>& op_types() const { return op_types_; }

  const std::vector<std::vector<int64>>& shapes() const { return shapes_; }

 private:
  const string trace_;
  std::set<string> op_types_;
  int64 kernel_compute_micros_;
  int64 requested_bytes_;
  int64 float_ops_;

  std::set<string> devices_;
  std::vector<std::vector<int64>> shapes_;
  std::map<string, const TFGraphNode*> nodes_;
  std::map<string, std::unique_ptr<TFCodeNode>> children_;
};
}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_NODE_H_
