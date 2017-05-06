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

namespace tensorflow {
namespace tfprof {

class TFNode {
 public:
  TFNode(const NodeDef* node)
      : node_(node),
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

  TFNode() : TFNode(nullptr) {}

  void AddInput(TFNode* input) { inputs_[input->node_def()->name()] = input; }

  void AddOpType(const string& op_type) { op_types_.insert(op_type); }

  void AddStepStat(const string& device, const NodeExecStats* step_stat);

  // Add CostGraphDef::Node.
  void AddNodeStat(const CostGraphDef::Node* cost_node);

  void AddFloatOps(int64 float_ops) { float_ops_ = float_ops; }

  const NodeDef* node_def() { return node_; }
  const std::map<string, TFNode*>& inputs() { return inputs_; }
  int64 op_start_micros() { return op_start_micros_; }
  // This is time spent in Op::Compute(), which is GPU kernel schedule time.
  // Currently not used.
  int64 op_schedule_micros() { return op_schedule_micros_; }
  // This is time spent in kernel execution.
  int64 kernel_compute_micros() { return kernel_compute_micros_; }
  int64 all_spent_micros() { return all_spent_micros_; }
  int64 requested_byptes() { return requested_bytes_; }
  int64 float_ops() { return float_ops_; }
  string device() { return device_; }
  const std::set<string>& op_types() { return op_types_; }

  const std::vector<int64>& shape() { return shape_; }

 private:
  void update_shape(const std::vector<int64>& shape) { shape_ = shape; }

  std::map<string, TFNode*> inputs_;
  const NodeDef* node_;
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

}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_NODE_H_
