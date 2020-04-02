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

#ifndef TENSORFLOW_CORE_GRAPPLER_COSTS_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_COSTS_UTILS_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/grappler/costs/cost_estimator.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {

// Returns a vector of InputProperties for 'node'. The vector will contain one
// entry for each input of 'node'.
// For each node in the graph, the 'name_to_cost' map stores a pointer to the
// corresponding cost graph node indexed by node name. The 'name_to_node' maps a
// node name to its node definition.
std::vector<OpInfo::TensorProperties> FindInputFeatures(
    const NodeDef& node,
    const std::unordered_map<string, const CostGraphDef::Node*>& name_to_cost,
    const std::unordered_map<string, const NodeDef*>& name_to_node);

// Returns the size of tensor (unit: bytes). For tensor shape with unknown rank,
// it assumes the tensor to be scalar. For any unknown dimension, it assumes
// size one.
int64 CalculateTensorSize(const OpInfo::TensorProperties& prop);

// Returns the size of output at port_num (unit: bytes). A special case is
// port_num -1, which is for control dependency and assumed to be 4 bytes.
int64 CalculateOutputSize(
    const std::vector<OpInfo::TensorProperties>& output_properties,
    int port_num);

// Returns the DeviceProperties of the device on which 'node' runs.
DeviceProperties GetDeviceInfo(const CostGraphDef::Node& node);
DeviceProperties GetDeviceInfo(const string& device_str);

// Return a string describing a node given a nodeinfo.
string GetOpDescription(const OpInfo& op_info);

// Builds the OpInfo for node without filling its device information, given all
// nodes in the graph and its input properties.
OpInfo BuildOpInfoWithoutDevice(
    const NodeDef& node,
    const std::unordered_map<string, const NodeDef*>& name_to_node,
    const std::vector<OpInfo::TensorProperties>& inputs);

// Gather performance data from a cost graph.
OpPerformanceList CostGraphToOpPerformanceData(const CostGraphDef& cost_graph,
                                               const GraphDef& graph);

// Simple histogram for profiling Tensor size; histogram uses logarithmic
// buckets.
class TensorSizeHistogram {
 public:
  TensorSizeHistogram() : buckets_(kMaxBuckets, 0) {}

  void Add(const uint64 value);
  void Merge(const TensorSizeHistogram& src);
  double Average() const {
    if (num_elem_ > 0) {
      return static_cast<double>(sum_elem_) / num_elem_;
    } else {
      return 0.0;
    }
  }
  uint64 Min() const { return min_; }
  uint64 Max() const { return max_; }
  uint64 NumElem() const { return num_elem_; }
  uint64 SumElem() const { return sum_elem_; }
  string ToString() const;

 protected:
  const int Index(const uint64 value) const;
  const std::vector<uint64>& GetBuckets() const { return buckets_; }

 private:
  const int kMaxBuckets = 64;
  uint64 num_elem_ = 0;
  uint64 sum_elem_ = 0;
  // min_ and max_ are initialized to a very large value and zero, respectively,
  // so that any value added can replace the initial min_ and max_.
  uint64 min_ = kuint64max;
  uint64 max_ = 0;
  // Buckets are logarithmic:
  // 0B, 1B, 2-3B, 4-7B, 8-15B, ..., 2^N - 2^(N+1)-1B, ...
  std::vector<uint64> buckets_;
};

// Helper functions for aggregating per-device stats into per-device-class
// stats.
string GetDeviceClassForNonChannelDevice(const string& device_name);
string GetDeviceClass(const string& device_name);

// Get stats in string format from RunMetadata.
string GetStatsStringFromRunMetadata(const RunMetadata& run_metadata,
                                     bool verbosity);

// This method calculates the execution time depending on whether IO can
// overlap with computation. It assumes the memory and the compute times have
// already been calculated.
void CombineCostsAndUpdateExecutionTime(bool compute_memory_overlap,
                                        Costs* costs);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_UTILS_H_
