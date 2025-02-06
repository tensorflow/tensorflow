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

#ifndef TENSORFLOW_CORE_GRAPPLER_CLUSTERS_VIRTUAL_CLUSTER_H_
#define TENSORFLOW_CORE_GRAPPLER_CLUSTERS_VIRTUAL_CLUSTER_H_

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/analytical_cost_estimator.h"
#include "tensorflow/core/grappler/costs/op_level_cost_estimator.h"
#include "tensorflow/core/grappler/costs/virtual_scheduler.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {

// Create a simple cluster that lists the devices (and their properties)
// available in a TensorFlow session. This cluster simulates the execution of
// actual graphs.
class VirtualCluster : public Cluster {
 public:
  explicit VirtualCluster(
      const std::unordered_map<string, DeviceProperties>& devices);
  VirtualCluster(const std::unordered_map<string, DeviceProperties>& devices,
                 std::unique_ptr<OpLevelCostEstimator> node_estimator,
                 std::unique_ptr<ReadyNodeManager> node_manager);
  explicit VirtualCluster(const DeviceSet* device_set);

  ~VirtualCluster() override;

  string type() const override { return "virtual"; }

  absl::Status Provision() override;
  absl::Status Initialize(const GrapplerItem& item) override;
  absl::Status Run(const GraphDef& graph,
                   const std::vector<std::pair<string, Tensor>>& feed,
                   const std::vector<string>& fetch,
                   RunMetadata* metadata) override;
  absl::Status Run(const GrapplerItem& item, RunMetadata* metadata) override;
  const DeviceSet* GetDeviceSet() const override { return device_set_; }

 private:
  std::unique_ptr<AnalyticalCostEstimator> estimator_;
  const DeviceSet* device_set_ = nullptr;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_CLUSTERS_VIRTUAL_CLUSTER_H_
