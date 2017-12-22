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

#ifndef TENSORFLOW_GRAPPLER_CLUSTERS_VIRTUAL_CLUSTER_H_
#define TENSORFLOW_GRAPPLER_CLUSTERS_VIRTUAL_CLUSTER_H_

#include <unordered_map>
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/op_level_cost_estimator.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {

// Create a simple cluster that lists the devices (and their properties)
// available in a TensorFlow session. This cluster simulates the execution of
// actual graphs.
class VirtualCluster : public Cluster {
 public:
  VirtualCluster(const std::unordered_map<string, DeviceProperties>& devices);
  VirtualCluster(const std::unordered_map<string, DeviceProperties>& devices,
                 OpLevelCostEstimator* node_estimator);

  ~VirtualCluster() override;

  string type() const override { return "virtual"; }

  Status Provision() override;
  Status Initialize(const GrapplerItem& item) override;
  Status Run(const GraphDef& item,
             const std::vector<std::pair<string, Tensor>>& feed,
             const std::vector<string>& fetch, RunMetadata* metadata) override;

 private:
  std::unique_ptr<OpLevelCostEstimator> node_estimator_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_CLUSTERS_VIRTUAL_CLUSTER_H_
