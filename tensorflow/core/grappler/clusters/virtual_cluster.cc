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

#include "tensorflow/core/grappler/clusters/virtual_cluster.h"

namespace tensorflow {
namespace grappler {

VirtualCluster::VirtualCluster(
    const std::unordered_map<string, DeviceProperties>& devices)
    : Cluster(0) {
  devices_ = devices;
}

VirtualCluster::~VirtualCluster() {}

Status VirtualCluster::Provision() { return Status::OK(); }

Status VirtualCluster::Initialize(const GrapplerItem& item) {
  return Status::OK();
}

Status VirtualCluster::Run(const GraphDef& item,
                           const std::vector<std::pair<string, Tensor>>& feed,
                           const std::vector<string>& fetch,
                           RunMetadata* metadata) {
  return Status::OK();

}

}  // namespace grappler
}  // namespace tensorflow
