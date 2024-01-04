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

#include "xla/service/gpu/mock_nccl_config.h"

#include "absl/container/inlined_vector.h"
#include "xla/service/gpu/mock_nccl_config.pb.h"

namespace xla {
namespace gpu {

// TODO(b/304829426): Sync the nccl graph config with actual GCP setup.
absl::InlinedVector<MockNcclTopoGraphConfig, 6> GetNcclTopoGraphConfig() {
  absl::InlinedVector<MockNcclTopoGraphConfig, 6> graphs;
  MockNcclTopoGraphConfig tree_graph, ring_graph, collNet_graph, nvls_graph;
  tree_graph.set_type_inter(MOCK_PATH_PHB);
  tree_graph.set_type_intra(MOCK_PATH_NVL);
  tree_graph.set_bw_inter(20);
  tree_graph.set_bw_intra(40);
  tree_graph.set_pattern(MOCK_NCCL_TOPO_PATTERN_BALANCED_TREE);
  tree_graph.set_num_channels(1);
  tree_graph.set_same_channels(true);
  tree_graph.set_latency_inter(8.0);

  ring_graph.set_type_inter(MOCK_PATH_PHB);
  ring_graph.set_type_intra(MOCK_PATH_NVL);
  ring_graph.set_bw_inter(20);
  ring_graph.set_bw_intra(20);
  ring_graph.set_pattern(MOCK_NCCL_TOPO_PATTERN_RING);
  ring_graph.set_num_channels(1);
  ring_graph.set_same_channels(true);
  ring_graph.set_latency_inter(8.0);

  graphs.push_back(tree_graph);
  graphs.push_back(ring_graph);
  // no collNet/NVLS support on GCP
  graphs.push_back(collNet_graph);
  graphs.push_back(collNet_graph);
  graphs.push_back(nvls_graph);
  graphs.push_back(nvls_graph);

  return graphs;
}

MockNcclTopoCpuNode GetNcclTopoCpuNode() {
  MockNcclTopoCpuNode node;
  node.set_arch(MOCK_NCCL_TOPO_CPU_ARCH_X86);
  node.set_vendor(MOCK_NCCL_TOPO_CPU_VENDOR_AMD);
  return node;
}

}  // namespace gpu
}  // namespace xla
