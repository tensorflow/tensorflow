/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_ONEDNN_FUSION_GRAPH_H_
#define XLA_BACKENDS_CPU_ONEDNN_FUSION_GRAPH_H_

// oneDNN-fusion-related defines that depend on oneDNN Graph API.
// For anything independent of Graph API, put it in onednn_fusion.h.

#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"

namespace xla::cpu {

// oneDNN fusion encapsulates logical tensors corresponding to fusion parameters
// and results, and oneDNN graph constructed from an XLA fusion computation,
// where each HLO op has a corresponding oneDNN operation in the graph.
struct OneDnnFusion {
  std::vector<dnnl::graph::logical_tensor> parameters;
  std::vector<dnnl::graph::logical_tensor> results;
  dnnl::graph::graph graph;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_ONEDNN_FUSION_GRAPH_H_
