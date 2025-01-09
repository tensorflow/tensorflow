// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_GRAPH_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_GRAPH_H_

#include <cstddef>
#include <map>

#include "third_party/odml/infra/southbound/sb_api.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"

class LiteRtDispatchGraphT {
 public:
  LiteRtDispatchGraphT(ThrGraph* thr_graph,
                       LiteRtDispatchDeviceContext device_context)
      : thr_graph_(thr_graph), device_context_(device_context) {}

  ThrGraph* thr_graph() { return thr_graph_; }

  LiteRtDispatchDeviceContext device_context() { return device_context_; }

  int NextNodeInputIndex(LiteRtDispatchNodeId node_id) {
    return NextNodeIoIndex(node_id, next_node_input_index_);
  }

  int NextNodeOutputIndex(LiteRtDispatchNodeId node_id) {
    return NextNodeIoIndex(node_id, next_node_output_index_);
  }

  int NextGraphInputIndex() { return next_graph_input_index_++; }

  int NextGraphOutputIndex() { return next_graph_output_index_++; }

  void AddInputEdge(int input_index, LiteRtDispatchEdgeId edge_id) {
    input_edges_[input_index] = edge_id;
  }

  void AddOutputEdge(int output_index, LiteRtDispatchEdgeId edge_id) {
    output_edges_[output_index] = edge_id;
  }

  litert::Expected<LiteRtDispatchEdgeId> InputEdge(int input_index) const {
    return IoEdge(input_index, input_edges_);
  }

  litert::Expected<LiteRtDispatchEdgeId> OutputEdge(int output_index) const {
    return IoEdge(output_index, output_edges_);
  }

  size_t NumOutputs() const { return output_edges_.size(); }

 private:
  using NextNodeIoIndexMap = std::map<LiteRtDispatchNodeId, int>;
  using IoIndexToEdgeIdMap = std::map<int, LiteRtDispatchEdgeId>;

  int NextNodeIoIndex(LiteRtDispatchNodeId node_id, NextNodeIoIndexMap& map) {
    return map[node_id]++;
  }

  litert::Expected<LiteRtDispatchEdgeId> IoEdge(
      int io_index, const IoIndexToEdgeIdMap& map) const {
    auto iter = map.find(io_index);
    if (iter == map.end()) {
      return litert::Unexpected(kLiteRtStatusErrorNotFound,
                                "Unexpected graph input/output index");
    }
    return iter->second;
  }

  ThrGraph* thr_graph_;
  LiteRtDispatchDeviceContext device_context_;
  NextNodeIoIndexMap next_node_input_index_;
  NextNodeIoIndexMap next_node_output_index_;
  int next_graph_input_index_ = 0;
  int next_graph_output_index_ = 0;
  IoIndexToEdgeIdMap input_edges_;
  IoIndexToEdgeIdMap output_edges_;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_LITERT_DISPATCH_GRAPH_H_
