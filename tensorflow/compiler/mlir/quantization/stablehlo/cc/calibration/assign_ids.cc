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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/assign_ids.h"

#include <cstdint>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/graph_def.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace stablehlo::quantization {
namespace {

using ::tensorflow::GraphDef;
using ::tensorflow::NodeDef;
using ::tensorflow::calibrator::CalibratorSingleton;

}  // namespace

void AssignIdsToCustomAggregatorOps(GraphDef& graph_def) {
  MutateNodeDefs(graph_def, [](NodeDef& node_def) {
    if (node_def.op() == "CustomAggregator") {
      const int64_t new_id = CalibratorSingleton::IssueNewId();
      (*node_def.mutable_attr())["id"].set_s(absl::StrCat(new_id));
    }
  });
}

}  // namespace stablehlo::quantization
