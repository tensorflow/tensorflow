/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
vcyou may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_KERNELS_HEXAGON_HEXAGON_OPS_DEFINITIONS_H_
#define TENSORFLOW_CORE_KERNELS_HEXAGON_HEXAGON_OPS_DEFINITIONS_H_

#include <unordered_map>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/i_remote_fused_graph_ops_definitions.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// HexagonOpsDefinitions provides ops definitions supported in hexagon library
// TODO(satok): add a functionality to call functions in hexagon library
class HexagonOpsDefinitions final : public IRemoteFusedGraphOpsDefinitions {
 public:
  static const IRemoteFusedGraphOpsDefinitions& getInstance();

  int GetTotalOpsCount() const final;
  int GetOpIdFor(const string& op_type, const DataTypeVector& dt) const final;

 private:
  enum class SupportedOpType;
  using DataTypeToOp = std::tuple<DataTypeVector, SupportedOpType>;

  HexagonOpsDefinitions();

  static void EmplaceOpType(
      const string& op_type, const DataTypeVector& dt_vec,
      const SupportedOpType supported_op_type,
      std::unordered_map<string, std::vector<DataTypeToOp>>* map);

  static std::unordered_map<string, std::vector<DataTypeToOp>>
  BuildOpNameToSocOpTypeMap();

  const std::unordered_map<string, std::vector<DataTypeToOp>>
      op_name_to_soc_op_type_map_;

  TF_DISALLOW_COPY_AND_ASSIGN(HexagonOpsDefinitions);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_HEXAGON_HEXAGON_OPS_DEFINITIONS_H_
