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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_HEXAGON_OPS_DEFINITIONS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_HEXAGON_OPS_DEFINITIONS_H_

#include "i_graph_transfer_ops_definitions.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// HexagonOpsDefinitions provides ops definitons supported in hexagon library
// TODO(satok): add a functionality to call functions in hexagon library
class HexagonOpsDefinitions final : public IGraphTransferOpsDefinitions {
 public:
  static const IGraphTransferOpsDefinitions& getInstance();

  int GetTotalOpsCount() const final;
  int GetInputNodeOpId() const final;
  int GetOutputNodeOpId() const final;
  int GetOpIdFor(const string& op_type) const final;

 private:
  HexagonOpsDefinitions() = default;

  TF_DISALLOW_COPY_AND_ASSIGN(HexagonOpsDefinitions);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_HEXAGON_OPS_DEFINITIONS_H
