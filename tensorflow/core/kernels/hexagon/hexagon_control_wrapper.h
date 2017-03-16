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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_CONTROL_WRAPPER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_CONTROL_WRAPPER_H_

#include <vector>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/hexagon/graph_transferer.h"
#include "tensorflow/core/kernels/i_remote_fused_graph_executor.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

/*
  HexagonControlWrapper is implementing interfaces in IRemoteFusedGraphExecutor.
  This class calls APIs on hexagon via hexagon control binary.
  TODO(satok): Add more documents about hexagon control binary.
 */
class HexagonControlWrapper final : public IRemoteFusedGraphExecutor {
 public:
  HexagonControlWrapper() = default;
  int GetVersion() final;
  bool Init(const RemoteFusedGraphExecuteInfo& info) final;
  bool Finalize() final;
  bool SetupGraph() final;
  bool ExecuteGraph() final;
  bool TeardownGraph() final;
  bool FillInputNode(const string& node_name, const Tensor& tensor) final;
  bool ReadOutputNode(const string& node_name,
                      TensorAllocatorFunc tensor_allocator) final;
  bool ReadOutputNode(const string& node_name, std::vector<ByteArray>* outputs);

 private:
  bool FillInputNode(const string& node_name, const ConstByteArray bytes);

  // CAVEAT: Need offset as HVX library reserves some ids
  static constexpr int NODE_ID_OFFSET = 0x10000;

  static GraphTransferInfo::NodeInfo* FindNodeInfo(
      const string& node_name, GraphTransferInfo* graph_transfer_info);

  const RemoteFusedGraphExecuteInfo* execute_info_{};
  GraphTransferer graph_transferer_{};
  // Dummy float array for input node.
  // TODO(satok): Use actual data passed by FillInputNode and remove
  std::vector<float> dummy_input_float_{};
  // Dummy byte array for cosnt node.
  // TODO(satok): Remove
  std::unordered_map<int, std::vector<uint8>> dummy_const_data_{};

  TF_DISALLOW_COPY_AND_ASSIGN(HexagonControlWrapper);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_CONTROL_WRAPPER_H_
