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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_I_SOC_CONTROL_WRAPPER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_I_SOC_CONTROL_WRAPPER_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/hexagon/graph_transferer.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class ISocControlWrapper {
 public:
  using ByteArray =
      std::tuple<uint8* /* data */, uint64 /* size */, DataType /* type */>;
  using ConstByteArray = std::tuple<const uint8* /* data */, uint64 /* size */,
                                    DataType /* type */>;

  ISocControlWrapper() = default;
  virtual ~ISocControlWrapper() = default;

  // Return version of SOC controller library.
  // This function is mainly for a debug purpose to verify SOC controller.
  virtual int GetVersion() = 0;

  // Initialize SOC. This function should be called before
  // starting graph transfer.
  virtual bool Init() = 0;

  // Finalize SOC. This function should be called when all graph executions
  // are finished.
  virtual bool Finalize() = 0;

  // Setup graph on SOC
  virtual bool SetupGraph(const GraphTransferer &graph_transferer) = 0;

  // Execute graph on SOC
  virtual bool ExecuteGraph() = 0;

  // Teardown Graph on SOC
  virtual bool TeardownGraph() = 0;

  // Fill input node's output on SOC with ByteArray
  virtual bool FillInputNode(const string& node_name,
                             const ConstByteArray bytes) = 0;

  // Fill input node's output on SOC with Tensor
  virtual bool FillInputNode(const string& node_name, const Tensor& tensor) = 0;

  // Read output node's outputs on SOC
  virtual bool ReadOutputNode(string node_name,
                              std::vector<ByteArray> *outputs) = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ISocControlWrapper);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_I_SOC_CONTROL_WRAPPER_H_
