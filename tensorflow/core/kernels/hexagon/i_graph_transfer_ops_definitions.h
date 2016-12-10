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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_I_GRAPH_TRANSFER_OPS_DEFINITIONS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_I_GRAPH_TRANSFER_OPS_DEFINITIONS_H_

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// IGraphTransferOpsDefinitions is an interface class which provides interfaces
// about ops supported by SOC.
// TODO(satok): Provide ways to transfer graph definitions into SOC
class IGraphTransferOpsDefinitions {
 public:
  // op id which is not supported by SOC
  static constexpr int INVALID_OP_ID = -1;
  // Custom op name for input node
  static constexpr const char* const INPUT_OP_NAME = "INPUT";
  // Custom op name for output node
  static constexpr const char* const OUTPUT_OP_NAME = "OUTPUT";
  // Custom op name for flatten node
  static constexpr const char* const FLATTEN_OP_NAME = "FLATTEN";

  IGraphTransferOpsDefinitions() = default;
  virtual ~IGraphTransferOpsDefinitions() = default;
  // Return total ops count supported by SOC
  virtual int GetTotalOpsCount() const = 0;
  // Return op id for input node
  // TODO(satok): Return vector isntead of integer
  virtual int GetInputNodeOpId() const = 0;
  // Return op id for output node
  // TODO(satok): Return vector isntead of integer
  virtual int GetOutputNodeOpId() const = 0;
  // Return op id for given string op name
  virtual int GetOpIdFor(const string& op_name) const = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IGraphTransferOpsDefinitions);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_I_GRAPH_TRANSFER_OPS_DEFINITIONS_H
