/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_I_REMOTE_GRAPH_EXECUTOR_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_I_REMOTE_GRAPH_EXECUTOR_H_

#include "tensorflow/core/framework/remote_fused_graph_execute_info.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class IRemoteFusedGraphExecutor {
 public:
  using TensorAllocatorFunc = std::function<Tensor*(const TensorShape& shape)>;

  IRemoteFusedGraphExecutor() = default;
  virtual ~IRemoteFusedGraphExecutor() = default;

  // Return version of executor.
  // This function is mainly for a debug purpose to verify version of
  // executor info.
  virtual int GetVersion() = 0;

  // Initialize executor. This function is called before
  // starting graph transfer.
  virtual bool Init(const RemoteFusedGraphExecuteInfo& info) = 0;

  // Finalize executor. This function is called when all graph executions
  // are finished.
  virtual bool Finalize() = 0;

  // Setup graph
  virtual bool SetupGraph() = 0;

  // Execute graph
  virtual bool ExecuteGraph() = 0;

  // Teardown Graph
  virtual bool TeardownGraph() = 0;

  // Fill input node's output with Tensor
  virtual bool FillInputNode(const string& node_name, const Tensor& tensor) = 0;

  // Read output node's outputs as ByteArrays
  virtual bool ReadOutputNode(const string& node_name,
                              TensorAllocatorFunc tensor_allocator) = 0;

  virtual Status FuseRemoteGraph(const GraphDef& original_graph_def,
                                 const std::vector<string>& inputs,
                                 const std::vector<string>& outputs,
                                 GraphDef* fused_graph_def) = 0;

  virtual bool IsEnabled() const = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IRemoteFusedGraphExecutor);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_I_REMOTE_GRAPH_EXECUTOR_H_
