/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/hexagon/hexagon_control_wrapper.h"
#include "tensorflow/core/kernels/remote_fused_graph_execute_utils.h"

namespace tensorflow {
namespace hexagon_remote_fused_graph_executor_build {

Status BuildRemoteFusedGraphExecutor(
    std::unique_ptr<IRemoteFusedGraphExecutor>* executor) {
  executor->reset(new HexagonControlWrapper());
  return Status::OK();
}

static RemoteFusedGraphExecuteUtils::ExecutorBuildRegistrar
    k_hexagon_remote_fused_graph_executor_build(
        HexagonControlWrapper::REMOTE_FUSED_GRAPH_EXECUTOR_NAME,
        BuildRemoteFusedGraphExecutor);

}  // namespace hexagon_remote_fused_graph_executor_build
}  // namespace tensorflow
