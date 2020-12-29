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

#include "tensorflow/core/kernels/i_remote_fused_graph_executor.h"
#include "tensorflow/core/kernels/remote_fused_graph_execute_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace hexagon_remote_fused_graph_executor_build {

Status BuildRemoteFusedGraphExecutor(
    std::unique_ptr<IRemoteFusedGraphExecutor>* executor);

namespace {

TEST(HexagonBuildRemoteFusedGraphExecutorTest, BasicRun) {
  std::unique_ptr<IRemoteFusedGraphExecutor> executor;
  ASSERT_FALSE(static_cast<bool>(executor));
  TF_ASSERT_OK(BuildRemoteFusedGraphExecutor(&executor));
  ASSERT_TRUE(static_cast<bool>(executor));
  ASSERT_NE(RemoteFusedGraphExecuteUtils::GetExecutorBuildFunc(
                "build_hexagon_remote_fused_graph_executor"),
            nullptr);
}

}  // namespace

}  // namespace hexagon_remote_fused_graph_executor_build
}  // namespace tensorflow
