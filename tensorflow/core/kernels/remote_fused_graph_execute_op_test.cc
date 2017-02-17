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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

class RemoteFusedGraphExecuteTest : public OpsTestBase {};

TEST_F(RemoteFusedGraphExecuteTest, ExecuteAddGraph) {
  TF_ASSERT_OK(
      NodeDefBuilder("remote_fused_graph_execute_op", "RemoteFusedGraphExecute")
          .Input(FakeInput(2, DT_FLOAT))
          .Attr("M", 2)
          .Attr("N", 1)
          .Attr("T", DataTypeToEnum<float>::v())
          .Attr("serialized_graph_transfer_info", "")
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // TODO(satok): Add benchmark
}

}  // namespace tensorflow
