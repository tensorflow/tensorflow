// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/lrt/tools/dump.h"

#include <sstream>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/lrt/core/model.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"

namespace {

using ::lrt::internal::Dump;
using ::lrt::internal::DumpOptions;
using ::lrt::testing::LoadTestFileModel;

TEST(DumpTest, TestDump) {
  auto model = LoadTestFileModel("one_mul.tflite");

  {
    std::ostringstream model_dump;
    Dump(*model, model_dump);
    EXPECT_EQ(model_dump.view(), "LrtModel : [ #subgraphs=1 ]\n");
  }

  {
    const LrtTensorT& in_tensor = *model->subgraphs.front().inputs.front();
    std::ostringstream in_tensor_dump;
    Dump(in_tensor, in_tensor_dump);
    EXPECT_EQ(in_tensor_dump.view(), "LrtTensor : <2x2xf32> [ * ] (TFL_MUL)\n");
  }

  {
    const LrtTensorT& out_tensor = *model->subgraphs.front().outputs.front();
    std::ostringstream out_tensor_dump;
    Dump(out_tensor, out_tensor_dump);
    EXPECT_EQ(out_tensor_dump.view(), "LrtTensor : <2x2xf32> [ TFL_MUL ] ()\n");
  }

  {
    const LrtOpT& op = *model->subgraphs.front().ops.front();
    std::ostringstream op_dump;
    Dump(op, op_dump);
    EXPECT_EQ(op_dump.view(),
              "LrtOp : [ TFL_MUL ] (<2x2xf32>, <2x2xf32>) -> <2x2xf32>\n");
  }

  {
    const LrtSubgraphT& subgraph = model->subgraphs.front();
    std::ostringstream subgraph_dump;
    Dump(subgraph, subgraph_dump);
    EXPECT_EQ(subgraph_dump.view(),
              "LrtSubgraph : [ #ops=1 #tensors=3 ] (<2x2xf32>, <2x2xf32>) -> "
              "<2x2xf32>\n");
  }
}

TEST(DumpTest, TestDumpOptions) {
  auto model = LoadTestFileModel("simple_strided_slice_op.tflite");
  const LrtOpT& op = *model->subgraphs.front().ops.front();
  std::ostringstream op_dump;
  DumpOptions(op, op_dump);
  EXPECT_EQ(op_dump.view(),
            "begin_mask: 0\n"
            "end_mask: 0\n"
            "ellipsis_mask: 0\n"
            "new_axis_mask: 0\n"
            "shrink_axis_mask: 0\n"
            "offset: 0\n");
}

}  // namespace
