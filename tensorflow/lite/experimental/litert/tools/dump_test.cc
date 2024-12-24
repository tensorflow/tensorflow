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

#include "tensorflow/lite/experimental/litert/tools/dump.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace {

using ::litert::internal::Dump;
using ::litert::internal::DumpOptions;
using ::litert::testing::LoadTestFileModel;

TEST(DumpTest, TestDump) {
  auto model = LoadTestFileModel("one_mul.tflite");

  {
    std::ostringstream model_dump;
    Dump(*model.Get(), model_dump);
    EXPECT_EQ(model_dump.view(), "LiteRtModel : [ #subgraphs=1 ]\n");
  }

  {
    const LiteRtTensorT& in_tensor = model.Get()->Subgraph(0).Input(0);
    std::ostringstream in_tensor_dump;
    Dump(in_tensor, in_tensor_dump);
    EXPECT_EQ(in_tensor_dump.view(),
              "LiteRtTensor : <2x2xf32> [ * ] (TFL_MUL)\n");
  }

  {
    const LiteRtTensorT& out_tensor = model.Get()->Subgraph(0).Output(0);
    std::ostringstream out_tensor_dump;
    Dump(out_tensor, out_tensor_dump);
    EXPECT_EQ(out_tensor_dump.view(),
              "LiteRtTensor : <2x2xf32> [ TFL_MUL ] ()\n");
  }

  {
    const LiteRtOpT& op = model.Get()->Subgraph(0).Op(0);
    std::ostringstream op_dump;
    Dump(op, op_dump);
    EXPECT_EQ(op_dump.view(),
              "LiteRtOp : [ TFL_MUL ] (<2x2xf32>, <2x2xf32>) -> <2x2xf32>\n");
  }

  {
    const LiteRtSubgraphT& subgraph = model.Get()->Subgraph(0);
    std::ostringstream subgraph_dump;
    Dump(subgraph, subgraph_dump);
    EXPECT_EQ(
        subgraph_dump.view(),
        "LiteRtSubgraph : [ #ops=1 #tensors=3 ] (<2x2xf32>, <2x2xf32>) -> "
        "<2x2xf32>\n");
  }
}

TEST(DumpTest, TestDumpOptions) {
  auto model = LoadTestFileModel("simple_strided_slice_op.tflite");
  const LiteRtOpT& op = model.Get()->Subgraph(0).Op(0);
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

TEST(DumpTest, TestDumpPerTensorQuantization) {
  QuantizationDetail per_tensor_detail;
  per_tensor_detail.per_tensor.scale = 1.0;
  per_tensor_detail.per_tensor.zero_point = 2;
  std::ostringstream q_dump;
  Dump(std::make_pair(kLiteRtQuantizationPerTensor, per_tensor_detail), q_dump);
  EXPECT_EQ(q_dump.view(), " <q PerTensor [ .z = 2, .s = 1.000000 ]>");
}

TEST(DumpTest, TestDumpPerChannelQuantization) {
  static constexpr size_t kRank = 2;
  static constexpr size_t kQuantizedDimension = 1;
  static constexpr float kScales[kRank] = {1.0, 2.0};
  static constexpr int64_t kZps[kRank] = {2, 3};
  QuantizationDetail per_channel_detail;
  per_channel_detail.per_channel.scales = const_cast<float*>(kScales);
  per_channel_detail.per_channel.zero_points = const_cast<int64_t*>(kZps);
  per_channel_detail.per_channel.quantized_dimension = kQuantizedDimension;
  per_channel_detail.per_channel.num_channels = kRank;
  std::ostringstream q_dump;
  Dump(std::make_pair(kLiteRtQuantizationPerChannel, per_channel_detail),
       q_dump);
  EXPECT_FALSE(q_dump.view().empty());
}

TEST(DumpTest, TestDumpNoQuantization) {
  QuantizationDetail none_detail;
  std::ostringstream q_dump;
  Dump(std::make_pair(kLiteRtQuantizationNone, none_detail), q_dump);
  EXPECT_TRUE(q_dump.view().empty());
}

TEST(DumpTest, TestDumpUnknownQuantization) {
  QuantizationDetail detail;
  std::ostringstream q_dump;
  Dump(std::make_pair(kLiteRtQuantizationBlockWise, detail), q_dump);
  EXPECT_EQ(q_dump.view(), " <q UNKNOWN>");
}

}  // namespace
