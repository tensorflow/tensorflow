/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cudnn_sdpa_score_mod.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"
#include "json/json.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

namespace stream_executor {
namespace gpu {

using tsl::testing::IsOk;

TEST(CudnnSdpaScoreModTest, CompileFwd) {
  absl::string_view hlo = R"(
  HloModule jit__unnamed_wrapped_function_

  %fwd (Arg_0.15: f32[4,4,1024,1024]) -> f32[4,4,1024,1024] {
    %Arg_0.15 = f32[4,4,1024,1024]{3,2,1,0} parameter(0)
    %constant.1 = f32[] constant(3)
    %broadcast.1 = f32[4,4,1024,1024]{3,2,1,0} broadcast(%constant.1), dimensions={}
    ROOT %multiply.1 = f32[4,4,1024,1024]{3,2,1,0} multiply(%Arg_0.15, %broadcast.1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          xla::ParseAndReturnUnverifiedModule(hlo));
  xla::HloComputation* comp = module->GetComputationWithName("fwd");
  ASSERT_NE(comp, nullptr);
  int64_t uid = 0;
  auto next_uid = [&]() -> int64_t { return uid++; };
  Graph graph = std::make_shared<cudnn_frontend::graph::Graph>();
  auto score_mod = std::make_shared<ScoreModFunc>(comp, nullptr);
  EXPECT_THAT(score_mod->UpdateCudnnMap(*graph, next_uid), IsOk());
  Tensor attn_score =
      graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_dim({4, 4, 1024, 1024})
                        .set_stride({4194304, 1048576, 1024, 1})
                        .set_data_type(cudnn_frontend::DataType_t::FLOAT)
                        .set_uid(next_uid())
                        .set_is_virtual(true));
  auto output = score_mod->Forward(graph, attn_score);
  std::string json_string = graph->print();
  Json::Value parsed_json;
  Json::Reader json_reader;
  json_reader.parse(json_string, parsed_json,
                    /* collectComments */ false);
  EXPECT_TRUE(parsed_json.isObject());
  // one mul node
  EXPECT_THAT(parsed_json["nodes"].size(), 1);
  // 2 inputs and 1 output in the graph
  EXPECT_THAT(parsed_json["tensors"].size(), 3);
  EXPECT_THAT(parsed_json["nodes"][0]["mode"], "MUL");
}

TEST(CudnnSdpaScoreModTest, CompileBwd) {
  absl::string_view hlo = R"(
  HloModule jit__unnamed_wrapped_function_

  %fwd (Arg_0.15: f32[4,4,1024,1024]) -> f32[4,4,1024,1024] {
    %Arg_0.15 = f32[4,4,1024,1024]{3,2,1,0} parameter(0)
    %constant.1 = f32[] constant(3)
    %broadcast.1 = f32[4,4,1024,1024]{3,2,1,0} broadcast(%constant.1), dimensions={}
    ROOT %multiply.1 = f32[4,4,1024,1024]{3,2,1,0} multiply(%Arg_0.15, %broadcast.1)
  }

  %bwd (Arg_0.2: f32[4,4,1024,1024], Arg_1.0: f32[4,4,1024,1024]) -> f32[4,4,1024,1024] {
    %Arg_1.0 = f32[4,4,1024,1024]{3,2,1,0} parameter(1)
    %Arg_0.2 = f32[4,4,1024,1024]{3,2,1,0} parameter(0)
    %constant_1_0 = f32[] constant(3)
    %broadcast.1.0 = f32[4,4,1024,1024]{3,2,1,0} broadcast(%constant_1_0), dimensions={}
    ROOT %multiply.1.0 = f32[4,4,1024,1024]{3,2,1,0} multiply(%Arg_0.2, %broadcast.1.0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          xla::ParseAndReturnUnverifiedModule(hlo));
  xla::HloComputation* fwd_comp = module->GetComputationWithName("fwd");
  xla::HloComputation* bwd_comp = module->GetComputationWithName("bwd");
  ASSERT_NE(fwd_comp, nullptr);
  ASSERT_NE(bwd_comp, nullptr);
  int64_t uid = 0;
  auto next_uid = [&]() -> int64_t { return uid++; };
  Graph graph = std::make_shared<cudnn_frontend::graph::Graph>();
  auto score_mod = std::make_shared<ScoreModFunc>(fwd_comp, bwd_comp);
  EXPECT_THAT(score_mod->UpdateCudnnMap(*graph, next_uid), IsOk());
  Tensor attn_score =
      graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_dim({4, 4, 1024, 1024})
                        .set_stride({4194304, 1048576, 1024, 1})
                        .set_data_type(cudnn_frontend::DataType_t::FLOAT)
                        .set_uid(next_uid())
                        .set_is_virtual(true));
  Tensor score_grad =
      graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_dim({4, 4, 1024, 1024})
                        .set_stride({4194304, 1048576, 1024, 1})
                        .set_data_type(cudnn_frontend::DataType_t::FLOAT)
                        .set_uid(next_uid())
                        .set_is_virtual(true));
  auto fwd_output = score_mod->Forward(graph, attn_score);
  auto bwd_output = score_mod->Backward(graph, score_grad);
  std::string json_string = graph->print();
  Json::Value parsed_json;
  Json::Reader json_reader;
  json_reader.parse(json_string, parsed_json,
                    /* collectComments */ false);
  EXPECT_TRUE(parsed_json.isObject());
  // two mul for fwd and bwd
  EXPECT_THAT(parsed_json["nodes"].size(), 2);
  // 2 inputs and 1 output for single mul
  EXPECT_THAT(parsed_json["tensors"].size(), 6);
  EXPECT_THAT(parsed_json["nodes"][0]["mode"], "MUL");
  EXPECT_THAT(parsed_json["nodes"][1]["mode"], "MUL");
}

}  // namespace gpu
}  // namespace stream_executor
