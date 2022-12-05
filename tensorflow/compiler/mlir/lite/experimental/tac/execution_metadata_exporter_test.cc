// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/mlir/lite/experimental/tac/execution_metadata_exporter.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/runtime_metadata_generated.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace tflite {

std::string CreateRuntimeMetadata() {
  flatbuffers::FlatBufferBuilder fb_builder;

  std::vector<flatbuffers::Offset<flatbuffers::String>> device_names = {
      fb_builder.CreateString("GPU"), fb_builder.CreateString("CPU")};

  const auto hardwares =
      CreateHardwareMetadata(fb_builder, fb_builder.CreateVector(device_names));
  const auto ops = {
      CreateOpMetadata(fb_builder, 0, 0,
                       fb_builder.CreateVector(std::vector<float>({1.0, 5.0}))),
      CreateOpMetadata(fb_builder, 1, 0,
                       fb_builder.CreateVector(std::vector<float>({1.0, 5.0}))),
      CreateOpMetadata(fb_builder, 2, 0,
                       fb_builder.CreateVector(std::vector<float>({1.0, 5.0}))),
      CreateOpMetadata(
          fb_builder, 3, 1,
          fb_builder.CreateVector(std::vector<float>({-1.0, 2.0}))),
  };
  const auto subgraphs = {CreateSubgraphMetadata(
      fb_builder, fb_builder.CreateVector(ops.begin(), ops.size()))};

  const auto metadata = CreateRuntimeMetadata(
      fb_builder, hardwares,
      fb_builder.CreateVector(subgraphs.begin(), subgraphs.size()));
  fb_builder.Finish(metadata);

  return std::string(
      reinterpret_cast<const char*>(fb_builder.GetBufferPointer()),
      fb_builder.GetSize());
}

void Verify(const RuntimeMetadata* result, const RuntimeMetadata* expected) {
  EXPECT_EQ(result->subgraph_metadata()->size(),
            expected->subgraph_metadata()->size());
  for (int i = 0; i < result->subgraph_metadata()->size(); ++i) {
    auto result_subgraph_metadata =
        result->subgraph_metadata()->GetAs<SubgraphMetadata>(i);
    auto expected_subgraph_metadata =
        expected->subgraph_metadata()->GetAs<SubgraphMetadata>(i);
    if (expected_subgraph_metadata->op_metadata() == nullptr &&
        result_subgraph_metadata->op_metadata() == nullptr) {
      return;
    }
    ASSERT_EQ(expected_subgraph_metadata->op_metadata()->size(),
              result_subgraph_metadata->op_metadata()->size());
    for (int j = 0; j < expected_subgraph_metadata->op_metadata()->size();
         ++j) {
      auto result_op_metadata =
          result_subgraph_metadata->op_metadata()->GetAs<OpMetadata>(j);
      auto expected_op_metadata =
          expected_subgraph_metadata->op_metadata()->GetAs<OpMetadata>(j);
      EXPECT_EQ(result_op_metadata->index(), expected_op_metadata->index());
      EXPECT_EQ(result_op_metadata->hardware(),
                expected_op_metadata->hardware());

      EXPECT_EQ(result_op_metadata->op_costs()->size(),
                expected_op_metadata->op_costs()->size());
      for (int i = 0; i < result_op_metadata->op_costs()->size(); ++i) {
        EXPECT_FLOAT_EQ(result_op_metadata->op_costs()->Get(i),
                        expected_op_metadata->op_costs()->Get(i));
      }
    }
  }
}

TEST(ExporterTest, Valid) {
  const std::string kMLIR = R"(
func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>) -> tensor<2x1xf32> {
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU6",  per_device_costs = {CPU = 5.0 : f32, GPU = 1.0 : f32}, tac.device = "GPU"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "tfl.mul"(%0, %arg2) {fused_activation_function = "RELU6", per_device_costs = {CPU = 5.0 : f32, GPU = 1.0 : f32}, tac.device = "GPU"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %2 = "tfl.add"(%arg0, %arg3) {fused_activation_function = "RELU6", per_device_costs = {CPU = 5.0 : f32, GPU = 1.0 : f32}, tac.device = "GPU"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.pack"(%1, %2) {axis = 0 : i32, per_device_costs = {CPU = 2.0 : f32, GPU = -1.0 : f32}, values_count = 2 : i32, tac.device = "CPU"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
  func.return %3 : tensor<2x1xf32>
})";
  const std::string kExpectedFB = CreateRuntimeMetadata();
  mlir::DialectRegistry registry;
  registry.insert<mlir::TFL::TensorFlowLiteDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::OwningOpRef<mlir::ModuleOp>(
      mlir::parseSourceString<mlir::ModuleOp>(kMLIR, &context));
  auto module_op = module.get();
  auto serialized_result_fb = ExportRuntimeMetadata(module_op);
  const auto* result = GetRuntimeMetadata(serialized_result_fb.value().c_str());
  const auto* expected = GetRuntimeMetadata(kExpectedFB.c_str());
  ASSERT_TRUE(result != nullptr);
  ASSERT_TRUE(result->subgraph_metadata() != nullptr);
  ASSERT_TRUE(expected->subgraph_metadata() != nullptr);
  Verify(result, expected);
}

}  // namespace tflite
