/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tfrt/transforms/update_op_cost_in_tfrt_mlir.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"
#include "tfrt/init_tfrt_dialects.h"  // from @tf_runtime

namespace tensorflow {
namespace {

constexpr char kCostAttrName[] = "_tfrt_cost";
constexpr char kOpKeyAttrName[] = "op_key";

absl::flat_hash_map<int64_t, uint64_t> GetOpCostMap(mlir::ModuleOp op) {
  absl::flat_hash_map<int64_t, uint64_t> op_cost_map;
  op.walk([&](mlir::Operation* op) {
    const auto cost_attr = op->getAttrOfType<mlir::IntegerAttr>(kCostAttrName);
    if (!cost_attr) return;
    const auto op_key_attr =
        op->getAttrOfType<mlir::IntegerAttr>(kOpKeyAttrName);
    if (!op_key_attr) return;
    op_cost_map[op_key_attr.getInt()] = cost_attr.getInt();
  });
  return op_cost_map;
}

TEST(CostUpdateTest, Basic) {
  std::string saved_model_mlir_path = tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/tfrt/tests/analysis/testdata/test.mlir");

  mlir::DialectRegistry registry;
  tfrt::RegisterTFRTDialects(registry);
  registry.insert<tfrt::fallback_async::FallbackAsyncDialect>();
  registry.insert<tfrt::fallback_sync::FallbackSyncDialect>();
  mlir::MLIRContext context(registry);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(saved_model_mlir_path, &context);
  ASSERT_TRUE(module);

  // Create a cost recorder with fake cost records.
  auto expected_op_cost_map = GetOpCostMap(module.get());
  EXPECT_EQ(expected_op_cost_map.size(), 1);
  unsigned int seed = 23579;
  for (auto& [op_key, cost] : expected_op_cost_map) {
    cost = rand_r(&seed) % 1000;
  }
  tensorflow::tfrt_stub::CostRecorder cost_recorder;
  for (const auto& [op_key, cost] : expected_op_cost_map) {
    cost_recorder.RecordCost(op_key, cost);
  }

  // Update the TFRT MLIR with the cost recorder.
  tfrt_compiler::UpdateOpCostInTfrtMlir(module.get(), cost_recorder);

  // Check the updated costs.
  const auto got_op_cost_map = GetOpCostMap(module.get());
  EXPECT_THAT(got_op_cost_map, ::testing::ContainerEq(expected_op_cost_map));
}

}  // namespace
}  // namespace tensorflow
