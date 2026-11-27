/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"

namespace tensorflow {
namespace {

using ::testing::SizeIs;

TEST(GetTfrtPipelineOptions, BatchPaddingPolicy) {
  tensorflow::TfrtCompileOptions options;
  options.batch_padding_policy = "PAD_TEST_OPTION";
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->batch_padding_policy, "PAD_TEST_OPTION");
}

TEST(GetTfrtPipelineOptions, NumBatchThreads) {
  tensorflow::TfrtCompileOptions options;
  options.batch_options.set_num_batch_threads(2);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->num_batch_threads, 2);
}

TEST(GetTfrtPipelineOptions, MaxBatchSize) {
  tensorflow::TfrtCompileOptions options;
  options.batch_options.set_max_batch_size(8);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->max_batch_size, 8);
}

TEST(GetTfrtPipelineOptions, BatchTimeoutMicros) {
  tensorflow::TfrtCompileOptions options;
  options.batch_options.set_batch_timeout_micros(5000);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->batch_timeout_micros, 5000);
}

TEST(GetTfrtPipelineOptions, AllowedBatchSizes) {
  tensorflow::TfrtCompileOptions options;
  options.batch_options.add_allowed_batch_sizes(2);
  options.batch_options.add_allowed_batch_sizes(4);
  options.batch_options.add_allowed_batch_sizes(8);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_THAT(pipeline_options->allowed_batch_sizes, SizeIs(3));
}

TEST(GetTfrtPipelineOptions, MaxEnqueuedBatches) {
  tensorflow::TfrtCompileOptions options;
  options.batch_options.set_max_enqueued_batches(250);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->max_enqueued_batches, 250);
}

TEST(GetTfrtPipelineOptions, LowPriorityMaxBatchSize) {
  tensorflow::TfrtCompileOptions options;
  options.batch_options.set_low_priority_max_batch_size(16);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->low_priority_max_batch_size, 16);
}

TEST(GetTfrtPipelineOptions, LowPriorityBatchTimeoutMicros) {
  tensorflow::TfrtCompileOptions options;
  options.batch_options.set_low_priority_batch_timeout_micros(10000);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->low_priority_batch_timeout_micros, 10000);
}

TEST(GetTfrtPipelineOptions, LowPriorityAllowedBatchSizes) {
  tensorflow::TfrtCompileOptions options;
  options.batch_options.add_low_priority_allowed_batch_sizes(4);
  options.batch_options.add_low_priority_allowed_batch_sizes(8);
  options.batch_options.add_low_priority_allowed_batch_sizes(16);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_THAT(pipeline_options->low_priority_allowed_batch_sizes, SizeIs(3));
}

TEST(GetTfrtPipelineOptions, LowPriorityMaxEnqueuedBatches) {
  tensorflow::TfrtCompileOptions options;
  options.batch_options.set_low_priority_max_enqueued_batches(100);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->low_priority_max_enqueued_batches, 100);
}

TEST(GetTfrtPipelineOptions, NumWarmupBatchThreads) {
  tensorflow::TfrtCompileOptions options;
  options.batch_options.set_num_warmup_batch_threads(4);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->num_warmup_batch_threads, 4);
}

TEST(GetTfrtPipelineOptions, EnableLargeBatchSplitting) {
  tensorflow::TfrtCompileOptions options;
  options.batch_options.set_enable_large_batch_splitting(true);
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_TRUE(pipeline_options->enable_large_batch_splitting);
}

TEST(GetTfrtPipelineOptions, MixedPriorityBatchingPolicy) {
  tensorflow::TfrtCompileOptions options;
  options.batch_options.set_mixed_priority_batching_policy("priority_merge");
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->mixed_priority_batching_policy, "priority_merge");
}

TEST(GetTfrtPipelineOptions, PriorityAwareBatchSchedulerUseGlobalScheduler) {
  tensorflow::TfrtCompileOptions default_options;
  EXPECT_FALSE(GetTfrtPipelineOptions(default_options)
                   ->priority_aware_batch_scheduler_use_global_scheduler);

  tensorflow::TfrtCompileOptions options;
  options.priority_aware_batch_scheduler_use_global_scheduler = true;
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_TRUE(
      pipeline_options->priority_aware_batch_scheduler_use_global_scheduler);
}

mlir::OwningOpRef<mlir::ModuleOp> ParseModuleWithBatchFunctionOp(
    mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  constexpr absl::string_view kMlirModule = R"mlir(
    func.func private @batched_function(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
      %0 = "tf.Identity"(%arg0) : (tensor<1x3xf32>) -> tensor<1x3xf32>
      func.return %0 : tensor<1x3xf32>
    }
    func.func @main(%arg0: tensor<1x3xf32>) -> tensor<*xf32> {
      %0 = "tf.BatchFunction"(%arg0) {
        allowed_batch_sizes = [1, 2],
        batch_padding_policy = "PAD_UP",
        batch_timeout_micros = 2 : i64,
        batching_queue = "",
        container = "",
        device = "/device:CPU:0",
        enable_large_batch_splitting = false,
        f = @batched_function,
        max_batch_size = 2 : i64,
        max_enqueued_batches = 2 : i64,
        num_batch_threads = 2 : i64,
        operandSegmentSizes = array<i32: 1, 0>,
        shared_name = "batch/"
      } : (tensor<1x3xf32>) -> tensor<*xf32>
      func.return %0 : tensor<*xf32>
    }
  )mlir";
  return mlir::parseSourceString<mlir::ModuleOp>(kMlirModule, &context);
}

mlir::TF::BatchFunctionOp FindBatchFunctionOp(mlir::ModuleOp module) {
  mlir::TF::BatchFunctionOp result;
  module.walk([&](mlir::TF::BatchFunctionOp op) { result = op; });
  return result;
}

TEST(ReconfigBatchOpPassTest, GlobalSchedulerOptionAloneDoesNotRewriteBatchOp) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      ParseModuleWithBatchFunctionOp(context);
  ASSERT_TRUE(module);

  tfrt_compiler::ReconfigBatchOpPassOptions options;
  options.min_num_batch_threads = 0;
  options.min_max_enqueued_batches = 0;
  options.priority_aware_batch_scheduler_use_global_scheduler = true;

  mlir::PassManager pm(&context);
  pm.addPass(tfrt_compiler::CreateReconfigBatchOpPass(options));
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  mlir::TF::BatchFunctionOp batch_op = FindBatchFunctionOp(*module);
  ASSERT_TRUE(batch_op);
  EXPECT_FALSE(batch_op.getEnablePriorityAwareBatchScheduler());
  EXPECT_EQ(batch_op.getNumBatchThreads(), 2);
}

TEST(ReconfigBatchOpPassTest,
     RewritesBatchFunctionOpWhenGlobalPrioritizationThreadsSetWithZeroMin) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      ParseModuleWithBatchFunctionOp(context);
  ASSERT_TRUE(module);

  tfrt_compiler::ReconfigBatchOpPassOptions options;
  options.min_num_batch_threads = 0;
  options.min_max_enqueued_batches = 0;
  options.batch_queue_global_prioritization_num_threads = 4;

  mlir::PassManager pm(&context);
  pm.addPass(tfrt_compiler::CreateReconfigBatchOpPass(options));
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  mlir::TF::BatchFunctionOp batch_op = FindBatchFunctionOp(*module);
  ASSERT_TRUE(batch_op);
  EXPECT_TRUE(batch_op.getEnablePriorityAwareBatchScheduler());
  EXPECT_EQ(batch_op.getNumBatchThreads(), 4);
  EXPECT_GT(batch_op.getNumWarmupBatchThreads(), 0);
}

TEST(ReconfigBatchOpPassTest, RewritesBatchFunctionOpWhenMinOptionsAreZero) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      ParseModuleWithBatchFunctionOp(context);
  ASSERT_TRUE(module);

  tfrt_compiler::ReconfigBatchOpPassOptions options;
  options.min_num_batch_threads = 0;
  options.min_max_enqueued_batches = 0;
  options.enable_large_batch_splitting = true;
  options.mixed_priority_batching_policy = "priority_merge";

  mlir::PassManager pm(&context);
  pm.addPass(tfrt_compiler::CreateReconfigBatchOpPass(options));
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  mlir::TF::BatchFunctionOp batch_op = FindBatchFunctionOp(*module);
  ASSERT_TRUE(batch_op);
  EXPECT_TRUE(batch_op.getEnableLargeBatchSplitting());
  EXPECT_EQ(batch_op.getMixedPriorityPolicy(), "priority_merge");
}

}  // namespace
}  // namespace tensorflow
