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

}  // namespace
}  // namespace tensorflow
