/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/experimental/auto_shard_dataset_op.h"

#include "tensorflow/core/kernels/data/rewrite_utils.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr const char* const AutoShardDatasetOp::kAutoShardPolicy;
/* static */ constexpr const char* const AutoShardDatasetOp::kDatasetType;
/* static */ constexpr const char* const AutoShardDatasetOp::kInputDataset;
/* static */ constexpr const char* const AutoShardDatasetOp::kNumWorkers;
/* static */ constexpr const char* const AutoShardDatasetOp::kIndex;
/* static */ constexpr const char* const AutoShardDatasetOp::kOutputTypes;
/* static */ constexpr const char* const AutoShardDatasetOp::kOutputShapes;

constexpr char kOptimizerName[] = "tf_auto_shard";

AutoShardDatasetOp::AutoShardDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx), auto_shard_policy_(0) {
  if (ctx->HasAttr("auto_shard_policy")) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("auto_shard_policy", &auto_shard_policy_));
  }
}

void AutoShardDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                     DatasetBase** output) {
  int64 index, num_workers, auto_shard_policy;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kNumWorkers, &num_workers));
  OP_REQUIRES(
      ctx, num_workers > 0,
      errors::InvalidArgument("num_workers must be greater than zero."));

  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kIndex, &index));
  OP_REQUIRES(
      ctx, index >= 0 && index < num_workers,
      errors::InvalidArgument("index must be between 0 and ", num_workers - 1));
  auto_shard_policy = auto_shard_policy_;

  auto config_factory = [num_workers, index, auto_shard_policy]() {
    return CreateConfig(num_workers, index, auto_shard_policy);
  };

  // We only want to optimize functions for some particular datasets like
  // FlatMapDataset, InterleaveDataset etc. So we disable generalized
  // function optimization and explicitly handle function modifications
  // for those datasets in the rewrite.
  OP_REQUIRES_OK(ctx, RewriteDataset(ctx, input, std::move(config_factory),
                                     /*optimize_function_library=*/false,
                                     /*record_fingerprint=*/false, output));
}

RewriterConfig AutoShardDatasetOp::CreateConfig(int64 num_workers, int64 index,
                                                int64 auto_shard_policy) {
  RewriterConfig rewriter_config;
  rewriter_config.set_fail_on_optimizer_errors(true);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);

  rewriter_config.add_optimizers(kOptimizerName);
  auto custom_optimizer = rewriter_config.add_custom_optimizers();
  custom_optimizer->set_name(kOptimizerName);
  AttrValue num_workers_attr;
  num_workers_attr.set_i(num_workers);
  (*custom_optimizer->mutable_parameter_map())[kNumWorkers] = num_workers_attr;
  AttrValue index_attr;
  index_attr.set_i(index);
  (*custom_optimizer->mutable_parameter_map())[kIndex] = index_attr;
  AttrValue auto_shard_policy_attr;
  auto_shard_policy_attr.set_i(auto_shard_policy);
  (*custom_optimizer->mutable_parameter_map())[kAutoShardPolicy] =
      auto_shard_policy_attr;

  return rewriter_config;
}

namespace {
REGISTER_KERNEL_BUILDER(Name("AutoShardDataset").Device(DEVICE_CPU),
                        AutoShardDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalAutoShardDataset").Device(DEVICE_CPU),
                        AutoShardDatasetOp);
}  // anonymous namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
