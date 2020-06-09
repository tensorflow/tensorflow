/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/optimize_dataset_op.h"

#include <map>

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/rewrite_utils.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const OptimizeDatasetOp::kDatasetType;
/* static */ constexpr const char* const OptimizeDatasetOp::kInputDataset;
/* static */ constexpr const char* const OptimizeDatasetOp::kOptimizations;
/* static */ constexpr const char* const OptimizeDatasetOp::kOutputTypes;
/* static */ constexpr const char* const OptimizeDatasetOp::kOutputShapes;
/* static */ constexpr const char* const
    OptimizeDatasetOp::kOptimizationConfigs;

constexpr char kOptimizerName[] = "tf_data_meta_optimizer";
constexpr char kOptimizers[] = "optimizers";
constexpr char kOptimizerConfigs[] = "optimizer_configs";

OptimizeDatasetOp::OptimizeDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(kOptimizationConfigs, &optimization_configs_));
}

void OptimizeDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {
  std::vector<tstring> optimizations;
  OP_REQUIRES_OK(
      ctx, ParseVectorArgument<tstring>(ctx, kOptimizations, &optimizations));

  auto config_factory = [this, &optimizations]() {
    return CreateConfig(optimizations, optimization_configs_);
  };
  Status s = RewriteDataset(ctx, input, std::move(config_factory),
                            /*record_fingerprint=*/true, output);
  if (errors::IsDeadlineExceeded(s)) {
    // Ignore DeadlineExceeded as it implies that the attempted rewrite took too
    // long which should not prevent further computation.
    LOG(WARNING) << s.ToString();

    *output = input;
    input->Ref();
    return;
  }
  OP_REQUIRES_OK(ctx, s);
}

RewriterConfig OptimizeDatasetOp::CreateConfig(
    std::vector<tstring> optimizations,
    std::vector<string> optimizations_configs) {
  RewriterConfig rewriter_config;
  rewriter_config.add_optimizers(kOptimizerName);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);
  rewriter_config.set_fail_on_optimizer_errors(true);
  auto custom_optimizer = rewriter_config.add_custom_optimizers();
  custom_optimizer->set_name(kOptimizerName);
  auto* custom_optimizations_list =
      (*custom_optimizer->mutable_parameter_map())[kOptimizers].mutable_list();
  for (const auto& opt : optimizations) {
    custom_optimizations_list->add_s(opt.data(), opt.size());
  }
  auto* config_list =
      (*custom_optimizer->mutable_parameter_map())[kOptimizerConfigs]
          .mutable_list();
  for (const auto& config : optimizations_configs) {
    config_list->add_s(config.data(), config.size());
  }
  return rewriter_config;
}

namespace {
REGISTER_KERNEL_BUILDER(Name("OptimizeDataset").Device(DEVICE_CPU),
                        OptimizeDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
