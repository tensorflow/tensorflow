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

// On mobile we do not provide optimize dataset op because not all of its
// dependencies are available there. The op is replaced with a no-op.
#if !defined(IS_MOBILE_PLATFORM)
#include <map>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/rewrite_utils.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const OptimizeDatasetOp::kDatasetType;
/* static */ constexpr const char* const OptimizeDatasetOp::kInputDataset;
/* static */ constexpr const char* const OptimizeDatasetOp::kOptimizations;
/* static */ constexpr const char* const
    OptimizeDatasetOp::kOptimizationsEnabled;
/* static */ constexpr const char* const
    OptimizeDatasetOp::kOptimizationsDisabled;
/* static */ constexpr const char* const
    OptimizeDatasetOp::kOptimizationsDefault;
/* static */ constexpr const char* const OptimizeDatasetOp::kOutputTypes;
/* static */ constexpr const char* const OptimizeDatasetOp::kOutputShapes;
/* static */ constexpr const char* const
    OptimizeDatasetOp::kOptimizationConfigs;
/* static */ constexpr const char* const OptimizeDatasetOp::kOptimizeDatasetV1;
/* static */ constexpr const char* const OptimizeDatasetOp::kOptimizeDatasetV2;

namespace {

// Applies given optimizations and optimizatin_config in dataset graph rewrite
// to return the OptimizeDataset.
void MakeDatasetHelper(OpKernelContext* ctx,
                       std::vector<tstring>& optimizations,
                       const std::vector<string>& optimization_configs,
                       DatasetBase* input, DatasetBase** output) {
  // The vector stores the graduated experiment names which will be turned on
  // for all input pipelines.
  // clang-format off
  std::vector<string> graduated_experiments = {
    "disable_intra_op_parallelism",
    "use_private_thread_pool"
  };
  // clang-format on

  // Add the graduated experiments to the optimization list and log them.
  for (auto& experiment : graduated_experiments) {
    if (std::find(optimizations.begin(), optimizations.end(), experiment) ==
        optimizations.end()) {
      optimizations.push_back(experiment);
    }
    VLOG(1) << "The graduated experiment \"" << experiment << "\" is applied.";
  }

  // If there are no optimizations to be applied, directly return the input.
  if (optimizations.empty()) {
    *output = input;
    input->Ref();
    return;
  }

  auto config_factory = [&optimizations, &optimization_configs]() {
    return CreateRewriterConfig(optimizations, optimization_configs);
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

}  // namespace

// static
void OptimizeDatasetOp::MakeDatasetFromOptions(
    OpKernelContext* ctx, DatasetBase* input,
    const std::vector<tstring>& optimizations_enabled,
    const std::vector<tstring>& optimizations_disabled,
    const std::vector<tstring>& optimizations_default,
    const std::vector<string>& optimization_configs, DatasetBase** output) {
  auto experiments = GetExperiments();
  LogAndRecordExperiments(experiments);
  auto optimizations =
      SelectOptimizations(experiments, optimizations_enabled,
                          optimizations_disabled, optimizations_default);
  MakeDatasetHelper(ctx, optimizations, optimization_configs, input, output);
}

OptimizeDatasetOp::OptimizeDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  auto& op_name = ctx->def().op();
  if (op_name == kOptimizeDatasetV1) {
    op_version_ = 1;
  } else if (op_name == kOptimizeDatasetV2) {
    op_version_ = 2;
  }
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(kOptimizationConfigs, &optimization_configs_));
}

void OptimizeDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {
  std::vector<tstring> optimizations;
  if (op_version_ == 1) {
    OP_REQUIRES_OK(
        ctx, ParseVectorArgument<tstring>(ctx, kOptimizations, &optimizations));
  } else if (op_version_ == 2) {
    std::vector<tstring> optimizations_enabled, optimizations_disabled,
        optimizations_default;
    OP_REQUIRES_OK(ctx, ParseVectorArgument<tstring>(ctx, kOptimizationsEnabled,
                                                     &optimizations_enabled));
    OP_REQUIRES_OK(ctx,
                   ParseVectorArgument<tstring>(ctx, kOptimizationsDisabled,
                                                &optimizations_disabled));
    OP_REQUIRES_OK(ctx, ParseVectorArgument<tstring>(ctx, kOptimizationsDefault,
                                                     &optimizations_default));
    auto experiments = GetExperiments();
    LogAndRecordExperiments(experiments);
    optimizations =
        SelectOptimizations(experiments, optimizations_enabled,
                            optimizations_disabled, optimizations_default);
  }
  MakeDatasetHelper(ctx, optimizations, optimization_configs_, input, output);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("OptimizeDataset").Device(DEVICE_CPU),
                        OptimizeDatasetOp);
REGISTER_KERNEL_BUILDER(Name("OptimizeDatasetV2").Device(DEVICE_CPU),
                        OptimizeDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
#else   // !IS_MOBILE_PLATFORM
namespace tensorflow {
namespace data {

// static
void OptimizeDatasetOp::MakeDatasetFromOptions(
    OpKernelContext* ctx, DatasetBase* input,
    const std::vector<tstring>& optimizations_enabled,
    const std::vector<tstring>& optimizations_disabled,
    const std::vector<tstring>& optimizations_default,
    const std::vector<string>& optimization_configs, DatasetBase** output) {
  input->Ref();
  *output = input;
}

OptimizeDatasetOp::OptimizeDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}

void OptimizeDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {
  input->Ref();
  *output = input;
}

namespace {
REGISTER_KERNEL_BUILDER(Name("OptimizeDataset").Device(DEVICE_CPU),
                        OptimizeDatasetOp);
REGISTER_KERNEL_BUILDER(Name("OptimizeDatasetV2").Device(DEVICE_CPU),
                        OptimizeDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
#endif  // !IS_MOBILE_PLATFORM
