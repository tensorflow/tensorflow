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

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/rewrite_utils.h"
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

constexpr char kOptimizerName[] = "tf_data_meta_optimizer";
constexpr char kOptimizers[] = "optimizers";
constexpr char kOptimizerConfigs[] = "optimizer_configs";

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

    string job_name = port::JobName();
    // The map that stores the live experiment names and for how much percentage
    // of the Borg jobs, the experiments will be randomly turned on.
    // clang-format off
    absl::flat_hash_map<string, uint64> live_experiments = {
        {"enable_gradient_descent", 0},
        {"map_parallelization", 100}
    };
    // clang-format on
    auto hash_func = [](const string& str) { return Hash64(str); };
    optimizations = SelectOptimizations(
        job_name, live_experiments, optimizations_enabled,
        optimizations_disabled, optimizations_default, hash_func);

    // Log and record the live experiments that will be applied.
    if (!job_name.empty() && !live_experiments.empty()) {
      VLOG(1) << "The input pipeline is subject to tf.data experiment. "
                 "Please see `go/tf-data-experiments` for more details.";

      for (auto& pair : live_experiments) {
        string experiment = pair.first;
        if (std::find(optimizations.begin(), optimizations.end(), experiment) !=
            optimizations.end()) {
          VLOG(1) << "The live experiment \"" << experiment << "\" is applied.";
          metrics::RecordTFDataExperiment(experiment);
        }
      }
    }
  }

  // The vector stores the graduated experiment names which will be turned on
  // for all input pipelines.
  // clang-format off
  std::vector<string> graduated_experiments = {"disable_intra_op_parallelism"};
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
REGISTER_KERNEL_BUILDER(Name("OptimizeDatasetV2").Device(DEVICE_CPU),
                        OptimizeDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
#else  // !IS_MOBILE_PLATFORM
namespace tensorflow {
namespace data {

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
