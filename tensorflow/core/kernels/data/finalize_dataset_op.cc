/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/finalize_dataset_op.h"

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/experimental/threadpool_dataset_op.h"
#include "tensorflow/core/kernels/data/model_dataset_op.h"
#include "tensorflow/core/kernels/data/optimize_dataset_op.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const FinalizeDatasetOp::kDatasetType;
/* static */ constexpr const char* const FinalizeDatasetOp::kInputDataset;
/* static */ constexpr const char* const FinalizeDatasetOp::kOutputTypes;
/* static */ constexpr const char* const FinalizeDatasetOp::kOutputShapes;
/* static */ constexpr const char* const FinalizeDatasetOp::kHasCapturedRef;

namespace {

void GetModelDatasetParams(const Options& options,
                           model::AutotuneAlgorithm* algorithm,
                           bool* cpu_budget, bool* ram_budget) {
  *algorithm = model::AutotuneAlgorithm::HILL_CLIMB;
  if (options.optimization_options().autotune_buffers()) {
    *algorithm = model::AutotuneAlgorithm::GRADIENT_DESCENT;
  }
  *cpu_budget = options.optimization_options().autotune_cpu_budget();
  *ram_budget = options.optimization_options().autotune_ram_budget();
}

void MakeDatasetHelper(OpKernelContext* ctx, bool has_captured_ref,
                       DatasetBase* input, DatasetBase** output) {
  *output = input;
  input->Ref();
  const Options& options = input->options();
  if (ShouldConfigureMaxIntraOpParallelism(options)) {
    experimental::MaxIntraOpParallelismDatasetOp::MakeDatasetFromOptions(
        ctx, input, options.threading_options().max_intra_op_parallelism(),
        output);
    input->Unref();
    input = *output;
  }
  if (ShouldUsePrivateThreadPool(options)) {
    experimental::PrivateThreadPoolDatasetOp::MakeDatasetFromOptions(
        ctx, input, options.threading_options().private_threadpool_size(),
        output);
    input->Unref();
    input = *output;
  }
  if (ShouldUseAutotuning(options)) {
    model::AutotuneAlgorithm algorithm;
    bool cpu_budget;
    bool ram_budget;
    GetModelDatasetParams(options, &algorithm, &cpu_budget, &ram_budget);
    ModelDatasetOp::MakeDatasetFromOptions(ctx, input, algorithm, cpu_budget,
                                           ram_budget, output);
    input->Unref();
    input = *output;
  }
  absl::flat_hash_set<tstring> optimizations_enabled;
  absl::flat_hash_set<tstring> optimizations_disabled;
  absl::flat_hash_set<tstring> optimizations_default;
  GetOptimizations(options, &optimizations_enabled, &optimizations_disabled,
                   &optimizations_default);
  if (ShouldApplyOptimizations(options, optimizations_enabled,
                               optimizations_default)) {
    if (has_captured_ref &&
        (!optimizations_enabled.empty() || !optimizations_default.empty())) {
      LOG(WARNING)
          << "tf.data graph rewrites are not compatible with reference "
             "variables. The following rewrites will be disabled: "
          << absl::StrJoin(optimizations_enabled, ", ") << ", "
          << absl::StrJoin(optimizations_default, ", ") << ". "
          << "To enable rewrites, use resource variables instead by calling "
             "`tf.enable_resource_variables()` at the start of the program.";
    } else {
      auto optimization_configs = CreateGraphRewriteConfigs(options);
      OptimizeDatasetOp::MakeDatasetFromOptions(
          ctx, input, optimizations_enabled, optimizations_disabled,
          optimizations_default, optimization_configs, output);
      input->Unref();
      input = *output;
    }
  }
}

}  // namespace

FinalizeDatasetOp::FinalizeDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  if (ctx->HasAttr(kHasCapturedRef)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kHasCapturedRef, &has_captured_ref_));
  } else {
    has_captured_ref_ = false;
  }
}

void FinalizeDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {
  MakeDatasetHelper(ctx, has_captured_ref_, input, output);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("FinalizeDataset").Device(DEVICE_CPU).Priority(2),
                        FinalizeDatasetOp);
REGISTER_KERNEL_BUILDER(Name("FinalizeDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("input_dataset")
                            .HostMemory("handle")
                            .Priority(1),
                        FinalizeDatasetNoopOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
