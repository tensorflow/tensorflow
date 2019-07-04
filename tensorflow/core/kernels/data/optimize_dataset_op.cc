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
#include <map>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kOptimizerName[] = "tf_data_meta_optimizer";

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.
class OptimizeDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit OptimizeDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("optimization_configs", &optimization_configs_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    std::vector<string> optimizations;
    OP_REQUIRES_OK(
        ctx, ParseVectorArgument<string>(ctx, "optimizations", &optimizations));

    auto config_factory = [this, &optimizations]() {
      return CreateConfig(optimizations, optimization_configs_);
    };
    OP_REQUIRES_OK(ctx,
                   RewriteDataset(ctx, input, std::move(config_factory),
                                  /*optimize_function_library=*/true, output));
  }

 private:
  static RewriterConfig CreateConfig(
      std::vector<string> optimizations,
      std::vector<string> optimizations_configs) {
    RewriterConfig rewriter_config;
    rewriter_config.add_optimizers(kOptimizerName);
    rewriter_config.set_meta_optimizer_iterations(
        RewriterConfig_NumIterationsType_ONE);
    rewriter_config.set_fail_on_optimizer_errors(true);
    auto custom_optimizer = rewriter_config.add_custom_optimizers();
    custom_optimizer->set_name(kOptimizerName);
    auto* custom_optimizations_list =
        (*custom_optimizer->mutable_parameter_map())["optimizers"]
            .mutable_list();
    for (const auto& opt : optimizations) {
      custom_optimizations_list->add_s(opt);
    }
    auto* config_list =
        (*custom_optimizer->mutable_parameter_map())["optimizer_configs"]
            .mutable_list();
    for (const auto& config : optimizations_configs) {
      config_list->add_s(config);
    }
    return rewriter_config;
  }

  std::vector<string> optimization_configs_;
};

REGISTER_KERNEL_BUILDER(Name("OptimizeDataset").Device(DEVICE_CPU),
                        OptimizeDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
