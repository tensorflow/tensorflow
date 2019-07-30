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
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kOptimizerName[] = "tf_data_rebatcher";
constexpr char kUseFallbackAttr[] = "use_fallback";

class RebatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit RebatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    if (ctx->HasAttr(kUseFallbackAttr)) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr(kUseFallbackAttr, &use_fallback_));
    }
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 num_workers;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_workers", &num_workers));
    OP_REQUIRES(
        ctx, num_workers > 0,
        errors::InvalidArgument("num_workers must be greater than zero."));

    auto config_factory = [num_workers, this]() {
      return CreateConfig(num_workers, this->use_fallback_);
    };

    // We only want to optimize functions for some particular datasets like
    // FlatMapDataset, InterleaveDataset etc. So we disable generalized
    // function optimization and explicitly handle function modifications
    // for those datasets in the rewrite.
    OP_REQUIRES_OK(ctx,
                   RewriteDataset(ctx, input, std::move(config_factory),
                                  /*optimize_function_library=*/false, output));
  }

 private:
  static RewriterConfig CreateConfig(int64 num_workers, bool use_fallback) {
    RewriterConfig rewriter_config;
    rewriter_config.set_fail_on_optimizer_errors(true);
    rewriter_config.add_optimizers(kOptimizerName);
    rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);
    auto custom_optimizer = rewriter_config.add_custom_optimizers();
    custom_optimizer->set_name(kOptimizerName);
    AttrValue num_workers_attr;
    num_workers_attr.set_i(num_workers);
    (*custom_optimizer->mutable_parameter_map())["num_workers"] =
        num_workers_attr;
    AttrValue use_fallback_attr;
    use_fallback_attr.set_b(use_fallback);
    (*custom_optimizer->mutable_parameter_map())["use_fallback"] =
        use_fallback_attr;
    return rewriter_config;
  }

  bool use_fallback_ = true;
};

REGISTER_KERNEL_BUILDER(Name("RebatchDataset").Device(DEVICE_CPU),
                        RebatchDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalRebatchDataset").Device(DEVICE_CPU),
                        RebatchDatasetOp);

}  // anonymous namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
