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
#include "tensorflow/core/kernels/data/graph_rewrite_dataset.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kOptimizerName[] = "tf_auto_shard";

class AutoShardDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit AutoShardDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 index;
    int64 num_workers;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_workers", &num_workers));
    OP_REQUIRES(
        ctx, num_workers > 0,
        errors::InvalidArgument("num_workers must be greater than zero."));

    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "index", &index));
    OP_REQUIRES(ctx, index >= 0 && index < num_workers,
                errors::InvalidArgument("index must be between 0 and ",
                                        num_workers - 1));

    Dataset* dataset = new Dataset(ctx, input, num_workers, index,
                                   output_types_, output_shapes_);
    const Status s = dataset->Optimize(ctx);

    if (s.ok()) {
      *output = dataset;
    } else {
      dataset->Unref();
      OP_REQUIRES_OK(ctx, s);
    }
  }

 private:
  class Dataset : public GraphRewriteDataset {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const int64 num_workers, const int64 index,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : GraphRewriteDataset(ctx, input, output_types, output_shapes),
          num_workers_(num_workers),
          index_(index) {}

    string DebugString() const override {
      return "AutoShardDatasetOp::Dataset";
    }

   private:
    bool ShouldOptimizeFunctions() override {
      // We only want to optimize functions for some particular datasets like
      // FlatMapDataset, InterleaveDataset etc. So we disable generalized
      // function optimization and explicitly handle function modifications
      // for those datasets in the rewrite.
      return false;
    }

    RewriterConfig CreateGrapplerRewriteConfig() override {
      RewriterConfig rewriter_config;
      rewriter_config.set_fail_on_optimizer_errors(true);
      rewriter_config.add_optimizers(kOptimizerName);
      rewriter_config.set_meta_optimizer_iterations(
          RewriterConfig_NumIterationsType_ONE);
      auto custom_optimizer = rewriter_config.add_custom_optimizers();
      custom_optimizer->set_name(kOptimizerName);
      AttrValue num_workers_attr;
      num_workers_attr.set_i(num_workers_);
      (*custom_optimizer->mutable_parameter_map())["num_workers"] =
          num_workers_attr;

      AttrValue index_attr;
      index_attr.set_i(index_);
      (*custom_optimizer->mutable_parameter_map())["index"] = index_attr;

      return rewriter_config;
    }

    const int64 num_workers_;
    const int64 index_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("ExperimentalAutoShardDataset").Device(DEVICE_CPU),
                        AutoShardDatasetOp);

}  // anonymous namespace
}  // namespace data
}  // namespace tensorflow
