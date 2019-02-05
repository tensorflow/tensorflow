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

constexpr char kOptimizerName[] = "tf_data_rebatcher";

class RebatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit RebatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 num_workers;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_workers", &num_workers));
    OP_REQUIRES(ctx, num_workers > 0,
                errors::InvalidArgument(
                    "num_parallel_calls must be greater than zero."));

    Dataset* dataset =
        new Dataset(ctx, input, num_workers, output_types_, output_shapes_);
    Status s = dataset->Optimize(ctx);
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
            const int64 num_workers, const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : GraphRewriteDataset(ctx, input, output_types, output_shapes),
          num_workers_(num_workers) {}

    string DebugString() const override { return "RebatchDatasetOp::Dataset"; }

   private:
    RewriterConfig CreateGrapplerRewriteConfig() override {
      RewriterConfig rewriter_config;
      rewriter_config.add_optimizers(kOptimizerName);
      rewriter_config.set_meta_optimizer_iterations(
          RewriterConfig_NumIterationsType_ONE);
      auto custom_optimizer = rewriter_config.add_custom_optimizers();
      custom_optimizer->set_name(kOptimizerName);
      AttrValue num_workers_attr;
      num_workers_attr.set_i(num_workers_);
      (*custom_optimizer->mutable_parameter_map())["num_workers"] =
          num_workers_attr;
      return rewriter_config;
    }

    const int64 num_workers_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("ExperimentalRebatchDataset").Device(DEVICE_CPU),
                        RebatchDatasetOp);

}  // anonymous namespace
}  // namespace data
}  // namespace tensorflow
