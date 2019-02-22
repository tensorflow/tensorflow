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
#include "tensorflow/core/kernels/data/graph_rewrite_dataset.h"
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
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    std::vector<string> optimizations;
    OP_REQUIRES_OK(
        ctx, ParseVectorArgument<string>(ctx, "optimizations", &optimizations));
    Dataset* dataset =
        new Dataset(ctx, input, optimizations, output_types_, output_shapes_);
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
            const std::vector<string>& optimizations,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : GraphRewriteDataset(ctx, input, output_types, output_shapes),
          optimizations_(optimizations) {}

    string DebugString() const override { return "OptimizeDatasetOp::Dataset"; }

   private:
    RewriterConfig CreateGrapplerRewriteConfig() override {
      RewriterConfig rewriter_config;
      rewriter_config.add_optimizers(kOptimizerName);
      rewriter_config.set_meta_optimizer_iterations(
          RewriterConfig_NumIterationsType_ONE);
      auto custom_optimizer = rewriter_config.add_custom_optimizers();
      custom_optimizer->set_name(kOptimizerName);
      auto* custom_optimizations_list =
          (*custom_optimizer->mutable_parameter_map())["optimizers"]
              .mutable_list();
      for (const auto& opt : optimizations_) {
        custom_optimizations_list->add_s(opt);
      }
      return rewriter_config;
    }

    const std::vector<string> optimizations_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("OptimizeDataset").Device(DEVICE_CPU),
                        OptimizeDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
