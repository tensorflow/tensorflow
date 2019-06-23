/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/parallel_map_dataset_op.h"

#include <deque>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const ParallelMapDatasetOp::kDatasetType;
/* static */ constexpr const char* const ParallelMapDatasetOp::kInputDataset;
/* static */ constexpr const char* const ParallelMapDatasetOp::kOtherArguments;
/* static */ constexpr const char* const
    ParallelMapDatasetOp::kNumParallelCalls;
/* static */ constexpr const char* const ParallelMapDatasetOp::kFunc;
/* static */ constexpr const char* const ParallelMapDatasetOp::kTarguments;
/* static */ constexpr const char* const ParallelMapDatasetOp::kOutputTypes;
/* static */ constexpr const char* const ParallelMapDatasetOp::kOutputShapes;
/* static */ constexpr const char* const
    ParallelMapDatasetOp::kUseInterOpParallelism;
/* static */ constexpr const char* const ParallelMapDatasetOp::kSloppy;
/* static */ constexpr const char* const
    ParallelMapDatasetOp::kPreserveCardinality;

class ParallelMapDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          int32 num_parallel_calls, const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes, bool sloppy,
          std::unique_ptr<CapturedFunction> captured_func,
          bool preserve_cardinality)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        num_parallel_calls_(num_parallel_calls),
        output_types_(output_types),
        output_shapes_(output_shapes),
        sloppy_(sloppy),
        preserve_cardinality_(preserve_cardinality),
        captured_func_(std::move(captured_func)) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    std::unique_ptr<ParallelMapFunctor> parallel_map_functor =
        absl::make_unique<ParallelMapDatasetFunctor>(this);
    return NewParallelMapIterator(
        {this, name_utils::IteratorPrefix(kDatasetType, prefix)}, input_,
        std::move(parallel_map_functor), num_parallel_calls_, sloppy_,
        preserve_cardinality_);
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(ParallelMapDatasetOp::kDatasetType);
  }

  int64 Cardinality() const override { return input_->Cardinality(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    // Input: input_dataset
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

    // Input: other_arguments
    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));

    // Input: num_parallel_calls
    Node* num_parallel_calls = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(num_parallel_calls_, &num_parallel_calls));

    // Attr: f
    AttrValue f_attr;
    b->BuildAttrValue(captured_func_->func(), &f_attr);

    // Attr: Targuments
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

    // Attr: use_inter_op_parallelism
    AttrValue use_inter_op_parallelism_attr;
    b->BuildAttrValue(captured_func_->use_inter_op_parallelism(),
                      &use_inter_op_parallelism_attr);

    // Attr: sloppy
    AttrValue sloppy_attr;
    b->BuildAttrValue(sloppy_, &sloppy_attr);

    // Attr: preserve_cardinality
    AttrValue preserve_cardinality_attr;
    b->BuildAttrValue(preserve_cardinality_, &preserve_cardinality_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        {std::make_pair(0, input_graph_node),
         std::make_pair(2, num_parallel_calls)},  // Single tensor inputs.
        {std::make_pair(1, other_arguments)},     // Tensor list inputs.
        {std::make_pair(kFunc, f_attr),
         std::make_pair(kTarguments, other_arguments_types_attr),
         std::make_pair(kUseInterOpParallelism, use_inter_op_parallelism_attr),
         std::make_pair(kSloppy, sloppy_attr),
         std::make_pair(kPreserveCardinality,
                        preserve_cardinality_attr)},  // Attrs
        output));
    return Status::OK();
  }

 private:
  class ParallelMapDatasetFunctor : public ParallelMapFunctor {
   public:
    explicit ParallelMapDatasetFunctor(const Dataset* dataset)
        : dataset_(dataset) {}

    Status InitFunc(IteratorContext* ctx) override {
      return dataset_->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_);
    }

    void MapFunc(IteratorContext* ctx, const string& prefix,
                 std::vector<Tensor> input_element, std::vector<Tensor>* result,
                 StatusCallback done) override {
      auto map_func = [this](IteratorContext* ctx, const string& prefix,
                             std::vector<Tensor> input_element,
                             std::vector<Tensor>* result, StatusCallback done) {
        instantiated_captured_func_->RunAsync(ctx, std::move(input_element),
                                              result, std::move(done), prefix);
      };
      if (!dataset_->captured_func_->use_inter_op_parallelism()) {
        (*ctx->runner())(std::bind(map_func, ctx, prefix,
                                   std::move(input_element), result,
                                   std::move(done)));
      } else {
        map_func(ctx, prefix, std::move(input_element), result,
                 std::move(done));
      }
    }

   private:
    const Dataset* const dataset_;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
  };

  const DatasetBase* const input_;
  const int32 num_parallel_calls_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  const bool sloppy_;
  const bool preserve_cardinality_;
  const std::unique_ptr<CapturedFunction> captured_func_;
};

ParallelMapDatasetOp::ParallelMapDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  FunctionMetadata::Params params;
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kUseInterOpParallelism,
                                   &params.use_inter_op_parallelism));
  params.is_multi_device_function = true;
  OP_REQUIRES_OK(ctx,
                 FunctionMetadata::Create(ctx, kFunc, params, &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kSloppy, &sloppy_));
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(kPreserveCardinality, &preserve_cardinality_));
}

void ParallelMapDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                       DatasetBase** output) {
  int32 num_parallel_calls;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument(ctx, kNumParallelCalls, &num_parallel_calls));
  OP_REQUIRES(
      ctx, num_parallel_calls > 0 || num_parallel_calls == model::kAutoTune,
      errors::InvalidArgument("num_parallel_calls must be greater than zero."));

  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));

  if (num_parallel_calls == model::kAutoTune) {
    metrics::RecordTFDataAutotune(kDatasetType);
  }

  *output =
      new Dataset(ctx, input, num_parallel_calls, output_types_, output_shapes_,
                  sloppy_, std::move(captured_func), preserve_cardinality_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ParallelMapDataset").Device(DEVICE_CPU),
                        ParallelMapDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("ParallelMapDataset");
}  // namespace
}  // namespace data
}  // namespace tensorflow
