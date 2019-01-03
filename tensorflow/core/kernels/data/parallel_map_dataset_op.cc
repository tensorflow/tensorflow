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
#include <deque>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/parallel_map_iterator.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

class ParallelMapDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ParallelMapDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_inter_op_parallelism",
                                     &use_inter_op_parallelism_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sloppy", &sloppy_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("preserve_cardinality", &preserve_cardinality_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int32 num_parallel_calls;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_parallel_calls",
                                            &num_parallel_calls));
    OP_REQUIRES(
        ctx, num_parallel_calls > 0 || num_parallel_calls == model::kAutoTune,
        errors::InvalidArgument(
            "num_parallel_calls must be greater than zero."));

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(func_, ctx, "other_arguments",
                                                 use_inter_op_parallelism_,
                                                 &captured_func));

    std::vector<int> indices;
    OP_REQUIRES_OK(ctx, ComputeShortCircuitIndices(ctx, func_, &indices));

    *output =
        new Dataset(ctx, input, func_, num_parallel_calls, output_types_,
                    output_shapes_, use_inter_op_parallelism_, sloppy_,
                    std::move(captured_func), indices, preserve_cardinality_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const NameAttrList& func, int32 num_parallel_calls,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            bool use_inter_op_parallelism, bool sloppy,
            std::unique_ptr<CapturedFunction> captured_func,
            const std::vector<int> indices, bool preserve_cardinality)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          func_(func),
          num_parallel_calls_(num_parallel_calls),
          output_types_(output_types),
          output_shapes_(output_shapes),
          use_inter_op_parallelism_(use_inter_op_parallelism),
          sloppy_(sloppy),
          preserve_cardinality_(preserve_cardinality),
          captured_func_(std::move(captured_func)),
          indices_(indices),
          can_move_(indices.empty() ? std::vector<bool>()
                                    : ComputeMoveVector(indices)) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      std::unique_ptr<ParallelMapFunctor> parallel_map_functor(nullptr);
      if (indices_.empty()) {
        parallel_map_functor.reset(new ParallelMapDatasetFunctor(this));
      } else {
        parallel_map_functor.reset(new ShortCircuitFunctor(this));
      }
      return NewParallelMapIterator(
          {this, strings::StrCat(prefix, "::ParallelMap")}, input_,
          std::move(parallel_map_functor), num_parallel_calls_, sloppy_,
          preserve_cardinality_);
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "ParallelMapDatasetOp::Dataset";
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
      DataTypeVector other_arguments_types;
      other_arguments_types.reserve(captured_func_->captured_inputs().size());
      std::vector<Node*> other_arguments;
      other_arguments.reserve(captured_func_->captured_inputs().size());
      for (const Tensor& t : captured_func_->captured_inputs()) {
        Node* node;
        DatasetBase* input;
        Status s = GetDatasetFromVariantTensor(t, &input);
        if (s.ok()) {
          TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input, &node));
        } else {
          TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
        }
        other_arguments.emplace_back(node);
        other_arguments_types.emplace_back(t.dtype());
      }

      // Input: num_parallel_calls
      Node* num_parallel_calls = nullptr;
      TF_RETURN_IF_ERROR(
          b->AddScalar(num_parallel_calls_, &num_parallel_calls));

      // Attr: f
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, func_.name()));
      AttrValue f_attr;
      b->BuildAttrValue(func_, &f_attr);

      // Attr: Targuments
      AttrValue other_arguments_types_attr;
      b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

      // Attr: use_inter_op_parallelism
      AttrValue use_inter_op_parallelism_attr;
      b->BuildAttrValue(use_inter_op_parallelism_,
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
          {std::make_pair("f", f_attr),
           std::make_pair("Targuments", other_arguments_types_attr),
           std::make_pair("use_inter_op_parallelism",
                          use_inter_op_parallelism_attr),
           std::make_pair("sloppy", sloppy_attr),
           std::make_pair("preserve_cardinality",
                          preserve_cardinality_attr)},  // Attrs
          output));
      return Status::OK();
    }

   private:
    class ShortCircuitFunctor : public ParallelMapFunctor {
     public:
      explicit ShortCircuitFunctor(const Dataset* dataset)
          : dataset_(dataset) {}

      void MapFunc(IteratorContext* ctx, const string& prefix,
                   std::vector<Tensor> input_element,
                   std::vector<Tensor>* result, StatusCallback done) override {
        const std::vector<Tensor>& captured_inputs =
            dataset_->captured_func_->captured_inputs();
        size_t num_args = input_element.size();
        for (size_t i = 0; i < dataset_->indices_.size(); ++i) {
          if (dataset_->indices_[i] < num_args) {
            if (dataset_->can_move_[i]) {
              result->push_back(
                  std::move(input_element[dataset_->indices_[i]]));
            } else {
              result->push_back(input_element[dataset_->indices_[i]]);
            }
          } else {
            result->push_back(
                captured_inputs[dataset_->indices_[i] - num_args]);
          }
        }
        done(Status::OK());
      }

      const Dataset* const dataset_;
    };

    class ParallelMapDatasetFunctor : public ParallelMapFunctor {
     public:
      explicit ParallelMapDatasetFunctor(const Dataset* dataset)
          : dataset_(dataset) {}

      Status InitFunc(IteratorContext* ctx) override {
        return dataset_->captured_func_->Instantiate(
            ctx, &instantiated_captured_func_);
      }

      void MapFunc(IteratorContext* ctx, const string& prefix,
                   std::vector<Tensor> input_element,
                   std::vector<Tensor>* result, StatusCallback done) override {
        auto map_func = [this](IteratorContext* ctx, const string& prefix,
                               std::vector<Tensor> input_element,
                               std::vector<Tensor>* result,
                               StatusCallback done) {
          instantiated_captured_func_->RunAsync(
              ctx, std::move(input_element), result, std::move(done), prefix);
        };
        if (!dataset_->use_inter_op_parallelism_) {
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
    const NameAttrList func_;
    const int32 num_parallel_calls_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const bool use_inter_op_parallelism_;
    const bool sloppy_;
    const bool preserve_cardinality_;
    const std::unique_ptr<CapturedFunction> captured_func_;
    const std::vector<int> indices_;
    const std::vector<bool> can_move_;
  };

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  bool use_inter_op_parallelism_;
  bool sloppy_;
  bool preserve_cardinality_;
  NameAttrList func_;
};

REGISTER_KERNEL_BUILDER(Name("ParallelMapDataset").Device(DEVICE_CPU),
                        ParallelMapDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
