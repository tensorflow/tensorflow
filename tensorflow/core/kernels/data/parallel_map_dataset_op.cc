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
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/kernels/data/parallel_map_iterator.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class ParallelMapDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ParallelMapDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("other_arguments", &inputs));
    std::vector<Tensor> other_arguments;
    other_arguments.reserve(inputs.size());
    for (const Tensor& t : inputs) {
      other_arguments.push_back(t);
    }

    int32 num_parallel_calls;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "num_parallel_calls",
                                            &num_parallel_calls));
    OP_REQUIRES(ctx, num_parallel_calls > 0,
                errors::InvalidArgument(
                    "num_parallel_calls must be greater than zero."));

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(
                            func_, std::move(other_arguments), &captured_func));

    *output = new Dataset(ctx, input, func_, num_parallel_calls, output_types_,
                          output_shapes_, std::move(captured_func));
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const NameAttrList& func, int32 num_parallel_calls,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            std::unique_ptr<CapturedFunction> captured_func)
        : GraphDatasetBase(ctx),
          input_(input),
          func_(func),
          num_parallel_calls_(num_parallel_calls),
          output_types_(output_types),
          output_shapes_(output_shapes),
          captured_func_(std::move(captured_func)) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      auto map_func = [this](IteratorContext* ctx,
                             std::vector<Tensor> input_element,
                             std::vector<Tensor>* result, StatusCallback done) {
        captured_func_->RunAsync(ctx, std::move(input_element), result,
                                 std::move(done));
      };

      return NewParallelMapIterator(
          {this, strings::StrCat(prefix, "::ParallelMap")}, input_,
          std::move(map_func), num_parallel_calls_);
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

   protected:
    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      // Input: input_dataset
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input_, &input_graph_node));

      // Input: other_arguments
      DataTypeVector other_arguments_types;
      other_arguments_types.reserve(captured_func_->captured_inputs().size());
      std::vector<Node*> other_arguments;
      other_arguments.reserve(captured_func_->captured_inputs().size());
      for (const Tensor& t : captured_func_->captured_inputs()) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
        other_arguments.emplace_back(node);
        other_arguments_types.emplace_back(t.dtype());
      }

      // Input: num_parallel_calls
      Node* num_parallel_calls = nullptr;
      TF_RETURN_IF_ERROR(
          b->AddScalar(num_parallel_calls_, &num_parallel_calls));

      // Attr: f
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, func_.name()));
      AttrValue f;
      b->BuildAttrValue(func_, &f);

      // Attr: Targuments
      AttrValue other_arguments_types_attr;
      b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this,
          {std::make_pair(0, input_graph_node),
           std::make_pair(2, num_parallel_calls)},  // Single tensor inputs.
          {std::make_pair(1, other_arguments)},     // Tensor list inputs.
          {std::make_pair("f", f),
           std::make_pair("Targuments", other_arguments_types_attr)},  // Attrs
          output));
      return Status::OK();
    }

   private:
    const DatasetBase* const input_;
    const NameAttrList func_;
    const int32 num_parallel_calls_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const std::unique_ptr<CapturedFunction> captured_func_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList func_;
};

REGISTER_KERNEL_BUILDER(Name("ParallelMapDataset").Device(DEVICE_CPU),
                        ParallelMapDatasetOp);

}  // namespace

}  // namespace tensorflow
