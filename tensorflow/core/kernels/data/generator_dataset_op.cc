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
#include <iterator>
#include <vector>

#include "tensorflow/core/kernels/data/generator_dataset_op.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class GeneratorDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, std::unique_ptr<CapturedFunction> init_func,
          std::unique_ptr<CapturedFunction> next_func,
          std::unique_ptr<CapturedFunction> finalize_func,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        init_func_(std::move(init_func)),
        next_func_(std::move(next_func)),
        finalize_func_(std::move(finalize_func)),
        output_types_(output_types),
        output_shapes_(output_shapes) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::unique_ptr<IteratorBase>(
        new Iterator({this, strings::StrCat(prefix, "::Generator")}));
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override { return "GeneratorDatasetOp::Dataset"; }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    return errors::Unimplemented("%s does not support serialization",
                                 DebugString());
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    ~Iterator() override {
      if (!finalized_) {
        std::vector<Tensor> ignored;
        Status s = dataset()->finalize_func_->RunInstantiated(state_, &ignored);
        if (!s.ok()) {
          LOG(WARNING)
              << "Error occurred when finalizing GeneratorDataset iterator: "
              << s;
        }
      }
    }

    Status Initialize(IteratorContext* ctx) override {
      TF_RETURN_IF_ERROR(dataset()->init_func_->Instantiate(ctx));
      TF_RETURN_IF_ERROR(dataset()->next_func_->Instantiate(ctx));
      TF_RETURN_IF_ERROR(dataset()->finalize_func_->Instantiate(ctx));
      TF_RETURN_IF_ERROR(
          dataset()->init_func_->RunWithBorrowedArgs(ctx, {}, &state_));
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);

      if (finalized_) {
        *end_of_sequence = true;
        return Status::OK();
      }

      Status s =
          dataset()->next_func_->RunWithBorrowedArgs(ctx, state_, out_tensors);
      if (s.ok()) {
        *end_of_sequence = false;
      } else if (errors::IsOutOfRange(s)) {
        // `next_func` may deliberately raise `errors::OutOfRange`
        // to indicate that we should terminate the iteration.
        s = Status::OK();
        *end_of_sequence = true;

        // NOTE(mrry): We ignore any tensors returned by the
        // finalize function.
        std::vector<Tensor> ignored;
        TF_RETURN_IF_ERROR(
            dataset()->finalize_func_->RunInstantiated(state_, &ignored));
        finalized_ = true;
      }
      return s;
    }

   private:
    mutex mu_;
    bool finalized_ GUARDED_BY(mu_) = false;
    std::vector<Tensor> state_ GUARDED_BY(mu_);
  };

  const std::unique_ptr<CapturedFunction> init_func_;
  const std::unique_ptr<CapturedFunction> next_func_;
  const std::unique_ptr<CapturedFunction> finalize_func_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

GeneratorDatasetOp::GeneratorDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("init_func", &init_func_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("next_func", &next_func_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("finalize_func", &finalize_func_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
}

void GeneratorDatasetOp::MakeDataset(OpKernelContext* ctx,
                                     DatasetBase** output) {
  OpInputList init_func_other_args_input;
  OP_REQUIRES_OK(ctx, ctx->input_list("init_func_other_args",
                                      &init_func_other_args_input));
  std::vector<Tensor> init_func_other_args;
  init_func_other_args.reserve(init_func_other_args_input.size());
  for (const Tensor& t : init_func_other_args_input) {
    init_func_other_args.push_back(t);
  }
  std::unique_ptr<CapturedFunction> init_func;
  OP_REQUIRES_OK(
      ctx, CapturedFunction::Create(init_func_, std::move(init_func_other_args),
                                    &init_func));

  OpInputList next_func_other_args_input;
  OP_REQUIRES_OK(ctx, ctx->input_list("next_func_other_args",
                                      &next_func_other_args_input));
  std::vector<Tensor> next_func_other_args;
  next_func_other_args.reserve(next_func_other_args_input.size());
  for (const Tensor& t : next_func_other_args_input) {
    next_func_other_args.push_back(t);
  }
  std::unique_ptr<CapturedFunction> next_func;
  OP_REQUIRES_OK(
      ctx, CapturedFunction::Create(next_func_, std::move(next_func_other_args),
                                    &next_func));

  OpInputList finalize_func_other_args_input;
  OP_REQUIRES_OK(ctx, ctx->input_list("finalize_func_other_args",
                                      &finalize_func_other_args_input));
  std::vector<Tensor> finalize_func_other_args;
  finalize_func_other_args.reserve(finalize_func_other_args_input.size());
  for (const Tensor& t : finalize_func_other_args_input) {
    finalize_func_other_args.push_back(t);
  }
  std::unique_ptr<CapturedFunction> finalize_func;
  OP_REQUIRES_OK(ctx, CapturedFunction::Create(
                          finalize_func_, std::move(finalize_func_other_args),
                          &finalize_func));

  *output =
      new Dataset(ctx, std::move(init_func), std::move(next_func),
                  std::move(finalize_func), output_types_, output_shapes_);
}

REGISTER_KERNEL_BUILDER(Name("GeneratorDataset").Device(DEVICE_CPU),
                        GeneratorDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("GeneratorDataset").Device(DEVICE_GPU).HostMemory("handle"),
    GeneratorDatasetOp);

}  // namespace tensorflow
