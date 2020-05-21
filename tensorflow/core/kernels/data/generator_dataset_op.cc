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
#include "tensorflow/core/kernels/data/generator_dataset_op.h"

#include <iterator>
#include <vector>

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const GeneratorDatasetOp::kDatasetType;
/* static */ constexpr const char* const GeneratorDatasetOp::kInitFuncOtherArgs;
/* static */ constexpr const char* const GeneratorDatasetOp::kNextFuncOtherArgs;
/* static */ constexpr const char* const
    GeneratorDatasetOp::kFinalizeFuncOtherArgs;
/* static */ constexpr const char* const GeneratorDatasetOp::kInitFunc;
/* static */ constexpr const char* const GeneratorDatasetOp::kNextFunc;
/* static */ constexpr const char* const GeneratorDatasetOp::kFinalizeFunc;
/* static */ constexpr const char* const GeneratorDatasetOp::kTinitFuncArgs;
/* static */ constexpr const char* const GeneratorDatasetOp::kTnextFuncArgs;
/* static */ constexpr const char* const GeneratorDatasetOp::kTfinalizeFuncArgs;
/* static */ constexpr const char* const GeneratorDatasetOp::kOutputTypes;
/* static */ constexpr const char* const GeneratorDatasetOp::kOutputShapes;

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
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(init_func_->CheckExternalState());
    TF_RETURN_IF_ERROR(next_func_->CheckExternalState());
    return finalize_func_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    return errors::Unimplemented(DebugString(),
                                 " does not support serialization");
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    ~Iterator() override {
      if (!finalized_ && initialized_) {
        std::vector<Tensor> ignored;
        Status s =
            instantiated_finalize_func_->RunInstantiated(state_, &ignored);
        if (!s.ok()) {
          LOG(WARNING)
              << "Error occurred when finalizing GeneratorDataset iterator: "
              << s;
        }
      }
    }

    Status Initialize(IteratorContext* ctx) override {
      TF_RETURN_IF_ERROR(
          dataset()->init_func_->Instantiate(ctx, &instantiated_init_func_));
      TF_RETURN_IF_ERROR(
          dataset()->next_func_->Instantiate(ctx, &instantiated_next_func_));
      TF_RETURN_IF_ERROR(dataset()->finalize_func_->Instantiate(
          ctx, &instantiated_finalize_func_));
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);

      if (!initialized_) {
        TF_RETURN_IF_ERROR(
            instantiated_init_func_->RunWithBorrowedArgs(ctx, {}, &state_));
        initialized_ = true;
      }

      if (finalized_) {
        *end_of_sequence = true;
        return Status::OK();
      }

      Status s = instantiated_next_func_->RunWithBorrowedArgs(ctx, state_,
                                                              out_tensors);
      if (s.ok()) {
        *end_of_sequence = false;
      } else if (errors::IsOutOfRange(s)) {
        // `next_func` may deliberately raise `errors::OutOfRange`
        // to indicate that we should terminate the iteration.
        s = Status::OK();
        *end_of_sequence = true;

        // NOTE(mrry): We ignore any tensors returned by the finalize function.
        std::vector<Tensor> ignored;
        TF_RETURN_IF_ERROR(instantiated_finalize_func_->RunWithBorrowedArgs(
            ctx, state_, &ignored));
        finalized_ = true;
      }
      return s;
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      return errors::Unimplemented(
          "GeneratorDataset does not support checkpointing.");
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return errors::Unimplemented(
          "GeneratorDataset does not support checkpointing.");
    }

   private:
    mutex mu_;
    bool initialized_ TF_GUARDED_BY(mu_) = false;
    bool finalized_ TF_GUARDED_BY(mu_) = false;
    std::vector<Tensor> state_ TF_GUARDED_BY(mu_);
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_init_func_;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_next_func_;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_finalize_func_;
  };

  const std::unique_ptr<CapturedFunction> init_func_;
  const std::unique_ptr<CapturedFunction> next_func_;
  const std::unique_ptr<CapturedFunction> finalize_func_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

GeneratorDatasetOp::GeneratorDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kInitFunc, /*params=*/{},
                                               &init_func_metadata_));
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kNextFunc, /*params=*/{},
                                               &next_func_metadata_));
  OP_REQUIRES_OK(ctx,
                 FunctionMetadata::Create(ctx, kFinalizeFunc, /*params=*/{},
                                          &finalize_func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void GeneratorDatasetOp::MakeDataset(OpKernelContext* ctx,
                                     DatasetBase** output) {
  std::unique_ptr<CapturedFunction> init_func;
  OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, init_func_metadata_,
                                               kInitFuncOtherArgs, &init_func));

  std::unique_ptr<CapturedFunction> next_func;
  OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, next_func_metadata_,
                                               kNextFuncOtherArgs, &next_func));

  std::unique_ptr<CapturedFunction> finalize_func;
  OP_REQUIRES_OK(
      ctx, CapturedFunction::Create(ctx, finalize_func_metadata_,
                                    kFinalizeFuncOtherArgs, &finalize_func));

  *output =
      new Dataset(ctx, std::move(init_func), std::move(next_func),
                  std::move(finalize_func), output_types_, output_shapes_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("GeneratorDataset").Device(DEVICE_CPU).Priority(2),
                        GeneratorDatasetOp);
REGISTER_KERNEL_BUILDER(Name("GeneratorDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("handle")
                            .Priority(1),
                        GeneratorDatasetOp);
}  // namespace

}  // namespace data
}  // namespace tensorflow
