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
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

class ShardDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ShardDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 index = 0;
    int64 num_shards = 0;

    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<int64>(ctx, "num_shards", &num_shards));
    OP_REQUIRES(
        ctx, num_shards > 0,
        errors::InvalidArgument("Number of shards must be greater than zero "
                                "(currently num_shards = ",
                                num_shards, ")."));

    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, "index", &index));
    OP_REQUIRES(
        ctx, index >= 0 && index < num_shards,
        errors::InvalidArgument("Index must be between 0 and ", num_shards - 1,
                                " (currently index = ", index, ")."));

    *output = new Dataset(ctx, num_shards, index, input);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, int64 num_shards, int64 index,
            const DatasetBase* input)
        : DatasetBase(DatasetContext(ctx)),
          num_shards_(num_shards),
          index_(index),
          input_(input) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::Shard")});
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override {
      return strings::StrCat("ShardDatasetOp(", num_shards_, ", ", index_,
                             ")::Dataset");
    }

    int64 Cardinality() const override {
      int64 n = input_->Cardinality();
      if (n == kInfiniteCardinality || n == kUnknownCardinality) {
        return n;
      }
      return n / num_shards_ + (index_ < n % num_shards_ ? 1 : 0);
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node* num_shards = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(num_shards_, &num_shards));
      Node* index = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(index_, &index));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {input_graph_node, num_shards, index}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params), next_index_(0) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);

        if (!input_impl_) {
          *end_of_sequence = true;
          return Status::OK();
        }

        std::vector<Tensor> result;
        do {
          result.clear();
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, &result, end_of_sequence));
          if (*end_of_sequence) {
            input_impl_.reset();
            return Status::OK();
          }
        } while ((next_index_++ % dataset()->num_shards_) != dataset()->index_);

        *out_tensors = std::move(result);
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         dataset()->num_shards_);
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (!input_impl_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impl_empty"), ""));
        } else {
          TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("next_index"), next_index_));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (!reader->Contains(full_name("input_impl_empty"))) {
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("next_index"), &next_index_));
        } else {
          input_impl_.reset();
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      int64 next_index_ GUARDED_BY(mu_);
    };

    const int64 num_shards_;
    const int64 index_;
    const DatasetBase* const input_;
  };
};

REGISTER_KERNEL_BUILDER(Name("ShardDataset").Device(DEVICE_CPU),
                        ShardDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
