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
#include "tensorflow/core/kernels/data/shard_dataset_op.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const ShardDatasetOp::kDatasetType;
/* static */ constexpr const char* const ShardDatasetOp::kInputDataset;
/* static */ constexpr const char* const ShardDatasetOp::kNumShards;
/* static */ constexpr const char* const ShardDatasetOp::kIndex;
/* static */ constexpr const char* const ShardDatasetOp::kRequireNonEmpty;
/* static */ constexpr const char* const ShardDatasetOp::kOutputTypes;
/* static */ constexpr const char* const ShardDatasetOp::kOutputShapes;

constexpr char kInputImplEmpty[] = "input_impl_empty";
constexpr char kNextIndex[] = "next_index";

class ShardDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, int64 num_shards, int64 index,
          bool require_non_empty, const DatasetBase* input)
      : DatasetBase(DatasetContext(ctx)),
        num_shards_(num_shards),
        index_(index),
        input_(input),
        require_non_empty_(require_non_empty),
        traceme_metadata_(
            {{"index", strings::Printf("%lld", static_cast<long long>(index))},
             {"num_shards",
              strings::Printf("%lld", static_cast<long long>(num_shards))}}) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.set_args(num_shards_, index_);
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  int64 Cardinality() const override {
    int64 n = input_->Cardinality();
    if (n == kInfiniteCardinality || n == kUnknownCardinality) {
      return n;
    }
    return n / num_shards_ + (index_ < n % num_shards_ ? 1 : 0);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
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

    AttrValue require_non_empty_attr;
    b->BuildAttrValue(require_non_empty_, &require_non_empty_attr);

    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node, num_shards, index},
                      {{kRequireNonEmpty, require_non_empty_attr}}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params), next_index_(0) {}

    Status Initialize(IteratorContext* ctx) override {
      if (dataset()->num_shards_ == kShardHint) {
        return errors::FailedPrecondition(
            "`tf.data.Dataset.shard(SHARD_HINT, ...)` can only be used in "
            "combiantion with "
            "`tf.distribute.Strategy.experimental_distribute_dataset()`.");
      }
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);

      *end_of_sequence = false;
      if (!input_impl_) {
        *end_of_sequence = true;
        return Status::OK();
      }

      int num_to_skip =
          (dataset()->index_ - next_index_) % dataset()->num_shards_;
      if (num_to_skip < 0) {
        num_to_skip += dataset()->num_shards_;
      }
      int num_skipped;
      TF_RETURN_IF_ERROR(
          input_impl_->Skip(ctx, num_to_skip, end_of_sequence, &num_skipped));
      next_index_ += num_skipped;
      if (*end_of_sequence) {
        input_impl_.reset();
        return Status::OK();
      }

      std::vector<Tensor> result;
      TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &result, end_of_sequence));
      if (*end_of_sequence) {
        input_impl_.reset();
        return Status::OK();
      }
      next_index_++;

      if (dataset()->require_non_empty_ &&
          next_index_ < dataset()->num_shards_) {
        int num_skipped;
        Status s = input_impl_->Skip(ctx, dataset()->num_shards_ - next_index_,
                                     end_of_sequence, &num_skipped);
        if (*end_of_sequence || errors::IsOutOfRange(s)) {
          // `dataset()->require_non_empty_` implies that this transformation
          // was introduced by auto_sharding rewrite, so it's acceptable
          // produce an error message that assumes auto-sharding context.
          return errors::InvalidArgument(
              "Could not apply FILE based sharding: the dataset only has ",
              next_index_, " file(s), which is not enough for the required ",
              dataset()->num_shards_,
              " shards/workers."
              "If you are using datasets with distribution strategy, "
              "consider setting the auto sharding policy to either DATA or "
              "OFF using the `experimental_distribute.auto_shard_policy` option"
              "of `tf.data.Options()`. Or, split your input files into a "
              "larger number of small files such that number of files is "
              "greater than number of shards/workers.");
        } else if (!s.ok()) {
          return s;
        }

        next_index_ = dataset()->num_shards_;
      }

      *out_tensors = std::move(result);
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), dataset()->num_shards_);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      if (!input_impl_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kInputImplEmpty), ""));
      } else {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kNextIndex), next_index_));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      if (!reader->Contains(full_name(kInputImplEmpty))) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name(kNextIndex), &next_index_));
      } else {
        input_impl_.reset();
      }
      return Status::OK();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
      return dataset()->traceme_metadata_;
    }

   private:
    mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    int64 next_index_ TF_GUARDED_BY(mu_);
  };

  const int64 num_shards_;
  const int64 index_;
  const DatasetBase* const input_;
  const bool require_non_empty_;
  const TraceMeMetadata traceme_metadata_;
};

ShardDatasetOp::ShardDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kRequireNonEmpty, &require_non_empty_));
}

void ShardDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                 DatasetBase** output) {
  int64 index = 0;
  int64 num_shards = 0;

  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kNumShards, &num_shards));
  OP_REQUIRES(
      ctx, num_shards > 0 || num_shards == kShardHint,
      errors::InvalidArgument("Number of shards must be greater than zero "
                              "(currently num_shards = ",
                              num_shards, ")."));

  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kIndex, &index));
  OP_REQUIRES(
      ctx, (index >= 0 && index < num_shards) || num_shards == kShardHint,
      errors::InvalidArgument("Index must be between 0 and ", num_shards - 1,
                              " (currently index = ", index, ")."));

  *output = new Dataset(ctx, num_shards, index, require_non_empty_, input);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ShardDataset").Device(DEVICE_CPU),
                        ShardDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
