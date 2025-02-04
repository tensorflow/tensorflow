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
#include "tensorflow/core/kernels/data/skip_dataset_op.h"

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/data/global_shuffle_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const SkipDatasetOp::kDatasetType;
/* static */ constexpr const char* const SkipDatasetOp::kInputDataset;
/* static */ constexpr const char* const SkipDatasetOp::kCount;
/* static */ constexpr const char* const SkipDatasetOp::kOutputTypes;
/* static */ constexpr const char* const SkipDatasetOp::kOutputShapes;

constexpr char kEmptySkip[] = "EmptySkip";
constexpr char kFiniteSkip[] = "FiniteSkip";
constexpr char kCurIndex[] = "i";
constexpr char kInputImplEmpty[] = "input_impl_empty";

class SkipDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, int64_t count, const DatasetBase* input)
      : DatasetBase(DatasetContext(ctx)), count_(count), input_(input) {
    input_->Ref();
    if (input_ != nullptr && count >= 0) {
      random_indexing_compatible_ = input_->RandomIndexingCompatible();
    } else {
      random_indexing_compatible_ = absl::FailedPreconditionError(
          absl::StrCat("Global shuffling does not support empty dataset or "
                       "skipping the entire dataset. Got skip(",
                       count, ")."));
    }
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    if (count_ < 0) {
      return std::make_unique<EmptyIterator>(EmptyIterator::Params{
          this, name_utils::IteratorPrefix(kEmptySkip, prefix)});
    } else {
      return std::make_unique<FiniteIterator>(FiniteIterator::Params{
          this, name_utils::IteratorPrefix(kFiniteSkip, prefix)});
    }
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    int64_t n = input_->Cardinality(options);
    if (n == kInfiniteCardinality || n == kUnknownCardinality) {
      return n;
    }
    return count_ < 0 ? 0 : std::max(int64_t{0}, n - count_);
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

  absl::Status Get(OpKernelContext* ctx, int64 index,
                   std::vector<Tensor>* out_tensors) const override {
    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    return input_->Get(ctx, index + count_, out_tensors);
  }

  absl::Status RandomIndexingCompatible() const override {
    return random_indexing_compatible_;
  }

 protected:
  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* count = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(count_, &count));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node, count}, output));
    return absl::OkStatus();
  }

 private:
  class EmptyIterator : public DatasetIterator<Dataset> {
   public:
    explicit EmptyIterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    bool SymbolicCheckpointCompatible() const override { return true; }

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      *end_of_sequence = true;
      return absl::OkStatus();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      return absl::OkStatus();
    }
  };

  class FiniteIterator : public DatasetIterator<Dataset> {
   public:
    explicit FiniteIterator(const Params& params)
        : DatasetIterator<Dataset>(params), i_(0) {}

    bool SymbolicCheckpointCompatible() const override { return true; }

    absl::Status Initialize(IteratorContext* ctx) override {
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      if (ctx->index_mapper() != nullptr) {
        return Get(ctx, out_tensors, end_of_sequence);
      }

      mutex_lock l(mu_);  // TODO(mrry): Make locking less conservative.
      if (!input_impl_) {
        *end_of_sequence = true;
        return absl::OkStatus();
      }

      if (i_ < dataset()->count_) {
        int num_skipped;
        TF_RETURN_IF_ERROR(input_impl_->Skip(ctx, dataset()->count_ - i_,
                                             end_of_sequence, &num_skipped));
        i_ += num_skipped;
        if (*end_of_sequence) {
          // We reached the end before the count was reached.
          input_impl_.reset();
          return absl::OkStatus();
        }
      }

      // Return GetNext() on the underlying iterator.
      TF_RETURN_IF_ERROR(
          input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
      if (*end_of_sequence) {
        input_impl_.reset();
      }
      return absl::OkStatus();
    }

    absl::Status Get(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) {
      mutex_lock l(mu_);
      if (!input_impl_) {
        *end_of_sequence = true;
        return absl::OkStatus();
      }

      IteratorContextWithIndexMapper ctx_with_index_mapper(ctx, this);
      TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx_with_index_mapper.Get(),
                                              out_tensors, end_of_sequence));
      ctx_with_index_mapper.MergeCheckpoint();
      return absl::OkStatus();
    }

    IndexMapperFn GetIndexMapper(
        IndexMapperFn parent_index_mapper) const override {
      int64_t skip_count = dataset()->count_;
      return [parent_index_mapper,
              skip_count](size_t element_position) -> absl::StatusOr<size_t> {
        TF_ASSIGN_OR_RETURN(size_t shuffled_element_position,
                            parent_index_mapper(element_position));
        return shuffled_element_position + skip_count;
      };
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kCurIndex, i_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          prefix(), kInputImplEmpty, static_cast<int64_t>(!input_impl_)));
      if (input_impl_) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      }
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      if (ctx->restored_element_count().has_value()) {
        mutex_lock l(mu_);
        return RestoreInput(ctx, reader, input_impl_);
      }

      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kCurIndex, &i_));
      int64_t input_empty;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kInputImplEmpty, &input_empty));
      if (!static_cast<bool>(input_empty)) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      } else {
        input_impl_.reset();
      }
      return absl::OkStatus();
    }

   private:
    mutex mu_;
    int64_t i_ TF_GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
  };

  const int64_t count_;
  const DatasetBase* const input_;
  absl::Status random_indexing_compatible_;
};

SkipDatasetOp::SkipDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}

void SkipDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                DatasetBase** output) {
  // Create a new SkipDatasetOp::Dataset, and return it as the output.
  int64_t count;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kCount, &count));

  *output = new Dataset(ctx, count, input);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("SkipDataset").Device(DEVICE_CPU), SkipDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
