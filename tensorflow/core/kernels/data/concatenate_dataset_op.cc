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
#include "tensorflow/core/kernels/data/concatenate_dataset_op.h"

#include <utility>

#include "absl/status/status.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const ConcatenateDatasetOp::kDatasetType;
/* static */ constexpr const char* const ConcatenateDatasetOp::kInputDataset;
/* static */ constexpr const char* const ConcatenateDatasetOp::kAnotherDataset;
/* static */ constexpr const char* const ConcatenateDatasetOp::kOutputTypes;
/* static */ constexpr const char* const ConcatenateDatasetOp::kOutputShapes;

constexpr char kIndex[] = "i";
constexpr char kInputImplUninitialized[] = "input_impl_uninitialized";

class ConcatenateDatasetOp::Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext* ctx, const DatasetBase* input,
                   const DatasetBase* to_concatenate)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        to_concatenate_(to_concatenate),
        input_cardinality_(input->Cardinality()),
        to_concatenate_cardinality_(to_concatenate_->Cardinality()) {
    input_->Ref();
    to_concatenate_->Ref();

    auto os_input = input->output_shapes();
    auto os_concatenate = to_concatenate->output_shapes();
    for (int i = 0; i < os_input.size(); i++) {
      PartialTensorShape output_tensorshape({});
      OP_REQUIRES_OK(ctx,
                     MostSpecificCompatibleShape(os_input[i], os_concatenate[i],
                                                 &output_tensorshape));
      output_shapes_.push_back(output_tensorshape);
    }
    if (input_ != nullptr && !input_->RandomIndexingCompatible().ok()) {
      random_indexing_compatible_ = input->RandomIndexingCompatible();
    } else if (to_concatenate_ != nullptr &&
               !to_concatenate_->RandomIndexingCompatible().ok()) {
      random_indexing_compatible_ = to_concatenate_->RandomIndexingCompatible();
    }
  }
  ~Dataset() override {
    input_->Unref();
    to_concatenate_->Unref();
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>*
                                split_providers) const override {
    TF_ASSIGN_OR_RETURN(*split_providers, GetSplitProviders(this));
    return absl::OkStatus();
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    int64_t input_cardinality = input_->Cardinality(options);
    int64_t to_concatenate_cardinality = to_concatenate_->Cardinality(options);

    if (input_cardinality == kInfiniteCardinality ||
        to_concatenate_cardinality == kInfiniteCardinality) {
      return kInfiniteCardinality;
    }
    if (input_cardinality == kUnknownCardinality ||
        to_concatenate_cardinality == kUnknownCardinality) {
      return kUnknownCardinality;
    }
    return input_cardinality + to_concatenate_cardinality;
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    inputs->push_back(to_concatenate_);
    return absl::OkStatus();
  }

  Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(input_->CheckExternalState());
    return to_concatenate_->CheckExternalState();
  }

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    if (index < input_cardinality_) {
      TF_RETURN_IF_ERROR(input_->Get(ctx, index, out_tensors));
    } else {
      TF_RETURN_IF_ERROR(
          to_concatenate_->Get(ctx, index - input_cardinality_, out_tensors));
    }
    return absl::OkStatus();
  }

  absl::Status RandomIndexingCompatible() const override {
    return random_indexing_compatible_;
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph));
    Node* to_concatenate_graph = nullptr;
    TF_RETURN_IF_ERROR(
        b->AddInputDataset(ctx, to_concatenate_, &to_concatenate_graph));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph, to_concatenate_graph}, output));
    return absl::OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params), i_(0) {}

    bool SymbolicCheckpointCompatible() const override { return true; }

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      input_impls_.resize(2);
      std::fill(end_of_sequence_, end_of_sequence_ + 2, false);

      TF_ASSIGN_OR_RETURN(input_contexts_,
                          CreateInputIteratorContexts(ctx, dataset()));
      TF_RETURN_IF_ERROR(dataset()->input_->MakeIterator(
          &input_contexts_[0], this, strings::StrCat(prefix(), "[0]"),
          &input_impls_[0]));

      TF_RETURN_IF_ERROR(dataset()->to_concatenate_->MakeIterator(
          &input_contexts_[1], this, strings::StrCat(prefix(), "[1]"),
          &input_impls_[1]));

      ctx->MergeCheckpoint(input_contexts_[0].checkpoint());
      ctx->MergeCheckpoint(input_contexts_[1].checkpoint());
      return absl::OkStatus();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      if (!input_impls_[0] && !input_impls_[1]) {
        *end_of_sequence = true;
        return absl::OkStatus();
      }
      // Global shuffling
      if (ctx->index_mapper()) {
        if (input_contexts_[0].index_mapper() == nullptr) {
          IndexMapperFn new_index_mapper1 =
              [index_mapper = ctx->index_mapper(),
               input_cardinality = dataset()->input_cardinality_,
               to_concat_cardinality = dataset()->to_concatenate_cardinality_](
                  size_t idx) -> absl::StatusOr<size_t> {
            TF_ASSIGN_OR_RETURN(size_t pidx, index_mapper(idx));
            if (pidx >= input_cardinality + to_concat_cardinality) {
              return absl::OutOfRangeError("Out of range");
            }
            if (pidx >= input_cardinality) {
              // This has to return a status so that upstream global shuffle
              // iterator will not treat it as an end of sequence.
              return absl::NotFoundError("Skip this element.");
            }
            return pidx;
          };

          IndexMapperFn new_index_mapper2 =
              [index_mapper = ctx->index_mapper(),
               input_cardinality = dataset()->input_cardinality_,
               to_concat_cardinality = dataset()->to_concatenate_cardinality_](
                  size_t idx) -> absl::StatusOr<size_t> {
            TF_ASSIGN_OR_RETURN(size_t pidx, index_mapper(idx));
            if (pidx >= input_cardinality + to_concat_cardinality) {
              return absl::OutOfRangeError("Out of range");
            }
            if (pidx < input_cardinality) {
              // This has to return a status so that upstream global shuffle
              // iterator will not treat it as an end of sequence.
              return absl::NotFoundError("Skip this element.");
            }
            return pidx - input_cardinality;
          };

          input_contexts_[0].SetIndexMapper(new_index_mapper1);
          input_contexts_[1].SetIndexMapper(new_index_mapper2);
        }

        // Just for debugging
        // TF_ASSIGN_OR_RETURN(size_t new_index,
        //                     ctx->index_mapper()(element_count_++));
        // LOG(ERROR) << "input_cardinality: " << dataset()->input_cardinality_;
        // LOG(ERROR) << "new_index: " << new_index;

        // Both iterators' GetNext must be called to make sure
        // their underlying element count is moved.

        std::vector<Tensor> out_tensors1, out_tensors2;
        absl::Status status1 = absl::OkStatus(), status2 = absl::OkStatus();
        if (!end_of_sequence_[0]) {
          auto status1 = input_impls_[0]->GetNext(
              &input_contexts_[0], &out_tensors1, &(end_of_sequence_[0]));
          if (status1.ok() && !end_of_sequence_[0]) {
            *out_tensors = std::move(out_tensors1);
          }
        }
        if (!end_of_sequence_[1]) {
          auto status2 = input_impls_[1]->GetNext(
              &input_contexts_[1], &out_tensors2, &end_of_sequence_[1]);
          if (status2.ok() && !end_of_sequence_[1]) {
            *out_tensors = std::move(out_tensors2);
          }
        }
        // LOG(ERROR) << "end_of_sequence_[0]: " << end_of_sequence_[0];
        // LOG(ERROR) << "end_of_sequence_[1]: " << end_of_sequence_[1];

        // If we have not reached the end of sequence, one of `input_impls_`
        // must return an element but not both.
        if (status1.ok() && !end_of_sequence_[0]) {
          return absl::OkStatus();
        }
        if (status2.ok() && !end_of_sequence_[1]) {
          return absl::OkStatus();
        }

        if (end_of_sequence_[0] && end_of_sequence_[1]) {
          *end_of_sequence = true;
          return absl::OkStatus();
        }
        status1.Update(status2);
        return status1;
      }

      for (; i_ < 2; ++i_) {
        TF_RETURN_IF_ERROR(input_impls_[i_]->GetNext(
            &input_contexts_[i_], out_tensors, end_of_sequence));
        ctx->MergeCheckpoint(input_contexts_[i_].checkpoint());
        if (!*end_of_sequence) {
          return absl::OkStatus();
        }
      }
      *end_of_sequence = true;
      input_impls_[0].reset();
      input_impls_[1].reset();
      return absl::OkStatus();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kIndex, i_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          prefix(), absl::StrFormat("%s[%d]", kInputImplUninitialized, 0),
          static_cast<int64_t>(!input_impls_[0])));
      if (input_impls_[0]) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impls_[0]));
      }
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          prefix(), absl::StrFormat("%s[%d]", kInputImplUninitialized, 1),
          static_cast<int64_t>(!input_impls_[1])));
      if (input_impls_[1]) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impls_[1]));
      }
      return absl::OkStatus();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);

      if (ctx->restored_element_count()) {
        if (input_impls_.size() != 2) {
          return absl::FailedPreconditionError(
              "`Initialize` should be called before restoring from the "
              "checkpoint.");
        }
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impls_[0]));
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impls_[1]));
        return absl::OkStatus();
      }

      TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kIndex, &i_));
      int64_t input_uninitialized;
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          prefix(), absl::StrFormat("%s[%d]", kInputImplUninitialized, 0),
          &input_uninitialized));
      if (static_cast<bool>(input_uninitialized)) {
        input_impls_[0].reset();
      }
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          prefix(), absl::StrFormat("%s[%d]", kInputImplUninitialized, 1),
          &input_uninitialized));
      if (static_cast<bool>(input_uninitialized)) {
        input_impls_[1].reset();
      }

      if (!TF_PREDICT_TRUE(i_ >= 0 && i_ <= 2))
        return errors::InvalidArgument("i_ must be in range [0, 2].");

      if (input_impls_[0]) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impls_[0]));
      }
      if (input_impls_[1]) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impls_[1]));
      }

      return absl::OkStatus();
    }

   private:
    mutex mu_;
    int64_t i_ TF_GUARDED_BY(mu_);
    std::vector<std::unique_ptr<IteratorBase>> input_impls_ TF_GUARDED_BY(mu_);
    std::vector<IteratorContext> input_contexts_ TF_GUARDED_BY(mu_);
    bool end_of_sequence_[2] TF_GUARDED_BY(mu_);
    // For debugging
    // int64_t element_count_ TF_GUARDED_BY(mu_) = 0;
  };

  Status MostSpecificCompatibleShape(const PartialTensorShape& ts1,
                                     const PartialTensorShape& ts2,
                                     PartialTensorShape* output_tensorshape) {
    if (ts1.dims() != ts2.dims() || ts1.unknown_rank() || ts2.unknown_rank())
      return absl::OkStatus();
    auto dims1 = ts1.dim_sizes();
    auto dims2 = ts2.dim_sizes();
    for (int d = 0; d < ts1.dims(); d++) {
      if (dims1[d] == dims2[d])
        TF_RETURN_IF_ERROR(output_tensorshape->AddDimWithStatus(dims1[d]));
      else
        TF_RETURN_IF_ERROR(output_tensorshape->AddDimWithStatus(-1));
    }
    return absl::OkStatus();
  }

  const DatasetBase* input_;
  const DatasetBase* to_concatenate_;
  const int64_t input_cardinality_;
  const int64_t to_concatenate_cardinality_;
  std::vector<PartialTensorShape> output_shapes_;
  absl::Status random_indexing_compatible_ = absl::OkStatus();
};

ConcatenateDatasetOp::ConcatenateDatasetOp(OpKernelConstruction* ctx)
    : BinaryDatasetOpKernel(ctx) {}

void ConcatenateDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                       DatasetBase* to_concatenate,
                                       DatasetBase** output) {
  OP_REQUIRES(ctx, input->output_dtypes() == to_concatenate->output_dtypes(),
              errors::InvalidArgument(
                  "input dataset and dataset to concatenate"
                  " have different output_types %s and %s",
                  (DataTypeVectorString(input->output_dtypes()),
                   DataTypeVectorString(to_concatenate->output_dtypes()))));
  *output = new Dataset(ctx, input, to_concatenate);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ConcatenateDataset").Device(DEVICE_CPU),
                        ConcatenateDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
