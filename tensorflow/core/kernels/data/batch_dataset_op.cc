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
#include "tensorflow/core/kernels/data/batch_dataset_op.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/global_shuffle_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/util/batch_util.h"
#include "tsl/platform/mutex.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const BatchDatasetOp::kDatasetType;
/* static */ constexpr const char* const BatchDatasetOp::kInputDataset;
/* static */ constexpr const char* const BatchDatasetOp::kBatchSize;
/* static */ constexpr const char* const BatchDatasetOp::kDropRemainder;
/* static */ constexpr const char* const BatchDatasetOp::kParallelCopy;
/* static */ constexpr const char* const BatchDatasetOp::kOutputTypes;
/* static */ constexpr const char* const BatchDatasetOp::kOutputShapes;

constexpr char kInputImplEmpty[] = "input_impl_empty";
constexpr char kBatchDataset[] = "BatchDataset";

class BatchDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, int64_t batch_size, bool drop_remainder,
          bool parallel_copy, const DatasetBase* input, int op_version)
      : DatasetBase(DatasetContext(ctx)),
        batch_size_(batch_size),
        // Dataset batch is sometimes used to stack all elements in the
        // dataset. In such cases, a very large batch size (e.g., INT32_MAX)
        // is passed with drop_remainder set to false. Avoid OOM in such case
        // by limiting `reserve()` size by 2**16.
        reserve_size_(drop_remainder ? batch_size
                                     : std::min<int64_t>(batch_size, 1 << 16)),
        drop_remainder_(drop_remainder),
        parallel_copy_(parallel_copy),
        input_(input),
        op_version_(op_version),
        traceme_metadata_(
            {{"batch_size",
              strings::Printf("%lld", static_cast<long long>(batch_size))},
             {"drop_remainder", drop_remainder ? "true" : "false"},
             {"parallel_copy", parallel_copy ? "true" : "false"}}) {
    input_->Ref();

    // NOTE(mrry): Currently we implement "batch up to" semantics. If
    // we could tell statically that the input dataset is infinite,
    // then we could always report `batch_size` as the 0th dimension.
    const auto& input_shapes = input_->output_shapes();
    output_shapes_.reserve(input_shapes.size());
    for (const auto& input_shape : input_shapes) {
      if (drop_remainder_ || input_->Cardinality() == kInfiniteCardinality) {
        output_shapes_.emplace_back(
            PartialTensorShape({batch_size_}).Concatenate(input_shape));
      } else {
        output_shapes_.emplace_back(
            PartialTensorShape({-1}).Concatenate(input_shape));
      }
    }

    random_indexing_compatible_ = absl::OkStatus();
    if (!drop_remainder_) {
      random_indexing_compatible_ = absl::FailedPreconditionError(absl::StrCat(
          type_string(),
          " does not support global shuffling with `drop_remainder=False`."));
    } else if (input_ != nullptr) {
      random_indexing_compatible_ = input_->RandomIndexingCompatible();
    }
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    params.op_version = op_version_;
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix, params)});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.op_version = op_version_;
    params.set_args(batch_size_);
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    int64_t n = input_->Cardinality(options);
    if (n == kInfiniteCardinality || n == kUnknownCardinality) {
      return n;
    }
    return n / batch_size_ + (n % batch_size_ == 0 || drop_remainder_ ? 0 : 1);
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
    const int64 cardinality = Cardinality();
    if (index < 0 || index >= cardinality) {
      return errors::OutOfRange("Index out of range [0, ", cardinality,
                                "):", index);
    }
    int batch_start_index = batch_size_ * index;
    std::vector<std::vector<Tensor>> batch_elements;
    int input_cardinality = input_->Cardinality();
    for (int i = batch_start_index;
         i < batch_start_index + batch_size_ && i < input_cardinality; ++i) {
      std::vector<Tensor> batch_element_tuple;
      TF_RETURN_IF_ERROR(input_->Get(ctx, i, &batch_element_tuple));
      batch_elements.emplace_back(std::move(batch_element_tuple));
    }
    TF_RETURN_IF_ERROR(CopyBatch(AnyContext(ctx), std::move(batch_elements),
                                 parallel_copy_, out_tensors));
    return absl::OkStatus();
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
    Node* batch_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));
    Node* drop_remainder = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(drop_remainder_, &drop_remainder));
    AttrValue parallel_copy;
    b->BuildAttrValue(parallel_copy_, &parallel_copy);
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node, batch_size, drop_remainder},
                      {{kParallelCopy, parallel_copy}}, output));
    return absl::OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    bool SymbolicCheckpointCompatible() const override { return true; }

    absl::Status Initialize(IteratorContext* ctx) override {
      tsl::mutex_lock l(mu_);
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      // Each row of `batch_elements` is a tuple of tensors from the
      // input iterator.
      std::vector<std::vector<Tensor>> batch_elements;
      {
        mutex_lock l(mu_);
        if (!input_impl_) {
          *end_of_sequence = true;
          return absl::OkStatus();
        }
        batch_elements.reserve(dataset()->reserve_size_);
        *end_of_sequence = false;
        IteratorContextWithIndexMapper ctx_with_index_mapper(ctx, this);
        for (int i = 0; i < dataset()->batch_size_ && !*end_of_sequence; ++i) {
          std::vector<Tensor> batch_element_tuple;
          TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx_with_index_mapper.Get(),
                                                  &batch_element_tuple,
                                                  end_of_sequence));
          if (!*end_of_sequence) {
            batch_elements.emplace_back(std::move(batch_element_tuple));
          } else {
            input_impl_.reset();
          }
        }
        ctx_with_index_mapper.MergeCheckpoint();
      }

      if (batch_elements.empty()) {
        DCHECK(*end_of_sequence);
        return absl::OkStatus();
      }

      if (dataset()->drop_remainder_ &&
          batch_elements.size() < dataset()->batch_size_) {
        *end_of_sequence = true;
        return absl::OkStatus();
      }

      // Copy the retrieved batch elements into one output tensor per tuple
      // component.
      //
      // NOTE(mrry): If the input or output sizes are statically known, we
      // could potentially read the input values in-place into their
      // respective slice locations. This would require a different GetNext()
      // overload that supports zero-copy, and might make sense in an
      // optimization pass.
      TF_RETURN_IF_ERROR(CopyBatch(AnyContext(ctx), std::move(batch_elements),
                                   dataset()->parallel_copy_, out_tensors));

      *end_of_sequence = false;
      return absl::OkStatus();
    }

    IndexMapperFn GetIndexMapper(
        IndexMapperFn parent_index_mapper) const override {
      int64_t batch_size = dataset()->batch_size_;
      return [parent_index_mapper,
              batch_size](size_t element_position) -> absl::StatusOr<size_t> {
        size_t batch_element_position = element_position / batch_size;
        size_t input_element_offset = element_position % batch_size;
        TF_ASSIGN_OR_RETURN(size_t shuffled_element_position,
                            parent_index_mapper(batch_element_position));
        return shuffled_element_position * batch_size + input_element_offset;
      };
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), dataset()->batch_size_);
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          prefix(), kInputImplEmpty, static_cast<int64_t>(!input_impl_)));
      if (input_impl_) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      }
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      int64_t input_empty;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kInputImplEmpty, &input_empty));

      if (ctx->restored_element_count().has_value()) {
        IteratorContext::Params params(ctx);
        params.restored_element_count =
            *ctx->restored_element_count() * dataset()->batch_size_;
        IteratorContext ctx_copy(params);
        if (!static_cast<bool>(input_empty)) {
          TF_RETURN_IF_ERROR(RestoreInput(&ctx_copy, reader, input_impl_));
          ctx->MergeCheckpoint(ctx_copy.checkpoint());
        } else {
          input_impl_.reset();
        }
        return absl::OkStatus();
      }

      if (!static_cast<bool>(input_empty)) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      } else {
        input_impl_.reset();
      }
      return absl::OkStatus();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
      return dataset()->traceme_metadata_;
    }

   private:
    mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
  };

  const int64_t batch_size_;
  const int64_t reserve_size_;
  const bool drop_remainder_;
  const bool parallel_copy_;
  const DatasetBase* const input_;
  const int op_version_;
  std::vector<PartialTensorShape> output_shapes_;
  absl::Status random_indexing_compatible_;
  const TraceMeMetadata traceme_metadata_;
};

BatchDatasetOp::BatchDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx),
      op_version_(ctx->def().op() == kBatchDataset ? 1 : 2) {
  if (ctx->HasAttr(kParallelCopy)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kParallelCopy, &parallel_copy_));
  }
}

void BatchDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                 DatasetBase** output) {
  int64_t batch_size = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, kBatchSize, &batch_size));
  OP_REQUIRES(ctx, batch_size > 0,
              errors::InvalidArgument("Batch size must be greater than zero."));

  bool drop_remainder = false;
  if (op_version_ > 1) {
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<bool>(ctx, kDropRemainder, &drop_remainder));
  }

  *output = new Dataset(ctx, batch_size, drop_remainder, parallel_copy_, input,
                        op_version_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("BatchDataset").Device(DEVICE_CPU),
                        BatchDatasetOp);

REGISTER_KERNEL_BUILDER(Name("BatchDatasetV2").Device(DEVICE_CPU),
                        BatchDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
