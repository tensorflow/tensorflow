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
#include "tensorflow/core/kernels/data/range_dataset_op.h"

#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/types.h"
#include "tensorflow/core/data/global_shuffle_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tsl/platform/mutex.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const RangeDatasetOp::kDatasetType;
/* static */ constexpr const char* const RangeDatasetOp::kStart;
/* static */ constexpr const char* const RangeDatasetOp::kStop;
/* static */ constexpr const char* const RangeDatasetOp::kStep;
/* static */ constexpr const char* const RangeDatasetOp::kOutputTypes;
/* static */ constexpr const char* const RangeDatasetOp::kOutputShapes;
/* static */ constexpr const char* const RangeDatasetOp::kReplicateOnSplit;

namespace {
constexpr char kNext[] = "next";
constexpr char kHasSplitProvider[] = "has_split_provider";
constexpr char kSlash[] = "/";
constexpr char kSplitProvider[] = "split_provider";

absl::Status ConvertOutputTypes(const tensorflow::DataTypeVector& output_dtypes,
                                std::vector<Tensor>* out_tensors, int64 value) {
  switch (output_dtypes[0]) {
#define HANDLE_TYPE(type)                                \
  case DataTypeToEnum<type>::value: {                    \
    out_tensors->emplace_back(static_cast<type>(value)); \
    break;                                               \
  }
    TF_CALL_NUMBER_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::InvalidArgument("Unsupported data type: ",
                                     DataTypeString(output_dtypes[0]));
  }
  return absl::OkStatus();
}

int64_t sgn(int64_t val) { return (0 < val) - (val < 0); }

int64_t RangeCardinality(int64_t start, int64_t stop, int64_t step) {
  // `enumerate` uses int max to simulate an infinite range dataset.
  if (stop >= tsl::kint64max) {
    return kInfiniteCardinality;
  }

  // If the signs of `stop - start` and `step` are different or either of
  // the values is zero, the range will be empty.
  if (sgn(stop - start) * sgn(step) <= 0) {
    return 0;
  } else if (step > 0) {
    // Invariant: stop - start > 0 && step > 0
    return (stop - start - 1) / step + 1;
  } else {
    // Invariant: start - stop > 0 && step < 0
    return (start - stop - 1) / -step + 1;
  }
}

// Class which produces the elements of `range(start, stop, step)`. Threadsafe.
class RangeCounter {
 public:
  RangeCounter(int64_t start, int64_t stop, int64_t step)
      : start_(start), stop_(stop), step_(step), next_(start) {}

  // Returns the next value for the counter. Sets `*end_of_counter` to indicate
  // whether the end of the counter was reached.
  int64_t GetNext(bool* end_of_counter) {
    mutex_lock l(mu_);
    if ((step_ > 0 && next_ >= stop_) || (step_ < 0 && next_ <= stop_)) {
      *end_of_counter = true;
      return -1;
    }
    *end_of_counter = false;
    int64_t result = next_;
    next_ += step_;
    return result;
  }

  int64_t Peek() const {
    mutex_lock l(mu_);
    return next_;
  }

  void Reset() {
    mutex_lock l(mu_);
    next_ = start_;
  }

  void SetNext(int64_t value) {
    mutex_lock l(mu_);
    next_ = value;
  }

  int64_t Cardinality() const { return RangeCardinality(start_, stop_, step_); }

 private:
  const int64_t start_;
  const int64_t stop_;
  const int64_t step_;
  mutable mutex mu_;
  int64_t next_ TF_GUARDED_BY(mu_);
};
}  // namespace

// Split provider where splits are individual outputs from RangeDataset.
// For example, the "splits" of range(0, 10, 2) will be {0, 2, 4, 6, 8}.
// The split tensors are scalars of type DT_INT64.
class RangeDatasetOp::RangeSplitProvider : public SplitProvider {
 public:
  RangeSplitProvider(int64_t start, int64_t stop, int64_t step)
      : counter_(start, stop, step) {}

  absl::Status GetNext(Tensor* split, bool* end_of_splits) override {
    int64_t next = counter_.GetNext(end_of_splits);
    if (*end_of_splits) {
      return absl::OkStatus();
    }
    *split = Tensor(DT_INT64, TensorShape{});
    split->scalar<int64_t>()() = next;
    return absl::OkStatus();
  }

  absl::Status Reset() override {
    counter_.Reset();
    return absl::OkStatus();
  }

  absl::Status Save(std::function<std::string(std::string)> key_name_fn,
                    IteratorStateWriter* writer) override {
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(key_name_fn(kNext), counter_.Peek()));
    return absl::OkStatus();
  }

  absl::Status Restore(std::function<std::string(std::string)> key_name_fn,
                       IteratorStateReader* reader) override {
    int64_t next;
    TF_RETURN_IF_ERROR(reader->ReadScalar(key_name_fn(kNext), &next));
    counter_.SetNext(next);
    return absl::OkStatus();
  }

  int64_t Cardinality() const override { return counter_.Cardinality(); }

 private:
  RangeCounter counter_;
};

class RangeDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, int64_t start, int64_t stop, int64_t step,
          DataTypeVector output_dtypes, bool replicate_on_split)
      : DatasetBase(DatasetContext(ctx)),
        start_(start),
        stop_(stop),
        step_(step),
        output_dtypes_(output_dtypes),
        replicate_on_split_(replicate_on_split) {}

  absl::Status RandomIndexingCompatible() const override {
    return absl::OkStatus();
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
    return output_dtypes_;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    static std::vector<PartialTensorShape>* shapes =
        new std::vector<PartialTensorShape>({PartialTensorShape({})});
    return *shapes;
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.set_args(start_, stop_, step_);
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return RangeCardinality(start_, stop_, step_);
  }

  absl::Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>*
                                      split_providers) const override {
    split_providers->push_back(
        std::make_unique<RangeSplitProvider>(start_, stop_, step_));
    return absl::OkStatus();
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    inputs->clear();
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override { return absl::OkStatus(); }

  absl::Status Get(OpKernelContext* ctx, int64 index,
                   std::vector<Tensor>* out_tensors) const override {
    return Get(AnyContext(ctx), index, out_tensors);
  }

  absl::Status Get(AnyContext ctx, int64 index,
                   std::vector<Tensor>* out_tensors) const override {
    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    return ConvertOutputTypes(output_dtypes(), out_tensors,
                              start_ + (index * step_));
  }

 protected:
  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    Node* start = nullptr;
    Node* stop = nullptr;
    Node* step = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(start_, &start));
    TF_RETURN_IF_ERROR(b->AddScalar(stop_, &stop));
    TF_RETURN_IF_ERROR(b->AddScalar(step_, &step));
    AttrValue replicate_on_split;
    b->BuildAttrValue(replicate_on_split_, &replicate_on_split);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {start, stop, step},                                // Inputs
        {std::make_pair(kReplicateOnSplit, replicate_on_split)},  // Attrs
        output));
    return absl::OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          global_shuffle_iterator_(dataset()) {}

    bool SymbolicCheckpointCompatible() const override { return true; }

    absl::Status Initialize(IteratorContext* ctx) override {
      if (ctx->split_providers().empty() || dataset()->replicate_on_split_) {
        counter_ = std::make_unique<RangeCounter>(
            dataset()->start_, dataset()->stop_, dataset()->step_);
      } else {
        TF_ASSIGN_OR_RETURN(split_provider_,
                            GetSingleSplitProvider(ctx, dataset()));
      }
      return absl::OkStatus();
    }

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      if (ctx->index_mapper() != nullptr) {
        return global_shuffle_iterator_.GetNext(ctx, out_tensors,
                                                end_of_sequence);
      }
      int64_t value;
      if (split_provider_ != nullptr) {
        Tensor split;
        TF_RETURN_IF_ERROR(split_provider_->GetNext(&split, end_of_sequence));
        if (*end_of_sequence) {
          return absl::OkStatus();
        }
        value = split.scalar<int64_t>()();
      } else {
        value = counter_->GetNext(end_of_sequence);
        if (*end_of_sequence) {
          return absl::OkStatus();
        }
      }
      out_tensors->reserve(1);
      return ConvertOutputTypes(output_dtypes(), out_tensors, value);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      if (split_provider_) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(prefix(), kHasSplitProvider, true));
        TF_RETURN_IF_ERROR(split_provider_->Save(
            [this](const std::string& key) {
              return SplitProviderKeyNameFn(key);
            },
            writer));
      } else {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(prefix(), kNext, counter_->Peek()));
      }
      TF_RETURN_IF_ERROR(global_shuffle_iterator_.Save(prefix(), ctx, writer));
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      if (ctx->restored_element_count().has_value()) {
        return global_shuffle_iterator_.Restore(prefix(), ctx, reader);
      }
      if (reader->Contains(prefix(), kHasSplitProvider)) {
        TF_RETURN_IF_ERROR(split_provider_->Restore(
            [this](const std::string& key) {
              return SplitProviderKeyNameFn(key);
            },
            reader));
      } else {
        int64_t next;
        TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kNext, &next));
        counter_->SetNext(next);
      }
      return absl::OkStatus();
    }

    std::string SplitProviderKeyNameFn(const std::string& key) {
      return full_name(absl::StrCat(kSplitProvider, kSlash, key));
    }

   private:
    std::unique_ptr<RangeCounter> counter_;
    std::shared_ptr<SplitProvider> split_provider_;
    GlobalShuffleIterator global_shuffle_iterator_;
  };

  const int64_t start_;
  const int64_t stop_;
  const int64_t step_;
  const DataTypeVector output_dtypes_;
  const bool replicate_on_split_;
};

RangeDatasetOp::RangeDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  if (ctx->HasAttr(kReplicateOnSplit)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kReplicateOnSplit, &replicate_on_split_));
  }
}

void RangeDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
  int64_t start;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kStart, &start));

  int64_t stop;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kStop, &stop));

  int64_t step;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kStep, &step));
  OP_REQUIRES(ctx, step != 0,
              errors::InvalidArgument("step must be a non-zero integer."));

  *output =
      new Dataset(ctx, start, stop, step, output_types_, replicate_on_split_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("RangeDataset").Device(DEVICE_CPU),
                        RangeDatasetOp);
}  // namespace

}  // namespace data
}  // namespace tensorflow
