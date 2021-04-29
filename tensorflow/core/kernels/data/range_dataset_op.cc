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

#include "absl/memory/memory.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/name_utils.h"

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

namespace {
constexpr char kNext[] = "next";
constexpr char kHasSplitProvider[] = "has_split_provider";
constexpr char kSlash[] = "/";
constexpr char kSplitProvider[] = "split_provider";

// Class which produces the elements of `range(start, stop, step)`. Threadsafe.
class RangeCounter {
 public:
  RangeCounter(int64 start, int64 stop, int64 step)
      : start_(start), stop_(stop), step_(step), next_(start) {}

  // Returns the next value for the counter. Sets `*end_of_counter` to indicate
  // whether the end of the counter was reached.
  int64 GetNext(bool* end_of_counter) {
    mutex_lock l(mu_);
    if ((step_ > 0 && next_ >= stop_) || (step_ < 0 && next_ <= stop_)) {
      *end_of_counter = true;
      return -1;
    }
    *end_of_counter = false;
    int result = next_;
    next_ += step_;
    return result;
  }

  int64 Peek() const {
    mutex_lock l(mu_);
    return next_;
  }

  void Reset() {
    mutex_lock l(mu_);
    next_ = start_;
  }

  void SetNext(int64 value) {
    mutex_lock l(mu_);
    next_ = value;
  }

 private:
  const int64 start_;
  const int64 stop_;
  const int64 step_;
  mutable mutex mu_;
  int64 next_ TF_GUARDED_BY(mu_);
};
}  // namespace

// Split provider where splits are individual outputs from RangeDataset.
// For example, the "splits" of range(0, 10, 2) will be {0, 2, 4, 6, 8}.
// The split tensors are scalars of type DT_INT64.
class RangeDatasetOp::RangeSplitProvider : public SplitProvider {
 public:
  RangeSplitProvider(int64 start, int64 stop, int64 step)
      : counter_(start, stop, step) {}

  Status GetNext(Tensor* split, bool* end_of_splits) override {
    int64 next = counter_.GetNext(end_of_splits);
    if (*end_of_splits) {
      return Status::OK();
    }
    *split = Tensor(DT_INT64, TensorShape{});
    split->scalar<int64>()() = next;
    return Status::OK();
  }

  Status Reset() override {
    counter_.Reset();
    return Status::OK();
  }

  Status Save(std::function<std::string(std::string)> key_name_fn,
              IteratorStateWriter* writer) override {
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(key_name_fn(kNext), counter_.Peek()));
    return Status::OK();
  }

  Status Restore(std::function<std::string(std::string)> key_name_fn,
                 IteratorStateReader* reader) override {
    int64 next;
    TF_RETURN_IF_ERROR(reader->ReadScalar(key_name_fn(kNext), &next));
    counter_.SetNext(next);
    return Status::OK();
  }

 private:
  RangeCounter counter_;
};

class RangeDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, int64 start, int64 stop, int64 step,
          DataTypeVector output_dtypes)
      : DatasetBase(DatasetContext(ctx)),
        start_(start),
        stop_(stop),
        step_(step),
        output_dtypes_(output_dtypes) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
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

  int64 Cardinality() const override {
    if (step_ > 0) {
      return std::max(int64{0}, (stop_ - start_ - 1) / step_ + 1);
    } else {
      return std::max(int64{0}, (start_ - stop_ - 1) / -step_ + 1);
    }
  }

  Status MakeSplitProvider(
      std::unique_ptr<SplitProvider>* split_provider) const override {
    *split_provider =
        absl::make_unique<RangeSplitProvider>(start_, stop_, step_);
    return Status::OK();
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->clear();
    return Status::OK();
  }

  Status CheckExternalState() const override { return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* start = nullptr;
    Node* stop = nullptr;
    Node* step = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(start_, &start));
    TF_RETURN_IF_ERROR(b->AddScalar(stop_, &stop));
    TF_RETURN_IF_ERROR(b->AddScalar(step_, &step));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {start, stop, step}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    Status Initialize(IteratorContext* ctx) override {
      split_provider_ = ctx->split_provider();
      if (!split_provider_) {
        counter_ = absl::make_unique<RangeCounter>(
            dataset()->start_, dataset()->stop_, dataset()->step_);
      }
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      int64 value;
      if (split_provider_ != nullptr) {
        Tensor split;
        TF_RETURN_IF_ERROR(split_provider_->GetNext(&split, end_of_sequence));
        if (*end_of_sequence) {
          return Status::OK();
        }
        value = split.scalar<int64>()();
      } else {
        value = counter_->GetNext(end_of_sequence);
        if (*end_of_sequence) {
          return Status::OK();
        }
      }
      out_tensors->reserve(1);
      switch (dataset()->output_dtypes()[0]) {
#define HANDLE_TYPE(type)                                \
  case DataTypeToEnum<type>::value: {                    \
    out_tensors->emplace_back(static_cast<type>(value)); \
    break;                                               \
  }
        TF_CALL_NUMBER_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
        default:
          return errors::InvalidArgument(
              "Unsupported data type: ",
              DataTypeString(dataset()->output_dtypes()[0]));
      }
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      if (split_provider_) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kHasSplitProvider), true));
        TF_RETURN_IF_ERROR(split_provider_->Save(
            [this](const std::string& key) {
              return SplitProviderKeyNameFn(key);
            },
            writer));
      } else {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kNext), counter_->Peek()));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      if (reader->Contains(full_name(kHasSplitProvider))) {
        TF_RETURN_IF_ERROR(split_provider_->Restore(
            [this](const std::string& key) {
              return SplitProviderKeyNameFn(key);
            },
            reader));
      } else {
        int64 next;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kNext), &next));
        counter_->SetNext(next);
      }
      return Status::OK();
    }

    std::string SplitProviderKeyNameFn(const std::string& key) {
      return full_name(absl::StrCat(kSplitProvider, kSlash, key));
    }

   private:
    std::unique_ptr<RangeCounter> counter_;
    std::shared_ptr<SplitProvider> split_provider_;
  };

  const int64 start_;
  const int64 stop_;
  const int64 step_;
  const DataTypeVector output_dtypes_;
};

RangeDatasetOp::RangeDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
}

void RangeDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
  int64 start;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kStart, &start));

  int64 stop;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kStop, &stop));

  int64 step;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kStep, &step));
  OP_REQUIRES(ctx, step != 0,
              errors::InvalidArgument("step must be a non-zero integer."));

  *output = new Dataset(ctx, start, stop, step, output_types_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("RangeDataset").Device(DEVICE_CPU),
                        RangeDatasetOp);
}  // namespace

}  // namespace data
}  // namespace tensorflow
