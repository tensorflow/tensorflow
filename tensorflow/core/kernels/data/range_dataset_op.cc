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

constexpr char kNext[] = "next";

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
    explicit Iterator(const Params& params) : DatasetIterator<Dataset>(params) {
      next_ = params.dataset->start_;
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      if ((dataset()->step_ > 0 && next_ >= dataset()->stop_) ||
          (dataset()->step_ < 0 && next_ <= dataset()->stop_)) {
        *end_of_sequence = true;
        return Status::OK();
      }
      out_tensors->reserve(1);
      Tensor result(dataset()->output_dtypes()[0]);
      switch (dataset()->output_dtypes()[0]) {
#define HANDLE_TYPE(type)                                \
  case DataTypeToEnum<type>::value: {                    \
    out_tensors->emplace_back(static_cast<type>(next_)); \
    break;                                               \
  }
        TF_CALL_NUMBER_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
        default:
          return errors::InvalidArgument(
              "Unsupported data type: ",
              DataTypeString(dataset()->output_dtypes()[0]));
      }
      *end_of_sequence = false;
      next_ += dataset()->step_;

      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kNext), next_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kNext), &next_));
      return Status::OK();
    }

   private:
    mutex mu_;
    int64 next_ GUARDED_BY(mu_);
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
