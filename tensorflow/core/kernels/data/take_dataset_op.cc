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
#include "tensorflow/core/kernels/data/take_dataset_op.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/name_utils.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const TakeDatasetOp::kDatasetType;
/* static */ constexpr const char* const TakeDatasetOp::kInputDataset;
/* static */ constexpr const char* const TakeDatasetOp::kCount;
/* static */ constexpr const char* const TakeDatasetOp::kOutputTypes;
/* static */ constexpr const char* const TakeDatasetOp::kOutputShapes;

constexpr char kCurIndex[] = "i";
constexpr char kInputImplEmpty[] = "input_impl_empty";
constexpr char kEmptyTake[] = "EmptyTake";
constexpr char kFiniteTake[] = "FiniteTake";

TakeDataset::TakeDataset(OpKernelContext* ctx, int64 count,
                         const DatasetBase* input)
    : DatasetBase(DatasetContext(ctx)), count_(count), input_(input) {
  input_->Ref();
}

TakeDataset::TakeDataset(DatasetContext::Params params, int64 count,
                         const DatasetBase* input)
    : DatasetBase(DatasetContext(std::move(params))),
      count_(count),
      input_(input) {
  input_->Ref();
}

TakeDataset::~TakeDataset() { input_->Unref(); }

const DataTypeVector& TakeDataset::output_dtypes() const {
  return input_->output_dtypes();
}

const std::vector<PartialTensorShape>& TakeDataset::output_shapes() const {
  return input_->output_shapes();
}

string TakeDataset::DebugString() const {
  return name_utils::DatasetDebugString(TakeDatasetOp::kDatasetType);
}

int64 TakeDataset::Cardinality() const {
  int64 n = input_->Cardinality();
  if (n == kUnknownCardinality) {
    return kUnknownCardinality;
  }
  if (n == kInfiniteCardinality) {
    return count_;
  } else if (count_ == kInfiniteCardinality) {
    return n;
  }

  return std::min(n, count_);
}

Status TakeDataset::CheckExternalState() const {
  return input_->CheckExternalState();
}

class TakeDataset::EmptyIterator : public DatasetIterator<TakeDataset> {
 public:
  explicit EmptyIterator(const Params& params)
      : DatasetIterator<TakeDataset>(params) {}
  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    *end_of_sequence = true;
    return Status::OK();
  }

 protected:
  std::shared_ptr<model::Node> CreateNode(
      IteratorContext* ctx, model::Node::Args args) const override {
    return model::MakeKnownRatioNode(std::move(args),
                                     /*ratio=*/1);
  }

  Status SaveInternal(IteratorStateWriter* writer) override {
    return Status::OK();
  }

  Status RestoreInternal(IteratorContext* ctx,
                         IteratorStateReader* reader) override {
    return Status::OK();
  }
};

class TakeDataset::FiniteIterator : public DatasetIterator<TakeDataset> {
 public:
  explicit FiniteIterator(const Params& params)
      : DatasetIterator<TakeDataset>(params), i_(0) {}

  Status Initialize(IteratorContext* ctx) override {
    return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
  }

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
    mutex_lock l(mu_);  // TODO(mrry): Make locking less conservative.
    if (!input_impl_) {
      *end_of_sequence = true;
      return Status::OK();
    }
    while (dataset()->count_ < 0 || i_ < dataset()->count_) {
      TF_RETURN_IF_ERROR(
          input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
      if (!*end_of_sequence) {
        ++i_;
        return Status::OK();
      }
      break;
    }
    *end_of_sequence = true;
    input_impl_.reset();
    return Status::OK();
  }

 protected:
  std::shared_ptr<model::Node> CreateNode(
      IteratorContext* ctx, model::Node::Args args) const override {
    return model::MakeKnownRatioNode(std::move(args),
                                     /*ratio=*/1);
  }

  Status SaveInternal(IteratorStateWriter* writer) override {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurIndex), i_));
    if (input_impl_) {
      TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
    } else {
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kInputImplEmpty), ""));
    }
    return Status::OK();
  }

  Status RestoreInternal(IteratorContext* ctx,
                         IteratorStateReader* reader) override {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurIndex), &i_));
    if (!reader->Contains(full_name(kInputImplEmpty))) {
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
    } else {
      input_impl_.reset();
    }
    return Status::OK();
  }

 private:
  mutex mu_;
  int64 i_ GUARDED_BY(mu_);
  std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
};

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.
std::unique_ptr<IteratorBase> TakeDataset::MakeIteratorInternal(
    const string& prefix) const {
  if (count_ == 0) {
    return absl::make_unique<EmptyIterator>(EmptyIterator::Params{
        this, name_utils::IteratorPrefix(kEmptyTake, prefix)});
  } else {
    return absl::make_unique<FiniteIterator>(FiniteIterator::Params{
        this, name_utils::IteratorPrefix(kFiniteTake, prefix)});
  }
}

Status TakeDataset::AsGraphDefInternal(SerializationContext* ctx,
                                       DatasetGraphDefBuilder* b,
                                       Node** output) const {
  Node* input_graph_node = nullptr;
  TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
  Node* count = nullptr;
  TF_RETURN_IF_ERROR(b->AddScalar(count_, &count));
  TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node, count}, output));
  return Status::OK();
}

TakeDatasetOp::TakeDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}

void TakeDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                DatasetBase** output) {
  // Create a new TakeDatasetOp::Dataset, and return it as the output.
  int64 count;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64>(ctx, kCount, &count));
  *output = new TakeDataset(ctx, count, input);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("TakeDataset").Device(DEVICE_CPU), TakeDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
