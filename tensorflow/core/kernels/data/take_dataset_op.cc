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

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"

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

TakeDataset::TakeDataset(OpKernelContext* ctx, int64_t count,
                         const DatasetBase* input)
    : DatasetBase(DatasetContext(ctx)), count_(count), input_(input) {
  input_->Ref();
}

TakeDataset::TakeDataset(DatasetContext::Params params, int64_t count,
                         const DatasetBase* input)
    : DatasetBase(DatasetContext(std::move(params))),
      count_(count),
      input_(input) {
  input_->Ref();
  random_indexing_compatible_ = absl::OkStatus();
  if (input_ != nullptr) {
    random_indexing_compatible_ = input_->RandomIndexingCompatible();
  }
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

int64_t TakeDataset::CardinalityInternal(CardinalityOptions options) const {
  int64_t n = input_->Cardinality(options);
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

absl::Status TakeDataset::InputDatasets(
    std::vector<const DatasetBase*>* inputs) const {
  inputs->push_back(input_);
  return absl::OkStatus();
}

absl::Status TakeDataset::CheckExternalState() const {
  return input_->CheckExternalState();
}

absl::Status TakeDataset::Get(OpKernelContext* ctx, int64 index,
                              std::vector<Tensor>* out_tensors) const {
  TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
  return input_->Get(ctx, index, out_tensors);
}

absl::Status TakeDataset::RandomIndexingCompatible() const {
  return random_indexing_compatible_;
}

class TakeDataset::EmptyIterator : public DatasetIterator<TakeDataset> {
 public:
  explicit EmptyIterator(const Params& params)
      : DatasetIterator<TakeDataset>(params) {}

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

class TakeDataset::FiniteIterator : public DatasetIterator<TakeDataset> {
 public:
  explicit FiniteIterator(const Params& params)
      : DatasetIterator<TakeDataset>(params), i_(0) {}

  bool SymbolicCheckpointCompatible() const override { return true; }

  absl::Status Initialize(IteratorContext* ctx) override {
    return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
  }

  absl::Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override {
    mutex_lock l(mu_);  // TODO(mrry): Make locking less conservative.
    if (!input_impl_) {
      *end_of_sequence = true;
      return absl::OkStatus();
    }
    while (dataset()->count_ < 0 || i_ < dataset()->count_) {
      TF_RETURN_IF_ERROR(
          input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
      if (!*end_of_sequence) {
        ++i_;
        return absl::OkStatus();
      }
      break;
    }
    *end_of_sequence = true;
    input_impl_.reset();
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
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kCurIndex, i_));
    TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kInputImplEmpty,
                                           static_cast<int64_t>(!input_impl_)));
    if (input_impl_) {
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
    }
    return absl::OkStatus();
  }

  absl::Status RestoreInternal(IteratorContext* ctx,
                               IteratorStateReader* reader) override {
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

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.
std::unique_ptr<IteratorBase> TakeDataset::MakeIteratorInternal(
    const string& prefix) const {
  if (count_ == 0) {
    return std::make_unique<EmptyIterator>(EmptyIterator::Params{
        this, name_utils::IteratorPrefix(kEmptyTake, prefix)});
  } else {
    return std::make_unique<FiniteIterator>(FiniteIterator::Params{
        this, name_utils::IteratorPrefix(kFiniteTake, prefix)});
  }
}

absl::Status TakeDataset::AsGraphDefInternal(SerializationContext* ctx,
                                             DatasetGraphDefBuilder* b,
                                             Node** output) const {
  Node* input_graph_node = nullptr;
  TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
  Node* count = nullptr;
  TF_RETURN_IF_ERROR(b->AddScalar(count_, &count));
  TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node, count}, output));
  return absl::OkStatus();
}

TakeDatasetOp::TakeDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {}

void TakeDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                DatasetBase** output) {
  // Create a new TakeDatasetOp::Dataset, and return it as the output.
  int64_t count;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kCount, &count));
  *output = new TakeDataset(ctx, count, input);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("TakeDataset").Device(DEVICE_CPU), TakeDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
