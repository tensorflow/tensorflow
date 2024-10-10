/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/experimental/assert_cardinality_dataset_op.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"
#include "tsl/platform/errors.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr const char* const
    AssertCardinalityDatasetOp::kInputDataset;
/* static */ constexpr const char* const
    AssertCardinalityDatasetOp::kDatasetType;
/* static */ constexpr const char* const
    AssertCardinalityDatasetOp::kCardinality;
/* static */ constexpr const char* const
    AssertCardinalityDatasetOp::kOutputTypes;
/* static */ constexpr const char* const
    AssertCardinalityDatasetOp::kOutputShapes;

class AssertCardinalityDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int64_t cardinality,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        cardinality_(cardinality),
        output_types_(output_types),
        output_shapes_(output_shapes) {
    input_->Ref();
    random_indexing_compatible_ = absl::OkStatus();
    if (input_ != nullptr) {
      random_indexing_compatible_ = input_->RandomIndexingCompatible();
    }
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }
  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return cardinality_;
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

  absl::Status RandomIndexingCompatible() const override {
    return random_indexing_compatible_;
  }

  absl::Status Get(OpKernelContext* ctx, int64_t index,
                   std::vector<Tensor>* out_tensors) const override {
    return GetInternal(ctx, index, out_tensors);
  }

  absl::Status Get(AnyContext ctx, int64_t index,
                   std::vector<Tensor>* out_tensors) const override {
    return GetInternal(ctx, index, out_tensors);
  }

 protected:
  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* cardinality_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(cardinality_, &cardinality_node));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node, cardinality_node}, output));
    return absl::OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params), num_elements_(0) {}

    bool SymbolicCheckpointCompatible() const override { return true; }

    absl::Status Initialize(IteratorContext* ctx) override {
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      TF_RETURN_IF_ERROR(
          input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
      if (!*end_of_sequence) {
        num_elements_++;
      }
      if (*end_of_sequence && num_elements_ != dataset()->cardinality_) {
        if (ctx->index_mapper()) {
          return absl::FailedPreconditionError(
              absl::StrCat("Input dataset was expected to contain ",
                           ElementString(dataset()->cardinality_), "."));
        }
        return errors::FailedPrecondition(
            "Input dataset was expected to contain ",
            ElementString(dataset()->cardinality_), " but contained only ",
            ElementString(num_elements_), ".");
      }
      if (dataset()->cardinality_ != kInfiniteCardinality &&
          num_elements_ > dataset()->cardinality_) {
        return errors::FailedPrecondition(
            "Input dataset was expected to contain ",
            ElementString(dataset()->cardinality_), " but contained at least ",
            ElementString(num_elements_), ".");
      }
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
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name("num_elements"), num_elements_));
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name("num_elements"), &num_elements_));
      return RestoreInput(ctx, reader, input_impl_);
    }

   private:
    std::unique_ptr<IteratorBase> input_impl_;
    int64_t num_elements_;
  };

  static string ElementString(int64_t n) {
    if (n == kInfiniteCardinality) {
      return absl::StrCat("an infinite number of elements");
    }
    return absl::StrCat(n, " element", n != 1 ? "s" : "");
  }

  template <typename Ctx>
  absl::Status GetInternal(Ctx&& ctx, int64_t index,
                           std::vector<Tensor>* out_tensors) const {
    if (index < 0) {
      return absl::OutOfRangeError(
          absl::StrCat("Dataset index ", index, " must be non-negative"));
    }
    if (cardinality_ != kInfiniteCardinality && index >= cardinality_) {
      return absl::OutOfRangeError(absl::StrCat(
          "Index ", index, " greater than dataset size ", cardinality_));
    }

    absl::Status result =
        input_->Get(std::forward<Ctx>(ctx), index, out_tensors);

    if (absl::IsOutOfRange(result)) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Input dataset was expected to contain ", ElementString(cardinality_),
          " but index ", index, " could not be retrieved: ", result.message()));
    }

    return result;
  }

  const DatasetBase* input_;
  const int64_t cardinality_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  absl::Status random_indexing_compatible_;
};

AssertCardinalityDatasetOp::AssertCardinalityDatasetOp(
    OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void AssertCardinalityDatasetOp::MakeDataset(OpKernelContext* ctx,
                                             DatasetBase* input,
                                             DatasetBase** output) {
  int64_t cardinality;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, kCardinality, &cardinality));
  *output = new Dataset(ctx, input, cardinality, output_types_, output_shapes_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("AssertCardinalityDataset").Device(DEVICE_CPU),
                        AssertCardinalityDatasetOp);
}  // namespace

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
