/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/experimental/assert_next_dataset_op.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr const char* const AssertNextDatasetOp::kInputDataset;
/* static */ constexpr const char* const AssertNextDatasetOp::kDatasetType;
/* static */ constexpr const char* const AssertNextDatasetOp::kTransformations;
/* static */ constexpr const char* const AssertNextDatasetOp::kOutputTypes;
/* static */ constexpr const char* const AssertNextDatasetOp::kOutputShapes;

class AssertNextDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          const std::vector<tstring>& transformations,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        transformations_(transformations),
        output_types_(output_types),
        output_shapes_(output_shapes) {
    input_->Ref();
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
    return input_->Cardinality(options);
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* transformations_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddVector(transformations_, &transformations_node));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node, transformations_node}, output));
    return absl::OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    absl::Status Initialize(IteratorContext* ctx) override {
      std::vector<string> tokens =
          absl::StrSplit(prefix(), ':', absl::SkipEmpty());
      if (dataset()->transformations_.size() > tokens.size() - 2) {
        return errors::InvalidArgument(
            "Asserted next ", dataset()->transformations_.size(),
            " transformations but encountered only ", tokens.size() - 2, ".");
      }
      int n = tokens.size();
      for (size_t i = 0; i < dataset()->transformations_.size(); ++i) {
        if (!MatchesAnyVersion(dataset()->transformations_[i],
                               tokens[n - 2 - i])) {
          return errors::InvalidArgument("Asserted transformation matching ",
                                         dataset()->transformations_[i],
                                         " at offset ", i, " but encountered ",
                                         tokens[n - 2 - i],
                                         " transformation instead.");
        }
      }
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      return absl::OkStatus();
    }

   private:
    std::unique_ptr<IteratorBase> input_impl_;
  };

  const DatasetBase* input_;
  const std::vector<tstring> transformations_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

AssertNextDatasetOp::AssertNextDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void AssertNextDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                      DatasetBase** output) {
  std::vector<tstring> transformations;
  OP_REQUIRES_OK(ctx, ParseVectorArgument<tstring>(ctx, kTransformations,
                                                   &transformations));
  *output =
      new Dataset(ctx, input, transformations, output_types_, output_shapes_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("AssertNextDataset").Device(DEVICE_CPU),
                        AssertNextDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalAssertNextDataset").Device(DEVICE_CPU),
    AssertNextDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
