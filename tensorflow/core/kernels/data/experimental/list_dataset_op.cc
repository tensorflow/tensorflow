/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/experimental/list_dataset_op.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const ListDatasetOp::kDatasetType;
/* static */ constexpr const char* const ListDatasetOp::kTensors;
/* static */ constexpr const char* const ListDatasetOp::kTinputTypes;
/* static */ constexpr const char* const ListDatasetOp::kOutputTypes;
/* static */ constexpr const char* const ListDatasetOp::kOutputShapes;

class ListDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, std::vector<Tensor> tensors,
          const DataTypeVector& input_types, const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes,
          int num_components)
      : DatasetBase(DatasetContext(ctx)),
        tensors_(std::move(tensors)),
        num_elements_(tensors_.size() / num_components),
        num_components_(num_components),
        input_types_(input_types),
        output_types_(output_types),
        output_shapes_(output_shapes) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>*
                                split_providers) const override {
    split_providers->push_back(
        std::make_unique<IndexSplitProvider>(num_elements_));
    return OkStatus();
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return num_elements_;
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    return OkStatus();
  }

  Status CheckExternalState() const override { return OkStatus(); }

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    out_tensors->clear();
    out_tensors->reserve(num_components_);
    for (int i = 0; i < num_components_; ++i) {
      out_tensors->push_back(tensors_[i + num_components_ * index]);
    }
    return OkStatus();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    std::vector<Node*> tensors;
    tensors.reserve(tensors_.size());
    for (const Tensor& t : tensors_) {
      Node* node;
      if (!ctx->is_graph_rewrite()) {
        TF_RETURN_IF_ERROR(b->AddDatasetOrTensor(ctx, t, &node));
      } else {
        TF_RETURN_IF_ERROR(b->AddPlaceholder(t, &node));
        DCHECK_NE(ctx->input_list(), nullptr);
        ctx->input_list()->emplace_back(node->name(), t);
      }
      tensors.emplace_back(node);
    }
    AttrValue input_types;
    b->BuildAttrValue(input_types_, &input_types);
    TF_RETURN_IF_ERROR(b->AddDataset(this, {}, {{0, tensors}},
                                     {{kTinputTypes, input_types}}, output));
    return OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    bool SymbolicCheckpointCompatible() const override { return true; }

    Status Initialize(IteratorContext* ctx) override {
      if (ctx->split_providers().empty()) {
        split_provider_ =
            std::make_shared<IndexSplitProvider>(dataset()->num_elements_);
      } else {
        TF_ASSIGN_OR_RETURN(split_provider_,
                            GetSingleSplitProvider(ctx, dataset()));
      }
      return OkStatus();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      Tensor split;
      TF_RETURN_IF_ERROR(split_provider_->GetNext(&split, end_of_sequence));
      if (*end_of_sequence) {
        return OkStatus();
      }
      int64_t index = split.scalar<int64_t>()();
      out_tensors->reserve(dataset()->num_components_);
      for (size_t i = 0; i < dataset()->num_components_; ++i) {
        out_tensors->push_back(
            dataset()->tensors_[i + dataset()->num_components_ * index]);
      }
      *end_of_sequence = false;
      return OkStatus();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      return split_provider_->Save(
          [this](const std::string& key) { return full_name(key); }, writer);
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return split_provider_->Restore(
          [this](const std::string& key) { return full_name(key); }, reader);
    }

   private:
    std::shared_ptr<SplitProvider> split_provider_;
  };

  const std::vector<Tensor> tensors_;
  int64 num_elements_;
  size_t num_components_;
  DataTypeVector input_types_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

ListDatasetOp::ListDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kTinputTypes, &input_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void ListDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
  OpInputList inputs;
  OP_REQUIRES_OK(ctx, ctx->input_list(kTensors, &inputs));
  std::vector<Tensor> tensors(inputs.begin(), inputs.end());
  *output = new Dataset(ctx, std::move(tensors), input_types_, output_types_,
                        output_shapes_, output_shapes_.size());
  OP_REQUIRES_OK(ctx,
                 VerifyTypesMatch((*output)->output_dtypes(), output_types_));
  OP_REQUIRES_OK(
      ctx, VerifyShapesCompatible((*output)->output_shapes(), output_shapes_));
}

namespace {

REGISTER_KERNEL_BUILDER(Name("ListDataset").Device(DEVICE_CPU), ListDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
