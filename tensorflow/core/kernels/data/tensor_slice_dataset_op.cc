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
#include "tensorflow/core/kernels/data/tensor_slice_dataset_op.h"

#include <string>
#include <utility>

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

/* static */ constexpr const char* const TensorSliceDatasetOp::kDatasetType;
/* static */ constexpr const char* const TensorSliceDatasetOp::kComponents;
/* static */ constexpr const char* const TensorSliceDatasetOp::kToutputTypes;
/* static */ constexpr const char* const TensorSliceDatasetOp::kOutputShapes;
/* static */ constexpr const char* const TensorSliceDatasetOp::kIsFiles;
/* static */ constexpr const char* const
    TensorSliceDatasetOp::kReplicateOnSplit;

class TensorSliceDatasetOp::Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext* ctx, std::vector<Tensor> tensors,
                   bool is_files, bool replicate_on_split)
      : DatasetBase(DatasetContext(ctx)),
        tensors_(std::move(tensors)),
        is_files_(is_files),
        replicate_on_split_(replicate_on_split) {
    for (const Tensor& t : tensors_) {
      dtypes_.push_back(t.dtype());
      gtl::InlinedVector<int64_t, 4> element_dim_sizes;
      // Handle scalar here. Check that everyone matches here? Or fail
      // at runtime?
      for (int i = 1; i < t.dims(); ++i) {
        element_dim_sizes.push_back(t.dim_size(i));
      }
      partial_shapes_.emplace_back(element_dim_sizes);
      shapes_.emplace_back(std::move(element_dim_sizes));
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>*
                                split_providers) const override {
    split_providers->push_back(
        std::make_unique<IndexSplitProvider>(tensors_[0].dim_size(0)));
    return OkStatus();
  }

  const DataTypeVector& output_dtypes() const override { return dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return partial_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return tensors_[0].dim_size(0);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    return OkStatus();
  }

  Status CheckExternalState() const override { return OkStatus(); }

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    out_tensors->clear();
    out_tensors->reserve(tensors_.size());
    for (int i = 0; i < tensors_.size(); ++i) {
      out_tensors->push_back(MaybeCopySubSlice(tensors_[i], index));
    }
    return OkStatus();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    std::vector<Node*> components;
    components.reserve(tensors_.size());
    for (const Tensor& t : tensors_) {
      Node* node;
      if (!ctx->is_graph_rewrite()) {
        TF_RETURN_IF_ERROR(b->AddDatasetOrTensor(ctx, t, &node));
        if (is_files_) {
          Node* file_node;
          TF_RETURN_IF_ERROR(
              b->AddIdentity(ctx, "FileIdentity", &node, &file_node));
        }
      } else {
        TF_RETURN_IF_ERROR(b->AddPlaceholder(t, &node));
        DCHECK_NE(ctx->input_list(), nullptr);
        ctx->input_list()->emplace_back(node->name(), t);
      }
      components.emplace_back(node);
    }
    AttrValue dtypes;
    b->BuildAttrValue(dtypes_, &dtypes);
    AttrValue is_files;
    b->BuildAttrValue(is_files_, &is_files);
    AttrValue replicate_on_split;
    b->BuildAttrValue(replicate_on_split_, &replicate_on_split);
    TF_RETURN_IF_ERROR(b->AddDataset(this, {}, {{0, components}},
                                     {{kToutputTypes, dtypes},
                                      {kIsFiles, is_files},
                                      {kReplicateOnSplit, replicate_on_split}},
                                     output));
    return OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    bool SymbolicCheckpointCompatible() const override { return true; }

    Status Initialize(IteratorContext* ctx) override {
      if (ctx->split_providers().empty() || dataset()->replicate_on_split_) {
        split_provider_ = std::make_shared<IndexSplitProvider>(
            dataset()->tensors_[0].dim_size(0));
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
      out_tensors->reserve(dataset()->tensors_.size());
      for (size_t i = 0; i < dataset()->tensors_.size(); ++i) {
        out_tensors->push_back(
            MaybeCopySubSlice(dataset()->tensors_[i], index));
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
  DataTypeVector dtypes_;
  std::vector<TensorShape> shapes_;
  std::vector<PartialTensorShape> partial_shapes_;
  const bool is_files_;
  const bool replicate_on_split_;
};

TensorSliceDatasetOp::TensorSliceDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kToutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  if (ctx->HasAttr(kIsFiles)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kIsFiles, &is_files_));
  }
  if (ctx->HasAttr(kReplicateOnSplit)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kReplicateOnSplit, &replicate_on_split_));
  }
}

void TensorSliceDatasetOp::MakeDataset(OpKernelContext* ctx,
                                       DatasetBase** output) {
  OpInputList inputs;
  OP_REQUIRES_OK(ctx, ctx->input_list(kComponents, &inputs));
  std::vector<Tensor> components;
  components.reserve(inputs.size());
  OP_REQUIRES(
      ctx, inputs[0].dims() > 0,
      errors::InvalidArgument("All components must be at least 1-dimensional"));
  const int64_t num_slices = inputs[0].dim_size(0);
  for (const Tensor& t : inputs) {
    components.push_back(t);
    OP_REQUIRES(ctx, t.dims() > 0,
                errors::InvalidArgument(
                    "All components must be at least 1-dimensional"));
    OP_REQUIRES(
        ctx, t.dim_size(0) == num_slices,
        errors::InvalidArgument(
            "All components must have the same size in the 0th dimension"));
  }
  *output =
      new Dataset(ctx, std::move(components), is_files_, replicate_on_split_);
  OP_REQUIRES_OK(ctx,
                 VerifyTypesMatch((*output)->output_dtypes(), output_types_));
  OP_REQUIRES_OK(
      ctx, VerifyShapesCompatible((*output)->output_shapes(), output_shapes_));
}

namespace {

REGISTER_KERNEL_BUILDER(Name("TensorSliceDataset").Device(DEVICE_CPU),
                        TensorSliceDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
