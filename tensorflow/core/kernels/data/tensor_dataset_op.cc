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
#include "tensorflow/core/kernels/data/tensor_dataset_op.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/global_shuffle_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const TensorDatasetOp::kDatasetType;
/* static */ constexpr const char* const TensorDatasetOp::kComponents;
/* static */ constexpr const char* const TensorDatasetOp::kToutput_types;
/* static */ constexpr const char* const TensorDatasetOp::kOutputShapes;

constexpr char kFromTensor[] = "FromTensor";
constexpr char kProduced[] = "produced";

class TensorDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, std::vector<Tensor> tensors)
      : DatasetBase(DatasetContext(ctx)), tensors_(std::move(tensors)) {
    dtypes_.reserve(tensors_.size());
    shapes_.reserve(tensors_.size());
    for (const Tensor& t : tensors_) {
      dtypes_.push_back(t.dtype());
      shapes_.emplace_back(t.shape().dim_sizes());
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kFromTensor, prefix)});
  }

  Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>*
                                split_providers) const override {
    split_providers->push_back(std::make_unique<IndexSplitProvider>(1));
    return absl::OkStatus();
  }

  const DataTypeVector& output_dtypes() const override { return dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return 1LL;
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    return absl::OkStatus();
  }

  Status CheckExternalState() const override { return absl::OkStatus(); }

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
    return Get(AnyContext(ctx), index, out_tensors);
  }

  Status Get(AnyContext ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    *out_tensors = tensors_;
    return absl::OkStatus();
  }

  absl::Status RandomIndexingCompatible() const override {
    return absl::OkStatus();
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
      } else {
        TF_RETURN_IF_ERROR(b->AddPlaceholder(t, &node));
        DCHECK_NE(ctx->input_list(), nullptr);
        ctx->input_list()->emplace_back(node->name(), t);
      }
      components.emplace_back(node);
    }
    AttrValue dtypes;
    b->BuildAttrValue(dtypes_, &dtypes);
    TF_RETURN_IF_ERROR(b->AddDataset(this, {}, {{0, components}},
                                     {{kToutput_types, dtypes}}, output));
    return absl::OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          produced_(false),
          global_shuffle_iterator_(dataset()) {}

    bool SymbolicCheckpointCompatible() const override { return true; }

    Status Initialize(IteratorContext* ctx) override {
      if (!ctx->split_providers().empty()) {
        TF_ASSIGN_OR_RETURN(split_provider_,
                            GetSingleSplitProvider(ctx, dataset()));
      }
      return absl::OkStatus();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      if (ctx->index_mapper() != nullptr) {
        return global_shuffle_iterator_.GetNext(ctx, out_tensors,
                                                end_of_sequence);
      }

      mutex_lock l(mu_);
      if (split_provider_) {
        bool end_of_splits;
        Tensor split;
        TF_RETURN_IF_ERROR(split_provider_->GetNext(&split, &end_of_splits));
        if (end_of_splits) {
          produced_ = true;
        }
      }
      if (!produced_) {
        *out_tensors = dataset()->tensors_;
        produced_ = true;
        *end_of_sequence = false;
        return absl::OkStatus();
      } else {
        *end_of_sequence = true;
        return absl::OkStatus();
      }
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kProduced,
                                             static_cast<int64_t>(produced_)));
      return absl::OkStatus();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      if (ctx->restored_element_count().has_value()) {
        return global_shuffle_iterator_.Restore(ctx);
      }

      mutex_lock l(mu_);
      int64_t produced;
      TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kProduced, &produced));
      produced_ = static_cast<bool>(produced);
      return absl::OkStatus();
    }

   private:
    mutex mu_;
    std::shared_ptr<SplitProvider> split_provider_;
    bool produced_ TF_GUARDED_BY(mu_);
    GlobalShuffleIterator global_shuffle_iterator_;
  };

  const std::vector<Tensor> tensors_;
  DataTypeVector dtypes_;
  std::vector<PartialTensorShape> shapes_;
};

TensorDatasetOp::TensorDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kToutput_types, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void TensorDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
  OpInputList inputs;
  OP_REQUIRES_OK(ctx, ctx->input_list(kComponents, &inputs));
  std::vector<Tensor> components(inputs.begin(), inputs.end());
  *output = new Dataset(ctx, std::move(components));
  OP_REQUIRES_OK(ctx,
                 VerifyTypesMatch((*output)->output_dtypes(), output_types_));
  OP_REQUIRES_OK(
      ctx, VerifyShapesCompatible((*output)->output_shapes(), output_shapes_));
}

namespace {
REGISTER_KERNEL_BUILDER(Name("TensorDataset").Device(DEVICE_CPU),
                        TensorDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
