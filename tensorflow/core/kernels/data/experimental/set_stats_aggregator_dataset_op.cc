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
#include <memory>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/data/stats_utils.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class StatsAggregatorWithTagAndPrefix : public StatsAggregator {
 public:
  StatsAggregatorWithTagAndPrefix(
      std::shared_ptr<StatsAggregator> stats_aggregator, const string& tag,
      const string& prefix)
      : wrapped_(stats_aggregator), tag_(tag), prefix_(prefix) {}

  void AddToHistogram(const string& name, gtl::ArraySlice<double> values,
                      int64 steps) override {
    wrapped_->AddToHistogram(TaggedName(name), values, steps);
  }

  void AddScalar(const string& name, float value, int64 steps) override {
    wrapped_->AddScalar(TaggedName(name), value, steps);
  }

  void EncodeToProto(Summary* out_summary) override {
    wrapped_->EncodeToProto(out_summary);
  }

  void IncrementCounter(const string& name, const string& label,
                        int64 val) override {
    if (!prefix_.empty()) {
      wrapped_->IncrementCounter(
          strings::StrCat(prefix_, "/", TaggedName(name)), label, val);
    } else {
      wrapped_->IncrementCounter(
          strings::StrCat("/tensorflow/", TaggedName(name)), label, val);
    }
  }

  Status SetSummaryWriter(SummaryWriterInterface* summary_writer) override {
    return wrapped_->SetSummaryWriter(summary_writer);
  }

 private:
  string TaggedName(const string& name) const {
    if (!tag_.empty()) {
      string tagged_name = strings::StrCat(tag_, stats_utils::kDelimiter, name);
      return tagged_name;
    }
    return name;
  }

  std::shared_ptr<StatsAggregator> wrapped_;
  string tag_;
  string prefix_;
  TF_DISALLOW_COPY_AND_ASSIGN(StatsAggregatorWithTagAndPrefix);
};

class SetStatsAggregatorDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit SetStatsAggregatorDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    core::RefCountPtr<StatsAggregatorResource> resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &resource));
    tstring tag;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "tag", &tag));
    tstring prefix;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "counter_prefix", &prefix));

    *output =
        new Dataset(ctx, input, ctx->input(1), resource.get(), tag, prefix);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, const DatasetBase* input,
                     const Tensor& resource_handle,
                     StatsAggregatorResource* resource, const string& tag,
                     const string& prefix)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          resource_handle_(resource_handle),
          stats_aggregator_resource_(resource),
          tag_(tag),
          prefix_(prefix) {
      input_->Ref();
      stats_aggregator_resource_->Ref();
    }

    ~Dataset() override {
      input_->Unref();
      stats_aggregator_resource_->Unref();
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(Iterator::Params{
          this, strings::StrCat(prefix, "::SetStatsAggregator")});
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override {
      return "SetStatsAggregatorDatasetOp::Dataset";
    }

    int64 Cardinality() const override { return input_->Cardinality(); }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      inputs->push_back(input_);
      return Status::OK();
    }

    Status CheckExternalState() const override {
      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node* resource_handle_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddTensor(resource_handle_, &resource_handle_node));
      Node* tag_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(tag_, &tag_node));
      Node* prefix_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(prefix_, &prefix_node));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {input_graph_node, resource_handle_node, tag_node, prefix_node},
          output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        IteratorContext iter_ctx = ContextWithAggregator(ctx);
        return dataset()->input_->MakeIterator(&iter_ctx, this, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        IteratorContext iter_ctx = ContextWithAggregator(ctx);
        return input_impl_->GetNext(&iter_ctx, out_tensors, end_of_sequence);
      }

      IteratorContext ContextWithAggregator(IteratorContext* ctx) {
        StatsAggregatorResource* resource =
            dataset()->stats_aggregator_resource_;
        IteratorContext::Params params(ctx);
        params.stats_aggregator = std::shared_ptr<StatsAggregator>(
            new StatsAggregatorWithTagAndPrefix(resource->stats_aggregator(),
                                                dataset()->tag_,
                                                dataset()->prefix_));
        IteratorContext iter_ctx(std::move(params));
        return iter_ctx;
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        return SaveInput(ctx, writer, input_impl_);
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        return RestoreInput(ctx, reader, input_impl_);
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    const Tensor resource_handle_;
    StatsAggregatorResource* stats_aggregator_resource_;
    tstring tag_;
    tstring prefix_;
  };
};

REGISTER_KERNEL_BUILDER(Name("SetStatsAggregatorDataset").Device(DEVICE_CPU),
                        SetStatsAggregatorDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalSetStatsAggregatorDataset").Device(DEVICE_CPU),
    SetStatsAggregatorDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
