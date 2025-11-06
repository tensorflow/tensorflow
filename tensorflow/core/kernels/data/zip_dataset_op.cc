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
#include "tensorflow/core/kernels/data/zip_dataset_op.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const ZipDatasetOp::kDatasetType;
/* static */ constexpr const char* const ZipDatasetOp::kInputDatasets;
/* static */ constexpr const char* const ZipDatasetOp::kOutputTypes;
/* static */ constexpr const char* const ZipDatasetOp::kOutputShapes;
/* static */ constexpr const char* const ZipDatasetOp::kNumInputDatasets;

constexpr char kInputImplsEmpty[] = "input_impls_empty";

class ZipDatasetOp::Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext* ctx,
                   const std::vector<DatasetBase*>& inputs)
      : DatasetBase(DatasetContext(ctx)), inputs_(inputs) {
    for (const auto& input : inputs_) {
      input->Ref();
      for (DataType dt : input->output_dtypes()) {
        output_dtypes_.push_back(dt);
      }
      output_shapes_.insert(output_shapes_.end(),
                            input->output_shapes().begin(),
                            input->output_shapes().end());

      if (input != nullptr && random_indexing_compatible_.ok() &&
          !input->RandomIndexingCompatible().ok()) {
        random_indexing_compatible_ = input->RandomIndexingCompatible();
      }
    }
  }

  ~Dataset() override {
    for (const auto& input : inputs_) {
      input->Unref();
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  absl::Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>*
                                      split_providers) const override {
    TF_ASSIGN_OR_RETURN(*split_providers, GetSplitProviders(this));
    return absl::OkStatus();
  }

  const DataTypeVector& output_dtypes() const override {
    return output_dtypes_;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    int64_t result = kInfiniteCardinality;
    for (const auto& input : inputs_) {
      int64_t n = input->Cardinality(options);
      if (n == kUnknownCardinality) {
        return kUnknownCardinality;
      }
      if (n != kInfiniteCardinality &&
          (result == kInfiniteCardinality || n < result)) {
        result = n;
      }
    }
    return result;
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    for (const auto& input : inputs_) {
      inputs->push_back(input);
    }
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override {
    for (const auto& input : inputs_) {
      TF_RETURN_IF_ERROR(input->CheckExternalState());
    }
    return absl::OkStatus();
  }

  absl::Status Get(OpKernelContext* ctx, int64 index,
                   std::vector<Tensor>* out_tensors) const override {
    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    out_tensors->reserve(output_dtypes().size());
    for (int i = 0; i < inputs_.size(); ++i) {
      std::vector<Tensor> input_tensors;
      TF_RETURN_IF_ERROR(inputs_[i]->Get(ctx, index, &input_tensors));
      out_tensors->insert(out_tensors->end(), input_tensors.begin(),
                          input_tensors.end());
    }
    return absl::OkStatus();
  }

  absl::Status RandomIndexingCompatible() const override {
    return random_indexing_compatible_;
  }

 protected:
  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    std::vector<Node*> input_graph_nodes;
    input_graph_nodes.reserve(inputs_.size());
    for (const auto& input : inputs_) {
      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input, &input_node));
      input_graph_nodes.emplace_back(input_node);
    }
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {}, {std::make_pair(0, input_graph_nodes)}, {}, output));
    return absl::OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    bool SymbolicCheckpointCompatible() const override { return true; }

    absl::Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      TF_ASSIGN_OR_RETURN(input_contexts_,
                          CreateInputIteratorContexts(ctx, dataset()));
      input_impls_.resize(dataset()->inputs_.size());
      for (size_t i = 0; i < input_impls_.size(); ++i) {
        TF_RETURN_IF_ERROR(dataset()->inputs_[i]->MakeIterator(
            &input_contexts_[i], this, absl::StrCat(prefix(), "[", i, "]"),
            &input_impls_[i]));
        ctx->MergeCheckpoint(input_contexts_[i].checkpoint());
      }
      return absl::OkStatus();
    }

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      mutex_lock l(mu_);
      if (input_impls_.empty()) {
        *end_of_sequence = true;
        return absl::OkStatus();
      }
      out_tensors->clear();
      out_tensors->reserve(dataset()->output_dtypes().size());
      absl::Status status = absl::OkStatus();
      *end_of_sequence = false;

      if (TF_PREDICT_FALSE(ctx->index_mapper() && !input_contexts_.empty() &&
                           input_contexts_.back().index_mapper() == nullptr)) {
        for (IteratorContext& input_context : input_contexts_) {
          input_context.SetIndexMapper(ctx->index_mapper());
        }
      }

      for (int i = 0; i < input_impls_.size(); ++i) {
        const auto& input_impl = input_impls_[i];
        std::vector<Tensor> input_tensors;
        bool component_end_of_sequence = false;
        status.Update(input_impl->GetNext(&input_contexts_[i], &input_tensors,
                                          &component_end_of_sequence));
        ctx->MergeCheckpoint(input_contexts_[i].checkpoint());
        *end_of_sequence |= component_end_of_sequence;
        // Even if an error is encountered for one of the components,
        // we need to make sure to advance all components, to keep them in sync.
        if (!status.ok()) {
          continue;
        }
        if (*end_of_sequence) {
          // Fetch one last time from each input so that we call GetNext the
          // same number of times for each input. This will finalize caches
          // when cached datasets of the same size are zipped together.
          for (int j = i + 1; j < input_impls_.size(); ++j) {
            absl::Status s =
                input_impls_[j]->GetNext(&input_contexts_[j], &input_tensors,
                                         &component_end_of_sequence);
            ctx->MergeCheckpoint(input_contexts_[j].checkpoint());
          }
          break;
        }
        out_tensors->insert(out_tensors->end(), input_tensors.begin(),
                            input_tensors.end());
      }
      if (*end_of_sequence || !status.ok()) {
        out_tensors->clear();
      }
      if (*end_of_sequence) {
        input_impls_.clear();
      }
      return status;
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      // NOTE: Although this dataset may have multiple inputs, it always
      // consumes one element per input to produce an output.
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kInputImplsEmpty,
                              static_cast<int64_t>(input_impls_.empty())));
      for (auto& input_impl : input_impls_) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl));
      }
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      // Note: When restoring, `SaveInternal` would not be called
      // if there is a global_shuffle_dataset_op.cc above this op.
      int64_t inputs_empty;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(prefix(), kInputImplsEmpty, &inputs_empty));
      if (ctx->restored_element_count()) {
        if (input_impls_.size() != dataset()->inputs_.size()) {
          return absl::FailedPreconditionError(
              "`Initialize` should be called before restoring from the "
              "checkpoint.");
        }
        if (ctx->index_mapper() == nullptr) {
          return absl::FailedPreconditionError(
              "ctx->index_mapper() should be provided along with "
              "ctx->restored_element_count() when restoring.");
        }
        if (static_cast<bool>(inputs_empty)) {
          input_impls_.clear();
        } else {
          for (int i = 0; i < input_impls_.size(); ++i) {
            input_contexts_[i].set_restored_element_count(
                ctx->restored_element_count().value());
            TF_RETURN_IF_ERROR(
                RestoreInput(&input_contexts_[i], reader, input_impls_[i]));
            ctx->MergeCheckpoint(input_contexts_[i].checkpoint());
          }
        }
        return absl::OkStatus();
      }
      if (static_cast<bool>(inputs_empty)) {
        input_impls_.clear();
      } else {
        DCHECK_EQ(input_impls_.size(), dataset()->inputs_.size());
        for (auto& input_impl : input_impls_)
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl));
      }
      return absl::OkStatus();
    }

   private:
    mutex mu_;
    std::vector<std::unique_ptr<IteratorBase>> input_impls_ TF_GUARDED_BY(mu_);
    std::vector<IteratorContext> input_contexts_ TF_GUARDED_BY(mu_);
  };

  const std::vector<DatasetBase*> inputs_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  absl::Status random_indexing_compatible_ = absl::OkStatus();
};

ZipDatasetOp::ZipDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {}

void ZipDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
  std::vector<DatasetBase*> inputs;
  for (size_t i = 0; i < ctx->num_inputs(); ++i) {
    DatasetBase* input;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(i), &input));
    inputs.push_back(input);
  }
  *output = new Dataset(ctx, inputs);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ZipDataset").Device(DEVICE_CPU), ZipDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
