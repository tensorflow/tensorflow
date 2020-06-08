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

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/name_utils.h"

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
    }
  }

  ~Dataset() override {
    for (const auto& input : inputs_) {
      input->Unref();
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
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

  int64 Cardinality() const override {
    int64 result = kInfiniteCardinality;
    for (const auto& input : inputs_) {
      int64 n = input->Cardinality();
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

  Status CheckExternalState() const override {
    for (const auto& input : inputs_) {
      TF_RETURN_IF_ERROR(input->CheckExternalState());
    }
    return Status::OK();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
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
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      input_impls_.resize(dataset()->inputs_.size());
      for (size_t i = 0; i < input_impls_.size(); ++i) {
        TF_RETURN_IF_ERROR(dataset()->inputs_[i]->MakeIterator(
            ctx, this, strings::StrCat(prefix(), "[", i, "]"),
            &input_impls_[i]));
      }
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      if (input_impls_.empty()) {
        *end_of_sequence = true;
        return Status::OK();
      }
      out_tensors->clear();
      out_tensors->reserve(dataset()->output_dtypes().size());
      Status status = Status::OK();
      *end_of_sequence = false;
      for (const auto& input_impl : input_impls_) {
        std::vector<Tensor> input_tensors;
        bool component_end_of_sequence = false;
        status.Update(input_impl->GetNext(ctx, &input_tensors,
                                          &component_end_of_sequence));
        *end_of_sequence |= component_end_of_sequence;
        // Even if an error is encountered for one of the components,
        // we need to make sure to advance all components, to keep them in sync.
        if (!status.ok()) {
          continue;
        }
        if (*end_of_sequence) {
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

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      if (input_impls_.empty()) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kInputImplsEmpty), ""));
      } else {
        for (auto& input_impl : input_impls_)
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      if (reader->Contains(full_name(kInputImplsEmpty))) {
        input_impls_.clear();
      } else {
        DCHECK_EQ(input_impls_.size(), dataset()->inputs_.size());
        for (auto& input_impl : input_impls_)
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl));
      }
      return Status::OK();
    }

   private:
    mutex mu_;
    std::vector<std::unique_ptr<IteratorBase>> input_impls_ TF_GUARDED_BY(mu_);
  };

  const std::vector<DatasetBase*> inputs_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
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
