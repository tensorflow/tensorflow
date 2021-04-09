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
#include "tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr const char* const
    DirectedInterleaveDatasetOp::kDatasetType;
/* static */ constexpr const char* const
    DirectedInterleaveDatasetOp::kSelectorInputDataset;
/* static */ constexpr const char* const
    DirectedInterleaveDatasetOp::kDataInputDatasets;
/* static */ constexpr const char* const
    DirectedInterleaveDatasetOp::kOutputTypes;
/* static */ constexpr const char* const
    DirectedInterleaveDatasetOp::kOutputShapes;
/* static */ constexpr const char* const
    DirectedInterleaveDatasetOp::kNumInputDatasets;

class DirectedInterleaveDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* selector_input,
          std::vector<DatasetBase*> data_inputs)
      : DatasetBase(DatasetContext(ctx)),
        selector_input_(selector_input),
        data_inputs_(std::move(data_inputs)) {
    selector_input_->Ref();

    output_shapes_ = data_inputs_[0]->output_shapes();
    data_inputs_[0]->Ref();
    for (size_t i = 1; i < data_inputs_.size(); ++i) {
      const DatasetBase* data_input = data_inputs_[i];
      data_input->Ref();
      for (size_t j = 0; j < output_shapes_.size(); ++j) {
        output_shapes_[j] = MostSpecificCompatibleShape(
            output_shapes_[j], data_input->output_shapes()[j]);
      }
    }
  }

  ~Dataset() override {
    selector_input_->Unref();
    for (DatasetBase* data_input : data_inputs_) {
      data_input->Unref();
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
    return data_inputs_[0]->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64 Cardinality() const override {
    // As long as one of input dataset has infinite cardinality, the output
    // cardinality is infinite.
    for (const auto& input : data_inputs_) {
      int64 n = input->Cardinality();
      if (n == kInfiniteCardinality) {
        return n;
      }
    }
    return kUnknownCardinality;
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(selector_input_);
    for (const auto& data_input : data_inputs_) {
      inputs->push_back(data_input);
    }
    return Status::OK();
  }

  Status CheckExternalState() const override {
    for (const auto& input : data_inputs_) {
      TF_RETURN_IF_ERROR(input->CheckExternalState());
    }
    return selector_input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* selector_input_node;
    TF_RETURN_IF_ERROR(
        b->AddInputDataset(ctx, selector_input_, &selector_input_node));
    std::vector<Node*> data_input_nodes(data_inputs_.size());
    for (size_t i = 0; i < data_inputs_.size(); ++i) {
      TF_RETURN_IF_ERROR(
          b->AddInputDataset(ctx, data_inputs_[i], &data_input_nodes[i]));
    }
    TF_RETURN_IF_ERROR(b->AddDataset(this, {{0, selector_input_node}},
                                     {{1, data_input_nodes}}, {}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          num_active_inputs_(params.dataset->data_inputs_.size()) {}

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(dataset()->selector_input_->MakeIterator(
          ctx, this, prefix(), &selector_input_impl_));
      data_input_impls_.resize(dataset()->data_inputs_.size());
      for (size_t i = 0; i < data_input_impls_.size(); ++i) {
        const DatasetBase* data_input = dataset()->data_inputs_[i];
        TF_RETURN_IF_ERROR(data_input->MakeIterator(
            ctx, this, strings::StrCat(prefix(), "[", i, "]"),
            &data_input_impls_[i]));
      }
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      if (!selector_input_impl_) {
        *end_of_sequence = true;
        return Status::OK();
      }

      while (true) {
        std::vector<Tensor> selector_result;
        *end_of_sequence = false;
        TF_RETURN_IF_ERROR(selector_input_impl_->GetNext(ctx, &selector_result,
                                                         end_of_sequence));
        if (*end_of_sequence) {
          selector_input_impl_.reset();
          for (auto& data_input_impl : data_input_impls_) {
            data_input_impl.reset();
          }
          return Status::OK();
        }

        int64 selected_input = selector_result[0].scalar<int64>()();
        if (selected_input < 0 || selected_input >= data_input_impls_.size()) {
          return errors::InvalidArgument(
              "Selector index out of range: ", selected_input,
              " >= ", data_input_impls_.size());
        }

        if (data_input_impls_[selected_input]) {
          bool end_of_selected_input = false;
          TF_RETURN_IF_ERROR(data_input_impls_[selected_input]->GetNext(
              ctx, out_tensors, &end_of_selected_input));

          if (!end_of_selected_input) {
            return Status::OK();
          }

          data_input_impls_[selected_input].reset();
          --num_active_inputs_;

          if (num_active_inputs_ == 0) {
            selector_input_impl_.reset();
            *end_of_sequence = true;
            return Status::OK();
          }
        }

        VLOG(2) << "DirectedInterleave selected an exhausted input: "
                << selected_input;
      }
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeInterleaveManyNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      if (selector_input_impl_) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, selector_input_impl_));
      } else {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("selector_input_impl_empty"), ""));
      }
      for (size_t i = 0; i < data_input_impls_.size(); ++i) {
        const auto& data_input_impl = data_input_impls_[i];
        if (data_input_impl) {
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, data_input_impl));
        } else {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat("data_input_impl_empty[", i, "]")),
              ""));
        }
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      if (!reader->Contains(full_name("selector_input_impl_empty"))) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, selector_input_impl_));
      } else {
        selector_input_impl_.reset();
      }
      for (size_t i = 0; i < data_input_impls_.size(); ++i) {
        if (!reader->Contains(
                full_name(strings::StrCat("data_input_impl_empty[", i, "]")))) {
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, data_input_impls_[i]));
        } else {
          data_input_impls_[i].reset();
        }
      }
      return Status::OK();
    }

   private:
    mutex mu_;
    std::unique_ptr<IteratorBase> selector_input_impl_ TF_GUARDED_BY(mu_);
    std::vector<std::unique_ptr<IteratorBase>> data_input_impls_
        TF_GUARDED_BY(mu_);
    int64 num_active_inputs_ TF_GUARDED_BY(mu_);
  };

  static PartialTensorShape MostSpecificCompatibleShape(
      const PartialTensorShape& ts1, const PartialTensorShape& ts2) {
    PartialTensorShape output_tensorshape;
    if (ts1.dims() != ts2.dims() || ts1.unknown_rank() || ts2.unknown_rank())
      return output_tensorshape;
    auto dims1 = ts1.dim_sizes();
    auto dims2 = ts2.dim_sizes();
    for (int d = 0; d < ts1.dims(); ++d) {
      if (dims1[d] == dims2[d])
        output_tensorshape.Concatenate(dims1[d]);
      else
        output_tensorshape.Concatenate(-1);
    }
    return output_tensorshape;
  }

  const DatasetBase* const selector_input_;
  const std::vector<DatasetBase*> data_inputs_;
  std::vector<PartialTensorShape> output_shapes_;
};

DirectedInterleaveDatasetOp::DirectedInterleaveDatasetOp(
    OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {}

void DirectedInterleaveDatasetOp::MakeDataset(OpKernelContext* ctx,
                                              DatasetBase** output) {
  DatasetBase* selector_input;
  OP_REQUIRES_OK(ctx,
                 GetDatasetFromVariantTensor(ctx->input(0), &selector_input));

  OP_REQUIRES(
      ctx,
      selector_input->output_dtypes().size() == 1 &&
          selector_input->output_dtypes()[0] == DT_INT64 &&
          selector_input->output_shapes().size() == 1 &&
          selector_input->output_shapes()[0].IsCompatibleWith(
              PartialTensorShape({})),
      errors::InvalidArgument(
          "The selector input must be a dataset of scalar int64 elements."));

  std::vector<DatasetBase*> data_inputs;
  for (size_t i = 1; i < ctx->num_inputs(); ++i) {
    DatasetBase* input;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(i), &input));
    data_inputs.push_back(input);

    OP_REQUIRES(ctx, data_inputs[0]->output_dtypes() == input->output_dtypes(),
                errors::InvalidArgument(
                    "All inputs must have the same output_dtypes. First input "
                    "has types ",
                    DataTypeVectorString(data_inputs[0]->output_dtypes()),
                    ", and input ", i - 1, " has types ",
                    DataTypeVectorString(input->output_dtypes())));
  }
  *output = new Dataset(ctx, selector_input, std::move(data_inputs));
}

namespace {
REGISTER_KERNEL_BUILDER(Name("DirectedInterleaveDataset").Device(DEVICE_CPU),
                        DirectedInterleaveDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalDirectedInterleaveDataset").Device(DEVICE_CPU),
    DirectedInterleaveDatasetOp);
}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
