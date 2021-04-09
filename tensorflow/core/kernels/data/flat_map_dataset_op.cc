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
#include "tensorflow/core/kernels/data/flat_map_dataset_op.h"

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const FlatMapDatasetOp::kDatasetType;
/* static */ constexpr const char* const FlatMapDatasetOp::kInputDataset;
/* static */ constexpr const char* const FlatMapDatasetOp::kOtherArguments;
/* static */ constexpr const char* const FlatMapDatasetOp::kFunc;
/* static */ constexpr const char* const FlatMapDatasetOp::kTarguments;
/* static */ constexpr const char* const FlatMapDatasetOp::kOutputTypes;
/* static */ constexpr const char* const FlatMapDatasetOp::kOutputShapes;

constexpr char kElementIndex[] = "element_index";
constexpr char kCapturedFuncInputsSize[] = "captured_func_inputs_size";
constexpr char kCapturedFuncInputs[] = "captured_func_inputs";
constexpr char kCurrentElementIteratorUninitialized[] =
    "current_element_iterator_uninitialized";
constexpr char kExhausted[] = "exhausted";

class FlatMapDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          std::unique_ptr<CapturedFunction> captured_func,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        captured_func_(std::move(captured_func)),
        output_types_(output_types),
        output_shapes_(output_shapes) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));
    AttrValue f;
    b->BuildAttrValue(captured_func_->func(), &f);
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {std::make_pair(0, input_graph_node)},  // Single tensor inputs.
        {std::make_pair(1, other_arguments)},         // Tensor list inputs.
        {std::make_pair(kFunc, f),
         std::make_pair(kTarguments, other_arguments_types_attr)},  // Attrs
        output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    Status Initialize(IteratorContext* ctx) override {
      TF_RETURN_IF_ERROR(
          dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
      return dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      do {
        if (!input_impl_) {
          *end_of_sequence = true;
          return Status::OK();
        }
        if (current_element_iterator_) {
          // We are currently processing a mapped element, so try to get the
          // next subelement.
          bool end_of_element;
          TF_RETURN_IF_ERROR(current_element_iterator_->GetNext(
              ctx, out_tensors, &end_of_element));
          if (!end_of_element) {
            // Produce the subelement as output.
            *end_of_sequence = false;
            return Status::OK();
          }

          // We have reached the end of the current element, so maybe move on
          // to the next element.
          current_element_iterator_.reset();
        }

        // Get the next element from the input dataset.
        captured_func_inputs_.clear();
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, &captured_func_inputs_, end_of_sequence));
        if (*end_of_sequence) {
          input_impl_.reset();
          return Status::OK();
        }

        TF_RETURN_IF_ERROR(
            BuildCurrentElementIteratorLocked(ctx, /*is_get_next=*/true));
      } while (true);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeInterleaveManyNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
          dataset()->captured_func_->CheckExternalState()));
      mutex_lock l(mu_);
      if (input_impl_) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kElementIndex), element_index_));
        if (current_element_iterator_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(kCapturedFuncInputsSize),
                                  captured_func_inputs_.size()));
          for (int i = 0; i < captured_func_inputs_.size(); i++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat(kCapturedFuncInputs, "[", i, "]")),
                captured_func_inputs_[i]));
          }
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, current_element_iterator_));
        } else {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(kCurrentElementIteratorUninitialized), ""));
        }
      } else {
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kExhausted), ""));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      input_impl_.reset();
      element_index_ = 0;
      current_element_iterator_.reset();
      captured_func_inputs_.clear();
      if (!reader->Contains(full_name(kExhausted))) {
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        {
          int64 temp;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name(kElementIndex), &temp));
          element_index_ = temp;
        }
        if (!reader->Contains(
                full_name(kCurrentElementIteratorUninitialized))) {
          size_t captured_func_inputs_size;
          {
            int64 temp;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(full_name(kCapturedFuncInputsSize), &temp));
            captured_func_inputs_size = static_cast<size_t>(temp);
          }
          captured_func_inputs_.reserve(captured_func_inputs_size);
          for (int i = 0; i < captured_func_inputs_size; i++) {
            captured_func_inputs_.emplace_back();
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                full_name(strings::StrCat(kCapturedFuncInputs, "[", i, "]")),
                &captured_func_inputs_.back()));
          }
          element_index_--;
          TF_RETURN_IF_ERROR(
              BuildCurrentElementIteratorLocked(ctx, /*is_get_next=*/false));
          TF_RETURN_IF_ERROR(
              RestoreInput(ctx, reader, current_element_iterator_));
        }
      }
      return Status::OK();
    }

   private:
    Status BuildCurrentElementIteratorLocked(IteratorContext* ctx,
                                             bool is_get_next)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (is_get_next) {
        return MakeIteratorFromInputElement(
            ctx, this, captured_func_inputs_, element_index_++,
            *instantiated_captured_func_, prefix(), &current_element_iterator_,
            model_node());
      } else {
        // NOTE: We intentionally ignore resource modeling outside GetNext().
        return MakeIteratorFromInputElement(
            ctx, this, captured_func_inputs_, element_index_++,
            *instantiated_captured_func_, prefix(), &current_element_iterator_,
            /*node=*/nullptr);
      }
    }

    mutex mu_;
    size_t element_index_ TF_GUARDED_BY(mu_) = 0;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> current_element_iterator_ TF_GUARDED_BY(mu_);
    std::vector<Tensor> captured_func_inputs_ TF_GUARDED_BY(mu_);
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
  };

  const DatasetBase* const input_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

FlatMapDatasetOp::FlatMapDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kFunc, /*params=*/{},
                                               &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void FlatMapDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                   DatasetBase** output) {
  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));
  *output = new Dataset(ctx, input, std::move(captured_func), output_types_,
                        output_shapes_);
}

namespace {

REGISTER_KERNEL_BUILDER(Name("FlatMapDataset").Device(DEVICE_CPU),
                        FlatMapDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("FlatMapDataset");

}  // namespace
}  // namespace data
}  // namespace tensorflow
