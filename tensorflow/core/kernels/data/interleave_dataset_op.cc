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
#include "tensorflow/core/kernels/data/interleave_dataset_op.h"

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const InterleaveDatasetOp::kDatasetType;
/* static */ constexpr const char* const InterleaveDatasetOp::kInputDataset;
/* static */ constexpr const char* const InterleaveDatasetOp::kOtherArguments;
/* static */ constexpr const char* const InterleaveDatasetOp::kCycleLength;
/* static */ constexpr const char* const InterleaveDatasetOp::kBlockLength;
/* static */ constexpr const char* const InterleaveDatasetOp::kFunc;
/* static */ constexpr const char* const InterleaveDatasetOp::kTarguments;
/* static */ constexpr const char* const InterleaveDatasetOp::kOutputTypes;
/* static */ constexpr const char* const InterleaveDatasetOp::kOutputShapes;

constexpr char kCycleIndex[] = "cycle_index";
constexpr char kBlockIndex[] = "block_index";
constexpr char kEndOfInput[] = "end_of_input";
constexpr char kNumOpen[] = "num_open";
constexpr char kArgsSize[] = "args_size";
constexpr char kArgsList[] = "args_list_";

class InterleaveDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          std::unique_ptr<CapturedFunction> captured_func, int64 cycle_length,
          int64 block_length, const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        captured_func_(std::move(captured_func)),
        cycle_length_(cycle_length),
        block_length_(block_length),
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

  Status CheckExternalState() const override {
    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_node;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
    Node* cycle_length_node;
    TF_RETURN_IF_ERROR(b->AddScalar(cycle_length_, &cycle_length_node));
    Node* block_length_node;
    TF_RETURN_IF_ERROR(b->AddScalar(block_length_, &block_length_node));
    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));
    AttrValue f;
    b->BuildAttrValue(captured_func_->func(), &f);
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {{0, input_node}, {2, cycle_length_node}, {3, block_length_node}},
        {{1, other_arguments}},
        {{kFunc, f}, {kTarguments, other_arguments_types_attr}}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          current_elements_(params.dataset->cycle_length_),
          args_list_(params.dataset->cycle_length_) {}

    string BuildTraceMeName() override {
      return strings::StrCat(prefix(),
                             "#cycle_length=", dataset()->cycle_length_,
                             ",block_length=", dataset()->block_length_, "#");
    }

    Status Initialize(IteratorContext* ctx) override {
      TF_RETURN_IF_ERROR(
          dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
      return dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_);
    }

    void AdvanceToNextInCycle() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      block_index_ = 0;
      cycle_index_ = (cycle_index_ + 1) % dataset()->cycle_length_;
    }

    void AdvancePosition() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      ++block_index_;
      if (block_index_ == dataset()->block_length_) {
        AdvanceToNextInCycle();
      }
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      while (!end_of_input_ || num_open_ > 0) {
        if (current_elements_[cycle_index_]) {
          // We are currently processing a mapped element, so try to get the
          // next subelement.
          bool end_of_element;
          TF_RETURN_IF_ERROR(current_elements_[cycle_index_]->GetNext(
              ctx, out_tensors, &end_of_element));
          if (!end_of_element) {
            // Produce the subelement as output.
            AdvancePosition();
            *end_of_sequence = false;
            return Status::OK();
          }
          // We have reached the end of the current element, so move
          // on to the next element in the cycle.
          current_elements_[cycle_index_].reset();
          args_list_[cycle_index_].clear();
          --num_open_;
          AdvanceToNextInCycle();
        } else if (!end_of_input_) {
          // Get the next element from the input dataset, and create
          // an iterator from it.
          TF_RETURN_IF_ERROR(input_impl_->GetNext(
              ctx, &args_list_[cycle_index_], &end_of_input_));
          if (!end_of_input_) {
            TF_RETURN_IF_ERROR(MakeIteratorFromInputElement(
                ctx, args_list_[cycle_index_], cycle_index_,
                *instantiated_captured_func_, prefix(),
                &current_elements_[cycle_index_]));
            ++num_open_;
          }
        } else {
          AdvanceToNextInCycle();
        }
      }

      *end_of_sequence = true;
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeInterleaveManyNode(std::move(args));
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kCycleIndex), cycle_index_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kBlockIndex), block_index_));
      if (end_of_input_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kEndOfInput), ""));
      }
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kNumOpen), num_open_));
      TF_RETURN_IF_ERROR(SaveCurrentElements(writer));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      int64 cycle_index;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kCycleIndex), &cycle_index));
      cycle_index_ = size_t(cycle_index);
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kBlockIndex), &block_index_));
      if (reader->Contains(full_name(kEndOfInput))) end_of_input_ = true;
      int64 num_open;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kNumOpen), &num_open));
      num_open_ = size_t(num_open);
      TF_RETURN_IF_ERROR(RestoreCurrentElements(ctx, reader));
      return Status::OK();
    }

   private:
    Status SaveCurrentElements(IteratorStateWriter* writer)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      for (int idx = 0; idx < current_elements_.size(); idx++) {
        if (current_elements_[idx]) {
          TF_RETURN_IF_ERROR(SaveInput(writer, current_elements_[idx]));
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(kArgsSize, "[", idx, "]")),
              args_list_[idx].size()));
          for (int i = 0; i < args_list_[idx].size(); i++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat(kArgsList, "[", idx, "][", i, "]")),
                args_list_[idx][i]));
          }
        }
      }
      return Status::OK();
    }

    Status RestoreCurrentElements(IteratorContext* ctx,
                                  IteratorStateReader* reader)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      for (int idx = 0; idx < current_elements_.size(); idx++) {
        if (reader->Contains(
                full_name(strings::StrCat(kArgsSize, "[", idx, "]")))) {
          int64 args_size;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              full_name(strings::StrCat(kArgsSize, "[", idx, "]")),
              &args_size));
          args_list_[idx].resize(args_size);
          for (int i = 0; i < args_size; i++) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                full_name(strings::StrCat(kArgsList, "[", idx, "][", i, "]")),
                &args_list_[idx][i]));
          }
          TF_RETURN_IF_ERROR(MakeIteratorFromInputElement(
              ctx, args_list_[idx], idx, *instantiated_captured_func_, prefix(),
              &current_elements_[idx]));
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, current_elements_[idx]));
        } else {
          current_elements_[idx].reset();
        }
      }
      return Status::OK();
    }

    mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    std::vector<std::unique_ptr<IteratorBase>> current_elements_
        GUARDED_BY(mu_);
    std::vector<std::vector<Tensor>> args_list_ GUARDED_BY(mu_);
    size_t cycle_index_ GUARDED_BY(mu_) = 0;
    int64 block_index_ GUARDED_BY(mu_) = 0;
    bool end_of_input_ GUARDED_BY(mu_) = false;
    size_t num_open_ GUARDED_BY(mu_) = 0;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
  };

  const DatasetBase* const input_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const int64 cycle_length_;
  const int64 block_length_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

InterleaveDatasetOp::InterleaveDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
  FunctionMetadata::Params params;
  params.is_multi_device_function = true;
  OP_REQUIRES_OK(ctx,
                 FunctionMetadata::Create(ctx, kFunc, params, &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void InterleaveDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                      DatasetBase** output) {
  int64 cycle_length = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kCycleLength, &cycle_length));
  if (cycle_length == model::kAutotune) {
    cycle_length = port::MaxParallelism();
  }
  OP_REQUIRES(
      ctx, cycle_length > 0,
      errors::InvalidArgument("cycle_length must be greater than zero."));

  int64 block_length = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kBlockLength, &block_length));
  OP_REQUIRES(
      ctx, block_length > 0,
      errors::InvalidArgument("block_length must be greater than zero."));

  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));

  *output = new Dataset(ctx, input, std::move(captured_func), cycle_length,
                        block_length, output_types_, output_shapes_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("InterleaveDataset").Device(DEVICE_CPU),
                        InterleaveDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("InterleaveDataset");
}  // namespace
}  // namespace data
}  // namespace tensorflow
