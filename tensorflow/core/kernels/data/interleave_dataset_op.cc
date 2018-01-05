
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
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class InterleaveDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit InterleaveDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("other_arguments", &inputs));
    std::vector<Tensor> other_arguments;
    other_arguments.reserve(inputs.size());
    for (const Tensor& t : inputs) {
      other_arguments.push_back(t);
    }

    const Tensor* cycle_length_t;
    OP_REQUIRES_OK(ctx, ctx->input("cycle_length", &cycle_length_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(cycle_length_t->shape()),
                errors::InvalidArgument("cycle_length must be a scalar."));
    const int64 cycle_length = cycle_length_t->flat<int64>()(0);
    OP_REQUIRES(
        ctx, cycle_length > 0,
        errors::InvalidArgument("cycle_length must be greater than zero."));

    const Tensor* block_length_t;
    OP_REQUIRES_OK(ctx, ctx->input("block_length", &block_length_t));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(block_length_t->shape()),
                errors::InvalidArgument("block_length must be a scalar."));
    const int64 block_length = block_length_t->flat<int64>()(0);
    OP_REQUIRES(
        ctx, block_length > 0,
        errors::InvalidArgument("block_length must be greater than zero."));

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, func_, graph_def_version_,
                                                 std::move(other_arguments),
                                                 &captured_func));

    *output =
        new Dataset(ctx, input, func_, std::move(captured_func), cycle_length,
                    block_length, output_types_, output_shapes_);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const NameAttrList& func,
            std::unique_ptr<CapturedFunction> captured_func, int64 cycle_length,
            int64 block_length, const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : GraphDatasetBase(ctx),
          input_(input),
          func_(func),
          captured_func_(std::move(captured_func)),
          cycle_length_(cycle_length),
          block_length_(block_length),
          output_types_(output_types),
          output_shapes_(output_shapes) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::Interleave")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() override { return "InterleaveDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, func_.name()));
      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input_, &input_node));
      Node* cycle_length_node;
      TF_RETURN_IF_ERROR(b->AddScalar(cycle_length_, &cycle_length_node));
      Node* block_length_node;
      TF_RETURN_IF_ERROR(b->AddScalar(block_length_, &block_length_node));
      DataTypeVector other_arguments_types;
      other_arguments_types.reserve(captured_func_->captured_inputs().size());
      std::vector<Node*> other_arguments;
      other_arguments.reserve(captured_func_->captured_inputs().size());
      for (const Tensor& t : captured_func_->captured_inputs()) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
        other_arguments.emplace_back(node);
        other_arguments_types.emplace_back(t.dtype());
      }
      AttrValue f;
      b->BuildAttrValue(func_, &f);
      AttrValue other_arguments_types_attr;
      b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this,
          {{0, input_node}, {2, cycle_length_node}, {3, block_length_node}},
          {{1, other_arguments}},
          {{"f", f}, {"Targuments", other_arguments_types_attr}}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            input_impl_(params.dataset->input_->MakeIterator(params.prefix)),
            current_elements_(params.dataset->cycle_length_),
            args_list_(params.dataset->cycle_length_) {}

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
              TF_RETURN_IF_ERROR(dataset::MakeIteratorFromInputElement(
                  ctx, args_list_[cycle_index_], cycle_index_,
                  dataset()->captured_func_.get(), prefix(),
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
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveParent(writer, input_impl_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("cycle_index"), cycle_index_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("block_index"), block_index_));
        if (end_of_input_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("end_of_input"), ""));
        }
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("num_open"), num_open_));
        TF_RETURN_IF_ERROR(SaveCurrentElements(writer));
        return Status::OK();
      }

      Status RestoreInternal(OpKernelContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreParent(ctx, reader, input_impl_));
        int64 cycle_index;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("cycle_index"), &cycle_index));
        cycle_index_ = size_t(cycle_index);
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("block_index"), &block_index_));
        if (reader->Contains(full_name("end_of_input"))) end_of_input_ = true;
        int64 num_open;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("num_open"), &num_open));
        num_open_ = size_t(num_open);
        TF_RETURN_IF_ERROR(RestoreCurrentElements(ctx, reader));
        return Status::OK();
      }

     private:
      Status SaveCurrentElements(IteratorStateWriter* writer)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        for (int idx = 0; idx < current_elements_.size(); idx++) {
          if (current_elements_[idx]) {
            TF_RETURN_IF_ERROR(SaveParent(writer, current_elements_[idx]));
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("args_size[", idx, "]")),
                args_list_[idx].size()));
            for (int i = 0; i < args_list_[idx].size(); i++) {
              TF_RETURN_IF_ERROR(writer->WriteTensor(
                  full_name(strings::StrCat("args_list_[", idx, "][", i, "]")),
                  args_list_[idx][i]));
            }
          }
        }
        return Status::OK();
      }

      Status RestoreCurrentElements(OpKernelContext* ctx,
                                    IteratorStateReader* reader)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        IteratorContext::Params params;
        params.env = ctx->env();
        params.runner = *(ctx->runner());
        IteratorContext iter_ctx(std::move(params));
        for (int idx = 0; idx < current_elements_.size(); idx++) {
          if (reader->Contains(
                  full_name(strings::StrCat("args_size[", idx, "]")))) {
            int64 args_size;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("args_size[", idx, "]")),
                &args_size));
            args_list_[idx].resize(args_size);
            for (int i = 0; i < args_size; i++) {
              TF_RETURN_IF_ERROR(reader->ReadTensor(
                  full_name(strings::StrCat("args_list_[", idx, "][", i, "]")),
                  &args_list_[idx][i]));
            }
            TF_RETURN_IF_ERROR(dataset::MakeIteratorFromInputElement(
                &iter_ctx, args_list_[idx], idx,
                dataset()->captured_func_.get(), prefix(),
                &current_elements_[idx]));
            TF_RETURN_IF_ERROR(
                RestoreParent(ctx, reader, current_elements_[idx]));
          } else {
            current_elements_[idx].reset();
          }
        }
        return Status::OK();
      }

      mutex mu_;
      const std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      std::vector<std::unique_ptr<IteratorBase>> current_elements_
          GUARDED_BY(mu_);
      std::vector<std::vector<Tensor>> args_list_ GUARDED_BY(mu_);
      size_t cycle_index_ GUARDED_BY(mu_) = 0;
      int64 block_index_ GUARDED_BY(mu_) = 0;
      bool end_of_input_ GUARDED_BY(mu_) = false;
      size_t num_open_ GUARDED_BY(mu_) = 0;
    };

    const DatasetBase* const input_;
    const NameAttrList func_;
    const std::unique_ptr<CapturedFunction> captured_func_;
    const int64 cycle_length_;
    const int64 block_length_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList func_;
};

REGISTER_KERNEL_BUILDER(Name("InterleaveDataset").Device(DEVICE_CPU),
                        InterleaveDatasetOp);

}  // namespace

}  // namespace tensorflow
