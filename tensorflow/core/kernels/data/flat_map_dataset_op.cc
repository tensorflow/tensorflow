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
namespace data {
namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class FlatMapDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit FlatMapDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(func_, ctx, "other_arguments",
                                                 &captured_func));
    *output = new Dataset(ctx, input, func_, std::move(captured_func),
                          output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const NameAttrList& func,
            std::unique_ptr<CapturedFunction> captured_func,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          func_(func),
          captured_func_(std::move(captured_func)),
          output_types_(output_types),
          output_shapes_(output_shapes) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::FlatMap")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override { return "FlatMapDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, func_.name()));
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

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
          this, {std::make_pair(0, input_graph_node)},  // Single tensor inputs.
          {std::make_pair(1, other_arguments)},         // Tensor list inputs.
          {std::make_pair("f", f),
           std::make_pair("Targuments", other_arguments_types_attr)},  // Attrs
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
            dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
        return dataset()->captured_func_->Instantiate(ctx);
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
            // We are currently precessing a mapped element, so try to get the
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
          TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &captured_func_inputs_,
                                                  end_of_sequence));
          if (*end_of_sequence) {
            input_impl_.reset();
            return Status::OK();
          }

          TF_RETURN_IF_ERROR(BuildCurrentElementIteratorLocked(ctx));
        } while (true);
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (input_impl_) {
          TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("element_index"), element_index_));
          if (current_element_iterator_) {
            TF_RETURN_IF_ERROR(
                writer->WriteScalar(full_name("captured_func_inputs_size"),
                                    captured_func_inputs_.size()));
            for (int i = 0; i < captured_func_inputs_.size(); i++) {
              TF_RETURN_IF_ERROR(writer->WriteTensor(
                  full_name(strings::StrCat("captured_func_inputs[", i, "]")),
                  captured_func_inputs_[i]));
            }
            TF_RETURN_IF_ERROR(SaveInput(writer, current_element_iterator_));
          } else {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name("current_element_iterator_uninitialized"), ""));
          }
        } else {
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("exhausted"), ""));
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
        if (!reader->Contains(full_name("exhausted"))) {
          TF_RETURN_IF_ERROR(
              dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
          {
            int64 temp;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(full_name("element_index"), &temp));
            element_index_ = temp;
          }
          if (!reader->Contains(
                  full_name("current_element_iterator_uninitialized"))) {
            size_t captured_func_inputs_size;
            {
              int64 temp;
              TF_RETURN_IF_ERROR(reader->ReadScalar(
                  full_name("captured_func_inputs_size"), &temp));
              captured_func_inputs_size = static_cast<size_t>(temp);
            }
            captured_func_inputs_.reserve(captured_func_inputs_size);
            for (int i = 0; i < captured_func_inputs_size; i++) {
              captured_func_inputs_.emplace_back();
              TF_RETURN_IF_ERROR(reader->ReadTensor(
                  full_name(strings::StrCat("captured_func_inputs[", i, "]")),
                  &captured_func_inputs_.back()));
            }
            element_index_--;
            TF_RETURN_IF_ERROR(BuildCurrentElementIteratorLocked(ctx));
            TF_RETURN_IF_ERROR(
                RestoreInput(ctx, reader, current_element_iterator_));
          }
        }
        return Status::OK();
      }

     private:
      Status BuildCurrentElementIteratorLocked(IteratorContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        return MakeIteratorFromInputElement(
            ctx, captured_func_inputs_, element_index_++,
            dataset()->captured_func_.get(), prefix(),
            &current_element_iterator_);
      }

      Status BuildCurrentElementIteratorLocked(OpKernelContext* ctx)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        IteratorContext::Params params;
        params.env = ctx->env();
        params.runner = *(ctx->runner());
        params.lib = ctx->function_library();
        IteratorContext iter_ctx(std::move(params));
        return BuildCurrentElementIteratorLocked(&iter_ctx);
      }

      mutex mu_;
      size_t element_index_ GUARDED_BY(mu_) = 0;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> current_element_iterator_ GUARDED_BY(mu_);
      std::vector<Tensor> captured_func_inputs_ GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    const NameAttrList func_;
    const std::unique_ptr<CapturedFunction> captured_func_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList func_;
};

REGISTER_KERNEL_BUILDER(Name("FlatMapDataset").Device(DEVICE_CPU),
                        FlatMapDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
