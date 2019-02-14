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
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

class UnbatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit UnbatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    *output = new Dataset(ctx, input);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, DatasetBase* input)
        : DatasetBase(DatasetContext(ctx)), input_(input) {
      input_->Ref();
      for (const PartialTensorShape& shape : input->output_shapes()) {
        if (!shape.unknown_rank()) {
          gtl::InlinedVector<int64, 4> partial_dim_sizes;
          for (int i = 1; i < shape.dims(); ++i) {
            partial_dim_sizes.push_back(shape.dim_size(i));
          }
          shapes_.emplace_back(std::move(partial_dim_sizes));
        } else {
          // If the input shape is unknown, the output shape will be unknown.
          shapes_.emplace_back();
        }
      }
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::Unbatch")});
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return shapes_;
    }

    string DebugString() const override { return "UnbatchDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            current_index_(0),
            current_batch_size_(0),
            shapes_(params.dataset->output_shapes().size()) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (!input_impl_) {
          *end_of_sequence = true;
          return Status::OK();
        }
        *end_of_sequence = false;
        while (!*end_of_sequence) {
          if (current_index_ < current_batch_size_) {
            out_tensors->clear();
            out_tensors->reserve(tensors_.size());
            for (int i = 0; i < tensors_.size(); ++i) {
              out_tensors->emplace_back(ctx->allocator({}), tensors_[i].dtype(),
                                        shapes_[i]);
              TF_RETURN_IF_ERROR(batch_util::MaybeMoveSliceToElement(
                  &tensors_[i], &out_tensors->back(), current_index_));
            }
            ++current_index_;
            *end_of_sequence = false;
            return Status::OK();
          }
          current_index_ = 0;
          current_batch_size_ = 0;
          tensors_.clear();
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, &tensors_, end_of_sequence));
          if (!*end_of_sequence) {
            for (size_t i = 0; i < tensors_.size(); ++i) {
              if (tensors_[i].dims() == 0) {
                return errors::InvalidArgument(
                    "Input element must have a non-scalar value in each "
                    "component.");
              }
              if (tensors_[i].dim_size(0) != tensors_[0].dim_size(0)) {
                return errors::InvalidArgument(
                    "Input element must have the same batch size in each "
                    "component. Component 0 had size ",
                    tensors_[0].dim_size(0), " but component ", i,
                    " had size, ", tensors_[i].dim_size(0), ".");
              }
              shapes_[i] = tensors_[i].shape();
              shapes_[i].RemoveDim(0);
            }
            current_batch_size_ = tensors_[0].dim_size(0);
          }
        }
        input_impl_.reset();
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        // Unbatch assumes that all input components have the same leading
        // dimension. If it is statically known for any component, we model the
        // transformation using `KnownRatio`. Otherwise, we use `UnknownRatio`.
        for (auto& shape : dataset()->input_->output_shapes()) {
          if (shape.dims() > 0 && shape.dim_size(0) > 0) {
            return model::MakeKnownRatioNode(
                std::move(args), 1.0 / static_cast<double>(shape.dim_size(0)));
          }
        }
        return model::MakeUnknownRatioNode(std::move(args));
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (input_impl_) {
          TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
        } else {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impl_empty"), ""));
        }
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("current_index"), current_index_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("n"), current_batch_size_));
        if (current_index_ < current_batch_size_) {
          for (size_t i = 0; i < tensors_.size(); ++i) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat("tensors[", i, "]")), tensors_[i]));
          }
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (!reader->Contains(full_name("input_impl_empty"))) {
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        } else {
          input_impl_.reset();
        }
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("current_index"), &current_index_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("n"), &current_batch_size_));
        tensors_.clear();
        tensors_.resize(dataset()->output_dtypes().size());
        if (current_index_ < current_batch_size_) {
          for (size_t i = 0; i < tensors_.size(); ++i) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                full_name(strings::StrCat("tensors[", i, "]")), &tensors_[i]));
            shapes_[i] = tensors_[i].shape();
            shapes_[i].RemoveDim(0);
          }
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      int64 current_index_ GUARDED_BY(mu_);
      int64 current_batch_size_ GUARDED_BY(mu_);
      std::vector<Tensor> tensors_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      std::vector<TensorShape> shapes_ GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    std::vector<PartialTensorShape> shapes_;
  };
};

REGISTER_KERNEL_BUILDER(Name("ExperimentalUnbatchDataset").Device(DEVICE_CPU),
                        UnbatchDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
