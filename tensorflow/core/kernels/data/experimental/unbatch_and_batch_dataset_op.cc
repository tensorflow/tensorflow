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
#include <atomic>
#include <utility>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/stats_utils.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kDatasetName[] = "ExperimentalUnbatchAndBatch";

class UnbatchAndBatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit UnbatchAndBatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 batch_size = 0;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "batch_size", &batch_size));
    OP_REQUIRES(
        ctx, batch_size > 0,
        errors::InvalidArgument("batch_size must be greater than zero."));

    bool drop_remainder;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument(ctx, "drop_remainder", &drop_remainder));

    *output = new Dataset(ctx, input, batch_size, drop_remainder,
                          output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            int64 batch_size,
            bool drop_remainder,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          batch_size_(batch_size),
          drop_remainder_(drop_remainder),
          output_types_(output_types),
          output_shapes_(output_shapes) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::", kDatasetName)});
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "UnbatchAndBatchDatasetOp::Dataset";
    }

    int64 Cardinality() const override {
      int64 n = input_->Cardinality();
      if (n == kInfiniteCardinality || n == kUnknownCardinality) {
        return n;
      }
      return n / batch_size_ +
             (n % batch_size_ == 0 || drop_remainder_ ? 0 : 1);
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
      Node* batch_size_node;
      TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size_node));
      Node* drop_remainder_node;
      TF_RETURN_IF_ERROR(b->AddScalar(drop_remainder_, &drop_remainder_node));

      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {input_graph_node, batch_size_node, drop_remainder_node}, output));
      return Status::OK();
    }

    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params)
          , current_index_(0)
	  , current_batch_size_(0) {}

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

        int64 chunk_read = 0;

        out_tensors->clear();
        std::vector<Tensor> elements;

        while (!*end_of_sequence) {
          if (current_index_ < current_batch_size_) {
            if (out_tensors->size() == 0) {

              out_tensors->reserve(tensors_.size());
              elements.reserve(tensors_.size());
              for (size_t i = 0; i < tensors_.size(); ++i) {
                TensorShape shape = tensors_[i].shape();

                shape.RemoveDim(0);
                elements.emplace_back(ctx->allocator({}), tensors_[i].dtype(), shape);

                shape.InsertDim(0, dataset()->batch_size_);
                out_tensors->emplace_back(ctx->allocator({}), tensors_[i].dtype(), shape);
              }
            }

            if (out_tensors->size() != tensors_.size()) {
              return errors::InvalidArgument("number tensors should match previous one, ", tensors_.size(), " vs. ", out_tensors->size());
            }

            int64 chunk_to_read = (current_batch_size_ - current_index_) < (dataset()->batch_size_ - chunk_read) ? (current_batch_size_ - current_index_) : (dataset()->batch_size_ - chunk_read);
            for (int i = 0; i < tensors_.size(); ++i) {
              // TODO: concurrent copy?
              for (int64 r = 0; r < chunk_to_read; ++r) {
                TF_RETURN_IF_ERROR(batch_util::MaybeMoveSliceToElement(
                    &tensors_[i], &elements[i], current_index_ + r));
                TF_RETURN_IF_ERROR(batch_util::CopyElementToSlice(
                    elements[i], &(*out_tensors)[i], chunk_read + r));
              }
            }

            chunk_read += chunk_to_read;
            current_index_ += chunk_to_read;
            if (chunk_read == dataset()->batch_size_) {
              *end_of_sequence = false;
              return Status::OK();
            }
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
            }
            current_batch_size_ = tensors_[0].dim_size(0);
          }
        }

        // Finally, resize if needed
        if (chunk_read > 0) {
          if (chunk_read < dataset()->batch_size_) {
            // No need to resieze with drop_reminder
            if (dataset()->drop_remainder_) {
              out_tensors->clear();
              input_impl_.reset();
              *end_of_sequence = true;
              return Status::OK();
	    }
            for (int i = 0; i < out_tensors->size(); ++i) {
              TensorShape shape = (*out_tensors)[i].shape();
              shape.set_dim(0, chunk_read);
              Tensor value_tensor;
	      value_tensor.CopyFrom((*out_tensors)[i], shape);
              (*out_tensors)[i] = std::move(value_tensor);
            }
          }
          *end_of_sequence = false;
          return Status::OK();
        }
        out_tensors->clear();
        input_impl_.reset();
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args), dataset()->batch_size_);
      }

     private:
      mutex mu_;
      int64 current_index_ GUARDED_BY(mu_);
      int64 current_batch_size_ GUARDED_BY(mu_);
      std::vector<Tensor> tensors_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    const int64 batch_size_;
    const bool drop_remainder_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(
    Name("ExperimentalUnbatchAndBatchDataset").Device(DEVICE_CPU),
    UnbatchAndBatchDatasetOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("ExperimentalUnbatchAndBatchDataset");

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
