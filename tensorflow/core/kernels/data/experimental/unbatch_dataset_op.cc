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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/batch_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/status.h"
#include "tsl/platform/strcat.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

using tsl::mutex;
using tsl::mutex_lock;
using tsl::Status;
using tsl::strings::StrCat;

constexpr char kInputImplEmpty[] = "input_impl_empty";

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
      batch_size_ = -1;
      for (const PartialTensorShape& shape : input->output_shapes()) {
        if (!shape.unknown_rank()) {
          if (batch_size_ < 0 && shape.dim_size(0) >= 0) {
            batch_size_ = shape.dim_size(0);
          }
          gtl::InlinedVector<int64_t, 4> partial_dim_sizes;
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
        const std::string& prefix) const override {
      return std::make_unique<Iterator>(
          Iterator::Params{this, StrCat(prefix, "::Unbatch")});
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return shapes_;
    }

    std::string DebugString() const override {
      return "UnbatchDatasetOp::Dataset";
    }

    int64_t CardinalityInternal(CardinalityOptions options) const override {
      int64_t n = input_->Cardinality(options);
      if (n == kInfiniteCardinality || n == kUnknownCardinality) {
        return n;
      }
      if (batch_size_ > 0) {
        return n * batch_size_;
      }
      return kUnknownCardinality;
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      inputs->push_back(input_);
      return absl::OkStatus();
    }

    Status CheckExternalState() const override {
      return input_->CheckExternalState();
    }

    absl::Status Get(OpKernelContext* ctx, int64_t index,
                     std::vector<Tensor>* out_tensors) const override {
      TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
      if (batch_size_ <= 0) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Random access for the `unbatch` dataset requires a known batch "
            "size. Got ",
            batch_size_, "."));
      }

      const int64_t input_index = index / batch_size_;
      const int64_t input_offset = index % batch_size_;
      std::vector<Tensor> input_tensors;
      TF_RETURN_IF_ERROR(input_->Get(ctx, input_index, &input_tensors));
      for (int64_t i = 0; i < input_tensors.size(); ++i) {
        const DataType& dtype = input_tensors[i].dtype();
        TensorShape shape = input_tensors[i].shape();
        shape.RemoveDim(0);

        out_tensors->emplace_back(ctx->get_allocator({}), dtype, shape);
        TF_RETURN_IF_ERROR(batch_util::MaybeMoveSliceToElement(
            &input_tensors[i], &out_tensors->back(), input_offset));
      }
      return absl::OkStatus();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, output));
      return absl::OkStatus();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            current_index_(0),
            current_batch_size_(0),
            shapes_(params.dataset->output_shapes().size()) {}

      bool SymbolicCheckpointCompatible() const override { return true; }

      Status Initialize(IteratorContext* ctx) override {
        mutex_lock l(mu_);
        input_ckpt_ = std::make_unique<MemoryCheckpoint>(ctx->id_registry());
        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (!input_impl_) {
          *end_of_sequence = true;
          return absl::OkStatus();
        }
        *end_of_sequence = false;
        while (!*end_of_sequence) {
          if (current_index_ < current_batch_size_) {
            out_tensors->clear();
            out_tensors->reserve(tensors_.size());
            for (int i = 0; i < tensors_.size(); ++i) {
              // TODO(b/201790899): Investigate why using MaybeCopySubSlice
              // may lead to a memory leak.
              out_tensors->emplace_back(ctx->allocator({}), tensors_[i].dtype(),
                                        shapes_[i]);
              TF_RETURN_IF_ERROR(batch_util::MaybeMoveSliceToElement(
                  &tensors_[i], &out_tensors->back(), current_index_));
            }
            ++current_index_;
            *end_of_sequence = false;
            if (current_index_ >= current_batch_size_) {
              ctx->MergeCheckpoint(input_ckpt_.get());
            }
            return absl::OkStatus();
          }
          current_index_ = 0;
          current_batch_size_ = 0;
          tensors_.clear();
          auto input_ctx = std::make_unique<IteratorContext>(*ctx);
          TF_RETURN_IF_ERROR(input_impl_->GetNext(input_ctx.get(), &tensors_,
                                                  end_of_sequence));
          input_ckpt_->Merge(input_ctx->checkpoint());
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
        return absl::OkStatus();
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

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(kInputImplEmpty), static_cast<int64_t>(!input_impl_)));
        if (input_impl_) {
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        }
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("current_index"), current_index_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("n"), current_batch_size_));
        if (current_index_ < current_batch_size_ &&
            !ctx->symbolic_checkpoint()) {
          for (size_t i = 0; i < tensors_.size(); ++i) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(StrCat("tensors[", i, "]")), tensors_[i]));
          }
        }
        return absl::OkStatus();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        int64_t input_empty;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name(kInputImplEmpty), &input_empty));
        if (!static_cast<bool>(input_empty)) {
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
          TF_RETURN_IF_ERROR(RestoreTensors(ctx, reader));
        }
        return absl::OkStatus();
      }

     private:
      // Restores the `tensors_` field (and its associated `shapes_`) from a
      // checkpoint.
      Status RestoreTensors(IteratorContext* ctx, IteratorStateReader* reader)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (ctx->symbolic_checkpoint()) {
          bool end_of_sequence;
          auto input_ctx = std::make_unique<IteratorContext>(*ctx);
          TF_RETURN_IF_ERROR(input_impl_->GetNext(input_ctx.get(), &tensors_,
                                                  &end_of_sequence));
          input_ckpt_->Merge(input_ctx->checkpoint());
          if (end_of_sequence) {
            return errors::FailedPrecondition(
                "Unexpected end of sequence while symbolically restoring "
                " UnbatchDataset. Please verify that the input produces data "
                " deterministically.");
          }
        } else {
          for (size_t i = 0; i < tensors_.size(); ++i) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                ctx->flr(), full_name(StrCat("tensors[", i, "]")),
                &tensors_[i]));
          }
        }
        for (size_t i = 0; i < tensors_.size(); ++i) {
          shapes_[i] = tensors_[i].shape();
          shapes_[i].RemoveDim(0);
        }
        return absl::OkStatus();
      }

      mutex mu_;
      int64_t current_index_ TF_GUARDED_BY(mu_);
      int64_t current_batch_size_ TF_GUARDED_BY(mu_);
      std::vector<Tensor> tensors_ TF_GUARDED_BY(mu_);
      // Checkpoint to use for operations on input_impl_. We maintain a
      // separate checkpoint from the one passed to unbatch so that we can
      // control when symbolic checkpoint state will be propagated. In
      // particular, we wait to propagate input checkpoint state until the
      // tensors being unbatched have been fully consumed, so that if we need to
      // restore the partially-unbatched tensors, we can do so by restoring the
      // input and then calling GetNext() on it.
      std::unique_ptr<MemoryCheckpoint> input_ckpt_ TF_GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      std::vector<TensorShape> shapes_ TF_GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    std::vector<PartialTensorShape> shapes_;
    // batch_size_ may or may not be known, with -1 as unknown
    int64_t batch_size_;
  };
};

REGISTER_KERNEL_BUILDER(Name("UnbatchDataset").Device(DEVICE_CPU),
                        UnbatchDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalUnbatchDataset").Device(DEVICE_CPU),
                        UnbatchDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
