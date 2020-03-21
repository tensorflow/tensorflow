/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

inline int64 CeilDiv(int64 dividend, int64 divisor) {
  return (dividend - 1 + divisor) / divisor;
}

constexpr const char* const kDatasetType = "Rebatch";

class RebatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit RebatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 num_replicas;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument(ctx, "num_replicas", &num_replicas));
    OP_REQUIRES(
        ctx, num_replicas > 0,
        errors::InvalidArgument("num_replicas must be greater than zero."));
    *output =
        new Dataset(ctx, input, num_replicas, output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const int64 num_replicas, const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          num_replicas_(num_replicas),
          output_types_(output_types),
          output_shapes_(output_shapes),
          traceme_metadata_(
              {{"num_replicas", strings::Printf("%lld", static_cast<long long>(
                                                            num_replicas))}}) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      name_utils::IteratorPrefixParams params;
      return absl::make_unique<Iterator>(Iterator::Params{
          this, name_utils::IteratorPrefix(kDatasetType, prefix, params)});
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      name_utils::DatasetDebugStringParams params;
      params.set_args(num_replicas_);
      return name_utils::DatasetDebugString(kDatasetType, params);
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
      Node* num_replicas = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(num_replicas_, &num_replicas));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {input_graph_node, num_replicas}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      ~Iterator() override {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        *end_of_sequence = false;
        if (slice_number_ % dataset()->num_replicas_ == 0) {
          input_descriptors_.clear();
          std::vector<Tensor> input_tensors;
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, &input_tensors, end_of_sequence));
          if (*end_of_sequence) {
            return Status::OK();
          }

          input_descriptors_.reserve(input_tensors.size());
          for (int i = 0; i < input_tensors.size(); ++i) {
            if (input_tensors[i].dims() == 0) {
              return errors::InvalidArgument(
                  "Cannot rebatch dataset: All components must have at least "
                  "one dimension. Perhaps your input dataset is not batched? "
                  "Component ",
                  i, " is scalar.");
            }

            int64 original_batch_dim = input_tensors[i].dim_size(0);
            int64 interval =
                CeilDiv(original_batch_dim, dataset()->num_replicas_);
            input_descriptors_.push_back(
                {std::move(input_tensors[i]), original_batch_dim, interval});
          }
        }

        out_tensors->reserve(input_descriptors_.size());

        // We slice each component independently because they may have
        // different batch dimensions.
        for (const auto& input_desc : input_descriptors_) {
          int64 start = input_desc.interval * slice_number_;
          int64 end = std::min(start + input_desc.interval,
                               input_desc.original_batch_dim);
          if (start >= end) {
            // We can get here if ceil(original_batch_dim_ / new batch dim) <
            // num_replicas_, i.e. the batch isn't big enough to distribute
            // over num replicas. In this case, we return empty tensors for
            // the remaining iterations that correspond to this batch.
            start = end;
          }
          Tensor slice = input_desc.whole_tensor.Slice(start, end);
          if (slice.IsAligned()) {
            out_tensors->push_back(std::move(slice));
          } else {
            out_tensors->push_back(tensor::DeepCopy(std::move(slice)));
          }
        }
        slice_number_ = (slice_number_ + 1) % dataset()->num_replicas_;
        return Status::OK();
      }

     protected:
      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (!input_impl_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impl_empty"), ""));
        } else {
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        }
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("slice_number"), slice_number_));

        if (slice_number_ % dataset()->num_replicas_ != 0) {
          // Save state of input tensors.
          for (int i = 0; i < input_descriptors_.size(); ++i) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat("tensors[", i, "]")),
                input_descriptors_[i].whole_tensor));
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
            reader->ReadScalar(full_name("slice_number"), &slice_number_));

        input_descriptors_.clear();
        input_descriptors_.resize(dataset()->output_dtypes().size());
        if (slice_number_ % dataset()->num_replicas_ != 0) {
          for (int i = 0; i < input_descriptors_.size(); ++i) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                full_name(strings::StrCat("tensors[", i, "]")),
                &input_descriptors_[i].whole_tensor));
            input_descriptors_[i].original_batch_dim =
                input_descriptors_[i].whole_tensor.dim_size(0);
            input_descriptors_[i].interval =
                CeilDiv(input_descriptors_[i].original_batch_dim,
                        dataset()->num_replicas_);
          }
        }
        return Status::OK();
      }

      TraceMeMetadata GetTraceMeMetadata() const override {
        return dataset()->traceme_metadata_;
      }

     private:
      // Describes one component of the input.
      struct InputDescriptor {
        InputDescriptor() {}
        InputDescriptor(Tensor&& whole_tensor, int64 original_batch_dim,
                        int64 interval)
            : whole_tensor(std::move(whole_tensor)),
              original_batch_dim(original_batch_dim),
              interval(interval) {}

        Tensor whole_tensor;
        int64 original_batch_dim;
        int64 interval;
      };

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_;
      std::vector<InputDescriptor> input_descriptors_ TF_GUARDED_BY(mu_);
      int64 slice_number_ TF_GUARDED_BY(mu_) = 0;
    };

    const DatasetBase* const input_;
    const int64 num_replicas_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const TraceMeMetadata traceme_metadata_;
  };

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("RebatchDataset").Device(DEVICE_CPU),
                        RebatchDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalRebatchDataset").Device(DEVICE_CPU),
                        RebatchDatasetOp);

}  // anonymous namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
