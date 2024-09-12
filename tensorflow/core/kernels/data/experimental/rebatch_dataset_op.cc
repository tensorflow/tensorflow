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

#include <optional>

#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

inline int64_t CeilDiv(int64_t dividend, int64_t divisor) {
  return (dividend - 1 + divisor) / divisor;
}

constexpr const char* const kDatasetTypeV1 = "Rebatch";
constexpr const char* const kDatasetTypeV2 = "RebatchV2";

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
    int64_t num_replicas;
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
            const int64_t num_replicas, const DataTypeVector& output_types,
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
      return std::make_unique<Iterator>(Iterator::Params{
          this, name_utils::IteratorPrefix(kDatasetTypeV1, prefix, params)});
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
      return name_utils::DatasetDebugString(kDatasetTypeV1, params);
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      inputs->push_back(input_);
      return absl::OkStatus();
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
      return absl::OkStatus();
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
            return absl::OkStatus();
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

            int64_t original_batch_dim = input_tensors[i].dim_size(0);
            int64_t interval =
                CeilDiv(original_batch_dim, dataset()->num_replicas_);
            input_descriptors_.push_back(
                {std::move(input_tensors[i]), original_batch_dim, interval});
          }
        }

        out_tensors->reserve(input_descriptors_.size());

        // We slice each component independently because they may have
        // different batch dimensions.
        for (const auto& input_desc : input_descriptors_) {
          int64_t start = input_desc.interval * slice_number_;
          int64_t end = std::min(start + input_desc.interval,
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
        return absl::OkStatus();
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
        return absl::OkStatus();
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
                ctx->flr(), full_name(strings::StrCat("tensors[", i, "]")),
                &input_descriptors_[i].whole_tensor));
            input_descriptors_[i].original_batch_dim =
                input_descriptors_[i].whole_tensor.dim_size(0);
            input_descriptors_[i].interval =
                CeilDiv(input_descriptors_[i].original_batch_dim,
                        dataset()->num_replicas_);
          }
        }
        return absl::OkStatus();
      }

      TraceMeMetadata GetTraceMeMetadata() const override {
        return dataset()->traceme_metadata_;
      }

     private:
      // Describes one component of the input.
      struct InputDescriptor {
        InputDescriptor() {}
        InputDescriptor(Tensor&& whole_tensor, int64_t original_batch_dim,
                        int64_t interval)
            : whole_tensor(std::move(whole_tensor)),
              original_batch_dim(original_batch_dim),
              interval(interval) {}

        Tensor whole_tensor;
        int64_t original_batch_dim;
        int64_t interval;
      };

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_;
      std::vector<InputDescriptor> input_descriptors_ TF_GUARDED_BY(mu_);
      int64_t slice_number_ TF_GUARDED_BY(mu_) = 0;
    };

    const DatasetBase* const input_;
    const int64_t num_replicas_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const TraceMeMetadata traceme_metadata_;
  };

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

// This dataset rebatches its input batches into batches of different size(s).
//
// This differs from RebatchDatasetOp. Namely, RebatchDatasetV2 rebatches
// incoming batches into batches whose new sizes are specified by the
// `batch_sizes` argument, while RebatchDataset splits its batches based
// on the (dynamic) input batch size and the given number of splits to make (its
// `num_replicas` argument). When used in tf.distribute, this allows
// RebatchDataset to split batches more correctly when the splits are
// distributed across multiple workers and replicas.
class RebatchDatasetV2Op : public UnaryDatasetOpKernel {
 public:
  explicit RebatchDatasetV2Op(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    const Tensor* batch_sizes_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("batch_sizes", &batch_sizes_tensor));
    OP_REQUIRES(
        ctx, batch_sizes_tensor->dims() <= 1,
        errors::InvalidArgument("`batch_sizes` must be a scalar or a vector."));

    std::vector<int64_t> batch_sizes;
    batch_sizes.reserve(batch_sizes_tensor->NumElements());
    for (int i = 0; i < batch_sizes_tensor->NumElements(); ++i) {
      batch_sizes.push_back(batch_sizes_tensor->flat<int64_t>()(i));
    }

    bool drop_remainder;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<bool>(ctx, "drop_remainder", &drop_remainder));

    *output = new Dataset(ctx, input, std::move(batch_sizes), drop_remainder,
                          output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            std::vector<int64_t>&& batch_sizes, bool drop_remainder,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          batch_sizes_(std::move(batch_sizes)),
          drop_remainder_(drop_remainder),
          output_types_(output_types),
          output_shapes_(output_shapes),
          traceme_metadata_(
              {{"batch_sizes", absl::StrJoin(batch_sizes, ",")}}) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      name_utils::IteratorPrefixParams params;
      return std::make_unique<Iterator>(Iterator::Params{
          this, name_utils::IteratorPrefix(kDatasetTypeV2, prefix, params)});
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return name_utils::DatasetDebugString(kDatasetTypeV2);
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      inputs->push_back(input_);
      return absl::OkStatus();
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
      Node* batch_sizes = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(batch_sizes_, &batch_sizes));
      Node* drop_remainder = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(drop_remainder_, &drop_remainder));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {input_graph_node, batch_sizes, drop_remainder}, output));
      return absl::OkStatus();
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
        if (end_of_sequence_) {
          *end_of_sequence = true;
          return absl::OkStatus();
        }

        *end_of_sequence = false;

        auto desired_batch_size = dataset()->batch_sizes_[batch_sizes_index_];
        // Tracks the size of the current batch as it's built up, possibly from
        // different input tensors.
        int64_t batch_size = 0;

        std::vector<std::vector<TensorSlice>> slices_to_concatenate;
        // Get slices from input tensors until they make up the whole batch
        // size or we run out of input.
        while (batch_size < desired_batch_size) {
          if (offset_ == -1) {
            // Get new input tensors.
            tensors_.clear();
            TF_RETURN_IF_ERROR(
                input_impl_->GetNext(ctx, &tensors_, &end_of_sequence_));
            if (end_of_sequence_) {
              // Break and return partial batch, if any.
              break;
            }
            TF_RETURN_IF_ERROR(ValidateInputTensors());
            offset_ = 0;
          }

          int64_t slice_end =
              std::min(offset_ + desired_batch_size - batch_size,
                       tensors_[0].dim_size(0));

          std::vector<TensorSlice> slices;
          slices.reserve(tensors_.size());
          for (const auto& tensor : tensors_) {
            slices.push_back(TensorSlice(tensor, offset_, slice_end));
          }
          slices_to_concatenate.push_back(std::move(slices));

          batch_size += (slice_end - offset_);
          offset_ = slice_end;
          if (offset_ == tensors_[0].dim_size(0)) {
            // Exhausted current input tensors, reset.
            offset_ = -1;
          }
        }

        batch_sizes_index_++;
        batch_sizes_index_ %= dataset()->batch_sizes_.size();

        // Return end_of_sequence if GetNext is expected to produce a non-empty
        // batch and there are no more inputs, or if drop_remainder is true and
        // we can't make a full batch.
        if ((batch_size == 0 && desired_batch_size > 0) ||
            (dataset()->drop_remainder_ && batch_size < desired_batch_size)) {
          DCHECK(end_of_sequence_);
          *end_of_sequence = true;
          return absl::OkStatus();
        }

        const size_t num_components = dataset()->output_dtypes().size();
        out_tensors->reserve(num_components);

        // Special case: desired batch size == 0. This may be the case when,
        // with distribution strategies, one of replicas expects an empty batch
        // so that the global batch size adds up correctly.
        if (desired_batch_size == 0) {
          DCHECK_EQ(batch_size, 0);
          DCHECK_EQ(slices_to_concatenate.size(), 0);
          for (int i = 0; i < dataset()->output_dtypes().size(); ++i) {
            if (dataset()->output_shapes()[i].unknown_rank()) {
              // For unknown rank tensors, we just create a empty Tensor since
              // it doesn't matter what shape it is.
              out_tensors->push_back(Tensor(dataset()->output_dtypes()[i]));
            } else {
              auto dim_sizes = dataset()->output_shapes()[i].dim_sizes();

              // The output batch size is always zero since the desired batch
              // size is zero.
              dim_sizes[0] = 0;

              // Handle unknown dimensions by setting any unknown dimensions to
              // zero since there isn't any data anyway.
              for (int j = 1; j < dim_sizes.size(); ++j) {
                if (dim_sizes[j] == -1) dim_sizes[j] = 0;
              }

              TensorShape tensor_shape(dim_sizes);
              out_tensors->push_back(
                  Tensor(dataset()->output_dtypes()[i], tensor_shape));
            }
          }
          return absl::OkStatus();
        }

        // Special case: when there's only one slice, we return the slice
        // directly where possible instead of copying the tensor data.
        if (slices_to_concatenate.size() == 1) {
          std::vector<Tensor> tensors;
          tensors.reserve(num_components);
          for (size_t i = 0; i < num_components; ++i) {
            Tensor& tensor = slices_to_concatenate[0][i].Slice();
            // If the slice is aligned, we return it directly.
            if (!tensor.IsAligned()) {
              tensor = tensor::DeepCopy(tensor);
            }
            tensors.push_back(std::move(tensor));
          }
          *out_tensors = std::move(tensors);
          return absl::OkStatus();
        }

        // For each component, concatenate slices into one tensor.
        for (size_t i = 0; i < num_components; ++i) {
          TensorShape component_shape({batch_size});
          TensorShape remaining_shape =
              slices_to_concatenate[0][i].Slice().shape();
          remaining_shape.RemoveDim(0);
          component_shape.AppendShape(remaining_shape);
          out_tensors->emplace_back(ctx->allocator({}),
                                    dataset()->output_dtypes()[i],
                                    component_shape);
          if (!out_tensors->back().IsInitialized()) {
            return errors::ResourceExhausted(
                "Failed to allocate memory for the batch of component ", i);
          }
          int64_t dst_offset = 0;
          for (size_t j = 0; j < slices_to_concatenate.size(); ++j) {
            auto num_slices =
                slices_to_concatenate[j][i].Slice().shape().dim_size(0);
            TensorSlice& slice = slices_to_concatenate[j][i];
            if (slice.OwnsTensor()) {
              slice.ClearSliceRef();
              // Instead of using the slice,
              // we directly use its parent tensor to make sure
              // the reference count is 1 and can move the data potentially.
              TF_RETURN_IF_ERROR(batch_util::MaybeMoveContiguousSlices(
                  slice.Parent(), slice.Start(), dst_offset, num_slices,
                  &(*out_tensors)[i]));
            } else if (slice.ParentRefCount() == 3 &&
                       j == slices_to_concatenate.size() - 1 &&
                       !tensors_.empty()) {
              // Special case:
              // When `tensors_` still holds a reference to the tensor buffer,
              // we could clear both parent and slice so that we can
              // potentially move the underlying data by dirctly using
              // `tensors_[i]`.
              //
              // For example:
              // B = 3, B_new = 2
              // tensors_:  [| e | e | e |, ...]
              //               v   v
              // new batch:  | e | e |
              slice.ClearAllRefs();
              Tensor& parent = tensors_[i];
              TF_RETURN_IF_ERROR(batch_util::MaybeMoveContiguousSlices(
                  parent, slice.Start(), dst_offset, num_slices,
                  &(*out_tensors)[i]));
            } else {
              // Other iterator ops are holding references,
              // we have to copy the underlying tensor buffer.
              TF_RETURN_IF_ERROR(batch_util::CopyContiguousSlices(
                  slice.Slice(), 0, dst_offset, num_slices,
                  &(*out_tensors)[i]));
            }
            dst_offset += num_slices;
          }
        }

        return absl::OkStatus();
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
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("batch_sizes_index"),
                                               batch_sizes_index_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("offset"), offset_));
        if (offset_ != -1) {
          for (int i = 0; i < tensors_.size(); ++i) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat("tensors[", i, "]")), tensors_[i]));
          }
        }
        return absl::OkStatus();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (!reader->Contains(full_name("input_impl_empty"))) {
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        } else {
          input_impl_.reset();
        }
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("batch_sizes_index"),
                                              &batch_sizes_index_));
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("offset"), &offset_));

        tensors_.clear();
        if (offset_ != -1) {
          tensors_.resize(dataset()->output_dtypes().size());
          for (int i = 0; i < tensors_.size(); ++i) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                ctx->flr(), full_name(strings::StrCat("tensors[", i, "]")),
                &tensors_[i]));
          }
        }
        return absl::OkStatus();
      }

      TraceMeMetadata GetTraceMeMetadata() const override {
        return dataset()->traceme_metadata_;
      }

     private:
      Status ValidateInputTensors() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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
                tensors_[0].dim_size(0), " but component ", i, " had size, ",
                tensors_[i].dim_size(0), ".");
          }
        }
        return absl::OkStatus();
      }

      class TensorSlice {
       public:
        TensorSlice(const Tensor& t, int64_t start, int64_t end)
            : start_(start),
              end_(end),
              parent_(std::make_unique<Tensor>(t)),
              slice_(std::make_unique<Tensor>(t.Slice(start, end))) {}
        bool OwnsTensor() {
          // If this iterator op owns this tensor,
          // there will be one reference from `parent_` and one from `slice_`.
          // Otherwise, some other iterator op might own this tensor.
          // For example, tensor_dataset_op.cc
          auto ref_count = ParentRefCount();
          if (ref_count) {
            return *ref_count == 2;
          } else {
            return false;
          }
        }
        std::optional<int> ParentRefCount() {
          if (parent_->data() == nullptr) {
            return std::nullopt;
          }
          return parent_->RefCount();
        }

        Tensor& Slice() { return *slice_; }
        Tensor& Parent() { return *parent_; }
        inline void ClearSliceRef() { slice_.reset(); }
        inline void ClearAllRefs() {
          parent_.reset();
          slice_.reset();
        }
        int64_t Start() { return start_; }
        int64_t End() { return end_; }

       private:
        const int64_t start_;
        const int64_t end_;
        std::unique_ptr<Tensor> parent_;
        // A slice taken from the first dimension of `parent`.
        std::unique_ptr<Tensor> slice_;
      };

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_;
      // Whether we have reached the end of the input.
      bool end_of_sequence_ TF_GUARDED_BY(mu_) = false;
      // Represents the current input tensor(s).
      std::vector<Tensor> tensors_ TF_GUARDED_BY(mu_);
      // Represents the offset into the current input tensor(s).
      // An offset of -1 indicates that there is no data left in the current
      // slice.
      int64_t offset_ TF_GUARDED_BY(mu_) = -1;
      // Represents the current index into the batch_sizes list.
      int64_t batch_sizes_index_ TF_GUARDED_BY(mu_) = 0;
    };

    const DatasetBase* const input_;
    const std::vector<int64_t> batch_sizes_;
    const bool drop_remainder_;
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

REGISTER_KERNEL_BUILDER(Name("RebatchDatasetV2").Device(DEVICE_CPU),
                        RebatchDatasetV2Op);

}  // anonymous namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
