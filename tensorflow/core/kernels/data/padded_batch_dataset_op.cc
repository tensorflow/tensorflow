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
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class PaddedBatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit PaddedBatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        op_version_(ctx->def().op() == "PaddedBatchDataset" ? 1 : 2) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 batch_size;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<int64>(ctx, "batch_size", &batch_size));
    OP_REQUIRES(
        ctx, batch_size > 0,
        errors::InvalidArgument("Batch size must be greater than zero."));

    bool drop_remainder = false;
    if (op_version_ > 1) {
      OP_REQUIRES_OK(ctx, ParseScalarArgument<bool>(ctx, "drop_remainder",
                                                    &drop_remainder));
    }

    OpInputList padded_shape_tensors;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("padded_shapes", &padded_shape_tensors));
    std::vector<PartialTensorShape> padded_shapes;
    padded_shapes.reserve(padded_shape_tensors.size());
    OP_REQUIRES(ctx,
                padded_shape_tensors.size() == input->output_shapes().size(),
                errors::InvalidArgument("Number of padded shapes (",
                                        padded_shape_tensors.size(),
                                        ") must match the number of components "
                                        "in the input dataset's elements (",
                                        input->output_shapes().size(), ")"));
    for (const Tensor& padded_shape_t : padded_shape_tensors) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(padded_shape_t.shape()),
                  errors::InvalidArgument("All padded shapes must be vectors"));
      PartialTensorShape padded_shape;
      OP_REQUIRES_OK(ctx, PartialTensorShape::MakePartialShape(
                              padded_shape_t.vec<int64>().data(),
                              padded_shape_t.NumElements(), &padded_shape));
      padded_shapes.push_back(std::move(padded_shape));
    }
    OpInputList padding_values_list;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("padding_values", &padding_values_list));
    std::vector<Tensor> padding_values;
    OP_REQUIRES(ctx,
                padding_values_list.size() == input->output_shapes().size(),
                errors::InvalidArgument(
                    "Number of padding values (", padding_values_list.size(),
                    ") must match the number of components in the input "
                    "dataset's elements (",
                    input->output_shapes().size(), ")"));
    for (int i = 0; i < padding_values_list.size(); ++i) {
      const Tensor& padding_value_t = padding_values_list[i];
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(padding_value_t.shape()),
          errors::InvalidArgument("All padding values must be scalars"));
      OP_REQUIRES(ctx, padding_value_t.dtype() == input->output_dtypes()[i],
                  errors::InvalidArgument(
                      "Mismatched type between padding value ", i,
                      " and input dataset's component ", i, ": ",
                      DataTypeString(padding_value_t.dtype()), " vs. ",
                      DataTypeString(input->output_dtypes()[i])));
      padding_values.push_back(tensor::DeepCopy(padding_value_t));
    }

    *output =
        new Dataset(ctx, batch_size, drop_remainder, std::move(padded_shapes),
                    std::move(padding_values), input);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, int64 batch_size, bool drop_remainder,
            std::vector<PartialTensorShape> padded_shapes,
            std::vector<Tensor> padding_values, const DatasetBase* input)
        : GraphDatasetBase(ctx),
          batch_size_(batch_size),
          drop_remainder_(drop_remainder),
          padded_shapes_(std::move(padded_shapes)),
          padding_values_(std::move(padding_values)),
          input_(input) {
      input_->Ref();

      // NOTE(mrry): Currently we implement "batch up to"
      // semantics. If we could tell statically that the input dataset
      // is infinite, then we could always report `batch_size` as the
      // 0th dimension.
      // TODO(mrry): Need to validate that the input shape and the
      // padded shape are "compatible" (i.e. that padded shape is >=
      // input shape, with both static and dynamic checks as appropriate).
      const auto& input_shapes = input_->output_shapes();
      output_shapes_.reserve(input_shapes.size());
      for (size_t i = 0; i < input_shapes.size(); ++i) {
        if (drop_remainder_) {
          output_shapes_.push_back(
              PartialTensorShape({batch_size_}).Concatenate(padded_shapes_[i]));
        } else {
          output_shapes_.push_back(
              PartialTensorShape({-1}).Concatenate(padded_shapes_[i]));
        }
      }
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::PaddedBatch")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return strings::StrCat("PaddedBatchDatasetOp(", batch_size_,
                             ")::Dataset");
    }

   protected:
    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input_, &input_graph_node));
      Node* batch_size = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));

      std::vector<Node*> padded_shapes;
      padded_shapes.reserve(padded_shapes_.size());
      for (int i = 0; i < padded_shapes_.size(); i++) {
        Node* node;
        Tensor t(DT_INT64, TensorShape({padded_shapes_[i].dims()}));
        for (int j = 0; j < padded_shapes_[i].dims(); j++) {
          t.vec<int64>()(j) = padded_shapes_[i].dim_size(j);
        }
        TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
        padded_shapes.emplace_back(node);
      }

      std::vector<Node*> padding_values;
      padding_values.reserve(padding_values_.size());
      for (const Tensor& t : padding_values_) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
        padding_values.emplace_back(node);
      }

      Node* drop_remainder = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(drop_remainder_, &drop_remainder));

      AttrValue output_types;
      b->BuildAttrValue(output_dtypes(), &output_types);

      AttrValue N;
      b->BuildAttrValue<int64>(padded_shapes_.size(), &N);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {{0, input_graph_node}, {1, batch_size}, {4, drop_remainder}},
          {{2, padded_shapes}, {3, padding_values}},
          {{"Toutput_types", output_types}, {"N", N}}, output));
      return Status::OK();
    }

   private:
    // Copies element into the index^th slice of parent (in the 0th dimension).
    //

    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        // Each row of `batch_elements` is a tuple of tensors from the
        // input iterator.
        std::vector<std::vector<Tensor>> batch_elements;
        {
          mutex_lock l(mu_);
          if (!input_impl_) {
            *end_of_sequence = true;
            return Status::OK();
          } else {
            *end_of_sequence = false;
            batch_elements.reserve(dataset()->batch_size_);
            for (int i = 0; i < dataset()->batch_size_ && !*end_of_sequence;
                 ++i) {
              std::vector<Tensor> batch_element_tuple;
              TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &batch_element_tuple,
                                                      end_of_sequence));
              if (!*end_of_sequence) {
                batch_elements.push_back(std::move(batch_element_tuple));
              }
            }
            if (*end_of_sequence) {
              input_impl_.reset();
            }
          }
        }

        if (batch_elements.empty()) {
          DCHECK(*end_of_sequence);
          return Status::OK();
        }

        if (dataset()->drop_remainder_ &&
            batch_elements.size() < dataset()->batch_size_) {
          *end_of_sequence = true;
          return Status::OK();
        }

        // Copy the retrieved batch elements into one output tensor
        // per tuple component.
        // NOTE(mrry): If the input or output sizes are statically
        // known, we could potentially read the input values in-place
        // into their respective slice locations. This would require a
        // different GetNext() overload that supports zero-copy, and might
        // make sense in an optimization pass.
        const size_t num_tuple_components = batch_elements[0].size();
        const int64 num_batch_elements = batch_elements.size();
        for (size_t component_index = 0; component_index < num_tuple_components;
             ++component_index) {
          // 1. Determine the shape of the padded tensor.
          TensorShape batch_component_shape({num_batch_elements});
          const PartialTensorShape& padded_shape =
              dataset()->padded_shapes_[component_index];

          for (int dim = 0; dim < padded_shape.dims(); ++dim) {
            if (padded_shape.dim_size(dim) == -1) {
              batch_component_shape.AddDim(0);
            } else {
              batch_component_shape.AddDim(padded_shape.dim_size(dim));
            }
          }

          for (int64 i = 0; i < num_batch_elements; ++i) {
            const TensorShape& element_shape =
                batch_elements[i][component_index].shape();
            // TODO(mrry): Perform this check in the shape function if
            // enough static information is available to do so.
            if (element_shape.dims() != padded_shape.dims()) {
              return errors::InvalidArgument(
                  "All elements in a batch must have the same rank as the "
                  "padded shape for component",
                  component_index, ": expected rank ", padded_shape.dims(),
                  " but got element with rank ", element_shape.dims());
            }
            for (int dim = 0; dim < padded_shape.dims(); ++dim) {
              if (padded_shape.dim_size(dim) == -1) {
                // Take the max of all batch elements in this dimension.
                if (batch_elements[i][component_index].shape().dim_size(dim) >
                    batch_component_shape.dim_size(dim + 1)) {
                  batch_component_shape.set_dim(
                      dim + 1,
                      batch_elements[i][component_index].shape().dim_size(dim));
                }
              } else {
                if (batch_elements[i][component_index].shape().dim_size(dim) >
                    batch_component_shape.dim_size(dim + 1)) {
                  return errors::DataLoss(
                      "Attempted to pad to a smaller size than the input "
                      "element.");
                }
              }
            }
          }

          // 2. Copy each batch element to the appropriate location in
          // the output component tensor.
          Tensor batch_component(ctx->allocator({}),
                                 output_dtypes()[component_index],
                                 batch_component_shape);
          TF_RETURN_IF_ERROR(batch_util::SetElementZero(
              &batch_component, dataset()->padding_values_[component_index]));

          // Build the output tuple component by copying one slice
          // from each input element in the batch.
          TensorShape component_shape({});
          for (int i = 1; i < batch_component_shape.dims(); ++i) {
            component_shape.AddDim(batch_component_shape.dim_size(i));
          }
          for (int64 i = 0; i < num_batch_elements; ++i) {
            // Take the fast path if possible.
            if (batch_elements[i][component_index].shape() == component_shape) {
              TF_RETURN_IF_ERROR(batch_util::CopyElementToSlice(
                  batch_elements[i][component_index], &batch_component, i));
            } else {
              TF_RETURN_IF_ERROR(batch_util::CopyElementToLargerSlice(
                  batch_elements[i][component_index], &batch_component, i));
            }
          }
          out_tensors->push_back(std::move(batch_component));
        }
        *end_of_sequence = false;
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (input_impl_)
          TF_RETURN_IF_ERROR(SaveParent(writer, input_impl_));
        else
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("exhausted"), ""));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (reader->Contains(full_name("exhausted"))) {
          input_impl_.reset();
        } else {
          TF_RETURN_IF_ERROR(
              dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
          TF_RETURN_IF_ERROR(RestoreParent(ctx, reader, input_impl_));
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
    };

    const int64 batch_size_;
    const bool drop_remainder_;
    const std::vector<PartialTensorShape> padded_shapes_;
    const std::vector<Tensor> padding_values_;
    const DatasetBase* const input_;
    std::vector<PartialTensorShape> output_shapes_;
  };

  const int op_version_;
};

REGISTER_KERNEL_BUILDER(Name("PaddedBatchDataset").Device(DEVICE_CPU),
                        PaddedBatchDatasetOp);

REGISTER_KERNEL_BUILDER(Name("PaddedBatchDatasetV2").Device(DEVICE_CPU),
                        PaddedBatchDatasetOp);

}  // namespace

}  // namespace tensorflow
