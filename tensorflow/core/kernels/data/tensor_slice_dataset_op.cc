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
#include "tensorflow/core/kernels/data/tensor_slice_dataset_op.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const TensorSliceDatasetOp::kDatasetType;
/* static */ constexpr const char* const TensorSliceDatasetOp::kComponents;
/* static */ constexpr const char* const TensorSliceDatasetOp::kToutputTypes;
/* static */ constexpr const char* const TensorSliceDatasetOp::kOutputShapes;

constexpr char kCurIndex[] = "i";

class TensorSliceDatasetOp::Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext* ctx, std::vector<Tensor> tensors)
      : DatasetBase(DatasetContext(ctx)), tensors_(std::move(tensors)) {
    for (const Tensor& t : tensors_) {
      dtypes_.push_back(t.dtype());
      gtl::InlinedVector<int64, 4> element_dim_sizes;
      // Handle scalar here. Check that everyone matches here? Or fail
      // at runtime?
      for (int i = 1; i < t.dims(); ++i) {
        element_dim_sizes.push_back(t.dim_size(i));
      }
      partial_shapes_.emplace_back(element_dim_sizes);
      shapes_.emplace_back(std::move(element_dim_sizes));
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override { return dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return partial_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64 Cardinality() const override { return tensors_[0].dim_size(0); }

  Status CheckExternalState() const override { return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    std::vector<Node*> components;
    components.reserve(tensors_.size());
    for (const Tensor& t : tensors_) {
      Node* node;
      if (ctx->serialize_data_tensors()) {
        TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
      } else {
        TF_RETURN_IF_ERROR(b->AddPlaceholder(t, &node));
        DCHECK_NE(ctx->input_list(), nullptr);
        ctx->input_list()->emplace_back(node->name(), t);
      }
      components.emplace_back(node);
    }
    AttrValue dtypes;
    b->BuildAttrValue(dtypes_, &dtypes);
    TF_RETURN_IF_ERROR(b->AddDataset(this, {}, {{0, components}},
                                     {{kToutputTypes, dtypes}}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          i_(0),
          n_(params.dataset->tensors_[0].dim_size(0)) {}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      int64 index = 0;
      {
        mutex_lock l(mu_);
        if (i_ < n_) {
          index = i_;
          ++i_;
        } else {
          *end_of_sequence = true;
          return Status::OK();
        }
      }
      out_tensors->clear();
      out_tensors->reserve(dataset()->tensors_.size());
      for (size_t i = 0; i < dataset()->tensors_.size(); ++i) {
        const Tensor& t = dataset()->tensors_[i];
        out_tensors->emplace_back(ctx->allocator({}), t.dtype(),
                                  dataset()->shapes_[i]);
        TF_RETURN_IF_ERROR(
            batch_util::CopySliceToElement(t, &out_tensors->back(), index));
      }
      *end_of_sequence = false;
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurIndex), i_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurIndex), &i_));
      return Status::OK();
    }

   private:
    mutex mu_;
    int64 i_ TF_GUARDED_BY(mu_);
    const int64 n_;
  };

  const std::vector<Tensor> tensors_;
  DataTypeVector dtypes_;
  std::vector<TensorShape> shapes_;
  std::vector<PartialTensorShape> partial_shapes_;
};

TensorSliceDatasetOp::TensorSliceDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kToutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void TensorSliceDatasetOp::MakeDataset(OpKernelContext* ctx,
                                       DatasetBase** output) {
  OpInputList inputs;
  OP_REQUIRES_OK(ctx, ctx->input_list(kComponents, &inputs));
  std::vector<Tensor> components;
  components.reserve(inputs.size());
  OP_REQUIRES(
      ctx, inputs[0].dims() > 0,
      errors::InvalidArgument("All components must be at least 1-dimensional"));
  const int64 num_slices = inputs[0].dim_size(0);
  for (const Tensor& t : inputs) {
    components.push_back(t);
    OP_REQUIRES(ctx, t.dims() > 0,
                errors::InvalidArgument(
                    "All components must be at least 1-dimensional"));
    OP_REQUIRES(
        ctx, t.dim_size(0) == num_slices,
        errors::InvalidArgument(
            "All components must have the same size in the 0th dimension"));
  }
  *output = new Dataset(ctx, std::move(components));
  OP_REQUIRES_OK(ctx,
                 VerifyTypesMatch((*output)->output_dtypes(), output_types_));
  OP_REQUIRES_OK(
      ctx, VerifyShapesCompatible((*output)->output_shapes(), output_shapes_));
}

namespace {
REGISTER_KERNEL_BUILDER(Name("TensorSliceDataset").Device(DEVICE_CPU),
                        TensorSliceDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
