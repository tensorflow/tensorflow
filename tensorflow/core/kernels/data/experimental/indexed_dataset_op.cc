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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace tensorflow {
namespace data {
namespace {

// TODO(saeta): Urgh, this is ugly.
class MaterializedIndexedDataset {
 public:
  virtual ~MaterializedIndexedDataset() = default;

  // Retrieve the element at a given index. The output tensors are stored in
  // out_tensors.
  //
  // If `index` is greater than `Size()`, tensorflow::errors::OutOfRangeError is
  // returned.
  //
  // Get is thread-safe.
  virtual Status Get(IteratorContext&& ctx, uint64 index,
                     std::vector<Tensor>* out_tensors) const = 0;

  // Size determines the number of elements in this IndexedDataset.
  //
  // Size is thread-safe.
  virtual Status Size(uint64* size) const = 0;

  // Returns a vector of DataType values, representing the respective
  // element types of each tuple component in the outputs of this dataset.
  virtual const DataTypeVector& output_dtypes() const = 0;

  // Returns a vector of tensor shapes, representing the respective
  // (and possibly partially defined) shapes of each tuple component
  // in the outputs of this dataset.
  virtual const std::vector<PartialTensorShape>& output_shapes() const = 0;
};

// IndexedDataset represents a dataset that supports random access in addition
// to iterator-based sequential access.
//
// Note: IndexedDatasets are HIGHLY experimental at this time. Expect
// significant (backwards incompatible) changes!
class IndexedDataset : public DatasetBase {
 public:
  explicit IndexedDataset(DatasetContext&& ctx) : DatasetBase(std::move(ctx)) {}

  // Materialize (if necessary) the dataset, and return a pointer.
  // TODO(saeta): Add in `IteratorContext* ctx` when materializing.
  virtual Status MaterializeDataset(
      std::shared_ptr<MaterializedIndexedDataset>* materialized) = 0;
};

// IndexedDatasetOpKernel abstracts away interfacing IndexedDatasets with the
// rest of the TensorFlow runtime.
//
// Most IndexedDataset's will be private members of classes inheriting from this
// class.
class IndexedDatasetOpKernel : public OpKernel {
 public:
  explicit IndexedDatasetOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) final;

 protected:
  // Subclasses should implement this method. It will be called during Compute
  // execution.
  virtual void MakeIndexedDataset(OpKernelContext* ctx,
                                  IndexedDataset** output) = 0;

  template <typename T>
  Status ParseScalarArgument(OpKernelContext* ctx,
                             const StringPiece argument_name, T* output) {
    const Tensor* argument_t;
    TF_RETURN_IF_ERROR(ctx->input(argument_name, &argument_t));
    if (!TensorShapeUtils::IsScalar(argument_t->shape())) {
      return errors::InvalidArgument(argument_name, " must be a scalar");
    }
    *output = argument_t->scalar<T>()();
    return Status::OK();
  }
};

class MaterializedDatasetResource : public ResourceBase {
 public:
  MaterializedDatasetResource(
      const DataTypeVector& output_dtypes,
      const std::vector<PartialTensorShape>& output_shapes)
      : output_dtypes_(output_dtypes), output_shapes_(output_shapes) {}

  string DebugString() const override {
    return "Materialized IndexedDataset resource";
  }

  Status Get(IteratorContext&& ctx, uint64 index,
             std::vector<Tensor>* out_tensors) {
    std::shared_ptr<MaterializedIndexedDataset> captured(materialized_);
    if (captured) {
      return captured->Get(std::move(ctx), index, out_tensors);
    } else {
      return errors::FailedPrecondition(
          "Get() failed because the MaterializedIndexedDataset has not been "
          "initialized. Ensure that you have run the materialization operation "
          "for this MaterializedIndexedDataset before retrieving elements.");
    }
  }

  // TODO(saeta): Implement Save and Restore

  const DataTypeVector& output_dtypes() const { return output_dtypes_; }
  const std::vector<PartialTensorShape>& output_shapes() const {
    return output_shapes_;
  }

  Status set_materialized_dataset(
      const std::shared_ptr<MaterializedIndexedDataset>& dataset) {
    if (dataset) {
      TF_RETURN_IF_ERROR(
          VerifyTypesMatch(output_dtypes_, dataset->output_dtypes()));
      TF_RETURN_IF_ERROR(
          VerifyShapesCompatible(output_shapes_, dataset->output_shapes()));
    }
    materialized_ = dataset;
    return Status::OK();
  }

 private:
  std::shared_ptr<MaterializedIndexedDataset> materialized_;
  const DataTypeVector output_dtypes_;
  const std::vector<PartialTensorShape> output_shapes_;
};

// A wrapper class for storing an `IndexedDataset` instance in a DT_VARIANT
// tensor. Objects of the wrapper class own a reference on an instance of an
// `IndexedTensor` and the wrapper's copy constructor and destructor take care
// of managing the reference count.
//
// NOTE: This is not a feature-complete implementation of the DT_VARIANT
// specification. In particular, we cannot currently serialize an arbitrary
// `IndexedDataset` object, so the `Encode()` and `Decode()` methods are not
// implemented.
//
// NOTE(saeta): When `IndexedDataset`s get merged into core, we can instead just
// use `tensorflow::DatasetVariantWrapper`.
class IndexedDatasetVariantWrapper {
 public:
  IndexedDatasetVariantWrapper() : dataset_(nullptr) {}

  // Transfers ownership of `dataset` to `*this`.
  explicit IndexedDatasetVariantWrapper(IndexedDataset* dataset)
      : dataset_(dataset) {}

  IndexedDatasetVariantWrapper(const IndexedDatasetVariantWrapper& other)
      : dataset_(other.dataset_) {
    if (dataset_) dataset_->Ref();
  }

  ~IndexedDatasetVariantWrapper() {
    if (dataset_) dataset_->Unref();
  }

  IndexedDataset* get() const { return dataset_; }

  string TypeName() const { return "tensorflow::IndexedDatasetVariantWrapper"; }
  string DebugString() const {
    if (dataset_) {
      return dataset_->DebugString();
    } else {
      return "<Uninitialized IndexedDatasetVariantWrapper>";
    }
  }

  void Encode(VariantTensorData* data) const {
    LOG(ERROR) << "The Encode() method is not implemented for "
                  "IndexedDatasetVariantWrapper objects.";
  }

  bool Decode(const VariantTensorData& data) {
    LOG(ERROR) << "The Decode() method is not implemented for "
                  "IndexedDatasetVariantWrapper objects.";
    return false;
  }

 private:
  IndexedDataset* const dataset_;  // Owns one reference.
};

Status GetIndexedDatasetFromVariantTensor(const Tensor& tensor,
                                          IndexedDataset** out_dataset) {
  if (!(tensor.dtype() == DT_VARIANT ||
        TensorShapeUtils::IsScalar(tensor.shape()))) {
    return errors::InvalidArgument(
        "IndexedDataset tensor must be a scalar of dtype DT_VARIANT.");
  }
  const Variant& variant = tensor.scalar<Variant>()();
  const IndexedDatasetVariantWrapper* wrapper =
      variant.get<IndexedDatasetVariantWrapper>();
  if (wrapper == nullptr) {
    return errors::InvalidArgument("Tensor must be an IndexedDataset object.");
  }
  *out_dataset = wrapper->get();
  if (*out_dataset == nullptr) {
    return errors::Internal("Read uninitialized IndexedDataset variant.");
  }
  return Status::OK();
}

Status StoreIndexedDatasetInVariantTensor(IndexedDataset* dataset,
                                          Tensor* tensor) {
  if (!(tensor->dtype() == DT_VARIANT ||
        TensorShapeUtils::IsScalar(tensor->shape()))) {
    return errors::InvalidArgument(
        "Dataset tensor must be a scalar of dtype DT_VARIANT.");
  }
  tensor->scalar<Variant>()() = IndexedDatasetVariantWrapper(dataset);
  return Status::OK();
}

void IndexedDatasetOpKernel::Compute(OpKernelContext* ctx) {
  IndexedDataset* dataset = nullptr;
  MakeIndexedDataset(ctx, &dataset);

  if (ctx->status().ok()) {
    OP_REQUIRES(ctx, dataset != nullptr,
                errors::Internal("MakeIndexedDataset did not correctly "
                                 "construct the IndexedDataset"));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES_OK(ctx, StoreIndexedDatasetInVariantTensor(dataset, output));
  }
}

class MaterializedHandleOp : public OpKernel {
 public:
  explicit MaterializedHandleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_dtypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  ~MaterializedHandleOp() override {
    if (resource_ != nullptr) {
      resource_->Unref();
      if (cinfo_.resource_is_private_to_kernel()) {
        if (!cinfo_.resource_manager()
                 ->template Delete<MaterializedDatasetResource>(
                     cinfo_.container(), cinfo_.name())
                 .ok()) {
          // Do nothing; the resource can have been deleted by session resets.
          // Note: cargo-culted from $tf/core/framework/resource_op_kernel.h
        }
      }
    }
  }

  void Compute(OpKernelContext* context) override LOCKS_EXCLUDED(mu_) {
    {
      mutex_lock l(mu_);
      if (resource_ == nullptr) {
        ResourceMgr* mgr = context->resource_manager();
        OP_REQUIRES_OK(context, cinfo_.Init(mgr, def()));

        MaterializedDatasetResource* resource;
        OP_REQUIRES_OK(context,
                       mgr->LookupOrCreate<MaterializedDatasetResource>(
                           cinfo_.container(), cinfo_.name(), &resource,
                           [this](MaterializedDatasetResource** ret)
                               EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                                 *ret = new MaterializedDatasetResource(
                                     output_dtypes_, output_shapes_);
                                 return Status::OK();
                               }));
        Status s = VerifyResource(resource);
        if (TF_PREDICT_FALSE(!s.ok())) {
          resource->Unref();
          context->SetStatus(s);
          return;
        }

        resource_ = resource;
      }
    }
    OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                                context, 0, cinfo_.container(), cinfo_.name(),
                                MakeTypeIndex<MaterializedDatasetResource>()));
  }

 private:
  // During the first Compute(), resource is either created or looked up using
  // shared_name. In the latter case, the resource found should be verified if
  // it is compatible with this op's configuration. The verification may fail in
  // cases such as two graphs asking queues of the same shared name to have
  // inconsistent capacities.
  Status VerifyResource(MaterializedDatasetResource* resource) {
    TF_RETURN_IF_ERROR(
        VerifyTypesMatch(output_dtypes_, resource->output_dtypes()));
    TF_RETURN_IF_ERROR(
        VerifyShapesCompatible(output_shapes_, resource->output_shapes()));
    return Status::OK();
  }

  mutex mu_;
  ContainerInfo cinfo_;  // Written once under mu_ then constant afterwards.
  MaterializedDatasetResource* resource_ GUARDED_BY(mu_) = nullptr;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
};

// TODO(saeta): Make async.
class MaterializeDatasetOp : public OpKernel {
 public:
  explicit MaterializeDatasetOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    IndexedDataset* dataset;
    OP_REQUIRES_OK(ctx,
                   GetIndexedDatasetFromVariantTensor(ctx->input(0), &dataset));

    MaterializedDatasetResource* materialized_resource;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 1),
                                       &materialized_resource));
    core::ScopedUnref unref(materialized_resource);
    std::shared_ptr<MaterializedIndexedDataset> materialized;
    OP_REQUIRES_OK(ctx, dataset->MaterializeDataset(&materialized));
    OP_REQUIRES_OK(
        ctx, materialized_resource->set_materialized_dataset(materialized));
  }
};

// TODO(saeta): Make async
class IndexedDatasetGet : public OpKernel {
 public:
  explicit IndexedDatasetGet(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    MaterializedDatasetResource* materialized_resource;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0),
                                       &materialized_resource));
    auto cleanup = gtl::MakeCleanup([materialized_resource] {
      materialized_resource->Unref();  // Note: can't use core::ScopedUnref.
    });

    const Tensor* index_t;
    OP_REQUIRES_OK(ctx, ctx->input("index", &index_t));
    // TODO(saeta): Support batch reads (indexes should be non-scalar!)
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(index_t->shape()),
                errors::InvalidArgument("index must be a scalar"));
    const uint64 index = index_t->scalar<uint64>()();

    std::vector<Tensor> out_tensors;
    Status s =
        materialized_resource->Get(IteratorContext(ctx), index, &out_tensors);

    // Note: Unref materialized_resource to avoid destruction races. (Important
    // in a [future] async op implementation.)
    cleanup.release()();

    if (!s.ok()) {
      ctx->SetStatus(s);
    } else {
      auto expected_shapes = materialized_resource->output_shapes();
      auto expected_types = materialized_resource->output_dtypes();
      for (size_t i = 0; i < out_tensors.size(); ++i) {
        OP_REQUIRES(
            ctx, expected_shapes[i].IsCompatibleWith(out_tensors[i].shape()),
            errors::Internal(
                "Materialized dataset output at index ", i,
                " is incompatible with the expected shape. (Expected: ",
                expected_shapes[i], ", got: ", out_tensors[i].shape(), ")"));
        OP_REQUIRES(ctx, out_tensors[i].dtype() == expected_types[i],
                    errors::Internal("Materialized dataset output at index ", i,
                                     " was not the expected dtype. (Expected: ",
                                     expected_types[i],
                                     ", got: ", out_tensors[i].dtype(), ")"));
        ctx->set_output(i, out_tensors[i]);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("ExperimentalMaterializedIndexDatasetHandle").Device(DEVICE_CPU),
    MaterializedHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalIndexedDatasetMaterialize").Device(DEVICE_CPU),
    MaterializeDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalIndexedDatasetGet").Device(DEVICE_CPU),
    IndexedDatasetGet);

class IdentityIndexedDatasetOp : public IndexedDatasetOpKernel {
 public:
  using IndexedDatasetOpKernel::IndexedDatasetOpKernel;

  void MakeIndexedDataset(OpKernelContext* ctx,
                          IndexedDataset** output) override {
    uint64 size = -1;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<uint64>(ctx, "size", &size));
    OP_REQUIRES(ctx, size > 0, errors::InvalidArgument("`size` must be > 0"));
    *output = new Dataset(ctx, size);
  }

  class Dataset : public IndexedDataset {
   public:
    Dataset(OpKernelContext* ctx, uint64 size)
        : IndexedDataset(DatasetContext(ctx)), size_(size) {}

    Status MaterializeDataset(
        std::shared_ptr<MaterializedIndexedDataset>* materialized) override {
      (*materialized) = std::make_shared<Materialized>(this);
      return Status::OK();
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_UINT64});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(Iterator::Params{
          this, strings::StrCat(prefix, "::IdentityIndexedDataset")});
    }

    string DebugString() const override {
      return "IdentityIndexedDataset::Dataset";
    }

    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** node) const override {
      return errors::Unimplemented(
          "identity_indexed_dataset.AsGraphDefInternal");
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (cur_ < dataset()->size_) {
          out_tensors->emplace_back(ctx->allocator({}), DT_UINT64,
                                    TensorShape({}));
          out_tensors->back().scalar<uint64>()() = cur_++;
          *end_of_sequence = false;
          return Status::OK();
        }
        *end_of_sequence = true;
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

     private:
      mutex mu_;
      uint64 cur_ GUARDED_BY(mu_);
    };

    class Materialized : public MaterializedIndexedDataset {
     public:
      explicit Materialized(Dataset* dataset) : dataset_(dataset) {
        dataset->Ref();
      }

      ~Materialized() override {
        // TODO(saeta): Pull this into MaterializedIndexedDataset
        dataset_->Unref();
      }

      const DataTypeVector& output_dtypes() const override {
        return dataset_->output_dtypes();
      }

      const std::vector<PartialTensorShape>& output_shapes() const override {
        return dataset_->output_shapes();
      }

      Status Get(IteratorContext&& ctx, uint64 index,
                 std::vector<Tensor>* out_tensors) const override {
        LOG(INFO) << "Materialized(" << dataset_->size_ << ")::Get(" << index
                  << ")";
        if (index >= dataset_->size_) {
          // Note: use InvalidArgument instead of OutOfRange error because many
          // things consider OutOfRange to be a "clean termination" error.
          return errors::InvalidArgument(
              "Index ", index,
              " is out of range for this dataset. (Size is: ", dataset_->size_,
              ".)");
        }
        out_tensors->emplace_back(ctx.allocator({}), DT_UINT64,
                                  TensorShape({}));
        out_tensors->back().scalar<uint64>()() = index;
        return Status::OK();
      }

      Status Size(uint64* size) const override {
        *size = dataset_->size_;
        return Status::OK();
      }

     private:
      const Dataset* const dataset_;  // Not owned.
    };

    const uint64 size_;
    std::shared_ptr<Materialized> materialized_;
  };
};

REGISTER_KERNEL_BUILDER(
    Name("ExperimentalIdentityIndexedDataset").Device(DEVICE_CPU),
    IdentityIndexedDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
