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
#include "tensorflow/contrib/data/kernels/indexed_dataset.h"

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace tensorflow {

namespace {

Status VerifyTypesMatch(const DataTypeVector& expected,
                        const DataTypeVector& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " types but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != received[i]) {
      return errors::InvalidArgument("Data type mismatch at component ", i,
                                     ": expected ", DataTypeString(expected[i]),
                                     " but got ", DataTypeString(received[i]),
                                     ".");
    }
  }
  return Status::OK();
}

Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<PartialTensorShape>& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " shapes but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!expected[i].IsCompatibleWith(received[i])) {
      return errors::InvalidArgument("Incompatible shapes at component ", i,
                                     ": expected ", expected[i].DebugString(),
                                     " but got ", received[i].DebugString(),
                                     ".");
    }
  }

  return Status::OK();
}

class MaterializedDatasetResource : public ResourceBase {
 public:
  MaterializedDatasetResource(
      const DataTypeVector& output_dtypes,
      const std::vector<PartialTensorShape>& output_shapes)
      : output_dtypes_(output_dtypes), output_shapes_(output_shapes) {}

  string DebugString() override {
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
// `IndexedTensor` and the wrapper's copy constructor and desctructor take care
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

}  // namespace

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

namespace {

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
    Name("MaterializedIndexDatasetHandle").Device(DEVICE_CPU),
    MaterializedHandleOp);
REGISTER_KERNEL_BUILDER(Name("IndexedDatasetMaterialize").Device(DEVICE_CPU),
                        MaterializeDatasetOp);
REGISTER_KERNEL_BUILDER(Name("IndexedDatasetGet").Device(DEVICE_CPU),
                        IndexedDatasetGet);
}  // namespace

}  // namespace tensorflow
