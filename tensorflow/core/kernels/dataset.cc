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

#include "tensorflow/core/kernels/dataset.h"

namespace tensorflow {

namespace {

// A wrapper class for storing a `DatasetBase` instance in a DT_VARIANT tensor.
// Objects of the wrapper class own a reference on an instance of `DatasetBase`,
// and the wrapper's copy constructor and destructor take care of managing the
// reference count.
//
// NOTE(mrry): This is not a feature-complete implementation of the DT_VARIANT
// specification. In particular, we cannot currently serialize an arbitrary
// `DatasetBase` object, so the `Encode()` and `Decode()` methods are not
// implemented.
class DatasetVariantWrapper {
 public:
  DatasetVariantWrapper() : dataset_(nullptr) {}

  // Transfers ownership of `dataset` to `*this`.
  explicit DatasetVariantWrapper(DatasetBase* dataset) : dataset_(dataset) {}

  DatasetVariantWrapper(const DatasetVariantWrapper& other)
      : dataset_(other.dataset_) {
    if (dataset_) dataset_->Ref();
  }

  ~DatasetVariantWrapper() {
    if (dataset_) dataset_->Unref();
  }

  DatasetBase* get() const { return dataset_; }

  string TypeName() const { return "tensorflow::DatasetVariantWrapper"; }
  string DebugString() const {
    if (dataset_) {
      return dataset_->DebugString();
    } else {
      return "<Uninitialized DatasetVariantWrapper>";
    }
  }
  void Encode(VariantTensorData* data) const {
    LOG(ERROR) << "The Encode() method is not implemented for "
                  "DatasetVariantWrapper objects.";
  }
  bool Decode(const VariantTensorData& data) {
    LOG(ERROR) << "The Decode() method is not implemented for "
                  "DatasetVariantWrapper objects.";
    return false;
  }

 private:
  DatasetBase* const dataset_;  // Owns one reference.
};

}  // namespace

Status GetDatasetFromVariantTensor(const Tensor& tensor,
                                   DatasetBase** out_dataset) {
  if (!(tensor.dtype() == DT_VARIANT ||
        TensorShapeUtils::IsScalar(tensor.shape()))) {
    return errors::InvalidArgument(
        "Dataset tensor must be a scalar of dtype DT_VARIANT.");
  }
  const Variant& variant = tensor.scalar<Variant>()();
  const DatasetVariantWrapper* wrapper = variant.get<DatasetVariantWrapper>();
  if (wrapper == nullptr) {
    return errors::InvalidArgument("Tensor must be a Dataset object.");
  }
  *out_dataset = wrapper->get();
  if (*out_dataset == nullptr) {
    return errors::Internal("Read uninitialized Dataset variant.");
  }
  return Status::OK();
}

Status StoreDatasetInVariantTensor(DatasetBase* dataset, Tensor* tensor) {
  if (!(tensor->dtype() == DT_VARIANT ||
        TensorShapeUtils::IsScalar(tensor->shape()))) {
    return errors::InvalidArgument(
        "Dataset tensor must be a scalar of dtype DT_VARIANT.");
  }
  tensor->scalar<Variant>()() = DatasetVariantWrapper(dataset);
  return Status::OK();
}

void DatasetOpKernel::Compute(OpKernelContext* ctx) {
  DatasetBase* dataset = nullptr;
  MakeDataset(ctx, &dataset);
  if (ctx->status().ok()) {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES_OK(ctx, StoreDatasetInVariantTensor(dataset, output));
  }
}

void UnaryDatasetOpKernel::MakeDataset(OpKernelContext* ctx,
                                       DatasetBase** output) {
  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  MakeDataset(ctx, input, output);
}

void BinaryDatasetOpKernel::MakeDataset(OpKernelContext* ctx,
                                        DatasetBase** output) {
  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  DatasetBase* another_input;
  OP_REQUIRES_OK(ctx,
                 GetDatasetFromVariantTensor(ctx->input(1), &another_input));
  MakeDataset(ctx, input, another_input, output);
}

const char GraphDatasetBase::kDatasetGraphKey[] = "_DATASET_GRAPH";
const char GraphDatasetBase::kDatasetGraphOutputNodeKey[] =
    "_DATASET_GRAPH_OUTPUT_NODE";

}  // namespace tensorflow
