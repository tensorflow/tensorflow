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
#ifndef TENSORFLOW_CONTRIB_DATA_KERNELS_INDEXED_DATASET_H_
#define TENSORFLOW_CONTRIB_DATA_KERNELS_INDEXED_DATASET_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace data {

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
  IndexedDataset(DatasetContext&& ctx) : DatasetBase(std::move(ctx)) {}

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
  IndexedDatasetOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) final;

 protected:
  // Subclasses should implement this method. It will be called during Compute
  // execution.
  virtual void MakeIndexedDataset(OpKernelContext* ctx,
                                  IndexedDataset** output) = 0;

  template <typename T>
  Status ParseScalarArgument(OpKernelContext* ctx,
                             const StringPiece& argument_name, T* output) {
    const Tensor* argument_t;
    TF_RETURN_IF_ERROR(ctx->input(argument_name, &argument_t));
    if (!TensorShapeUtils::IsScalar(argument_t->shape())) {
      return errors::InvalidArgument(argument_name, " must be a scalar");
    }
    *output = argument_t->scalar<T>()();
    return Status::OK();
  }
};

// Validates and extracts an `IndexedDataset` object from `tensor`.
//
// `tensor` must have been written by a call to
// `StoreIndexedDatasetInVariantTensor`
//
// The retrieved pointer isa  borrowed reference to the dataset, which is owned
// by the tensor. The consumer must either acquire its own reference to the
// dataset by calling `(*out_dataset)->Ref()`, or ensure that `tensor` is not
// destroyed or mutated while the retrieved pointer is in use.
Status GetIndexedDatasetFromVariantTensor(const Tensor& tensor,
                                          IndexedDataset** out_dataset);

// Stores an `IndexedDataset` object in `tensor.`
//
// The ownership of `dataset` is transferred to `tensor`.
Status StoreIndexedDatasetInVariantTensor(IndexedDataset* dataset,
                                          Tensor* tensor);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_DATA_KERNELS_INDEXED_DATASET_H_
