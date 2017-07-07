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
#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DATASET_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DATASET_H_

#include <memory>

#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

class ResourceMgr;

// A cut-down version of OpKernelContext for running computations in
// iterators. Note that we cannot simply use OpKernelContext here
// because we might run computation in an iterator whose lifetime is
// not nested within the lifetime of a single OpKernelContext
// (e.g. asynchronous prefetching).
//
// TODO(mrry): We will probably need to support more of
// OpKernelContext here. For example, should allocation be handled by
// the IteratorContext?
// TODO(mrry): We will need to fabricate step IDs for calls to ops
// that are not nested within a particular step.
// TODO(mrry): We're making some daring assumptions about the lifetime
// of the FunctionLibraryRuntime and runner passed in here. Once
// created, a FunctionLibraryRuntime should stay alive for the
// remainder of a session, so we copy the pointer. A runner will be
// deleted when the original step ends, but all existing runners only
// close over session-lifetime (or longer-lived) state, so we can make
// a copy of the function. There's nothing in the definition of either
// class to guarantee that what we are doing is safe. We should
// formalize the properties here.
class IteratorContext {
 public:
  struct Params {
    // Interface to operating system functionality.
    Env* env;

    // The step being executed.
    int64 step_id = 0;

    // Shared resources accessible by this iterator invocation.
    ResourceMgr* resource_manager = nullptr;

    // Function call support.
    std::function<void(std::function<void()>)> runner = nullptr;
  };

  explicit IteratorContext(Params params) : params_(std::move(params)) {}

  Env* env() const { return params_.env; }

  int64 step_id() const { return params_.step_id; }

  std::function<void(std::function<void()>)>* runner() {
    return &params_.runner;
  }

  ResourceMgr* resource_manager() const { return params_.resource_manager; }

 private:
  Params params_;
};

// Represents the current position in a range of outputs, where the
// range of outputs is typically represented by an `DatasetBase`,
// defined below.
class IteratorBase {
 public:
  virtual ~IteratorBase() {}

  // Gets the next output from the range that this iterator is traversing.
  //
  // If at least one output remains in this iterator's range, that
  // output will be stored in `*out_tensors` and `false` will be
  // stored in `*end_of_sequence`.
  //
  // If no more outputs remain in this iterator's range, `true` will
  // be stored in `*end_of_sequence`, and the content of
  // `*out_tensors` will be undefined.
  //
  // This method is thread-safe.
  //
  // TODO(mrry): Define `GetNextAsync()` or `GetNextManyAsync()`, and
  // potentially remove this method.
  virtual Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) = 0;

  // Returns a vector of DataType values, representing the respective
  // element types of each tuple component in the outputs of this
  // iterator.
  virtual const DataTypeVector& output_dtypes() const = 0;

  // Returns a vector of tensor shapes, representing the respective
  // (and possibly partially defined) shapes of each tuple component
  // in the outputs of this iterator.
  virtual const std::vector<PartialTensorShape>& output_shapes() const = 0;
};

// Represents a (potentially infinite) range of outputs, where each
// output is a tuple of tensors.
class DatasetBase : public ResourceBase {
 public:
  // Returns a new iterator for iterating over the range of elements in
  // this dataset.
  //
  // This method may be called multiple times on the same instance,
  // and the resulting iterators will have distinct state. Each
  // iterator will traverse all elements in this dataset from the
  // start.
  //
  // Ownership of the created iterator will be transferred to the caller.
  virtual std::unique_ptr<IteratorBase> MakeIterator() const = 0;

  // Returns a vector of DataType values, representing the respective
  // element types of each tuple component in the outputs of this
  // dataset.
  virtual const DataTypeVector& output_dtypes() const = 0;

  // Returns a vector of tensor shapes, representing the respective
  // (and possibly partially defined) shapes of each tuple component
  // in the outputs of this dataset.
  virtual const std::vector<PartialTensorShape>& output_shapes() const = 0;
};

// Represents an iterator that is associated with a particular parent dataset.
template <class DatasetType>
class DatasetIterator : public IteratorBase {
 public:
  explicit DatasetIterator(const DatasetType* dataset) : dataset_(dataset) {
    dataset_->Ref();
  }

  ~DatasetIterator() override { dataset_->Unref(); }

  // The dataset from which this iterator was created.
  const DatasetType* dataset() const { return dataset_; }

  const DataTypeVector& output_dtypes() const override {
    return dataset_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return dataset_->output_shapes();
  }

 private:
  const DatasetType* const dataset_;  // Owns one reference on the
                                      // shared dataset resource.
};

// Encapsulates the work required to plug a DatasetBase into the core TensorFlow
// graph execution engine.
class DatasetOpKernel : public OpKernel {
 public:
  DatasetOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) final;

 protected:
  // Subclasses should implement this method. It will be called during Compute
  // execution.
  virtual void MakeDataset(OpKernelContext* ctx, DatasetBase** output) = 0;

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

// Encapsulates the work required to plug unary Datasets into the core
// TensorFlow graph execution engine.
class UnaryDatasetOpKernel : public DatasetOpKernel {
 public:
  UnaryDatasetOpKernel(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) final;
  virtual void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                           DatasetBase** output) = 0;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DATASET_H_
