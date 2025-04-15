/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_
#define TENSORFLOW_CORE_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_

#include <atomic>

#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace lookup {

// Base class for lookup tables that require initialization.
class InitializableLookupTable : public LookupInterface {
 public:
  class InitTableIterator;
  class InitializerSerializer;

  // Performs batch lookups, for every element in the key tensor, Find returns
  // the corresponding value into the values tensor.
  // If an element is not present in the table, the given default value is used.
  //
  // For tables that require initialization, `Find` is available once the table
  // is marked as initialized.
  //
  // Returns the following statuses:
  // - OK: when the find finishes successfully.
  // - FailedPrecondition: if the table is not initialized.
  // - InvalidArgument: if any of the preconditions on the lookup key or value
  //   fails.
  // - In addition, other implementations may provide another non-OK status
  //   specific to their failure modes.
  absl::Status Find(OpKernelContext* ctx, const Tensor& keys, Tensor* values,
                    const Tensor& default_value) final;

  // Returns errors::Unimplemented.
  absl::Status Insert(OpKernelContext* ctx, const Tensor& keys,
                      const Tensor& values) final {
    return errors::Unimplemented(
        "Insert not supported by InitializableLookupTable implementations");
  }

  // Returns errors::Unimplemented.
  absl::Status Remove(OpKernelContext* ctx, const Tensor& keys) final {
    return errors::Unimplemented(
        "Remove not supported by InitializableLookupTable implementations");
  }

  absl::Status ExportValues(OpKernelContext* context) override {
    return errors::Unimplemented(
        "ExportValues not supported by InitializableLookupTable "
        "implementations");
  }

  absl::Status ImportValues(OpKernelContext* ctx, const Tensor& keys,
                            const Tensor& values) final;

  TensorShape key_shape() const final { return TensorShape(); }

  TensorShape value_shape() const final { return TensorShape(); }

  // Returns whether the table was initialized and is ready to serve lookups.
  bool is_initialized() const {
    return is_initialized_.load(std::memory_order_acquire);
  }

  // Initializes the table from the given init table iterator.
  //
  // Atomically, this operation prepares the table, populates it with the given
  // iterator, and marks the table as initialized.
  //
  // Returns the following statuses:
  // - OK: when the initialization was successful.
  // - InvalidArgument: if any of the preconditions on the lookup key or value
  //   fails.
  // - FailedPrecondition: if the table is already initialized and
  //   fail_if_initialized is set to true.
  // - In addition, other implementations may provide another non-OK status
  //   specific to their failure modes.
  absl::Status Initialize(InitTableIterator& iter);

  // Initializes the table from the given init table iterator. `serializer` may
  // specify how to serialize the table initializer, so that the table can be
  // serialized using its metadata (as opposed to serializing a handle to the
  // table).
  absl::Status Initialize(InitTableIterator& iter,
                          std::unique_ptr<InitializerSerializer> serializer);

  // Basic iterator to initialize lookup tables.
  // It yields a sequence of pairs of `keys()` and `values()` Tensors, so that
  // the consumer may insert key-value pairs in batches.
  //
  // Then the iterator is exhausted, valid returns false and status returns
  // Status::OutOfRange.
  //
  // This class is Thread-unsafe.
  class InitTableIterator {
   public:
    InitTableIterator() {}

    virtual ~InitTableIterator() {}

    // Prepares the next batch of key and value tensors.
    virtual void Next() = 0;

    // Returns true if keys and values point to valid tensors.
    virtual bool Valid() const = 0;

    // Returns a tensor that contains the current batch of 'key' values.
    virtual const Tensor& keys() const = 0;

    // Returns a tensor that contains the current batch of 'value' values.
    virtual const Tensor& values() const = 0;

    // Returns an error if one has occurred, otherwise returns Status::OK.
    virtual absl::Status status() const = 0;

    // Returns the total number of elements that the iterator will produce.
    // It might return -1 in case of error.
    virtual int64_t total_size() const = 0;

   private:
    InitTableIterator(const InitTableIterator&) = delete;
    void operator=(const InitTableIterator&) = delete;
  };

  InitializableLookupTable* GetInitializableLookupTable() override {
    return this;
  }

  // Logic specifying how to represent an initializer as a GraphDef, so that a
  // lookup table can be serialized using its metadata (as opposed to
  // serializing the content of the table, or a handle to the table).
  class InitializerSerializer {
   public:
    // A function which builds a graph so that executing `*out` will initialize
    // `table`.
    using SerializeFn = std::function<absl::Status(GraphDefBuilder* builder,
                                                   Node* table, Node** out)>;
    // A function which performs any necessary cleanup for the serializer.
    using CleanupFn = std::function<void()>;

    // Wraps serialization logic that requires no cleanup.
    explicit InitializerSerializer(SerializeFn serialize)
        : serialize_(std::move(serialize)), cleanup_([] {}) {}

    // Wraps serialization logic along with a cleanup function. `cleanup` will
    // be run when the serializer is destroyed.
    explicit InitializerSerializer(SerializeFn serialize, CleanupFn cleanup)
        : serialize_(std::move(serialize)), cleanup_(std::move(cleanup)) {}

    ~InitializerSerializer() { cleanup_(); }

    // Builds a graph so that executing `*out` will initialize `table`.
    absl::Status AsGraphDef(GraphDefBuilder* builder, Node* table, Node** out) {
      return serialize_(builder, table, out);
    }

   private:
    SerializeFn serialize_;
    CleanupFn cleanup_;
  };

 protected:
  // Prepares and allocates the underlying data structure to store the given
  // number of expected elements.
  virtual absl::Status DoPrepare(size_t expected_num_elements) = 0;

  // Same as DoPrepare() but derived implementations might choose to skip
  // calling get_expected_num_elements if size is not needed for DoPrepare.
  virtual absl::Status DoLazyPrepare(
      std::function<int64_t(void)> get_expected_num_elements) {
    int64_t expected_num_elements = get_expected_num_elements();
    if (expected_num_elements < 0) {
      return errors::FailedPrecondition("Got negative expected_num_elements.");
    }
    return DoPrepare(expected_num_elements);
  }

  // Populates the table in batches given keys and values as tensors into the
  // underlying data structure.
  virtual absl::Status DoInsert(const Tensor& keys, const Tensor& values) = 0;

  // Performs the batch find operation on the underlying data structure.
  virtual absl::Status DoFind(const Tensor& keys, Tensor* values,
                              const Tensor& default_value) = 0;

  virtual absl::Status AreEntriesSame(const InitTableIterator& iter,
                                      bool* result);

  mutex mu_;

 protected:
  // When set, provides a mechanism for serializing the table initializer as
  // GraphDef.
  std::unique_ptr<InitializerSerializer> initializer_serializer_;

 private:
  std::atomic<bool> is_initialized_{false};
};

// Iterator to initialize tables given 'keys' and 'values' tensors.
//
// The two tensors are returned in the first iteration. It doesn't loop
// over each element of the tensor since insertions in the lookup table can
// process batches.
class KeyValueTensorIterator
    : public InitializableLookupTable::InitTableIterator {
 public:
  // keys and values are not owned by the iterator.
  explicit KeyValueTensorIterator(const Tensor* keys, const Tensor* values)
      : keys_(keys), values_(values), valid_(true), status_(absl::OkStatus()) {
    TensorShape key_shape = keys_->shape();
    if (!key_shape.IsSameSize(values_->shape())) {
      valid_ = false;
      status_ = errors::InvalidArgument(
          "keys and values should have the same dimension.",
          key_shape.DebugString(), " vs ", values_->shape().DebugString());
    }
    if (key_shape.num_elements() == 0) {
      valid_ = false;
      status_ =
          errors::InvalidArgument("keys and values cannot be empty tensors.");
    }
  }

  bool Valid() const override { return valid_; }

  void Next() override {
    valid_ = false;
    status_ = errors::OutOfRange("No more data.");
  }

  const Tensor& keys() const override { return *keys_; }

  const Tensor& values() const override { return *values_; }

  absl::Status status() const override { return status_; }

  int64_t total_size() const override {
    return keys_ == nullptr ? -1 : keys_->NumElements();
  }

 private:
  KeyValueTensorIterator(const KeyValueTensorIterator&) = delete;
  void operator=(const KeyValueTensorIterator&) = delete;

  const Tensor* keys_;    // Doesn't own it.
  const Tensor* values_;  // Doesn't own it.
  bool valid_;            // true if the iterator points to an existing range.
  absl::Status status_;
};

}  // namespace lookup
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_
