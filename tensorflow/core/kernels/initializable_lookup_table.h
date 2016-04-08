/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_
#define TENSORFLOW_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_

#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace lookup {

// Base class for lookup tables that require initialization.
class InitializableLookupTable : public LookupInterface {
 public:
  class InitTableIterator;

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
  Status Find(const Tensor& keys, Tensor* values,
              const Tensor& default_value) final;

  // Returns whether the table was initialized and is ready to serve lookups.
  bool is_initialized() const { return is_initialized_; }

  // Initializes the table from the given init table iterator.
  //
  // Atomically, this operation prepares the table, populates it with the given
  // iterator, and mark the table as initialized.
  //
  // Returns the following statuses:
  // - OK: when the initialization was successful.
  // - InvalidArgument: if any of the preconditions on the lookup key or value
  //   fails.
  // - FailedPrecondition: if the table is already initialized and
  //   fail_if_initialized is set to true.
  // - In addition, other implementations may provide another non-OK status
  //   specific to their failure modes.
  Status Initialize(InitTableIterator& iter);

  // Basic iterator to initialize lookup tables.
  // It yields a sequence of pairs of `keys()` and `values()` Tensors, so that
  // the consumer may insert key-value pairs in batches.
  //
  // Then the iterator is exhausted, valid returns false and status returns
  // Status::OutOfRange.
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
    virtual Status status() const = 0;

    // Returns the total number of elements that the iterator will produce.
    virtual int64 total_size() const = 0;

   private:
    TF_DISALLOW_COPY_AND_ASSIGN(InitTableIterator);
  };

  InitializableLookupTable* GetInitializableLookupTable() override {
    return this;
  }

 protected:
  // Prepares and allocates the underlying data structure to store the given
  // number of expected elements.
  virtual Status DoPrepare(size_t expected_num_elements) = 0;

  // Populates the table in batches given keys and values as tensors into the
  // underlying data structure.
  virtual Status DoInsert(const Tensor& keys, const Tensor& values) = 0;

  // Performs the batch find operation on the underlying data structure.
  virtual Status DoFind(const Tensor& keys, Tensor* values,
                        const Tensor& default_value) = 0;

  mutex mu_;
  bool is_initialized_ = false;
};

}  // namespace lookup
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_INITIALIZABLE_LOOKUP_TABLE_H_
